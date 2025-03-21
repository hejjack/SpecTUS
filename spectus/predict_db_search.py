import sys
import shutil
import time
from pathlib import Path
import typer
import yaml
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import multiprocessing
import heapq
from glob import glob

from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
from matchms.similarity import ModifiedCosine, CosineGreedy
from matchms import Spectrum

from utils.spectra_process_utils import get_fp_generator, get_fp_simil_function
from utils.data_utils import filter_datapoints
from utils.general_utils import timestamp_to_readable, hours_minutes_seconds
from utils.general_utils import timestamp_to_readable, hours_minutes_seconds
from predict import open_files

RDLogger.DisableLog('rdApp.*')

app = typer.Typer(pretty_exceptions_enable=False)


def find_run_with_more_candidates(output_folder, data_range, config):
    """Check if there is a prediction file with more candidates"""
    db_search_name = f"db_search_{config['general']['ranking_function']}"
    range_str = (f"{data_range}") if data_range else "full"
    dataset_name = config["dataset"]["dataset_name"]
    data_split = config["dataset"]["data_split"]
    additional_info = config["general"]["additional_naming_info"]
    potential_folders = glob(str(output_folder / db_search_name / dataset_name / f"*{data_split}_{range_str}*{additional_info}"))

    for folder in potential_folders:
        log_file = Path(folder) / "log_file.yaml"
        if not log_file.exists():
            continue

        with open(log_file, "r", encoding="utf-8") as f:
            logs = yaml.safe_load(f)

        if logs["general"]["num_candidates"] > config["general"]["num_candidates"]:
            if logs.get("finished_time_utc", None):
                if logs["filtering_args"] == config["filtering_args"]:
                    print(f"Found a finished prediction run with the same filtering and more predicted candidates: {folder}")
                return folder
    print(f"No finished prediction run with more candidates found")
    return None


def extract_candidates(old_predictions_file, new_predictions_file, num_candidates):
    with open(old_predictions_file, "r") as old_f:
        with open(new_predictions_file, "w") as new_f:
            for line in tqdm(old_f):
                old_preds = json.loads(line)
                sorted_candidates = sorted(old_preds.items(), key=lambda x: x[1], reverse=True)
                new_candidates = {k: v for k, v in sorted_candidates[:num_candidates]}
                new_f.write(json.dumps(new_candidates) + "\n")


def find_candidates_in_database_for_one_query(query_row, df_references, ranking_function, similarity_measure, num_candidates):
    if ranking_function in ["hss", "sss"]:
        query = Spectrum(mz=np.array(query_row.mz),
                        intensities=np.array(query_row.intensity),
                        metadata={"precursor_mz": round(ExactMolWt(Chem.MolFromSmiles(query_row.smiles)))},
                        metadata_harmonization=False)
    else:
        query = query_row["fp"]
    candidates = []

    for i, ref_row in df_references.iterrows():
        if ranking_function in ["hss", "sss"]:
            similarity_score = similarity_measure(query, ref_row["spectrum"])
        else:
            similarity_score = similarity_measure(query, ref_row["fp"])

        if len(candidates) < num_candidates:
            heapq.heappush(candidates, (similarity_score, ref_row["smiles"]))
        else:
            if similarity_score > candidates[0][0]:
                heapq.heappushpop(candidates, (similarity_score, ref_row["smiles"]))

    return candidates


def find_candidates_in_database_for_all_queries(df_queries,
                                                df_references,
                                                ranking_function,
                                                num_candidates,
                                                outfile_path,
                                                process_id=0):
    """Find the top num_candidates most similar spectra in the database to each query spectrum"""

    if ranking_function == "hss":
        similarity_measure = lambda x,y: ModifiedCosine().pair(x, y)["score"].round(5).item()
        ref_spectra = [Spectrum(mz=np.array(ref_row.mz),
                        intensities=np.array(ref_row.intensity),
                        metadata={"precursor_mz": round(ExactMolWt(Chem.MolFromSmiles(ref_row.smiles)))},    # rounded exact mass came out as the most reliable
                        metadata_harmonization=False)
                        for _, ref_row in tqdm(df_references.iterrows(), desc="precomputing ref_spectra")]
        df_references["spectrum"] = ref_spectra
    elif ranking_function == "sss":
        similarity_measure = lambda x,y: CosineGreedy().pair(x, y)["score"].round(5).item()
        ref_spectra = [Spectrum(mz=np.array(ref_row.mz),
                        intensities=np.array(ref_row.intensity),
                        metadata_harmonization=False)
                        for _, ref_row in tqdm(df_references.iterrows(), desc="precomputing ref_spectra")]

        df_references["spectrum"] = ref_spectra
    elif ranking_function == "morgan_tanimoto":
        similarity_measure = lambda x, y: round(get_fp_simil_function("tanimoto")(x, y), 5)
        fpgen = get_fp_generator("morgan")

        ref_fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in tqdm(df_references["smiles"], desc="Reference data fps precomputing")]
        query_fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in tqdm(df_queries["smiles"], desc="Query data fps precomputing")]

        df_references["fp"], df_queries["fp"] = ref_fps, query_fps
    else:
        raise ValueError(f"Unknown similarity type: {ranking_function}")

    # go through all queries
    outfile = open(outfile_path, "w+")
    for i, query_row in tqdm(df_queries.iterrows(), desc=f"Process {process_id}"):
        candidates = find_candidates_in_database_for_one_query(query_row, df_references, ranking_function, similarity_measure, num_candidates)
        candidates_dict = {smiles: score for score, smiles in candidates}
        outfile.write(json.dumps(candidates_dict) + "\n")

    outfile.close()


def parallel_db_search(df_references,
                       df_queries,
                       config,
                       outfile_path,
                       num_workers):
    """Extract queries and references based on the ranking function and run the database search in parallel"""

    # split data
    idxs = np.array_split(np.arange(len(df_queries)), num_workers)

    # create file names
    tmp_folder_path = outfile_path.parent / "tmp"
    tmp_folder_path.mkdir(parents=True, exist_ok=True)
    tmp_paths = [tmp_folder_path / f"{outfile_path.stem}_{i}{outfile_path.suffix}" for i in range(num_workers)]

    # run multiprocess
    print("STARTING MULTIPROCESSING")
    processes = {}
    for i in range(num_workers):
        processes[f"process{i}"] = multiprocessing.Process(target=find_candidates_in_database_for_all_queries,
                                                           args=(df_queries.iloc[idxs[i]],
                                                                 df_references,
                                                                 config["general"]["ranking_function"],
                                                                 config["general"]["num_candidates"],
                                                                 tmp_paths[i]),
                                                           kwargs=dict(process_id=i))
    for process in processes.values():
        process.start()
    for process in processes.values():
        process.join()

    # concat files
    print("CONCATENATING FILES")
    with open(outfile_path, "w+") as outfile:
        for i in range(num_workers):
            with open(tmp_paths[i], "r") as f:
                for line in f:
                    outfile.write(line)

    # force remove tmp folder
    print("REMOVING TMP FILES")
    shutil.rmtree(str(tmp_folder_path))


@app.command()
def main(
    output_folder: Path = typer.Option(..., dir_okay=True, file_okay=True, exists=False, writable=True, help="Path to the folder where the predictions will be saved"),
    config_file: Path = typer.Option(..., help="Path to the config file"),
    data_range: str = typer.Option("", help="Range of data to generate predictions for. Format: <start>:<end>"),
    num_workers: int = typer.Option(1, help="Number of processes to use for the database search"),
) -> None:

    # load config
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    general_config = config["general"]
    dataset_config = config["dataset"]
    filtering_args = deepcopy(config["filtering_args"]) # deepcopy so tokenizer is not saved to logs

    config["command"] = " ".join(sys.argv)
    start_time = time.time()
    config["start_loading_time"] = timestamp_to_readable(start_time)
    additional_info = general_config["additional_naming_info"]

    # set output files
    db_search_name = f"db_search_{general_config['ranking_function']}"
    placeholder_path = Path(db_search_name) / "placeholder"  # in order to comply with the open_files function
    additional_info = f"{general_config['num_candidates']}cand" + additional_info
    log_file, outfile = open_files(output_folder, placeholder_path, dataset_config, data_range, additional_info)
    outfile_path = Path(outfile.name)
    outfile.close() # we dont need it yet

    yaml.dump(config, log_file)

    # check if there is a prediction file with more candidates
    run_with_more_candidates = find_run_with_more_candidates(output_folder, data_range, config)
    if run_with_more_candidates:
        print(f"Skipping prediction, using the existing prediction file to extract the candidates: {run_with_more_candidates}")
        start_generation_time = time.time()
        extract_candidates(run_with_more_candidates + "/predictions.jsonl", outfile_path, general_config["num_candidates"])
    else:
        # load data
        print("LOADING DATA")
        df_reference = pd.read_json(dataset_config["reference_data"], lines=True, orient="records")
        df_all_queries = pd.read_json(dataset_config["query_data"], lines=True, orient="records")
        tqdm.pandas(desc="Filtering queries")
        df_queries = df_all_queries[df_all_queries.progress_apply(lambda row: filter_datapoints(row, filtering_args), axis=1)]

        # load range
        if data_range:
            data_range_min, data_range_max = list(map(int, data_range.split(":")))
            df_queries = df_queries.iloc[data_range_min:data_range_max]
        else:
            data_range_min, data_range_max = None, None

        # Start generating
        start_generation_time = time.time()

        parallel_db_search(df_reference, df_queries, config, outfile_path, num_workers)

    finished_time = time.time()

    log_config = {
        "finished_time_utc": timestamp_to_readable(finished_time),
        "generation_time": f"{hours_minutes_seconds(finished_time - start_generation_time)}",
        "wall_time_utc": f"{hours_minutes_seconds(finished_time - start_time)}"}
    yaml.dump(log_config, log_file)
    log_file.close()


if __name__ == "__main__":
    app()
