# Precompute index of similarities with reference library for db_search evaluation
# For every sample in query dataset find the most similar molecule in the reference library,
# and save its index, spectral similarity and SMILES similarity.
# It takes a lot of time => we utilize parallelization.


import argparse
import json
import pandas as pd
import numpy as np
from matchms.similarity import CosineGreedy, ModifiedCosine
from matchms import Spectrum
from tqdm import tqdm
from rdkit import Chem, DataStructs
import multiprocessing
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()

from utils.spectra_process_utils import get_fp_generator, get_fp_simil_function


def find_best_indexes_and_similarities(df_query, ref_spectra, ref_fps, fpgen, fp_simil_function, outfile_path, process_id=None):
    """find the best match (according to spectra similarity) for every sample in df_query
      and add its index, spectral similarity and SMILES similarity to the row.
      Write the row immediately to a new file.

      Parameters
      ----------
      df_query : pd.DataFrame
          dataframe with query samples
      ref_spectra : list[Spectrum]
          list of reference spectra
      ref_fps : list[rdkit.ExplicitBitVect]
          list of reference fingerprints
      fpgen : rdkit.Chem.rdFingerprintGenerator.FingeprintGenerator64
          fingerprint generator
      fp_simil_function : function
          fction taking two fingerprints and returning their similarity
      num_db_candidates : int
          number of candidates to return when searching for the closest molecule in the database
      outfile_path : str
          path to the output file
      process_id : int, optional
          id of the process to print when multiprocessing, by default None"""

    if process_id is not None:
        print(f"process {process_id} started")

    outfile = open(outfile_path, "w+")
    cosine_greedy = CosineGreedy()

    for _, query_row in tqdm(df_query.iterrows(), desc="outer loop"):
        query_spec = Spectrum(mz=np.array(query_row.mz),
                                intensities=np.array(query_row.intensity),
                                metadata_harmonization=False)
        query_fp = fpgen.GetFingerprint(Chem.MolFromSmiles(query_row.smiles))
        best_spec_simil = 0
        best_index = 0
        best_spec_simils = []
        best_indexes = []
        best_smiles_simils = []
        for index, ref_spec in enumerate(ref_spectra):
            spec_score = float(cosine_greedy.pair(query_spec, ref_spec)["score"])
            if spec_score > best_spec_simil:
                best_spec_simil = spec_score
                best_index = index
        smiles_score = fp_simil_function(query_fp, ref_fps[best_index])

        best_spec_simils.append(best_spec_simil)
        best_indexes.append(best_index)
        best_smiles_simils.append(smiles_score)

        # update row and write it to file
        query_row["index_of_closest"] = best_index
        query_row["spectra_sim_of_closest"] = best_spec_simil
        query_row[f"smiles_sim_of_closest_{args.fingerprint_type}_{args.fp_simil_function}"] = smiles_score
        query_row = dict(query_row)
        outfile.write(json.dumps(query_row))
        outfile.write("\n")

    outfile.close()

    if process_id is not None:
        print(f"process {process_id} finished")


def db_search_preprocess_mp(df_reference, df_query, outfile_path, tmp_folder_path, fpgen, fp_simil_function, num_processes=1):
    # create fingerprints and spectra fo reference dataset
    ref_spectra = [Spectrum(mz=np.array(ref_row.mz),
                            intensities=np.array(ref_row.intensity),
                            metadata_harmonization=False)
                            for _, ref_row in tqdm(df_reference.iterrows(), desc="precomputing ref_spectra")]
    ref_fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(ref_row.smiles))
               for _, ref_row in tqdm(df_reference.iterrows(), desc="precomputing ref_fps")]
    assert len(ref_spectra) == len(df_reference), "ref_spectra and df have different lengths"
    assert len(ref_fps) == len(df_reference), "ref_fps and df have different lengths"

    # split data
    idxs = np.array_split(np.arange(len(df_query)), num_processes)

    # create file names
    tmp_paths = [tmp_folder_path / f"{outfile_path.stem}_{i}{outfile_path.suffix}" for i in range(num_processes)]

    # run multiprocess
    print("STARTING MULTIPROCESSING")
    processes = {}
    for i in range(num_processes):
        processes[f"process{i}"] = multiprocessing.Process(target=find_best_indexes_and_similarities,
                                                           args=(df_query.iloc[idxs[i]],
                                                                 ref_spectra,
                                                                 ref_fps,
                                                                 fpgen,
                                                                 fp_simil_function,
                                                                 tmp_paths[i]),
                                                           kwargs=dict(process_id=i))
    for process in processes.values():
        process.start()
    for process in processes.values():
        process.join()

    # concat files
    print("CONCATENATING FILES")
    with open(outfile_path, "w+") as outfile:
        for i in range(num_processes):
            with open(tmp_paths[i], "r") as f:
                    for line in f:
                        outfile.write(line)


def compute_simil_of_closest(row, df_reference, fpgen, sim_function) -> list:
    gt_smiles = row["smiles"]
    closest_smiless = df_reference["smiles"][row[f"index_of_closest"]]
    closest_fp = fpgen.GetFingerprint(Chem.MolFromSmiles(closest_smiless))
    gt_fp = fpgen.GetFingerprint(Chem.MolFromSmiles(gt_smiles))
    simil_of_closest = sim_function(gt_fp, closest_fp)
    simil_of_closest = np.array(simil_of_closest).round(5)
    return simil_of_closest.tolist()


def add_simils_to_existing_db_index_df(df_precomputed: pd.DataFrame, df_reference: pd.DataFrame, fp_type, fp_simil_type) -> pd.DataFrame:
    fpgen = get_fp_generator(fp_type)
    sim_function = get_fp_simil_function(fp_simil_type)
    df_enriched = df_precomputed.copy()
    simils = df_enriched.progress_apply(lambda row: compute_simil_of_closest(row, df_reference, fpgen, sim_function), axis=1)
    df_enriched[f"smiles_sim_of_closest_{fp_type}_{fp_simil_type}"] = simils
    return df_enriched


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, required=True, help="path to the reference (database) dataset")
    parser.add_argument("--query", type=str, required=True, help="path to the query dataset")
    parser.add_argument("--outfile", type=str, required=True, help="path to the output file")
    parser.add_argument("--num_processes", type=int, default=1, help="number of processes to use")
    parser.add_argument("--fingerprint_type", type=str, required=True, help="daylight or morgan - type of fingerpirnts to do the database search and compute simil of closest")
    parser.add_argument("--fp_simil_function", type=str, required=True, help="cosine or tanimoto - FP simil function to compute similarity of the closest candidates")
    parser.add_argument("--num_db_candidates", type=int, default=1, help="number of candidates to return when searching for the closest molecule in the database")
    args = parser.parse_args()

    # load data
    print("LOADING DATA")
    df_reference = pd.read_json(args.reference, lines=True, orient="records")
    df_query = pd.read_json(args.query, lines=True, orient="records")

    # if precomputed available, check for the appropriate column, if it's present, throw an error, if not, print a message and add the new fp-simil column
    precomputed_changed = False
    if Path(args.outfile).exists():
        print("The   outfile   already exists -> LOADING PRECOMPUTED DATA")
        df_precomputed = pd.read_json(args.outfile, lines=True, orient="records")
        if f"smiles_sim_of_closest_{args.fingerprint_type}_{args.fp_simil_function}" in df_precomputed.columns:
            raise ValueError(f"The precomputed file already contains the smiles_sim_of_closest_{args.fingerprint_type}_{args.fp_simil_function} column. Please choose a different fp-simil combination or manually remove the existing column to recompute it.")
        else:
            print(f"The precomputed file does not contain the smiles_sim_of_closest_{args.fingerprint_type}_{args.fp_simil_function} column yet. We'll add it.")
            df_precomputed_enriched = add_simils_to_existing_db_index_df(df_precomputed, df_reference, args.fingerprint_type, args.fp_simil_function)
            print("SAVING the updated query dataset to the original location.")
            df_precomputed_enriched.to_json(args.outfile, orient="records", lines=True)
            exit(0)


    # set FP generator and simil functions
    fpgen = get_fp_generator(args.fingerprint_type)
    fp_simil_function = get_fp_simil_function(args.fp_simil_function)
    print(f">> Setting up   {args.fingerprint_type}   fingerprint generator.")
    print(f">> Setting up   {args.fp_simil_function}   similarity function.")
    print(f">> Num workers:   {args.num_processes}")


    # create dirs
    outfile_path = Path(args.outfile)
    query_path = Path(args.query)
    outfile_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_folder_path = outfile_path.parent / f"tmp_{query_path.stem}"
    tmp_folder_path.mkdir(parents=True, exist_ok=True)


    # run precompute
    db_search_preprocess_mp(df_reference,
                         df_query,
                         outfile_path,
                         tmp_folder_path,
                         fpgen,
                         fp_simil_function,
                         num_processes=args.num_processes)