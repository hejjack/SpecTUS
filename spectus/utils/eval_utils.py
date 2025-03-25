from __future__ import annotations
import pandas as pd
import json
from pathlib import Path
import yaml
from typing import Optional
from tqdm import tqdm
tqdm.pandas()

from  utils.data_utils import build_single_datapipe, filter_datapoints, range_filter


def filter_predictions(old_predictions_path, original_data_path, old_config, new_config, save_path=None):
    """
    Function that takes predictions, original data, old and new config and filters the predicitons according to
    the new config - be careful, the new predictions will always be a subset of the old predictions. If the new
    config is not a subset of the old config, the data will be incomplete.

    Parameters
    ----------
    old_predictions_path: str
        Path to the old predictions.
    original_data_path: str
        Path to the original data.
    old_config: Dict[str, Any]
        Old preprocessing config.
    new_config: Dict[str, Any]
        New preprocessing config.
    save_path: str
        Path to save the new predictions.

    Returns
    -------
    List[Dict[str: float]]
        New predictions.
    """
    predictions_pipe = build_single_datapipe(old_predictions_path, False)

    # Load original data and filter it with old preprocessing config
    print(">>> Loading original data")
    original_data = pd.read_json(original_data_path, lines=True, orient="records")
    print("Original data len: ", len(original_data))

    # Filter original data with old preprocessing config
    old_filter_mask = original_data.progress_apply(lambda row: filter_datapoints(row, old_config), axis=1)
    old_filtered_original_data = original_data[old_filter_mask]
    print("Filtered original data len: ", len(old_filtered_original_data))

    # Combine filtered original data with loaded predictions
    old_filtered_original_data["predictions"] = list(iter(predictions_pipe))

    # Filter everything based on the new preprocessing config
    new_filter_mask = old_filtered_original_data.progress_apply(lambda row: filter_datapoints(row, new_config), axis=1)
    new_filtered_combined_data = old_filtered_original_data[new_filter_mask]
    print("Filtered combined data len: ", len(new_filtered_combined_data))

    new_predictions = new_filtered_combined_data["predictions"]

    # Potentially save the new predictions
    if save_path is not None:
        print(">>> Saving new predicitons")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as pred_file:
            pred_file.write('\n'.join(json.dumps(pred) for pred in new_predictions))

        with open(Path(save_path).parent / "log_file.yaml", "w") as log_file:
            yaml.dump({"created_by": "utils.data_utils/filter_predictions",
                            "origin_predictions": old_predictions_path,
                            "origin_data": original_data_path,
                            "old_config": old_config,
                            "new_config": new_config,
                            }, log_file)



    return new_predictions


def load_labels_from_dataset(dataset_path: Path,
                             data_range: range,
                             do_db_search: bool = False,
                             fp_type: str | None = None,
                             simil_func: str | None = None) -> tuple:
    """Load the labels from the dataset without on the fly filtering

    Parameters
    ----------
    dataset_path: Path
        Path to the dataset.
    data_range: range
        Range of the data to load.
    do_db_search: bool
        Whether we want to do db_search evaluation and thus load the fingerprint similarities too.
        For this you need a precomputed index of the closest spectrum from the reference library.
    fp_type: str
        Fingerprint type. In case of db_search evaluation specifies which metric to choose.
    simil_func: str
        Similarity function. In case of db_search evaluation specifies which metric to choose.

    Returns
    -------
    tuple
        Tuple of (1) the labels (SMILES strings) and (2) if do_db_search is True it returns the precomputed fingerprint similarities of the closest spectrum from the reference library"""
    df = pd.read_json(dataset_path, lines=True, orient="records")
    if not data_range:
        data_range = range(len(df))
    df_ranged = df.iloc[data_range.start:data_range.stop] # TODO
    simles_list = df_ranged["smiles"].tolist()

    smiles_sim_of_closest = None
    if do_db_search:
        assert f"smiles_sim_of_closest_{fp_type}_{simil_func}" in df_ranged.columns, "smiles_sim_of_closest column not found in labels, not able to do DENOVO evaluation"
        smiles_sim_of_closest = df_ranged[f"smiles_sim_of_closest_{fp_type}_{simil_func}"].tolist()

    return iter(simles_list), smiles_sim_of_closest


def load_labels_to_datapipe(dataset_path: Path,
                            data_range: range = range(0, 0),
                            do_db_search: bool = False,
                            fp_type: str | None = None,
                            simil_func: str | None = None,
                            filtering_args: dict = {"max_num_peaks": 300, "max_mz": 500, "max_mol_repr_len": 100, "mol_repr": "smiles"}) -> tuple:
    """Load the labels from the dataset with on the fly filtering

    Parameters
    ----------
    dataset_path: Path
        Path to the dataset.
    data_range: range
        Range of the data to load.
    do_db_search: bool
        Whether we want to do db_search evaluation and thus load the fingerprint similarities too.
        For this you need a precomputed index of the closest spectrum from the reference library.
    fp_type: str
        Fingerprint type. In case of db_search evaluation specifies which metric to choose.
    simil_func: str
        Similarity function. In case of db_search evaluation specifies which metric to choose.

    Returns
    -------
    tuple
        Tuple of (1) the labels (SMILES strings) and (2) if do_db_search is True it returns the precomputed fingerprint similarities of the closest spectrum from the reference library
    """

    assert set(["max_num_peaks", "max_mz", "max_mol_repr_len", "mol_repr"]).issubset(filtering_args.keys()), "filtering_args has to contain max_num_peaks, max_mz, max_mol_repr_len and mol_repr"

    datapipe = build_single_datapipe(str(dataset_path), shuffle=False)

    datapipe = datapipe.filter(filter_fn=lambda d: filter_datapoints(d, filtering_args)) # filter out too long data and stuff
    if data_range:
        datapipe = datapipe.header(data_range.stop)  # speeding up for slices near to the beginning
        datapipe = datapipe.filter(filter_fn=range_filter(data_range))  # actual slicing

    smiles_sim_of_closest_datapipe = None
    if do_db_search:
        assert fp_type is not None and simil_func is not None, "fp_type and simil_func have to be specified for db_search evaluation"
        smiles_datapipe, smiles_sim_of_closest_datapipe = datapipe.fork(num_instances=2, buffer_size=1e6,)  # 'copy' (fork) the datapipe into two 'new'
        smiles_sim_of_closest_datapipe = iter(smiles_sim_of_closest_datapipe.map(lambda d: d[f"smiles_sim_of_closest_{fp_type}_{simil_func}"]))
    else:
        smiles_datapipe = datapipe
    smiles_datapipe = iter(smiles_datapipe.map(lambda d: d["smiles"]))

    return smiles_datapipe, smiles_sim_of_closest_datapipe if do_db_search else None