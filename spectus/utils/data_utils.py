# Class representing custom dataset inheriting from torch Dataset
# crucial part: __getitem__ changes data to tensor (could not be
# stored as jsonl)

import json
import torch
from torch.utils.data import Dataset
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper, SampleMultiplexer, Concater, Header
from typing import Callable, Dict, Sized, Union, Any, Optional, List, Tuple, Iterator, Hashable, TypeVar
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
import selfies as sf
from tqdm import tqdm
from utils.spectra_process_utils import mol_repr_to_labels, cumsum_filtering, remove_stereochemistry_and_canonicalize

tqdm.pandas()
T_co = TypeVar("T_co", covariant=True)

class SpectroDataset(Dataset):
    # deprecated, using dynamically loaded (preprocessed) datapipes instead
    def __init__(self, df_or_pth_jsonl, inference_mode=False, restrict_intensities=False):
        """
        Parameters
        ----------
        df_or_pth_jsonl: pd.DataFrame
            dataframe with prepared data or a path to DFs stored as jsonl (prepared with run_prepare_data.sh)
        inference_mode: bool
            evaluation mode where we don't have the labels at hand
        restrict_intensities: bool
            if True, we restrict the intensities channel (we don't provide position_ids to the model)
        """
        if isinstance(df_or_pth_jsonl, pd.DataFrame):
            self.data = df_or_pth_jsonl
        else:
            self.data = pd.read_json(df_or_pth_jsonl, orient="records", lines=True)
        self.inference_mode = inference_mode
        self.restrict_intensities = restrict_intensities

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        out = {"input_ids": row["input_ids"]}

        if not self.restrict_intensities:
            out["position_ids"] = row["position_ids"]

        if not self.inference_mode:
            out["labels"] = row["labels"]

        return out


class SpectroDataCollator:
    """
    A class that from a list of dataset elements forms a batch
    """

    def __init__(self, inference_mode=False, restrict_intensities=False, keep_all_columns=False):
        self.inference_mode = inference_mode
        self.restrict_intensities = restrict_intensities
        self.keep_all_columns = keep_all_columns

    def __call__(
        self, batch: List[Dict[str, list]]
    ) -> Dict[str, torch.Tensor]:
        return self._collate_batch(batch)

    def _collate_batch(self, batch: List[Dict[str, list]]) -> Dict[str, torch.Tensor]:
        """Collate `examples` into a batch"""
        longest_encoder_sequence = max(len(e["input_ids"]) for e in batch)
        inputs = torch.tensor([e["input_ids"] + [0] * (longest_encoder_sequence - len(e["input_ids"])) for e in batch])   # adding padding
        out = {"input_ids": inputs}

        if not self.restrict_intensities: # add position_ids too
            out["position_ids"] = torch.tensor([e["position_ids"] + [0] * (longest_encoder_sequence - len(e["position_ids"])) for e in batch]) # adding padding

        if not self.inference_mode: # add labels too
            longest_decoder_sequence = max(len(e["labels"]) for e in batch)
            labels = torch.tensor([e["labels"] + [-100] * (longest_decoder_sequence - len(e["labels"])) for e in batch]) # adding padding
            out["attention_mask"] = (inputs != 0).int()
            out["decoder_attention_mask"] = (labels != -100).int()
            out["labels"] = labels

        if self.keep_all_columns:
            for k in batch[0].keys():
                if k not in out:
                    out[k] = [e[k] for e in batch]
        return out


def position_ids_creator(intensities, log_base=None, log_shift=None, do_log_binning=True, linear_bin_decimals=None):
    """create position ids for the Spectro Transformer model"""

    assert not do_log_binning or (log_base is not None and log_shift is not None), "log_base and log_shift must be provided for log binning"
    assert do_log_binning or linear_bin_decimals is not None, "linear_bin_decimals must be provided for linear binning"

    x = np.array(intensities) / max(intensities)  # normalize

    if do_log_binning:
        x = (np.log(x)/np.log(log_base)).astype(int) + log_shift # log binning
        x = x * (x > 0) # all the small intensities are mapped to 0
    else:
        x = np.around(x, decimals=linear_bin_decimals) * 10**linear_bin_decimals  # intensity rounded to 2 decimal places

    return list(x.astype("int32"))


def preprocess_datapoint(datadict, source_token, preprocess_args):
    """
    Preprocess a single datapoint.

    Parameters
    ----------
    datapoint: Dict[str, Any]
        A single datapoint - dictionary containing mzs, intensities and SMILES.
    preprocess_args: Dict[str, Any]
        tokenizer, restrict_intensities, inference_mode, log_base, log_shift, mol_repr, keep_all_columns (optional), max_cumsum (optional) ...

    Returns
    -------
    Dict[str, Any]
        Preprocessed datapoint - dictionary containing input_ids, ?position_ids? and ?labels?.
    """
    mzs, intensities = datadict.pop("mz"), datadict.pop("intensity")

    # remove zero peak (sometimes in NEIMS data)
    if mzs[0] == 0:
        mzs = mzs[1:]
        intensities = intensities[1:]
        # print("Warning: zero peak occured => removed")

    if preprocess_args.get("max_cumsum", None) is not None:
        mzs, intensities = cumsum_filtering(mzs, intensities, preprocess_args["max_cumsum"])

    out = {"input_ids": [round(mz) for mz in mzs]}

    if not preprocess_args.get("restrict_intensities"):
        out["position_ids"] = position_ids_creator(intensities,
                                                   preprocess_args.get("log_base", None),
                                                   preprocess_args.get("log_shift", None),
                                                   do_log_binning=preprocess_args.get("do_log_binning", True),
                                                   linear_bin_decimals=preprocess_args.get("linear_bin_decimals", None))

    if not preprocess_args["inference_mode"]:
        smiles = datadict.pop("smiles")
        canon_mol_repr = remove_stereochemistry_and_canonicalize(smiles)
        assert canon_mol_repr is not None, f"Corrupted SMILES: {smiles} not filtered out!"
        if preprocess_args["mol_repr"] == "selfies":  # if selfies, encode it
            canon_mol_repr = sf.encoder(canon_mol_repr)  # encode smiles to selfies
        out["mol_repr"] = canon_mol_repr
        source_id = preprocess_args["tokenizer"].encode(source_token)[0]
        out["labels"] = mol_repr_to_labels(canon_mol_repr, preprocess_args["tokenizer"], source_id)

    if preprocess_args.get("keep_all_columns", False):
        out.update(datadict)

    return out


def filter_datapoints(datadict, preprocess_args) -> bool:
    """
    Filter out datapoints that are too long.

    Parameters
    ----------
    datapoint: Dict[str, Any]
        A single datapoint - dictionary containing mzs, intensities and SMILES.
    preprocess_args: Dict[str, Any]
        max_num_peaks, max_mz, max_mol_repr_len, mol_repr, max_cumsum (optional) - parameters for filtering.

    Returns
    -------
    bool
        Whether the datapoint should be kept.
    """
    # # canonicalization + possible selfies transformation
    canon_mol_repr = remove_stereochemistry_and_canonicalize(datadict["smiles"])

    # filter corrupted
    if canon_mol_repr is None:
        # print(f"datapoint_out: Corrupted SMILES: {datadict['smiles']}")
        return False
    else:
        canon_mol_repr = canon_mol_repr.strip() # often is a blank space at the beginning
        # no simles filtering
        if canon_mol_repr == "":
            # print(f"datapoint_out: Corrupted SMILES: {datadict['smiles']}")
            return False
        # long simles filtering
        elif len(canon_mol_repr) > preprocess_args["max_mol_repr_len"]:
            # print(f"datapoint_out: Too long SMILES: {canon_mol_repr}")
            return False

    # if selfies, encode it
    if preprocess_args["mol_repr"] == "selfies" and canon_mol_repr is not None:
        try:
            canon_mol_repr = sf.encoder(canon_mol_repr)        # TODO?? try block?
        except:
            # print(f"datapoint_out: Corrupted SMILES: {datadict['smiles']}")
            return False

    # filter high MZ
    if max(datadict["mz"]) > preprocess_args["max_mz"]:
        # print(f"datapoint_out: Too high MZ: {max(datadict['mz'])}")
        return False

    # filter little peaks so it doesn't get kicked out
    if preprocess_args.get("max_cumsum", None) is not None:
        mz, _ = cumsum_filtering(datadict["mz"], datadict["intensity"], preprocess_args["max_cumsum"])
    else:
        mz, _ = datadict["mz"], datadict["intensity"]

    # filter long spectra
    if len(mz) > preprocess_args["max_num_peaks"]:
        # print(f"datapoint_out: Too long spectra: {len(mz)}")
        return False

    return True


def range_filter(data_range: range) -> Callable[[Any], bool]:
   """Filter function for the datapipe based on the range.
   usage: datapipe.filter(filter_fn=range_filter(range(100)))"""

   count = data_range.start
   def f(_: Any) -> bool:
        nonlocal count
        count += 1
        return count <= data_range.stop
   return f


def build_single_datapipe(json_file: str,
                          shuffle: bool,
                          buffer_size: Optional[int] = None,
                          limit: Optional[int] = None,
                          source_token: Optional[str] = None,
                          preprocess_args: Optional[Dict[str, Any]] = None,
                        ):
    """
    Build a single datapipe from a json file.

    Parameters
    ----------
    json_file: str
        Path to the jsonl data file.
    shuffle: bool
        Whether to shuffle the dataset.
    buffer_size: int
        Buffer size for shuffling.
    limit: int
        Limit the number of datapoints (0, limit).
    source_token: str
        Source token for the tokenizer.
    preprocess_args: Dict[str, Any]
        Preprocessing and filtering arguments.
    """

    if preprocess_args:
        assert source_token is not None, "source_token must be provided if preprocess_args are provided (and thus preprocessing is required)"

    datapipe = IterableWrapper([json_file])
    datapipe = datapipe.open_files(mode='rt')
    datapipe = datapipe.readlines()
    if buffer_size and shuffle:
        print(f"shuffling {json_file} with buffer_size={buffer_size}")
        datapipe = datapipe.shuffle(buffer_size=buffer_size)
    elif buffer_size or shuffle:
        warnings.warn("SHUFFLE and its buffer_size are IGNORED if either not specified")
    datapipe = datapipe.sharding_filter()  # after shuffle, before expensive operations
    if limit is not None:
        datapipe = datapipe.header(limit)
    if preprocess_args:
        datapipe = datapipe.map(lambda x: json.loads(x[1]))
        datapipe = datapipe.filter(filter_fn=lambda d: filter_datapoints(d, preprocess_args)) # filter out too long data and stuff
        datapipe = datapipe.map(lambda d: preprocess_datapoint(d, source_token, preprocess_args))
    else:
        datapipe = datapipe.map(lambda x: json.loads(x[1]))
    return datapipe


def build_datapipe_mixture(datapipes: List[IterDataPipe],
                           weights: Union[Tuple[float], None],
                           concat: bool = False,
                           seed: Optional[int] = 42):
    """
    Build a datapipe that samples from a mixture of datapipes.

    Parameters
    ----------
    datapipes: List[IterDataPipe]
        List of datapipes to sample from.
    weights: List[float]
        List of weights for each datapipe.
    concat: bool
        Whether to concatenate the datapipes (if False, the pipes will be interleaved).
    """
    if concat:
        return Concater(*datapipes) # type: ignore
    else:
        assert weights is not None, "weights must be provided if concat=False"
        assert len(datapipes) == len(weights)

        # filter zero weight datasets
        nonzero_datapipes = [(pipe, weight) for pipe, weight in zip(datapipes, weights) if weight]
        if len(nonzero_datapipes)==0:
            raise ValueError("All train weights are zero! Please provide at least one non-zero weight.")
        print("Number of non-zero weight datasets: ", len(nonzero_datapipes))

        return SampleMultiplexer({
            pipe.cycle(): weight for pipe, weight in nonzero_datapipes
        }, seed=seed)


def load_all_datapipes(data_args: Dict[str, Any], preprocess_args: Optional[Dict[str, Any]] = None) -> Dict[str, IterDataPipe]:
    """
    Load all datapipes from a dict of json files.

    Parameters
    ----------
    data_args: Dict[str,Dict]
        Dictionary containing datasets' paths, weights and other config info (like shuffle and buffer_size       Buffer size for shuffling.

    Returns
    -------
    Dict[str, IterDataPipe]
        Dictionary with train and valid datapipes.
        {"train": train_datapipe,
         "valid":
             {"dataset1": valid_datapipe1,
             "dataset2": valid_datapipe2}
        }
    """
    seed = data_args["data_seed"] if data_args.get("data_seed") else 42
    buffer_size = data_args.get("buffer_size", None)
    shuffle_train = data_args["shuffle_train"]
    datapipes = {}

    info = [(dataset_name,
             dataset["train_path"],
             dataset["valid_path"],
             dataset["weight"],
             dataset["limit_train_split"],
             dataset["limit_val_split"],
             dataset["limit_example_split"],
             dataset["source_token"]
             )
           for dataset_name, dataset in data_args["datasets"].items()]
    dataset_names, train_paths, valid_paths, weights, limit_train_splits, limit_val_splits, limit_example_splits, source_tokens = list(zip(*info))

    train_pipes = [build_single_datapipe(path,
                                         shuffle=shuffle_train,
                                         buffer_size=buffer_size,
                                         limit=limit,
                                         preprocess_args=preprocess_args,
                                         source_token=source_token)
                   for path, limit, source_token in zip(train_paths, limit_train_splits, source_tokens)]
    valid_pipes = {name: build_single_datapipe(path,
                                               shuffle=False,
                                               buffer_size=buffer_size,
                                               limit=limit,
                                               preprocess_args=preprocess_args,
                                               source_token=source_token)
                   for name, path, limit, source_token in zip(dataset_names, valid_paths, limit_val_splits, source_tokens)
                   if limit != 0}
    example_pipes = {name: build_single_datapipe(path,
                                                 shuffle=False,
                                                 buffer_size=buffer_size,
                                                 limit=limit,
                                                 preprocess_args=preprocess_args,
                                                 source_token=source_token)
                     for name, path, limit, source_token in zip(dataset_names, valid_paths, limit_example_splits, source_tokens)
                     if limit != 0}

    datapipes["train"] = build_datapipe_mixture(train_pipes, weights, concat=False, seed=seed)
    datapipes["valid"] = valid_pipes
    datapipes["example"] = example_pipes
    return datapipes