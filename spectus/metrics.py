from __future__ import annotations

import transformers
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
import numpy as np

import selfies as sf
from model.selfies_tokenizer import SelfiesTokenizer
from utils.spectra_process_utils import get_fp_generator, get_fp_simil_function

def compute_fp_simils(preds: list[str] | list[Chem.rdchem.Mol],
                      trues: list[str] | list[Chem.rdchem.Mol],
                      fp_type: str = "daylight",
                      fp_simil_function_type: str = "tanimoto",
                      fp_kwargs: dict = {},
                      input_mols: bool = False,
                      return_mols: bool = False):
    """
    Compute the average cosine similarity between the predicted and true SMILES strings (or molecules)

    Parameters
    ----------
    preds : list[str] | list[Chem.rdchem.Mol]
        List of predicted SMILES strings or RDKit molecules (if input mols flag is set)
    trues : list[str] | list[Chem.rdchem.Mol]
        List of ground truth SMILES strings RDKit molecules (if input mols flag is set)
    fp_type : str
        The type of fingerprint to use
    fp_simil_function_type : str
        The type of similarity function to use
    input_mols : bool
        Whether the inputs are lists of RDKit molecules or not
    return_mols : bool
        Whether to return the predicted and true molecules as well

    Returns
    -------
    list[float] | tuple[list[float], list[Chem.rdchem.Mol], list[Chem.rdchem.Mol]]
        List of cosine similarities between the predicted and true SMILES strings
        If return_mols is set to True, returns also the predicted and true RDKit molecules
    """

    assert len(preds) == len(trues)
    fpgen = get_fp_generator(fp_type, fp_kwargs)
    fp_simil_function = get_fp_simil_function(fp_simil_function_type)
    simils = []
    pred_mols = []
    true_mols = []
    for pred, true in zip(preds, trues):
        if not input_mols:
            pred_mol = Chem.MolFromSmiles(pred)
            true_mol = Chem.MolFromSmiles(true)
        else:
            pred_mol = pred
            true_mol = true
        if pred_mol is None or true_mol is None:
            simils.append(0.0)
            pred_mols.append(None)
            true_mols.append(None)
            continue
        pred_fp = fpgen.GetFingerprint(pred_mol)
        true_fp = fpgen.GetFingerprint(true_mol)
        simil = fp_simil_function(pred_fp, true_fp)
        simils.append(simil)
        pred_mols.append(pred_mol)
        true_mols.append(true_mol)
        # print("pred: " + pred, "true: " + true, "simil: " + str(simil), sep="\n")

    if return_mols:
        return simils, pred_mols, true_mols
    else:
        return simils


def compute_rate_matched_formulas(preds: list[Chem.rdchem.Mol], trues: list[Chem.rdchem.Mol]):
    """
    Compute the percentage of matched formulas between the predicted and true molecules

    Parameters
    ----------
    preds : list[Chem.rdchem.Mol]
        List of predicted RDKit molecules
    trues : list[Chem.rdchem.Mol]
        List of ground truth RDKit molecules
    """
    assert len(preds) == len(trues)

    matched = 0
    for pred, true in zip(preds, trues):
        if not pred or not true:
            continue

        if rdMolDescriptors.CalcMolFormula(pred) == rdMolDescriptors.CalcMolFormula(true):
            matched += 1
    return matched / len(preds)


def compute_rate_canons(pred_smiless: list[str],
                        pred_mols: list[Chem.rdchem.Mol] | None = None,
                        pred_canons: list[str] | None = None):
    """
    Compute the percentage of canonical SMILES in the predicted SMILES strings

    Parameters
    ----------
    pred_smiles : list[str]
        List of predicted SMILES strings
    pred_mols : list[Chem.rdchem.Mol]
        List of predicted RDKit molecules (if available for speedup)
    pred_canons : list[str]
        List of predicted canonical SMILES strings (if available for speedup)
    """
    if pred_mols is None and pred_canons is None:
        pred_mols = [Chem.MolFromSmiles(x) for x in pred_smiless]

    if pred_canons is None:
        pred_canons = [Chem.MolToSmiles(x) if x is not None else "" for x in pred_mols]

    assert len(pred_smiless) == len(pred_canons)

    canon = 0
    for pred_smiles, pred_canon in zip(pred_smiless, pred_canons):
        if  pred_smiles == pred_canon:
            canon += 1
    return canon / len(pred_smiless)


class SpectroMetrics:

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerFast | SelfiesTokenizer,
    ) -> None:
        self.tokenizer = tokenizer

    def __call__(self, eval_preds: transformers.EvalPrediction) -> dict[str, float]:
        preds_all = eval_preds.predictions
        trues_all = eval_preds.label_ids

        if isinstance(preds_all, tuple):
            preds_all = preds_all[0]


        pad_token_id = self.tokenizer.pad_token_id
        assert pad_token_id is not None
        preds_all = np.where(preds_all != -100, preds_all, pad_token_id)
        trues_all = np.where(trues_all != -100, trues_all, pad_token_id)

        preds_str_all = self.tokenizer.batch_decode(preds_all, skip_special_tokens=True)
        trues_str_all = self.tokenizer.batch_decode(trues_all, skip_special_tokens=True)

        if isinstance(self.tokenizer, SelfiesTokenizer):
            preds_str_all = [sf.decoder(x) for x in preds_str_all]
            trues_str_all = [sf.decoder(x) for x in trues_str_all]

        # strip decoded smiless
        preds_str_all = [x.strip() for x in preds_str_all]
        trues_str_all = [x.strip() for x in trues_str_all]

        daylight_tanimoto_simils, pred_mols, gt_mols = compute_fp_simils(preds_str_all, trues_str_all, return_mols=True)
        morgan_tanimoto_simils = compute_fp_simils(pred_mols, gt_mols, fp_type="morgan", fp_kwargs={"radius": 2, "fpSize": 2048}, input_mols=True, return_mols=False)
        precise_morgan_tanimoto_hits = np.sum(np.array(morgan_tanimoto_simils) == 1) / len (morgan_tanimoto_simils)
        precise_daylight_tanimoto_hits = np.sum(np.array(daylight_tanimoto_simils) == 1) / len (daylight_tanimoto_simils)

        pred_canons = [Chem.MolToSmiles(x) if x is not None else "" for x in pred_mols]


        rate_matched_formulas = compute_rate_matched_formulas(pred_mols, gt_mols)
        rate_pred_canon_smiles = compute_rate_canons(preds_str_all, pred_canons=pred_canons) # type: ignore   #!
        rate_exact_smiles = np.sum(np.array(preds_str_all) == np.array(trues_str_all)) / len(preds_str_all)
        rate_exact_mols = np.sum(np.array(pred_canons) == np.array(trues_str_all)) / len(pred_canons)         #!

        metrics = {}
        metrics["daylight_tanimoto_simil"] = np.mean(daylight_tanimoto_simils)
        metrics["morgan_tanimoto_simil"] = np.mean(morgan_tanimoto_simils)
        metrics["morgan_tanimoto_simil_equals_1"] = precise_morgan_tanimoto_hits
        metrics["daylight_tanimoto_hits_equals_1"] = precise_daylight_tanimoto_hits
        metrics["matched_formulas"] = rate_matched_formulas
        metrics["canon_smiles"] = rate_pred_canon_smiles
        metrics["exact_smiles"] = rate_exact_smiles
        metrics["exact_mols"] = rate_exact_mols

        return metrics