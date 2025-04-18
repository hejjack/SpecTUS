from __future__ import annotations

import transformers
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
import numpy as np

import selfies as sf
from spectus.model.selfies_tokenizer import SelfiesTokenizer
from spectus.utils.data_utils import parse_outputs_list
from spectus.utils.spectra_process_utils import get_fp_generator, get_fp_simil_function

def compute_fp_simils(preds: list[str] | list[Chem.rdchem.Mol],
                      labels: list[str] | list[Chem.rdchem.Mol],
                      fp_type: str = "daylight",
                      fp_simil_function_type: str = "tanimoto",
                      fp_kwargs: dict = {},
                      input_mols: bool = False,
                      return_mols: bool = False):
    """
    Compute the average cosine similarity between the predicted and label SMILES strings (or molecules)

    Parameters
    ----------
    preds : list[str] | list[Chem.rdchem.Mol]
        List of predicted SMILES strings or RDKit molecules (if input mols flag is set)
    labels : list[str] | list[Chem.rdchem.Mol]
        List of ground truth SMILES strings RDKit molecules (if input mols flag is set)
    fp_type : str
        The type of fingerprint to use
    fp_simil_function_type : str
        The type of similarity function to use
    input_mols : bool
        Whether the inputs are lists of RDKit molecules or not
    return_mols : bool
        Whether to return the predicted and label molecules as well

    Returns
    -------
    list[float] | tuple[list[float], list[Chem.rdchem.Mol], list[Chem.rdchem.Mol]]
        List of cosine similarities between the predicted and label SMILES strings
        If return_mols is set to True, returns also the predicted and label RDKit molecules
    """

    assert len(preds) == len(labels)
    fpgen = get_fp_generator(fp_type, fp_kwargs)
    fp_simil_function = get_fp_simil_function(fp_simil_function_type)
    simils = []
    pred_mols = []
    label_mols = []
    for pred, label in zip(preds, labels):
        pred_mol = Chem.MolFromSmiles(pred) if not input_mols else pred
        label_mol = Chem.MolFromSmiles(label) if not input_mols else label

        if pred_mol is None or label_mol is None:
            simils.append(0.0)
            pred_mols.append(None)
            label_mols.append(None)
            continue

        pred_fp = fpgen.GetFingerprint(pred_mol)
        label_fp = fpgen.GetFingerprint(label_mol)
        simil = fp_simil_function(pred_fp, label_fp)
        simils.append(simil)
        pred_mols.append(pred_mol)
        label_mols.append(label_mol)
        # print("pred: " + pred, "label: " + label, "simil: " + str(simil), sep="\n")

    if return_mols:
        return simils, pred_mols, label_mols
    else:
        return simils


def compute_rate_matched_formulas(preds: list[Chem.rdchem.Mol | str],
                                  labels: list[Chem.rdchem.Mol | str],
                                  input_formulas: bool = False):
    """
    Compute the percentage of matched formulas between the predicted and label molecules

    Parameters
    ----------
    preds : list[Chem.rdchem.Mol]
        List of predicted RDKit molecules
    labels : list[Chem.rdchem.Mol]
        List of ground truth RDKit molecules
    input_formulas : bool
        If yes, the inputs are already precomputed formulas
    """
    assert len(preds) == len(labels)

    matched = 0
    for pred, label in zip(preds, labels):
        if not pred or not label:
            continue
        if not input_formulas:
            pred, label = rdMolDescriptors.CalcMolFormula(pred), rdMolDescriptors.CalcMolFormula(label)
        if pred == label:
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
        output_format: str = "<mol_repr>",
    ) -> None:
        self.tokenizer = tokenizer
        self.output_format = output_format

    def __call__(self, eval_preds: transformers.EvalPrediction) -> dict[str, float]:
        preds_all = eval_preds.predictions
        labels_all = eval_preds.label_ids

        if isinstance(preds_all, tuple):
            preds_all = preds_all[0]


        pad_token_id = self.tokenizer.pad_token_id
        assert pad_token_id is not None
        preds_all = np.where(preds_all != -100, preds_all, pad_token_id)
        labels_all = np.where(labels_all != -100, labels_all, pad_token_id)

        preds_str_all = self.tokenizer.batch_decode(preds_all, skip_special_tokens=True)
        labels_str_all = self.tokenizer.batch_decode(labels_all, skip_special_tokens=True)

        if isinstance(self.tokenizer, SelfiesTokenizer):
            preds_str_all = [sf.decoder(x) for x in preds_str_all]
            labels_str_all = [sf.decoder(x) for x in labels_str_all]

        # strip decoded smiless
        preds_str_all = [x.strip() for x in preds_str_all]
        labels_str_all = [x.strip() for x in labels_str_all]

        if self.output_format == "<mol_repr>": # speed up for the basic case
            preds_mol_repr_all = preds_str_all
            labels_mol_repr_all = labels_str_all
            num_format_errors = 0
        else:
            parsed_outputs, num_format_errors = parse_outputs_list(preds_str_all, self.output_format)
            parsed_labels, _ = parse_outputs_list(labels_str_all, self.output_format)

            assert sorted(parsed_outputs.keys()) == sorted(parsed_labels.keys())
            assert len(parsed_outputs["mol_repr"]) == len(preds_str_all)

            preds_mol_repr_all = parsed_outputs["mol_repr"]
            labels_mol_repr_all = parsed_labels["mol_repr"]

        daylight_tanimoto_simils, pred_mols, gt_mols = compute_fp_simils(preds_mol_repr_all, labels_mol_repr_all, return_mols=True)
        morgan_tanimoto_simils = compute_fp_simils(pred_mols, gt_mols, fp_type="morgan", fp_kwargs={"radius": 2, "fpSize": 2048}, input_mols=True, return_mols=False)
        precise_morgan_tanimoto_hits = np.sum(np.array(morgan_tanimoto_simils) == 1) / len (morgan_tanimoto_simils)
        precise_daylight_tanimoto_hits = np.sum(np.array(daylight_tanimoto_simils) == 1) / len (daylight_tanimoto_simils)

        pred_canons = [Chem.MolToSmiles(x) if x is not None else "" for x in pred_mols]
        pred_formulas_computed_all = [rdMolDescriptors.CalcMolFormula(x) if x is not None else "" for x in pred_mols]


        rate_pred_canon_smiles = compute_rate_canons(preds_mol_repr_all, pred_canons=pred_canons) # type: ignore   #!
        rate_exact_smiles = np.sum(np.array(preds_mol_repr_all) == np.array(labels_mol_repr_all)) / len(preds_mol_repr_all)
        rate_exact_mols = np.sum(np.array(pred_canons) == np.array(labels_mol_repr_all)) / len(pred_canons)         #!

        # metric comparing formulas of generated SMILES to ground truth SMILES (use precomputed if directly available in label otherwise compute it)
        if "formula" in self.output_format:
            rate_matched_formulas = compute_rate_matched_formulas(pred_formulas_computed_all, parsed_labels["formula"], input_formulas=True)
        else:
            gt_formulas_computed_all = [rdMolDescriptors.CalcMolFormula(x) if x is not None else "" for x in gt_mols]
            rate_matched_formulas = compute_rate_matched_formulas(pred_formulas_computed_all, gt_formulas_computed_all, input_formulas=True)


        metrics = {}
        metrics["daylight_tanimoto_simil"] = np.mean(daylight_tanimoto_simils)
        metrics["morgan_tanimoto_simil"] = np.mean(morgan_tanimoto_simils)
        metrics["morgan_tanimoto_simil_equals_1"] = precise_morgan_tanimoto_hits
        metrics["daylight_tanimoto_hits_equals_1"] = precise_daylight_tanimoto_hits
        metrics["matched_formulas"] = rate_matched_formulas
        metrics["canon_smiles"] = rate_pred_canon_smiles
        metrics["exact_smiles"] = rate_exact_smiles
        metrics["exact_mols"] = rate_exact_mols
        metrics["correct_format_rate"] = 1 - num_format_errors / len(preds_str_all)


        # new metrics
        # - does the predicted formula match the formula of the generated molecule?
        # - does the predicted formula match the formula of the ground truth molecule?
        # - number of errors in the output format
        if "formula" in self.output_format:
            cot_formula_agreement = compute_rate_matched_formulas(parsed_outputs["formula"],
                                                                  pred_formulas_computed_all,
                                                                  input_formulas=True)
            cot_formula_accuracy = compute_rate_matched_formulas(parsed_outputs["formula"],
                                                                 parsed_labels["formula"],
                                                                 input_formulas=True)
            metrics["cot_formula_agreement"] = cot_formula_agreement
            metrics["cot_formula_accuracy"] = cot_formula_accuracy
        return metrics