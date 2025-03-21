from __future__ import annotations

import math
from typing import Callable, Iterable

from tqdm.auto import tqdm
import pandas as pd
import transformers
import wandb
import torch
import torch.utils.data
from rdkit.Chem import Draw

import selfies as sf

from model.modeling_spectus import SpectusForConditionalGeneration
from model.selfies_tokenizer import SelfiesTokenizer
from metrics import compute_fp_simils


class PredictionLogger(transformers.TrainerCallback):
    def __init__(
        self,
        datasets: list[torch.utils.data.Dataset] | list[torch.utils.data.IterableDataset],
        source_tokens: list[str],
        log_prefixes: list[str],
        log_every_n_steps: int,
        show_raw_preds: bool,
        collator: Callable,
        generate_kwargs: dict | None = None,
        **kwargs
    ) -> None:
        # super().__init__(**kwargs) # not needed?

        self.datasets = datasets
        self.source_tokens = source_tokens
        self.log_prefixes = log_prefixes
        self.logging_steps = log_every_n_steps
        self.show_raw_preds = show_raw_preds
        self.collator = collator

        if generate_kwargs is None:
            generate_kwargs = {}
        self.generate_kwargs = generate_kwargs

        self.num_examples = [sum(1 for _ in dataset) for dataset in self.datasets]

    def log_example_prediction(self,
                               dataset: torch.utils.data.Dataset | torch.utils.data.IterableDataset,
                               source_token: str,
                               log_prefix: str,
                               num_examples: int,
                               global_step: int,
                               args: transformers.TrainingArguments,
                               kwargs: dict) -> None:
        """
        Log predictions and stats of one valid dataset to wandb.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset | torch.utils.data.IterableDataset
            The dataset to log predictions for.
        log_prefix : str
            The prefix to use for the wandb table.
        global_step : int
            The current global step.
        """

        model: SpectusForConditionalGeneration = kwargs["model"]
        tokenizer: transformers.PreTrainedTokenizerFast | SelfiesTokenizer = kwargs["tokenizer"] # if missing add to the class args
        batch_size = args.per_device_eval_batch_size // 2   # to avoid OOM

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.collator,
        )

        model.eval()

        num_batches = math.ceil(num_examples / batch_size)
        progress = tqdm(dataloader, total=num_batches, desc="Generating preds for logging", leave=False)

        all_raw_preds = []
        all_preds = []
        all_decoded_labels = []
        all_pred_molecules = []
        all_gt_molecules = []
        all_daylight_tanimoto_simils = []
        all_morgan_tanimoto_simils = []

        gen_kwargs = self.generate_kwargs.copy()
        gen_kwargs["forced_decoder_ids"] = [[1, tokenizer.encode(source_token)[0]]]

        # decide wether to use selfies or smiles
        if isinstance(tokenizer, SelfiesTokenizer):
            mol_repr = "selfies"
            assert sf.get_semantic_constraints()["I"] == 5, "Selfies tokenizer constraints are not set properly!"
        else:
            mol_repr = "smiles"

        with torch.no_grad():
            for batch in progress:
                model_input = {key: value.to(args.device) for key, value in batch.items()} # move tensors from batch to device
                preds = model.generate(**model_input,
                                       **gen_kwargs)

                raw_preds_str = tokenizer.batch_decode(preds, skip_special_tokens=False)
                preds_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
                labels = batch["labels"][batch["labels"] == -100] = 2 # replace -100 with pad token
                gts_str = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

                all_raw_preds.extend(raw_preds_str)
                all_preds.extend(preds_str)
                all_decoded_labels.extend(gts_str)

                # if SELFIES, translate them to SMILES before continuing
                if mol_repr == "selfies":
                    preds_str = [sf.decoder(x) for x in preds_str]
                    gts_str = [sf.decoder(x) for x in gts_str]

                # compute SMILES simil
                daylight_tanimoto_simils, pred_mols, gt_mols = compute_fp_simils(preds_str, gts_str, return_mols=True)
                morgan_tanimoto_simils = compute_fp_simils(pred_mols, gt_mols, fp_type="morgan", fp_kwargs={"radius": 2, "fpSize": 2048}, input_mols=True, return_mols=False)

                all_daylight_tanimoto_simils.extend(daylight_tanimoto_simils) # type: ignore
                all_morgan_tanimoto_simils.extend(morgan_tanimoto_simils) # type: ignore

                # create images for logging
                for mol in pred_mols:
                    try:
                        img = wandb.Image(Draw.MolToImage(mol))
                    except ValueError:
                        img = None
                    all_pred_molecules.append(img)

                # create a mol for labels if it's valid, otherwise None (it should be, but things can happen)
                for mol in gt_mols:
                    try:
                        img = wandb.Image(Draw.MolToImage(mol))
                    except ValueError:
                        img = None
                    all_gt_molecules.append(img)

        # ic(len(all_raw_preds), len(all_preds), len(all_decoded_labels), len(all_pred_molecules), len(all_gt_molecules), len(all_simils))
        # ic(all_raw_preds, all_preds, all_decoded_labels, all_pred_molecules, all_gt_molecules, all_simils)
        df_log = pd.DataFrame({f"gt_{mol_repr}": all_decoded_labels,
                               f"predicted_{mol_repr}": all_preds,
                               "gt_molecule": all_gt_molecules,
                               "predicted_molecule": all_pred_molecules,
                               "daylight_tanimoto_simil": all_daylight_tanimoto_simils,
                               "morgan_tanimoto_simil": all_morgan_tanimoto_simils
                               })
        if self.show_raw_preds:
            df_log[f"raw_predicted_{mol_repr}"] = all_raw_preds
        table = wandb.Table(dataframe=df_log)

        # log either to WANDB or to STDOUT
        if args.report_to and "wandb" in args.report_to:
            wandb.log({f"eval_tables/{log_prefix}/global_step_{global_step}": table})
            wandb.log({f"eval/{log_prefix}/example_DLT_similarity": sum(all_daylight_tanimoto_simils)/len(all_daylight_tanimoto_simils)})
        else:
            print(f"Log {log_prefix} at global step {global_step}")
            df_log.pop("gt_molecule")
            df_log.pop("predicted_molecule")
            print(df_log)


    def on_step_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs
    ) -> None:

        if state.global_step % self.logging_steps != 0:
            return

        for dataset, source_token, log_prefix, num_examples in zip(self.datasets, self.source_tokens, self.log_prefixes, self.num_examples):
            self.log_example_prediction(dataset,
                                        source_token,
                                        log_prefix,
                                        num_examples,
                                        state.global_step,
                                        args,
                                        kwargs)
