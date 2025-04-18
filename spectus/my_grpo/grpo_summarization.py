# the code below is altered from https://gist.github.com/jogonba2/9bee8bb154a292b24850f1483daa6b71
from __future__ import annotations
import os
import gc
import yaml
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass

import evaluate
import torch
from torchdata.datapipes.iter import IterDataPipe
from torch import FloatTensor, LongTensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)

import wandb
import typer


from spectus.model.selfies_tokenizer import hardcode_build_selfies_tokenizer
from spectus.train_spectus import set_batch_size, get_spectro_config
from spectus.utils.general_utils import get_nice_time, build_tokenizer
from spectus.callbacks import PredictionLogger
from spectus.metrics import SpectroMetrics, compute_fp_simils
from spectus.utils.data_utils import SpectroDataCollator, load_all_datapipes
from spectus.model.modeling_spectus import SpectusForConditionalGeneration
from spectus.model.configuration_spectus import SpectusConfig


app = typer.Typer()


@dataclass
class TrainingArguments:
    steps: int
    batch_size: int
    learning_rate: float
    update_old_after: int
    group_size: int
    logging_steps: int
    max_new_tokens: int
    grpo_epsilon: float
    grpo_beta: float
    gradient_max_norm: float
    save_steps: int
    save_dir: str
    device: str
    gen_args: dict


@dataclass
class BatchRewards:
    rewards: FloatTensor


@dataclass
class GRPOOutput:
    loss: FloatTensor
    reward: FloatTensor
    kl: FloatTensor


def compute_rewards(
    token_ids: LongTensor, labels: LongTensor, tokenizer: PreTrainedTokenizer
) -> BatchRewards:
    """
    Compute rewards based on the ROUGE avg score between generated completions and reference summaries.

    Args:
        token_ids (LongTensor): Tensor containing token IDs of the generated completions.
        labels (LongTensor): Tensor containing token IDs of the reference summaries.
        tokenizer (PreTrainedTokenizer): Tokenizer used to decode the token IDs.

    Returns:
        BatchRewards: A tensor containing the computed rewards for each completion.
    """
    labels[labels == -100] = tokenizer.pad_token_id
    pred_smiless = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    label_smiless = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pred_smiless = [x.strip() for x in pred_smiless]
    label_smiless = [x.strip() for x in label_smiless]

    morgan_tanimoto_simils = compute_fp_simils(pred_smiless, label_smiless, fp_type="morgan", fp_kwargs={"radius": 2, "fpSize": 2048})
    rewards = torch.tensor(morgan_tanimoto_simils, device=token_ids.device)

    return BatchRewards(rewards)


def selective_log_softmax(
    logits: FloatTensor, index: LongTensor
) -> FloatTensor:
    """
    Computes the log softmax of the input logits selectively based on the provided indices.
    This function performs the same operation as applying `log_softmax` on the logits tensor
    along the last dimension and then gathering the results based on the provided indices.
    However, it processes the logits row by row to save memory by leveraging PyTorch internals.

    Taken from https://www.tylerromero.com/posts/2025-02-selective-log-softmax/

    Args:
        logits (FloatTensor): A tensor of shape (batch_size, num_classes) containing the raw
                              logits for each class.
        index (LongTensor): A tensor of shape (batch_size, num_indices) containing the indices
                            of the classes for which to compute the log softmax.

    Returns:
        FloatTensor: A tensor of shape (batch_size, num_indices) containing the log softmax
                     values for the specified indices.
    """

    token_logprobs = []
    for logits_row, index_row in zip(logits, index):
        logprobs_row = logits_row.log_softmax(dim=-1)
        token_logprobs_row = torch.gather(
            logprobs_row, dim=-1, index=index_row.unsqueeze(-1)
        ).squeeze(-1)
        token_logprobs.append(token_logprobs_row)
    return torch.stack(token_logprobs)


def gather_token_scores(
    logits: FloatTensor, generated_ids: LongTensor
) -> FloatTensor:
    """
    Gathers token scores from logits based on generated token IDs.

    Args:
        logits (FloatTensor): The logits output from the model. It can be a tuple of tensors or a single tensor.
        generated_ids (LongTensor): The IDs of the generated tokens.

    Returns:
        FloatTensor: The token scores after applying a selective log softmax on the logits.
    """

    if isinstance(logits, tuple):
        # Stack the logits (batch_size*group_size, output_length, vocab)
        logits = torch.stack(logits, axis=0).permute((1, 0, 2))

    # Logsoftmax the logits
    token_scores = selective_log_softmax(logits, generated_ids)

    return token_scores


def compute_token_scores(
    model: PreTrainedModel,
    encoder_input_ids: LongTensor,
    encoder_position_ids: LongTensor,
    encoder_attention_mask: LongTensor,
    decoder_input_ids: LongTensor,
    decoder_attention_mask: LongTensor,
    batch_size: int,
    group_size: int,
) -> FloatTensor:
    """
    Computes token scores for a given batch of input sequences using a pre-trained model.

    Args:
        model (PreTrainedModel): The pre-trained model to use for generating logits.
        encoder_input_ids (LongTensor): Tensor containing input IDs for the encoder.
        encoder_attention_mask (LongTensor): Tensor containing attention masks for the encoder inputs.
        decoder_input_ids (LongTensor): Tensor containing input IDs for the decoder.
        decoder_attention_mask (LongTensor): Tensor containing attention masks for the decoder inputs.
        batch_size (int): The size of the batch.
        group_size (int): The size of the group.

    Returns:
        FloatTensor: A tensor containing the computed token scores, reshaped to (batch_size, group_size, -1).
    """
    logits = model(
        input_ids=encoder_input_ids,
        position_ids=encoder_position_ids,
        attention_mask=encoder_attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
    ).logits
    scores = gather_token_scores(logits[:, :-1], decoder_input_ids[:, 1:])
    scores = scores.view(batch_size, group_size, -1)
    del logits
    torch.cuda.empty_cache()
    return scores


def grpo(
    generated_ids: LongTensor,
    old_scores: FloatTensor,
    current_scores: FloatTensor,
    reference_scores: FloatTensor,
    labels: LongTensor,
    tokenizer: PreTrainedTokenizer,
    epsilon: float,
    beta: float,
) -> GRPOOutput:
    """
    Compute the loss of Group Relative Policy Optimization (GRPO) on the given inputs of one batch.

    Args:
        generated_ids (LongTensor): Tensor of generated token IDs.
        old_scores (FloatTensor): Tensor of old policy scores.
        current_scores (FloatTensor): Tensor of current policy scores.
        reference_scores (FloatTensor): Tensor of reference policy scores.
        labels (LongTensor): Tensor of ground truth token IDs.
        tokenizer (PreTrainedTokenizer): Tokenizer used for encoding/decoding.
        epsilon (float): Clipping parameter for policy ratios.
        beta (float): Weighting factor for the Kullback-Leibler divergence term.

    Returns:
        GRPOOutput: A dataclass containing the mean loss, rewards and KL divergences.
    """
    losses = torch.zeros(generated_ids.shape[0])
    rewards = torch.zeros(generated_ids.shape[0])
    kls = torch.zeros(generated_ids.shape[0])

    for idx, (
        group_ids,
        group_labels,
        group_old_scores,
        group_current_scores,
        group_reference_scores,
    ) in enumerate(
        zip(generated_ids, labels, old_scores, current_scores, reference_scores)
    ):
        # Compute advantages
        group_rewards = compute_rewards(group_ids, group_labels, tokenizer)
        mean = group_rewards.rewards.mean()
        centered = group_rewards.rewards - mean
        std = group_rewards.rewards.std()
        if std < 1e-8:
            advantages = torch.zeros_like(centered)
        else:
            advantages = centered / (std + 1e-8)

        # Store the mean of each rewards for the group
        rewards[idx] = group_rewards.rewards.mean()

        # Compute the ratios
        ratios = torch.exp(group_current_scores - group_old_scores)

        # Compute the clipped ratios
        clipped_ratios = torch.clamp(
            ratios, min=1.0 - epsilon, max=1.0 + epsilon
        )

        # Compute kullback-leibler divergence between reference and current policy
        kl = (
            torch.exp(group_reference_scores - group_current_scores)
            - (group_reference_scores - group_current_scores)
            - 1
        )
        kls[idx] = kl.mean()

        # Compute mean loss of the group
        completion_mask = group_ids[:, 1:] != tokenizer.pad_token_id
        loss = (
            torch.min(
                ratios * advantages.unsqueeze(-1),
                clipped_ratios * advantages.unsqueeze(-1),
            )
            - beta * kl
        )
        loss = -(loss * completion_mask).sum() / completion_mask.sum()
        losses[idx] = loss

    return GRPOOutput(
        loss=losses.mean(),
        reward=rewards.mean(),
        kl=kls.mean(),
    )


def train(
    datapipes: dict[str, IterDataPipe],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    training_args: TrainingArguments,
    model_args: dict,
) -> None:
    """
    Train a language model using the GRPO (Group Relative Policy Optimization) objective.

    Args:
        dataset (Dataset): The dataset containing training data.
        model (PreTrainedModel): The model to be trained.
        tokenizer (PreTrainedTokenizer): The tokenizer used for encoding the data.
        training_args (TrainingArguments): The training arguments containing hyperparameters and configurations.
    """
    # Prepare the dataloader
    train_dataloader = DataLoader(
        datapipes["train"],
        collate_fn=SpectroDataCollator(restrict_intensities=model_args.get("restrict_intensities", False)),
        batch_size=training_args.batch_size,
    )

    # Prepare policies
    model.to(training_args.device)
    reference_model = deepcopy(model)
    old_model = deepcopy(model)
    reference_model.eval()
    old_model.eval()
    model.train()

    # Prepare optimizer and lr scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
    )

    scheduler = LinearLR(
        optimizer,
        start_factor=1,
        end_factor=0.1,
        total_iters=training_args.steps,
    )

    # Prepare the metrics
    running_metrics = {
        "loss": 0.0,
        "reward": 0.0,
        "completion_length": 0.0,
        "kl": 0.0,
    }

    # Let's train
    training_step = 0
    best_reward = 0.0
    for _ in range(training_args.steps):
        # Update the old policy
        old_model.load_state_dict(model.state_dict(), strict=False)
        for batch in tqdm(train_dataloader, desc="Training step", total=training_args.steps):
            # Prepare the batch data
            model_input = {key: value.to(model.device) for key, value in batch.items()} # move tensors from batch to device

            input_ids = model_input["input_ids"]
            attention_mask = model_input["attention_mask"]
            labels = model_input["labels"]
            position_ids = model_input["position_ids"]
            effective_batch_size = input_ids.shape[0]

            # Generate ids with the old policy
            generated_ids = old_model.generate(
                **model_input,
                **training_args.gen_args,
                num_beams=training_args.group_size,
                num_return_sequences=training_args.group_size,
            )

            # Prepare attention mask for computing current
            # and reference logits on the generated ids
            decoder_attention_mask = generated_ids != tokenizer.pad_token_id

            # Interleave input_ids and attention_mask to have
            # the same shape as the generated completions
            repeated_input_ids = input_ids.repeat_interleave(
                repeats=training_args.group_size, dim=0
            )
            repeated_position_ids = position_ids.repeat_interleave(
                repeats=training_args.group_size, dim=0
            )
            repeated_attention_mask = attention_mask.repeat_interleave(
                repeats=training_args.group_size, dim=0
            )

            # Compute the sequence scores of the old policy
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                old_scores = compute_token_scores(
                    old_model,
                    encoder_input_ids=repeated_input_ids,
                    encoder_position_ids=repeated_position_ids,
                    encoder_attention_mask=repeated_attention_mask,
                    decoder_input_ids=generated_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    batch_size=effective_batch_size,
                    group_size=training_args.group_size,
                )

            # Compute the sequence scores of the current policy
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.eval()
                current_scores = compute_token_scores(
                    model,
                    encoder_input_ids=repeated_input_ids,
                    encoder_position_ids=repeated_position_ids,
                    encoder_attention_mask=repeated_attention_mask,
                    decoder_input_ids=generated_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    batch_size=effective_batch_size,
                    group_size=training_args.group_size,
                )
                model.train()

            # Compute the sequence scores of the reference model
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                reference_scores = compute_token_scores(
                    reference_model,
                    encoder_input_ids=repeated_input_ids,
                    encoder_position_ids=repeated_position_ids,
                    encoder_attention_mask=repeated_attention_mask,
                    decoder_input_ids=generated_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    batch_size=effective_batch_size,
                    group_size=training_args.group_size,
                )

            # Group the generated ids (batch_size, group_size, output_length)
            generated_ids = generated_ids.view(
                effective_batch_size, training_args.group_size, -1
            )

            # Repeat the labels and group (batch_size, group_size)
            labels = labels.repeat_interleave(
                repeats=training_args.group_size, dim=0
            ).view(effective_batch_size, training_args.group_size, -1)

            # Compute GRPO objective
            with torch.autocast("cuda", dtype=torch.bfloat16):
                grpo_output = grpo(
                    generated_ids,
                    old_scores,
                    current_scores,
                    reference_scores,
                    labels,
                    tokenizer,
                    training_args.grpo_epsilon,
                    training_args.grpo_beta,
                )

            # Update the current policy
            grpo_output.loss.backward()
            clip_grad_norm_(
                model.parameters(),
                training_args.gradient_max_norm,
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Update old policy periodically
            if (training_step + 1) % training_args.update_old_after == 0:
                old_model.load_state_dict(model.state_dict(), strict=False)
                torch.cuda.empty_cache()

            # Update log metrics
            batch_metrics = {
                "loss": grpo_output.loss.item(),
                "reward": grpo_output.reward.item(),
                "kl": grpo_output.kl.item(),
                "completion_length": decoder_attention_mask.sum(-1)
                .float()
                .mean()
                .item(),
            }
            running_metrics = {
                key: running_metrics[key] + batch_metrics.get(key, 0)
                for key in running_metrics
            }

            # And report them periodically
            if (training_step + 1) % training_args.logging_steps == 0:
                wandb.log(
                    {
                        **{
                            key: val / (training_step + 1)
                            for key, val in running_metrics.items()
                        },
                        **{"lr": scheduler.get_last_lr()[0]},
                    }
                )

            # Save the model each periodically
            if (training_step + 1) % training_args.save_steps == 0:
                last_reward = running_metrics["loss"] / (training_step + 1)
                if last_reward > best_reward:
                    model.save_pretrained(f"{training_args.save_dir}")
                    best_reward = last_reward
                    print(
                        "Saving model with reward:",
                        best_reward,
                        f"step: {training_step+1}",
                    )
                else:
                    print(
                        f"Model not saved because didn't improve the reward at step {training_step+1}"
                    )

            # Free GPU memory at the end
            del (
                generated_ids,
                old_scores,
                input_ids,
                attention_mask,
                repeated_input_ids,
                repeated_attention_mask,
                current_scores,
                reference_scores,
                grpo_output,
                labels,
            )
            torch.cuda.empty_cache()
            gc.collect()
            training_step += 1


@app.command()
def main(config_file: Path = typer.Option(..., dir_okay=False, help="Path to the config file"),
         checkpoint: Path = typer.Option(None, help="Path to the checkpoint directory"),
         resume_id: str = typer.Option(None, help="Wandb id of the run to resume, if not None, resume will be attempted"),
         checkpoints_dir: Path = typer.Option("checkpoints", help="Path to the checkpoints directory"),
         additional_info: str = typer.Option(None, help="use format '_info'; additional info to add to run_name"),
         additional_tags: str = typer.Option(None, help="Tags to add to the wandb run, one string, delimited by ':'"),
         device: str = typer.Option("cuda", help="Device to use for training: 'cuda' or 'cpu' ... CPU currently not available "),
         wandb_group: str = typer.Option(..., help="Wandb group to use for logging"),
         ):
    """
    Main function for training an encoder-decoder model with
    GRPO to optimize MORGAN-TANIMOTO on the NIST dataset (altered from https://gist.github.com/jogonba2/9bee8bb154a292b24850f1483daa6b71).
    """
    # Instantiate current policy and reference model

    # hardcoded for testing
    checkpoint = "checkpoints/finetune_clean/youthful-wave-590_exp5_9M_448+296/checkpoint-294952"
    tokenizer_path = "tokenizer/tokenizer_mf10M.model"
    model = SpectusForConditionalGeneration.from_pretrained(checkpoint)
    tokenizer = build_tokenizer(tokenizer_path)
    wandb_group = "grpo"

    print("DEVICE", device)
    assert device in ["cuda", "cpu"], "ArgumentError: Device must be 'cuda' or 'cpu'"
    if device == "cpu":
        print("Training on CPU is currently not supported due to dependency issues.")
        return

    if additional_tags:
        add_tags = additional_tags.split(":")
    else:
        add_tags = []

    cvd = os.environ['CUDA_VISIBLE_DEVICES']
    print(f"CUDA_VISIBLE_DEVICES set to: {cvd}")
    if len(cvd) < 60:
        add_tags.append("CVD=" + cvd)
    else:
        add_tags.append("CVD=weird_meta_id")

    if device == "cuda":
        for i in range(torch.cuda.device_count()):
            print(f"device: {device}")
            print(torch.cuda.get_device_properties(i))


    # load config
    with open(config_file, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError("Error in configuration file:", exc) from exc

    hf_training_args = config["hf_training_args"]
    dataset_args = config["data_args"]
    preprocess_args = config.get("preprocess_args", {})
    model_args = config["model_args"]
    example_gen_args = config["example_generation_args"]
    grpo_args = config["grpo_args"]
    tokenizer_path = model_args["tokenizer_path"]
    report_to = hf_training_args.pop("report_to", "none")
    use_wandb = report_to == "wandb"

    hf_training_args = set_batch_size(hf_training_args)

    # load tokenizer, data
    if tokenizer_path == "selfies_tokenizer":
        tokenizer = hardcode_build_selfies_tokenizer()
    else:
        tokenizer = build_tokenizer(tokenizer_path)
    print(f"TOKENIZER vocab size: {len(tokenizer.get_vocab())}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # surpressing a warning

    if preprocess_args:
        print("Using  O N - T H E - F L Y  PREPROCESSING")
        preprocess_args = {
            "restrict_intensities": model_args.get("restrict_intensities", False),
            "inference_mode": False,
            "max_num_peaks": preprocess_args.get("max_num_peaks", 300),
            "max_mol_repr_len": preprocess_args.get("max_mol_repr_len", 100),
            "max_mz": model_args["max_mz"],
            "mol_repr": "selfies" if tokenizer_path == "selfies_tokenizer" else "smiles",
            "log_base": preprocess_args.get("log_base", 1.28),
            "log_shift": preprocess_args.get("log_shift", 29),
            "max_cumsum": preprocess_args.get("max_cumsum", None),
            "tokenizer": tokenizer,
            "do_log_binning": preprocess_args.get("do_log_binning", True),
            "linear_bin_decimals": preprocess_args.get("linear_bin_decimals", None),
            "output_format": preprocess_args.get("output_format", "<mol_repr>"),
        }

        if preprocess_args["do_log_binning"]:
            model_args["max_log_id"] = preprocess_args["log_shift"]
        else:
            if not preprocess_args.get("linear_bin_decimals", None):
                raise ValueError("linear_bin_decimals must be provided if do_log_binning is False. It's 2 for 100 bins, 3 for 1000 bins, ...")
            model_args["max_log_id"] = 10**preprocess_args["linear_bin_decimals"]

    datapipes = load_all_datapipes(dataset_args, preprocess_args)
    spectus_spectro_config = get_spectro_config(model_args, tokenizer)

    print("Loading model...")
    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}")
        model = SpectusForConditionalGeneration.from_pretrained(checkpoint)
    else:
        model = SpectusForConditionalGeneration(spectus_spectro_config)

    tuned_params = sum(p.shape.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.shape.numel() for p in model.parameters())

    # Init wandb
    if use_wandb:
        log_tags = [d for d in dataset_args["datasets"].keys()]
        log_tags.extend(add_tags)
        log_tags.append(wandb_group)
        log_tags.append(f"params={total_params}")
        log_tags.append(f"trained_params={tuned_params}")
        log_tags.append(f"trained_percentage={tuned_params/total_params*100:.2f}%")
        log_tags.append(f"lr={hf_training_args['learning_rate']}")
        log_tags.append(f"pd_bs={hf_training_args['per_device_train_batch_size']}")
        if additional_info:
            log_tags.append(additional_info)

        wandb.login()
        run = wandb.init(
                id=resume_id,
                resume="must" if resume_id else "never",
                entity="hajekad",
                project="BART_for_gcms",
                tags=log_tags,
                save_code=True,
                dir=checkpoints_dir.parent,
                config=config,
                group=wandb_group,
            )

        # to not add additional info to the run name if it is already there
        if run.name.endswith(additional_info):
            run_name = run.name
        else:
            run_name = run.name + additional_info
        run.name = run_name
        run.tags += (f"run_id={run.id}",)
    else:
        run_name = get_nice_time() + additional_info
    print(f"Run name: {run_name}")

    # Resume training
    if resume_id:
        if not checkpoint:
            raise ValueError("Checkpoint must be provided when resuming training")
        save_path = checkpoint.parent
    else:
        save_path = checkpoints_dir / wandb_group / run_name
    print(f"save path: {save_path}")

    # Define training arguments
    training_args = TrainingArguments(
        steps=hf_training_args["max_steps"],
        batch_size=hf_training_args["per_device_train_batch_size"],
        learning_rate=hf_training_args["learning_rate"],
        update_old_after=grpo_args["update_old_after"],
        group_size=grpo_args["group_size"],
        logging_steps=hf_training_args["logging_steps"],
        max_new_tokens=model_args["decoder_seq_len"] - 5, # -5 for special tokens, may be even less in fact
        grpo_epsilon=grpo_args["epsilon"],
        grpo_beta=grpo_args["beta"],
        gradient_max_norm=grpo_args["gradient_max_norm"],
        save_steps=hf_training_args["save_steps"],
        save_dir=str(save_path),
        gen_args=grpo_args["grpo_gen_args"],
        device=device,
    )


    # Let's train!
    train(datapipes, model, tokenizer, training_args, model_args)

    # Save the model and finish logging
    model.save_pretrained(save_path)
    wandb.finish()


if __name__ == "__main__":
    app()