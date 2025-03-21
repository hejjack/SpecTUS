import os
import wandb
from pathlib import Path
import torch
import transformers
import typer
import yaml
from pathlib import Path
from typing import Dict
from tokenizers import Tokenizer
import peft

# custom code
from callbacks import PredictionLogger
from metrics import SpectroMetrics
from utils.data_utils import SpectroDataCollator, load_all_datapipes
from model.modeling_spectus import SpectusForConditionalGeneration
from model.configuration_spectus import SpectusConfig
from model.selfies_tokenizer import hardcode_build_selfies_tokenizer
from utils.general_utils import get_nice_time, build_tokenizer

app = typer.Typer()


def enrich_best_metric_name(metric_name: str, dataset_name: str) -> str:
    subnames = metric_name.split("_")
    subnames = subnames[:1] + [dataset_name] + subnames[1:]
    metric_name = "_".join(subnames)
    return metric_name


def set_batch_size(hf_training_args: Dict):
    """
    Set batch sizes and gradient accumulation steps. If auto_bs is True, it computes the per device batch sizes and gas based on
    given 'effective_{train, eval}_batch_size', number of GPUs, GPU RAM, and the size of model ('base' or 'large'). If auto_bs is False
    or not provided, it uses the batch size and gas provided in hf_training_args - 'per_device_{train, eval}_batch_size',
    'gradient_accumulation_steps' and 'eval_accumulation_steps'.

    Parameters:
    -----------
    hf_training_args (Dict): dictionary with training arguments

    Returns:
    --------
    Dict: dictionary with updated training arguments
    """

    cvd = os.environ['CUDA_VISIBLE_DEVICES']
    gpu_ram = torch.cuda.get_device_properties(0).total_memory
    num_gpu = len(cvd.split(",")) if cvd else 0
    auto_bs = hf_training_args.pop("auto_bs", False)
    spectus_size = hf_training_args.pop("spectus_size", None)
    if auto_bs:
        print("\nUsing   A U T O M A T I C   batch size")
        print("relies heavily on POSSIBLE_TO_FIT_ON_GPU hardcoded constant")
        print("> it works well for SPECTUS base (354M parameters)")
        print("> it is computed from the effective BS)")
        print("> per device BS and GAS are overwritten")
        print("> it automatically distinguishes between 40GB and 80GB GPUs")

        # GPU specific batch size
        train_eff_bs = hf_training_args.pop("effective_train_batch_size")
        eval_eff_bs = hf_training_args.pop("effective_eval_batch_size")

        if spectus_size == "large": # experimental, not used in the paper
            if gpu_ram > 70*1e9:               # 80GB
                possible_to_fit_on_gpu = 32
                possible_to_fit_on_gpu_eval = 32
            else:                              # 40GB
                possible_to_fit_on_gpu = 16
                possible_to_fit_on_gpu_eval = 16

        elif spectus_size == "base":
            if gpu_ram > 70*1e9:               # 80GB
                possible_to_fit_on_gpu = 128
                possible_to_fit_on_gpu_eval = 64
            else:                              # 40GB
                possible_to_fit_on_gpu = 64
                possible_to_fit_on_gpu_eval = 32
        else:
            raise ValueError("spectus_size must be provided in hf_training_args if auto_bs is True")

        gas = train_eff_bs // (num_gpu * possible_to_fit_on_gpu)
        train_eff_bs = gas * num_gpu * possible_to_fit_on_gpu

        eas = eval_eff_bs // (num_gpu * possible_to_fit_on_gpu_eval)
        eval_eff_bs = eas * num_gpu * possible_to_fit_on_gpu_eval

        if gas == 0 or eas == 0:
            print("\nAUTO BS: Effective batch size (train or eval) is TOO SMALL for the type and num of GPUs")
            print("> use LESS GPUs or INCREASE effective batch size")
        else:
            print("\nAUTO BS")
            print(f"> GAS: {gas}")
            print(f"> effective train batch size: {gas * num_gpu * possible_to_fit_on_gpu}")
            print(f"> effective eval batch size: {eas * num_gpu * possible_to_fit_on_gpu_eval}")

        hf_training_args["per_device_train_batch_size"] = possible_to_fit_on_gpu
        hf_training_args["per_device_eval_batch_size"] = possible_to_fit_on_gpu_eval
        hf_training_args["gradient_accumulation_steps"] = gas
        hf_training_args["eval_accumulation_steps"] = gas

    else:
        print("Using   M A N U A L   batch size")
        assert hf_training_args.get("per_device_train_batch_size", None) and \
               hf_training_args.get("per_device_eval_batch_size", None) and \
               hf_training_args.get("gradient_accumulation_steps", None) is not None and \
               hf_training_args.get("eval_accumulation_steps", None) is not None, \
               "You must specify per_device_train_batch_size, per_device_eval_batch_size, \
               gradient_accumulation_steps and eval_accumulation_steps in config's \
               hf_training_args when auto_bs is False."

        hf_training_args.pop("effective_train_batch_size", None)  # these have to be removed if present
        hf_training_args.pop("effective_eval_batch_size", None)   # these have to be removed if present

    print(f"> train batch size per GPU: {hf_training_args['per_device_train_batch_size']}")
    print(f"> eval batch size per GPU: {hf_training_args['per_device_eval_batch_size']}")
    print(f"> gradient accumulation steps: {hf_training_args['gradient_accumulation_steps']}")
    print(f"> eval accumulation steps: {hf_training_args['eval_accumulation_steps']}")
    print(f"> num of GPUs: {num_gpu}")
    print(f"> GPU RAM: {gpu_ram}")

    return hf_training_args


def freeze_model(model, train_using_peft, train_fc1_only, custom_freeze):
    if train_using_peft:
        peft_config = peft.get_peft_config(peft_config_dict) # not working yet (wasn't tested)
        model = peft.get_peft_model(model, peft_config)

    if train_fc1_only:
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if "fc1" in name:
                param.requires_grad = True
        for name, param in model.get_decoder().named_parameters(): # type: ignore
            if "embed" in name:
                param.requires_grad = True

    if custom_freeze:
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True

        for name, param in model.get_encoder().named_parameters(): # type: ignore
            if "fc1" in name:
                param.requires_grad = True

        for name, param in model.get_decoder().named_parameters(): # type: ignore
            if "encoder_attn" in name or "self_attn" in name or "fc1" in name or "embed" in name:
                param.requires_grad = True



def get_spectro_config(model_args: Dict, tokenizer: transformers.PreTrainedTokenizerFast) -> SpectusConfig:
    assert not (bool(model_args.get("restrict_intensities", None)) ^ bool(model_args.get("encoder_seq_len", None))), "restrict_intensities and encoder_position_embeddings must be both provided or both None"

    return SpectusConfig(separate_encoder_decoder_embeds=model_args["separate_encoder_decoder_embeds"],
                             vocab_size=len(tokenizer.get_vocab()),
                             decoder_max_position_embeddings=model_args["decoder_seq_len"],
                             encoder_max_position_embeddings=model_args.get("encoder_seq_len", None), # specify only when restricting intensities, otherwise encoder embedding matrix is sized by max_log_id
                             max_length=model_args["decoder_seq_len"],
                             max_mz=model_args["max_mz"],
                             tie_word_embeddings=False,     # exrtremely important - enables two vocabs, don't change
                             min_len=0,
                             encoder_layers=model_args["encoder_layers"],
                             encoder_ffn_dim=model_args["encoder_ffn_dim"],
                             encoder_attention_heads=model_args["encoder_attention_heads"],
                             decoder_layers=model_args["decoder_layers"],
                             decoder_ffn_dim=model_args["decoder_ffn_dim"],
                             decoder_attention_heads=model_args["decoder_attention_heads"],
                             encoder_layerdrop=0.0,
                             decoder_layerdrop=0.0,
                             activation_function='gelu',
                             d_model=1024,
                             dropout=0.2,
                             attention_dropout=0.0,
                             activation_dropout=0.0,
                             init_std=0.02,
                             classifier_dropout=0.0,
                             scale_embedding=False,
                             use_cache=True,
                             pad_token_id=2,
                             bos_token_id=3,
                             eos_token_id=0,
                             is_encoder_decoder=True,
                             decoder_start_token_id=3,
                             forced_eos_token_id=0,
                             max_log_id=model_args.get("max_log_id", 9) if not model_args.get("restrict_intensities", False) else None # extremely important, don't change
                             )


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
    tokenizer_path = model_args["tokenizer_path"]
    report_to = hf_training_args.pop("report_to", "none")
    use_wandb = report_to == "wandb"

    # get freeze args
    train_using_peft = model_args.pop("train_using_peft", False)
    train_fc_only = model_args.pop("train_fc_only", False)
    custom_freeze = model_args.pop("custom_freeze", False)


    hf_training_args = set_batch_size(hf_training_args)

    # set the name for metric choosing the best model (add chosen dataset name)
    if dataset_args.get("dataset_for_choosing_best_model", None):
        hf_training_args["metric_for_best_model"] = enrich_best_metric_name(hf_training_args["metric_for_best_model"],
                                                                            dataset_args["dataset_for_choosing_best_model"])
        print(f"Metric for choosing best model: {hf_training_args['metric_for_best_model']}")
    else:
        raise ValueError("dataset_for_choosing_best_model must be provided in data_args.")

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

    print("max_log_id:", model.config.max_log_id)
    print("log_shift:", preprocess_args.get("log_shift", None))
    print("log_base:", preprocess_args.get("log_base", None))
    print("report_to:", report_to)

    model.to(device)

    # model freezing
    assert train_using_peft + train_fc_only + custom_freeze <= 1, "Only one of train_using_peft, train_fc_only, custom_freeze can be True"
    if train_fc_only or custom_freeze or train_using_peft:
        freeze_model(model, train_using_peft, train_fc_only, custom_freeze)
    tuned_params = sum(p.shape.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.shape.numel() for p in model.parameters())
    print(f"Number of trained parameters: {tuned_params}/{total_params} = {tuned_params/total_params*100:.2f}%")

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

    # set callbacks
    sorted_dataset_names = sorted([name for name in datapipes["example"].keys()])
    prediction_callback = PredictionLogger(datasets=[datapipes["example"][name] for name in sorted_dataset_names],
                                           source_tokens=[dataset_args["datasets"][name]["source_token"] for name in sorted_dataset_names],
                                           log_prefixes=sorted_dataset_names, # type: ignore
                                           collator=SpectroDataCollator(restrict_intensities=model_args.get("restrict_intensities", False)),
                                           log_every_n_steps=hf_training_args["eval_steps"],
                                           show_raw_preds=dataset_args["show_raw_preds"],
                                        #    log_to_wandb=use_wandb,
                                           generate_kwargs=example_gen_args,
                                        )

    compute_metrics = SpectroMetrics(tokenizer)
    seq2seq_training_args = transformers.Seq2SeqTrainingArguments(**hf_training_args,
                                                                    output_dir=str(save_path),
                                                                    run_name=run_name,
                                                                    data_seed=dataset_args["data_seed"],
                                                                    report_to=[report_to],
                                                                    # use_cpu= device == "cpu", # coming in future versions
                                                                    )


    trainer = transformers.Seq2SeqTrainer(
                    model=model,
                    args=seq2seq_training_args,
                    train_dataset=datapipes["train"],
                    eval_dataset=datapipes["valid"],
                    callbacks=[prediction_callback],
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                    data_collator = SpectroDataCollator(restrict_intensities=model_args.get("restrict_intensities", False)),
                )


    if checkpoint and resume_id:
        trainer.train(resume_from_checkpoint=str(checkpoint))
    else:
        trainer.train()


if __name__ == "__main__":
    app()