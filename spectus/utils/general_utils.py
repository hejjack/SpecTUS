import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from io import TextIOWrapper
import time
from tokenizers import Tokenizer
import transformers
import pathlib
import torch




def build_tokenizer(tokenizer_path: str) -> transformers.PreTrainedTokenizerFast:
    bpe_tokenizer = Tokenizer.from_file(tokenizer_path)

    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=bpe_tokenizer,
                                        bos_token="<bos>",
                                        eos_token="<eos>",
                                        unk_token="<unk>",
                                        pad_token="<pad>",
                                        is_split_into_words=True,
                                        clean_up_tokenization_spaces=True)
    return tokenizer


def move_file_pointer(num_lines: int, file_pointer: TextIOWrapper) -> None:
    """Move the file pointer a specified number of lines forward"""
    for _ in range(num_lines):
        file_pointer.readline()


def line_count(file_path: pathlib.Path):
    """Count number of lines in a file"""
    f = open(file_path, "r")
    file_len =  sum(1 for _ in f)
    f.close()
    return file_len


def dummy_generator(from_n_onwards=0):
    i = from_n_onwards
    while True:
        yield i
        i += 1


def get_sequence_probs(model,
                       generated_outputs: transformers.utils.ModelOutput,
                       batch_size: int,
                       is_beam_search: bool):
    """ collect the generation probability of all generated sequences """
    transition_scores = model.compute_transition_scores(
                                generated_outputs.sequences,
                                generated_outputs.scores,
                                beam_indices=generated_outputs.beam_indices if is_beam_search else None,
                                normalize_logits=True)  # type: ignore

    transition_scores = transition_scores.reshape(batch_size, -1, transition_scores.shape[-1])
    transition_scores[torch.isinf(transition_scores)] = 0
    if is_beam_search:
        output_length = (transition_scores < 0).sum(dim=-1)
        length_penalty = model.generation_config.length_penalty
        all_probs = transition_scores.sum(dim=2).exp() / output_length**length_penalty
    else:
        all_probs = transition_scores.sum(dim=2).exp()
    return all_probs


def timestamp_to_readable(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%d/%m/%Y %H:%M:%S")


def hours_minutes_seconds(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def get_nice_time():
    now = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    now = now.replace(":", "_").replace(" ", "-")
    return now


@contextmanager
def suppress_subprocess_output():
    with open(os.devnull, 'w') as fnull:
        original_popen = subprocess.Popen
        try:
            # Override subprocess.Popen to redirect stdout and stderr
            def custom_popen(*args, **kwargs):
                kwargs['stdout'] = fnull
                kwargs['stderr'] = fnull
                return original_popen(*args, **kwargs)

            subprocess.Popen = custom_popen
            yield
        finally:
            # Restore the original Popen function
            subprocess.Popen = original_popen