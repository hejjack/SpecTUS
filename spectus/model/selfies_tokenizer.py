from __future__ import annotations
import json
from pathlib import Path
import selfies as sf

from transformers import PreTrainedTokenizer


# we need
# encode
# decode
# batch_decode  ?? decode_batch
# pad_token_id
# eos_token_id
# unk_token
# add_special_tokens
# get_vocab
# num_special_tokens_to_add
# token_to_id

# batch_encode_plus

# allowing different constraints - motivated by NIST database
default_constraints = sf.get_semantic_constraints()
default_constraints["I"] = 5
default_constraints["Ti"] = 13
default_constraints["P"] = 6
default_constraints["P-1"] = 6
default_constraints["Co"] = 10
default_constraints["Mo"] = 12
sf.set_semantic_constraints(default_constraints)


class SelfiesTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab: dict[str, int] = {}, max_len: int | None = None, **kwargs):
        super().__init__(max_len=max_len, **kwargs)
        self.__token_ids: dict[str, int] = vocab
        self.__id_tokens: dict[int, str] = {value: key for key, value in vocab.items()}
        self.special_tokens = {}

    def add_special_tokens(self, *args, **kwargs):
        """The functionality stays the same, we just want to have the special tokens
        explicitly saved
        """
        out = super().add_special_tokens(*args, **kwargs)
        special_keys = []
        for value in self.special_tokens_map.values():
            if isinstance(value, list):
                special_keys.extend(value)
            else:
                special_keys.append(value)

        self.special_tokens = {key: self.__token_ids[key] for key in special_keys}
        return out

    def _tokenize(self, text: str, **kwargs):
        return sf.split_selfies(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.__token_ids[token] if token in self.__token_ids else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self.__id_tokens[index] if index in self.__id_tokens else self.unk_token

    def get_vocab(self) -> dict[str, int]:
        return self.__token_ids.copy()

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        if filename_prefix is None:
            filename_prefix = ''
        vocab_path = Path(save_directory, filename_prefix + 'vocab.json')
        json.dump(self.__token_ids, open(vocab_path, 'w'))
        return str(vocab_path),

    def get_vocab_size(self, with_added_tokens=True) -> int:
        return self.vocab_size

    def _decode(
        self,
        token_ids: int | list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None, # not implemented
        **kwargs,
    ) -> str:
        if skip_special_tokens:
            special_ids = list(self.special_tokens.values())
            if isinstance(token_ids, int):
                return self._convert_id_to_token(token_ids) if token_ids not in special_ids else ''
            tokens = [self._convert_id_to_token(token_id) for token_id in token_ids if token_id not in special_ids]

        else:
            if isinstance(token_ids, int):
                return self._convert_id_to_token(token_ids)
            tokens = [self._convert_id_to_token(token_id) for token_id in token_ids]
        return "".join(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.__token_ids)

    @property
    def vocab(self) -> dict[str, int]:
        return self.__token_ids

def hardcode_build_selfies_tokenizer() -> SelfiesTokenizer:
    vocab = sf.get_semantic_robust_alphabet()
    vocab = sorted(list(vocab))
    special_tokens = ['<eos>', '<unk>', '<pad>', '<bos>']
    sources = ['<nist>', '<rassp>', '<neims>', '<trafo>', '<source1>', '<source2>', '<source3>']
    vocab = special_tokens + vocab + sources


    vocab_dict = {key: i for i, key in enumerate(vocab)}

    sel_tokenizer = SelfiesTokenizer(vocab_dict, max_len=200)

    # tell your tokenizer about your special tokens
    sel_tokenizer.add_special_tokens({
        'unk_token': '<unk>',
        'pad_token': '<pad>',
        'bos_token': '<bos>',
        'eos_token': '<eos>',
        'additional_special_tokens': sources
    })
    return sel_tokenizer