"""The NIST train/val/test molecule folds, canonicalised to match SpecTUS.

Molecule identity uses the same canonicalisation as SpecTUS's data-loading path
(``utils.spectra_process_utils.remove_stereochemistry_and_canonicalize``): strip
stereochemistry, then canonical SMILES. On the stereo-free NIST ``.smi`` files
this is a no-op (verified byte-identical), but it pins our molecule set, dedup,
and leakage check to SpecTUS's by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def canonicalize(smiles: str) -> str | None:
    """SpecTUS-consistent canonical SMILES (stereo removed); None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def _load_unique(path: Path, limit: int | None) -> tuple[list[str], int]:
    seen: dict[str, None] = {}
    n_invalid = 0
    with open(path) as f:
        for line in f:
            smiles = line.strip()
            if not smiles:
                continue
            canon = canonicalize(smiles)
            if canon is None:
                n_invalid += 1
                continue
            seen.setdefault(canon, None)
            if limit is not None and len(seen) >= limit:
                break
    return list(seen), n_invalid


@dataclass
class Folds:
    """Unique canonical molecules per fold, plus the query universe (valid+test)."""

    train: list[str]
    valid: list[str]
    test: list[str]
    unparseable: dict[str, int]

    @classmethod
    def load(cls, data_dir: Path, limit_train: int | None = None, limit_queries: int | None = None) -> Folds:
        """
        Load and canonicalise train/valid/test ``.smi`` files.

        Args:
            data_dir (Path): directory with train.smi, valid.smi, test.smi.
            limit_train (int | None): cap unique train molecules (testing).
            limit_queries (int | None): cap unique molecules per query fold (testing).
        Returns:
            Folds: the loaded, de-duplicated folds.
        """
        train, inv_train = _load_unique(data_dir / "train.smi", limit_train)
        valid, inv_valid = _load_unique(data_dir / "valid.smi", limit_queries)
        test, inv_test = _load_unique(data_dir / "test.smi", limit_queries)
        return cls(train, valid, test, dict(train=inv_train, valid=inv_valid, test=inv_test))

    @property
    def n_train(self) -> int:
        return len(self.train)

    def queries(self) -> pd.DataFrame:
        """
        One row per unique query molecule.

        Returns:
            pd.DataFrame: columns ``smiles``, ``in_valid``, ``in_test``, ``status``
            (``exact_match`` for a training duplicate, else ``candidates``).
        """
        in_valid, in_test, train = set(self.valid), set(self.test), set(self.train)
        rows = [
            dict(smiles=s, in_valid=s in in_valid, in_test=s in in_test,
                 status="exact_match" if s in train else "candidates")
            for s in dict.fromkeys(self.valid + self.test)
        ]
        return pd.DataFrame(rows)
