"""Rigorous, vectorised lower bounds on the myopic-MCES distance.

MCES runs on the heavy-atom graph with bond-order edge weights (single 1, double
2, triple 3, aromatic 1.5; see ``myopic_mces.graph.construct_graph``). Each bound
maps a molecule to a feature vector such that

    bound(a, b) = scale * cityblock(feature(a), feature(b))  <=  MCES(a, b)

so a single ``cdist`` screens a whole reference set, and we only run the
expensive ILP on pairs that *could* be below the threshold. Two independent
bounds are combined (``CompositeBound``) by taking their max — still a valid
lower bound, but much tighter than either alone.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from rdkit import Chem
from scipy.spatial.distance import cdist


class RigorousBound(ABC):
    """A vectorisable lower bound: ``scale * cityblock(feature(a), feature(b))``."""

    scale: float

    @abstractmethod
    def feature_matrix(self, mols: list[Chem.Mol]) -> np.ndarray:
        """Build the [n_molecules, n_features] feature matrix for these molecules."""

    def cityblock_cutoff(self, ceiling: float) -> float:
        """Cityblock distance below which the bound is < ceiling (so a pair may be too)."""
        return ceiling / self.scale


class Filter1Bound(RigorousBound):
    """Degree bound (``myopic_mces.filter_MCES.filter1``), in vectorised form.

    For each element, the atoms' weighted degrees (sum of incident bond orders)
    are sorted descending and zero-padded to a global length; the half-cityblock
    of the concatenation equals filter1.
    """

    scale = 0.5

    def feature_matrix(self, mols: list[Chem.Mol]) -> np.ndarray:
        seqs = [self._element_degree_sequences(m) for m in mols]
        layout: dict[str, int] = {}
        for s in seqs:
            for element, degrees in s.items():
                layout[element] = max(layout.get(element, 0), len(degrees))
        offsets, total = {}, 0
        for element in sorted(layout):
            offsets[element] = total
            total += layout[element]
        mat = np.zeros((len(mols), total), dtype=np.float32)
        for row, s in enumerate(seqs):
            for element, degrees in s.items():
                off = offsets[element]
                mat[row, off : off + len(degrees)] = degrees
        return mat

    @staticmethod
    def _element_degree_sequences(mol: Chem.Mol) -> dict[str, list[float]]:
        seqs: dict[str, list[float]] = {}
        for atom in mol.GetAtoms():
            degree = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
            seqs.setdefault(atom.GetSymbol(), []).append(degree)
        for symbol in seqs:
            seqs[symbol].sort(reverse=True)
        return seqs


class BondTypeBound(RigorousBound):
    """Bond-type bound: edges only map between equal element-pair types, so the
    per-bond-type total-weight cityblock is a valid lower bound (each bond counted
    once, hence scale 1)."""

    scale = 1.0

    def feature_matrix(self, mols: list[Chem.Mol]) -> np.ndarray:
        weights = [self._bondtype_weights(m) for m in mols]
        types = sorted({t for w in weights for t in w})
        index = {t: i for i, t in enumerate(types)}
        mat = np.zeros((len(mols), len(types)), dtype=np.float32)
        for row, w in enumerate(weights):
            for t, value in w.items():
                mat[row, index[t]] = value
        return mat

    @staticmethod
    def _bondtype_weights(mol: Chem.Mol) -> dict[tuple[str, str], float]:
        w: dict[tuple[str, str], float] = {}
        for bond in mol.GetBonds():
            t = tuple(sorted((bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol())))
            w[t] = w.get(t, 0.0) + bond.GetBondTypeAsDouble()
        return w


class CompositeBound:
    """Several rigorous bounds combined by their max — a pair survives only if it
    is below the ceiling under *every* bound."""

    def __init__(self, bounds: list[RigorousBound]):
        self.bounds = bounds

    def feature_matrices(self, mols: list[Chem.Mol]) -> list[np.ndarray]:
        """One feature matrix per bound (build over all molecules to share the layout)."""
        return [b.feature_matrix(mols) for b in self.bounds]

    def candidate_mask(self, query_feats: list[np.ndarray], train_feats: list[np.ndarray],
                       ceiling: float) -> np.ndarray:
        """
        Boolean [n_query, n_train] mask of pairs that could be below the ceiling.

        Args:
            query_feats (list[np.ndarray]): per-bound query feature blocks.
            train_feats (list[np.ndarray]): per-bound train feature blocks.
            ceiling (float): MCES threshold.
        Returns:
            np.ndarray: True where every bound is below the ceiling.
        """
        mask: np.ndarray | None = None
        for bound, qf, tf in zip(self.bounds, query_feats, train_feats):
            below = cdist(qf, tf, metric="cityblock") < bound.cityblock_cutoff(ceiling)
            mask = below if mask is None else (mask & below)
        return mask
