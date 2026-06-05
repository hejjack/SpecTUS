"""MCES-distance split analysis for the NIST SMILES folds.

Identifies validation/test molecules whose nearest training-set neighbour is at
MCES distance >= a threshold (default 10), showing the train/val/test split
contains structurally novel molecules. Three stages: ``PreFilter`` (rigorous
bound screen -> candidate batches), ``BatchComputer`` (myopic-MCES), ``SplitReport``
(far/close split at any threshold). See ``README.md``.
"""
from mces_split.bounds import BondTypeBound, CompositeBound, Filter1Bound, RigorousBound
from mces_split.compute import BatchComputer
from mces_split.dataset import Folds
from mces_split.prefilter import PreFilter
from mces_split.split import SplitReport

__all__ = [
    "Folds",
    "RigorousBound",
    "Filter1Bound",
    "BondTypeBound",
    "CompositeBound",
    "PreFilter",
    "BatchComputer",
    "SplitReport",
]
