"""Shared I/O for the MCES-distance split: HDF5 batch read/write and the per-pair
time cap.

Batches use the ``myopic_mces`` ``--hdf5_mode`` layout: a ``computation_indices``
array of ``(id, query_idx, train_idx)`` rows plus a ``smiles`` array they index
into. Stage 2 adds ``mces`` (distance) and ``computation_modes`` to each batch.
"""
from __future__ import annotations

import signal
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa

NEAR_SCHEMA = pa.schema([("query", pa.string()), ("train", pa.string()), ("dist", pa.float64())])


def decode_smiles(arr) -> np.ndarray:
    """Decode an h5py SMILES array (variable-length bytes) to an object array of str."""
    return np.array([s.decode() if isinstance(s, bytes) else str(s) for s in arr], dtype=object)


def read_smiles(path: Path) -> np.ndarray:
    """Read the (decoded) SMILES array from a batch — identical across all batches."""
    with h5py.File(path, "r") as f:
        return decode_smiles(np.asarray(f["smiles"]))


def read_computed(path: Path) -> pd.DataFrame:
    """
    Read a computed batch into a (query, train, dist, mode) frame.

    Args:
        path (Path): batch HDF5 file that already carries the ``mces`` dataset.
    Returns:
        pd.DataFrame: one row per computed pair.
    """
    with h5py.File(path, "r") as f:
        if "mces" not in f:
            raise ValueError(f"{path} has no 'mces' dataset — run the compute stage on it first")
        idx = f["computation_indices"][:]
        smiles = decode_smiles(np.asarray(f["smiles"]))
        dist = np.asarray(f["mces"], dtype=float)
        mode = np.asarray(f["computation_modes"], dtype=int)
    return pd.DataFrame(dict(query=smiles[idx[:, 1]], train=smiles[idx[:, 2]], dist=dist, mode=mode))


def scan_error_query_smiles(batches_dir: Path) -> set[str]:
    """
    Distinct query SMILES that have at least one failed (dist<0) pair.

    These queries have an uncomputable distance (a blind spot), so their
    far/close status is unknowable at any threshold. Reads ``computation_indices``
    contiguously then indexes it — h5py fancy-indexing on gzip is very slow.

    Args:
        batches_dir (Path): directory of computed batch HDF5 files.
    Returns:
        set[str]: query SMILES with a blind spot.
    """
    paths = sorted(batches_dir.glob("batch*.hdf5"))
    smiles = read_smiles(paths[0])  # identical across batches
    bad: set[str] = set()
    for path in paths:
        with h5py.File(path, "r") as f:
            e = np.where(np.asarray(f["mces"]) < 0)[0]
            if len(e) == 0:
                continue
            idx = np.asarray(f["computation_indices"])[e]
        bad.update(smiles[idx[:, 1]])
    return bad


class BatchWriter:
    """Streams ``(id, query_idx, train_idx)`` rows into fixed-size HDF5 batches."""

    def __init__(self, out_dir: Path, smiles: list[str], batch_size: int, settings: dict):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.smiles = smiles
        self.batch_size = batch_size
        self.settings = settings
        self._buffer: list[np.ndarray] = []  # pending [k, 2] (query_idx, train_idx) blocks
        self._pending = 0
        self._next_id = 0
        self.n_batches = 0

    def add(self, query_idx: np.ndarray, train_idx: np.ndarray) -> None:
        """Queue candidate pairs, flushing whole batches as they fill."""
        self._buffer.append(np.column_stack([query_idx, train_idx]))
        self._pending += len(query_idx)
        while self._pending >= self.batch_size:
            self._flush(self.batch_size)

    def close(self) -> int:
        """Flush the remainder and return the number of batches written."""
        if self._pending:
            self._flush(self._pending)
        return self.n_batches

    def _flush(self, n: int) -> None:
        pairs = np.concatenate(self._buffer) if len(self._buffer) > 1 else self._buffer[0]
        take, rest = pairs[:n], pairs[n:]
        ids = np.arange(self._next_id, self._next_id + len(take))
        indices = np.column_stack([ids, take]).astype("int64")
        with h5py.File(self.out_dir / f"batch{self.n_batches}.hdf5", "w") as f:
            f.create_dataset("computation_indices", data=indices, dtype="int64", compression="gzip")
            f.create_dataset("smiles", data=self.smiles)
            group = f.create_group("settings")
            for key, value in self.settings.items():
                group[key] = value
        self._next_id += len(take)
        self.n_batches += 1
        self._buffer = [rest] if len(rest) else []
        self._pending = len(rest)


@contextmanager
def time_limit(seconds: float | None) -> Iterator[None]:
    """
    Raise ``TimeoutError`` if the wrapped block runs longer than ``seconds``.

    Bounds a single MCES pair: some pairs spend minutes in the (pure-Python)
    matching bound, which the solver's own time limit does not cover. ``None`` or
    0 disables the cap. Main-thread only (SIGALRM).
    """
    if not seconds:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"exceeded {seconds}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
