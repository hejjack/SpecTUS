"""Stage 2: compute myopic-MCES for one HDF5 batch.

Thin wrapper around ``myopic_mces.MCES`` (Fleming's algorithm) adding a per-pair
solver time limit and a hard wall-clock cap -- a tiny fraction of pairs otherwise
dominate the runtime, and some of that is in the (pure-Python) matching bound that
the solver limit does not cover. Results (``mces`` / ``computation_modes`` /
``computation_times``) are written back into the batch, matching ``--hdf5_mode``.

Distance modes: 1 = exact (<= ceiling); 2/4 = bound (>= ceiling); 9 = wall-capped
(treated as >= ceiling, flagged); dist < 0 = per-pair error. A capped or errored
pair never becomes a false "near", since it is excluded from the near pairs.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from joblib import Parallel, delayed
from myopic_mces.myopic_mces import MCES
from rdkit import RDLogger

from mces_split.io import time_limit

RDLogger.DisableLog("rdApp.*")

CAPPED_MODE = 9


def _run_pair(rid: int, smiles1: str, smiles2: str, threshold: float, solver: str,
              solver_options: dict, wall_cap: float | None) -> tuple:
    """Compute one pair under a hard wall-clock cap; cap hit -> ceiling, mode 9."""
    t0 = time.time()
    try:
        with time_limit(wall_cap):
            return MCES(smiles1, smiles2, threshold, rid, solver, solver_options,
                        always_stronger_bound=False, catch_errors=True)
    except TimeoutError:
        return (rid, float(threshold), time.time() - t0, CAPPED_MODE)


class BatchComputer:
    """Computes a candidate-pair batch with myopic-MCES under solver + wall caps."""

    def __init__(self, threshold: float = 10.0, solver: str = "PULP_CBC_CMD",
                 solver_time_limit: float | None = 30.0, wall_cap: float | None = 45.0, n_jobs: int = -1):
        self.threshold = threshold
        self.solver = solver
        self.solver_time_limit = solver_time_limit
        self.wall_cap = wall_cap
        self.n_jobs = n_jobs

    def compute(self, batch_path: Path) -> None:
        """Compute every pair in ``batch_path`` and write the results back in place."""
        with h5py.File(batch_path, "r") as f:
            idx = f["computation_indices"][:]
            smiles = [s.decode() if isinstance(s, bytes) else str(s) for s in np.asarray(f["smiles"])]

        options = dict(msg=False, threads=1)  # single-threaded solver under joblib parallelism
        if self.solver_time_limit:
            options["timeLimit"] = self.solver_time_limit
        results = Parallel(n_jobs=self.n_jobs, batch_size=32, pre_dispatch="10*n_jobs")(
            delayed(_run_pair)(int(rid), smiles[i1], smiles[i2], self.threshold, self.solver, options, self.wall_cap)
            for rid, i1, i2 in idx
        )
        dist = np.array([r[1] for r in results], dtype=float)
        times = np.array([r[2] for r in results], dtype=float)
        modes = np.array([r[3] for r in results], dtype="uint8")

        with h5py.File(batch_path, "a") as f:
            for key, data, kw in (("mces", dist, {}), ("computation_times", times, {}),
                                  ("computation_modes", modes, dict(dtype="uint8"))):
                if key in f:
                    del f[key]
                f.create_dataset(key, data=data, compression="gzip", **kw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True, help="HDF5 batch to compute in place")
    parser.add_argument("--threshold", type=float, default=10.0, help="MCES ceiling")
    parser.add_argument("--solver", default="PULP_CBC_CMD", help="PuLP solver (CPLEX_CMD if available)")
    parser.add_argument("--solver-time-limit", type=float, default=30.0, help="per-pair ILP cap (s; 0 disables)")
    parser.add_argument("--wall-cap", type=float, default=45.0, help="hard per-pair wall cap (s; 0 disables)")
    parser.add_argument("--num-jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    args = parser.parse_args()
    BatchComputer(args.threshold, args.solver, args.solver_time_limit or None,
                  args.wall_cap or None, args.num_jobs).compute(args.batch)


if __name__ == "__main__":
    main()
