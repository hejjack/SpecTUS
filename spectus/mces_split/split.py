"""Stage 3: derive the far/close split from the computed batches.

First streams every exact near pair (distance < ceiling) into ``near_pairs.parquet``
(bounded memory; skipped if it already exists). From that single file the split
can be reported at any threshold <= ceiling with no MCES recompute -- optionally
excluding molecules with an incomputable pair, which removes every blind spot so
the far set is certain at all thresholds at once.

A query is *far* at threshold T when its nearest train neighbour is at distance
>= T (matching MassSpecGym's single-linkage criterion). Only exact (mode-1) pairs
below the ceiling can make a query close; errors and wall-caps are excluded.

Outputs (in ``--results-dir``): near_pairs.parquet, nn_distances.parquet (per
query), incomputable.smi (if excluding), and far_{valid,test}_t{T}[_clean].smi.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mces_split.io import NEAR_SCHEMA, read_computed, scan_error_query_smiles


class SplitReport:
    """Builds the near-pairs file and reports the far/close split at any threshold."""

    def __init__(self, out_dir: Path, results_dir: Path):
        self.out_dir = out_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.ceiling = json.loads((out_dir / "manifest.json").read_text())["ceiling"]
        self.queries = pd.read_parquet(out_dir / "queries.parquet")
        self.near_path = results_dir / "near_pairs.parquet"

    def build_near_pairs(self) -> None:
        """Stream exact near pairs (< ceiling) from all batches into near_pairs.parquet."""
        if self.near_path.exists():
            print(f"using existing {self.near_path.name} (skipping batch streaming)")
            return
        writer = pq.ParquetWriter(self.near_path, NEAR_SCHEMA)
        n_pairs = n_err = n_capped = n_near = 0
        try:
            for path in sorted((self.out_dir / "batches").glob("batch*.hdf5")):
                df = read_computed(path)
                n_pairs += len(df)
                n_err += int((df["dist"] < 0).sum())
                n_capped += int((df["mode"] == 9).sum())
                near = df[(df["mode"] == 1) & (df["dist"] >= 0) & (df["dist"] < self.ceiling)]
                if not near.empty:
                    n_near += len(near)
                    writer.write_table(pa.Table.from_pandas(near[["query", "train", "dist"]],
                                                            schema=NEAR_SCHEMA, preserve_index=False))
        finally:
            writer.close()
        print(f"computed pairs: {n_pairs:,}  near(<{self.ceiling}): {n_near:,}  "
              f"errors: {n_err}  wall-capped: {n_capped}")

    def _annotate(self, exclude_incomputable: bool) -> pd.DataFrame:
        """Per-query table with nearest distance, witness, and (optionally) incomputable flag."""
        near = pd.read_parquet(self.near_path)
        nearest = near.loc[near.groupby("query")["dist"].idxmin()].set_index("query")
        q = self.queries.copy()
        q["nearest"] = q["smiles"].map(nearest["dist"])
        q["witness"] = q["smiles"].map(nearest["train"])
        q.loc[q["status"] == "exact_match", "nearest"] = 0.0
        if exclude_incomputable:
            bad = scan_error_query_smiles(self.out_dir / "batches")
            q["incomputable"] = q["smiles"].isin(bad)
            (self.results_dir / "incomputable.smi").write_text("\n".join(q.loc[q["incomputable"], "smiles"]) + "\n")
        return q

    def report(self, thresholds: list[float], exclude_incomputable: bool = False) -> pd.DataFrame:
        """
        Write the far sets and per-query table; print and return a summary.

        Args:
            thresholds (list[float]): thresholds to slice at (each <= ceiling).
            exclude_incomputable (bool): drop molecules with any failed pair.
        Returns:
            pd.DataFrame: rows of (threshold, far_valid, far_test, n_valid, n_test).
        """
        q = self._annotate(exclude_incomputable)
        q.to_parquet(self.results_dir / "nn_distances.parquet")
        usable = q[~q.get("incomputable", False)] if exclude_incomputable else q
        n_valid, n_test = int(usable["in_valid"].sum()), int(usable["in_test"].sum())
        suffix = "_clean" if exclude_incomputable else ""
        if exclude_incomputable:
            print(f"incomputable excluded: valid {int((q['in_valid'] & q['incomputable']).sum())}, "
                  f"test {int((q['in_test'] & q['incomputable']).sum())} (of {int(q['incomputable'].sum())})")
        print(f"clean universe: valid {n_valid}, test {n_test} (ceiling {self.ceiling})")

        rows = []
        for t in sorted(thresholds):
            if t > self.ceiling:
                print(f"  t={t}: skipped (> computed ceiling {self.ceiling})")
                continue
            is_far = ~(usable["nearest"] < t)  # NaN (no near neighbour) -> far
            fv = usable.loc[usable["in_valid"] & is_far, "smiles"]
            ft = usable.loc[usable["in_test"] & is_far, "smiles"]
            (self.results_dir / f"far_valid_t{int(t)}{suffix}.smi").write_text("\n".join(fv) + "\n")
            (self.results_dir / f"far_test_t{int(t)}{suffix}.smi").write_text("\n".join(ft) + "\n")
            print(f"  far >= {t:>2}: valid {len(fv):>6} ({100 * len(fv) / n_valid:5.2f}%)  "
                  f"test {len(ft):>6} ({100 * len(ft) / n_test:5.2f}%)")
            rows.append(dict(threshold=t, far_valid=len(fv), far_test=len(ft), n_valid=n_valid, n_test=n_test))
        return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="stage-1 dir (batches/, queries.parquet, manifest.json)")
    parser.add_argument("--results-dir", type=Path, required=True, help="where to write near pairs and far sets")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[10.0], help="thresholds to slice at (<= ceiling)")
    parser.add_argument("--exclude-incomputable", action="store_true",
                        help="drop molecules with any failed pair -> far set certain at all thresholds")
    args = parser.parse_args()
    report = SplitReport(args.out_dir, args.results_dir)
    report.build_near_pairs()
    report.report(args.thresholds, args.exclude_incomputable)


if __name__ == "__main__":
    main()
