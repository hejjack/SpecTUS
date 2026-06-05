"""Stage 1: screen every query against train with the rigorous bounds and emit
only the surviving candidate pairs as myopic-MCES HDF5 batches.

For each of the ~48k query (valid+test) molecules we want its minimum MCES
distance to ~195k train molecules -- ~9.5B pairs if done naively. The composite
bound discards the ~99% that are provably >= the ceiling at numpy speed; only the
structurally similar survivors are written out for the expensive ILP.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from tqdm import tqdm

from mces_split.bounds import BondTypeBound, CompositeBound, Filter1Bound
from mces_split.dataset import Folds
from mces_split.io import BatchWriter


class PreFilter:
    """Screens query x train pairs with a composite lower bound into HDF5 batches."""

    def __init__(self, bound: CompositeBound | None = None, ceiling: float = 10.0,
                 batch_size: int = 10_000_000, query_chunk: int = 512, seed: int = 0):
        self.bound = bound or CompositeBound([Filter1Bound(), BondTypeBound()])
        self.ceiling = ceiling
        self.batch_size = batch_size
        self.query_chunk = query_chunk
        self.seed = seed

    def run(self, folds: Folds, out_dir: Path) -> dict:
        """
        Screen the folds and write candidate batches + queries.parquet + manifest.json.

        Args:
            folds (Folds): the loaded train/valid/test molecules.
            out_dir (Path): output directory.
        Returns:
            dict: the manifest.
        """
        queries = folds.queries()
        candidates = list(queries.loc[queries["status"] == "candidates", "smiles"])
        smiles = folds.train + candidates  # global array; train first, then candidate queries
        n_train = folds.n_train
        mols = [Chem.MolFromSmiles(s) for s in smiles]

        feats = self.bound.feature_matrices(mols)  # one matrix per bound, over all molecules
        train_feats = [f[:n_train] for f in feats]
        query_feats = [f[n_train:] for f in feats]

        settings = dict(threshold=float(self.ceiling), choose_bound_dynamically=True)
        writer = BatchWriter(out_dir / "batches", smiles, self.batch_size, settings)
        has_candidate = np.zeros(len(candidates), dtype=bool)
        order = np.random.default_rng(self.seed).permutation(len(candidates))  # mix across batches
        n_pairs = 0
        for start in tqdm(range(0, len(order), self.query_chunk), desc="prefilter", unit="chunk"):
            block = order[start : start + self.query_chunk]
            mask = self.bound.candidate_mask([f[block] for f in query_feats], train_feats, self.ceiling)
            qi, ti = np.nonzero(mask)
            if len(qi) == 0:
                continue
            has_candidate[block[qi]] = True
            writer.add(block[qi] + n_train, ti)
            n_pairs += len(qi)
        n_batches = writer.close()

        # A candidate query with zero surviving pairs is provably far (no MCES needed).
        status = {candidates[i]: ("candidates" if has_candidate[i] else "no_candidates")
                  for i in range(len(candidates))}
        queries["status"] = queries.apply(lambda r: status.get(r["smiles"], r["status"]), axis=1)

        out_dir.mkdir(parents=True, exist_ok=True)
        queries.to_parquet(out_dir / "queries.parquet")
        manifest = dict(
            ceiling=self.ceiling, n_train=n_train, n_valid_unique=len(folds.valid),
            n_test_unique=len(folds.test), n_query_unique=len(queries),
            n_candidate_pairs=int(n_pairs), n_batches=n_batches, batch_size=self.batch_size,
            status_counts=queries["status"].value_counts().to_dict(), unparseable=folds.unparseable,
        )
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"candidate pairs: {n_pairs:,} in {n_batches} batches; status: {manifest['status_counts']}")
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/nist"), help="dir with {train,valid,test}.smi")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ceiling", type=float, default=10.0, help="MCES distance threshold")
    parser.add_argument("--batch-size", type=int, default=10_000_000, help="candidate pairs per HDF5 batch")
    parser.add_argument("--query-chunk", type=int, default=512, help="queries per cdist block")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=None, help="testing: cap train molecules")
    parser.add_argument("--limit-queries", type=int, default=None, help="testing: cap queries per fold")
    args = parser.parse_args()

    folds = Folds.load(args.data_dir, args.limit_train, args.limit_queries)
    print(f"unique canonical: train={folds.n_train} valid={len(folds.valid)} test={len(folds.test)} "
          f"(unparseable: {folds.unparseable})")
    PreFilter(ceiling=args.ceiling, batch_size=args.batch_size,
              query_chunk=args.query_chunk, seed=args.seed).run(folds, args.out_dir)


if __name__ == "__main__":
    main()
