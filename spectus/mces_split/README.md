# MCES-distance split analysis

Identifies validation/test molecules whose **nearest training-set neighbour is at
MCES distance ≥ T** (default 10), showing the NIST train/val/test split (built on
InChIKey14) contains structurally novel molecules — not just near-duplicates of
training compounds. Distance 10 matches the single-linkage criterion of the
MCES-based split in MassSpecGym; the metric is the myopic maximum-common-edge-subgraph
edit distance from [`myopic-mces`](https://github.com/AlBi-HHU/myopic-mces).

## Result (NIST, ceiling 10)

Molecules ≥ T MCES edits from *every* training molecule, over the clean universe
(molecules where every distance is computable):

| T | far_valid | far_test |
|---|-----------|----------|
| 6 | 2,164 (8.96%) | 2,170 (8.98%) |
| 8 | 1,005 (4.16%) | 1,002 (4.15%) |
| 10 | 446 (1.85%) | 469 (1.94%) |

357 molecules (0.7%) were excluded as MCES-intractable (very large structures
where the ILP cannot terminate); 0 false positives among all computable pairs.

## Why it is fast *and* rigorous

For ~48k query × ~195k train molecules (~9.5 B pairs), almost all are far. The
candidates are pruned with **two vectorised, rigorous MCES lower bounds**
(`bounds.py`):

* **`Filter1Bound`** — the degree bound (`myopic_mces.filter1`) as a half-cityblock
  of zero-padded sorted per-element weighted-degree sequences.
* **`BondTypeBound`** — edges only map between equal element-pair types, so the
  per-bond-type total-weight cityblock is a valid lower bound.

`CompositeBound` takes their max (still a true lower bound), so a discarded pair
is *provably* ≥ ceiling — no false "far". Only the ~9% of survivors reach the ILP.

## Architecture (`spectus/mces_split/`)

| Module | Class | Role |
|--------|-------|------|
| `bounds.py` | `RigorousBound` → `Filter1Bound`, `BondTypeBound`, `CompositeBound` | rigorous vectorised lower bounds |
| `dataset.py` | `Folds` | load/canonicalise (SpecTUS-consistent)/dedupe the `.smi` folds |
| `prefilter.py` | `PreFilter` | bound screen → candidate-pair HDF5 batches + manifest |
| `compute.py` | `BatchComputer` | myopic-MCES on a batch (solver + wall caps) |
| `split.py` | `SplitReport` | near-pairs → far/close split at any threshold, ± incomputable exclusion |
| `io.py` | — | shared HDF5 read/write, error-pair scan, per-pair `time_limit` |

## Workflow

```
prefilter   data/nist/{train,valid,test}.smi ─▶ batches/*.hdf5 + queries.parquet + manifest.json
compute     batches/*.hdf5 (one per array task) ─▶ same files, now with `mces` distances
report      batches + queries.parquet ─▶ near_pairs.parquet, nn_distances.parquet,
                                         incomputable.smi, far_{valid,test}_t{T}_clean.smi
```

### Local (small sample)

```bash
pip install -e ".[mces]"
python -m mces_split.prefilter --out-dir /tmp/run --limit-queries 200 --limit-train 10000
for b in /tmp/run/batches/batch*.hdf5; do python -m mces_split.compute --batch "$b"; done
python -m mces_split.split --out-dir /tmp/run --results-dir /tmp/run/results --thresholds 6 8 10 --exclude-incomputable
```

### Full run on MetaCentrum (plzen1)

```bash
bash config_runners/run_mces_split.sh   # prefilter ▸ compute array; then run the report it prints
```

Tests: `pytest tests/test_mces_split.py` runs the whole pipeline on a tiny fixture
and checks the far classification against a brute-force minimum.

## Notes

* The compute array is **idempotent** (a batch carrying `mces` is skipped), so a
  re-submit resumes after any walltime kill — durable per-batch checkpoints.
* Each pair is bounded by a solver time limit *and* a hard wall cap (the matching
  bound is pure-Python and the solver limit doesn't cover it).
* Dependencies are pinned in `setup.py[mces]` to the versions that produced the result.
