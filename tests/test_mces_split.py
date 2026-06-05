"""End-to-end check: the pipeline's far/close split must match a brute-force
minimum MCES over all training molecules (the rigour guarantee, in miniature)."""
from __future__ import annotations

from myopic_mces.myopic_mces import MCES

from mces_split import BatchComputer, Folds, PreFilter, SplitReport

TRAIN = ["CCO", "CCN", "CCC", "CCCC", "CCCCC", "CCCCCC", "c1ccccc1", "c1ccncc1",
         "CC(=O)O", "CCOCC", "CC(C)O", "CCCCO"]
VALID = ["CCCCCO", "c1ccccc1C"]                       # near a train molecule
TEST = ["CCCCCCCCCCCCCCCCCCCC", "O=S(=O)(O)c1ccc(N)cc1"]  # the C20 alkane is far from all
CEILING = 10.0


def _brute_force_far(query: str, train: list[str]) -> bool:
    """A query is far iff no training molecule is at exact MCES distance < ceiling."""
    options = dict(msg=False, threads=1)
    for t in train:
        _, dist, _, mode = MCES(query, t, CEILING, 0, "PULP_CBC_CMD", options, always_stronger_bound=False)
        if mode == 1 and 0 <= dist < CEILING:
            return False
    return True


def test_far_split_matches_bruteforce(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.smi").write_text("\n".join(TRAIN) + "\n")
    (data / "valid.smi").write_text("\n".join(VALID) + "\n")
    (data / "test.smi").write_text("\n".join(TEST) + "\n")
    out, results = tmp_path / "run", tmp_path / "results"

    folds = Folds.load(data)
    PreFilter(ceiling=CEILING, batch_size=10_000).run(folds, out)
    for batch in sorted((out / "batches").glob("batch*.hdf5")):
        BatchComputer(threshold=CEILING, n_jobs=1).compute(batch)
    report = SplitReport(out, results)
    report.build_near_pairs()
    report.report([CEILING])

    pipeline_far = set((results / "far_valid_t10.smi").read_text().split())
    pipeline_far |= set((results / "far_test_t10.smi").read_text().split())

    for query in dict.fromkeys(folds.valid + folds.test):
        assert (query in pipeline_far) == _brute_force_far(query, folds.train), query
    assert pipeline_far, "expected at least one far molecule (the C20 alkane)"
