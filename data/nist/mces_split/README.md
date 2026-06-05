# MCES-distance far molecules

Validation/test molecules at MCES distance **≥ T from *every* training molecule** —
the output of the MCES-distance split analysis (code: `spectus/mces_split/`). These
are structurally novel molecules: the InChIKey14 split guarantees no shared 2D
skeleton across folds, but most val/test molecules are still a few edits from a
training molecule; these are the ones that are genuinely far.

SMILES are RDKit-canonical with stereochemistry removed (matching SpecTUS's molecule
identity); one SMILES per line.

## Counts (clean universe: 24,150 valid / 24,152 test)

| T | far_valid | far_test |
|---|-----------|----------|
| 6 | 2,164 | 2,170 |
| 7 | 1,449 | 1,440 |
| 8 | 1,005 | 1,002 |
| 9 | 653 | 654 |
| 10 | 446 | 469 |

## Files

- `far_{valid,test}_t{6,7,8,9,10}_clean.smi` — far molecules per fold per threshold.
- `incomputable.smi` — 357 molecules excluded because MCES is computationally
  intractable for them (very large structures: long-chain lipids, polyaromatics,
  organometallics); their far/close status is unverifiable at any threshold.

**"clean"** = restricted to molecules where every distance is computable, so the far
set is **certain at every threshold** (0 false positives among all computable pairs).
