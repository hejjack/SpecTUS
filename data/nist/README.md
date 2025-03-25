# NIST dataset
This dataset made of NIST20 is destereochemicalized, canonicalized, and split into train, test, and validation sets (with no overlaps). For more info refer to the scripts.
It only contains jsonl files with `mz`, `intensity`, and `smiles` fields. The filtering and preprocessing is meant to be done on-the-fly (for more freedom in testing preprocessing methods).

## This dataset is created by the following steps:
1. Create msp splits using my updated version of FILIP JOZEFOV's notebook data/nist_cleaning_splitting.ipynb
   - it dorps ~60k spectra that don't have some form of proper identifier (smiles, inchikey)
   - split 0.8:0.1:0.1
2. Create jsonl files by the function msp2jsonl from utils.spectra_process_utils.py (for 'train' and then 'test' and 'valid'):

```python
import sys
sys.path.append("..")
from utils.spectra_process_utils import msp2jsonl
from pathlib import Path

tokenizer = None
for dataset_type in ["train", "valid", "test"]:
    dataset_path = Path("../data/datasets/NIST/NIST_split_filip")
    msp2jsonl(path_msp=dataset_path / f"{dataset_type}.msp",
                      tokenizer = tokenizer,
                      path_jsonl=dataset_path / f"{dataset_type}.jsonl",
                      keep_spectra=True,
                      do_preprocess=False
                      )
```

## No stats available (NO PREPROCESS)



################################
# PREPROCESSING STATS DEPRECATED
 - test.jsonl
   0 no smiles
   52 smiles too long
   0 spectra corrupted
   780 spectra w/ too high mz
   2463 spectra w/ too many peaks
   totally 3295 issues
   discarded 2957/29218 spectra
   LENGTH: 26261
 - valid.jsonl
   0 no smiles
   55 smiles too long
   0 spectra corrupted
   709 spectra w/ too high mz
   2339 spectra w/ too many peaks
   totally 3103 issues
   discarded 2809/29053 spectra
   LENGTH: 26244
 - train.jsonl
    0 no smiles
    406 smiles too long              ??? deprecated
    14 spectra corrupted             ??? deprecated
    6004 spectra w/ too high mz      ??? deprecated
    20226 spectra w/ too many peaks  ??? deprecated
    totally 26650 issues             ??? deprecated
    discarded 24049/237455 spectra   ??? deprecated
    LENGTH: 208448                   CORRECT