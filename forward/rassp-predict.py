#!/usr/bin/env python3
import argparse
import json
import rassp
from rassp import netutil
from rdkit import Chem
import numpy as np

from rassp import msutil,model
from rassp.msutil.masscompute import FragmentFormulaPeakEnumerator
import sys
sys.modules['msutil'] = msutil
sys.modules['model'] = model

use_gpu = True
batch_size = 8

p = argparse.ArgumentParser()
p.add_argument('-m', '--model')
p.add_argument('-e', '--meta')
p.add_argument('-s', '--smiles')
p.add_argument('-o', '--output')
p.add_argument('-d', '--discarded')
p.add_argument('-w', '--workers', default=4)

a = p.parse_args()

predictor = netutil.PredModel(
    a.meta,
    a.model,
    USE_CUDA=use_gpu,
    data_parallel=False,
)

with open(a.smiles) as sf:
	smiles = sf.read().splitlines()

# our "medium" version
#valid_atoms = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}

# original RASSP
valid_atoms = {1, 6, 7, 8, 9, 15, 16, 17}
num_peaks_per_formula = 12
max_formulae = 4096

ffe = FragmentFormulaPeakEnumerator(sorted(valid_atoms), use_highres=True, max_peak_num=num_peaks_per_formula)


def goodmol(mol):
    if len(mol.GetAtoms()) > 48:
        return False

    atoms = { a.GetAtomicNum() for a in mol.GetAtoms() }
    if not atoms < valid_atoms:
        return False

    f, m = ffe.get_frag_formulae(mol)
    if len(f) > max_formulae:
        return False
    return True


mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in smiles]
assert len(mols) == len(smiles)

good_map = list(map(goodmol, mols))

badsmiles = [smi for smi, good in zip(smiles, good_map) if not good]
goodmols = [mol for mol, good in zip(mols, good_map) if good]

if a.discarded:
    with open(a.discarded, 'w') as d:
        for s in badsmiles:
            d.write(s + '\n')

goodsmiles = [ smi for smi, good in zip(smiles, good_map) if good ]
assert len(goodmols) == len(goodsmiles)

pred = predictor.pred(
    goodmols,
    progress_bar=False,
    normalize_pred=True,
    output_hist_bins=True,
    batch_size=batch_size,  # XXX
    dataloader_config={
        'pin_memory': False,
        'num_workers': int(a.workers),  # XXX
        'persistent_workers': False,
    },
    benchmark_dataloader=False,
)['pred_binned']

assert len(pred) == len(goodsmiles)

with open(a.output, 'w') as out:
    for i,s in enumerate(pred):
        out.write(json.dumps({'mz': list(s[:, 0].astype(float)), 'intensity': list(s[:, 1].astype(float)), 'smiles': goodsmiles[i]}))
        out.write('\n')
