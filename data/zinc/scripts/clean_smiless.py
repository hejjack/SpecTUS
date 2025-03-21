
# this script is used to create a x% slice of a 1.8B ZINC15 database (2D-clean-annotated-druglike)
# it slices
# creates a random sample from each file acording to sample_ratio.
# (... the next step is to add the firstline and concat the files)

import sys
sys.path.append("../")

import random
import os
import glob
from multiprocessing import Pool
import typer
from pathlib import Path
import pandas as pd
from utils.spectra_process_utils import remove_stereochemistry_and_canonicalize

app = typer.Typer(pretty_exceptions_enable=False)

def clean_smiles(smiles):

    smiles = smiles.strip()
    canon_smiles = remove_stereochemistry_and_canonicalize(smiles)

    return canon_smiles if canon_smiles and len(canon_smiles) <= 100 else None



def clean_all_smiles_in_file(input_file, output_dir):
    print("file {} RUNNING".format(input_file))

    file_name = Path(input_file).name
    output_file = output_dir + f"/{file_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as inpf:
        with open(output_file, "w+", encoding="utf-8") as outf:
            for smiles in inpf:
                smiles = clean_smiles(smiles)
                if smiles:
                    outf.write(smiles + '\n')

    print(f"file {input_file} DONE")


@app.command()
def main(
    input_dir: str = typer.Option(..., help="Path to the directory with tranches to sample from"),
    output_dir: str = typer.Option(..., help="Path to the directory where to save the sampled tranches"),
    num_workers: int = typer.Option(8, help="Number of workers to use"),
) -> None:

    files = sorted(glob.glob(input_dir + "/*"))
    print(f"files: {files}")

    with Pool(processes=num_workers) as p:
        p.starmap(clean_all_smiles_in_file, [(file, output_dir) for file in files])

    print("all done")


if __name__ == "__main__":
    app()