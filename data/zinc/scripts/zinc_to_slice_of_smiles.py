
# this script is used to create a x% slice of a 1.8B ZINC15 database (2D-clean-annotated-druglike)
# it slices
# creates a random sample from each file acording to sample_ratio.
# (... the next step is to add the firstline and concat the files)

import random
import os
import glob
from multiprocessing import Pool
import typer
from pathlib import Path
import pandas as pd

app = typer.Typer(pretty_exceptions_enable=False)

def sample_data(subdir, output_dir, sample_ratio, seed):
    print("subdir {} RUNNING".format(subdir))

    subdir_name = Path(subdir).stem
    output_file = output_dir + f"/{subdir_name}.smi"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(subdir + "/*"))

    with open(output_file, "w+", encoding="utf-8") as outf:
        for input_file in files:
            df = pd.read_csv(input_file, sep=" ")
            df.drop("zinc_id", axis=1, inplace=True)
            df_sampled = df.sample(frac=sample_ratio, random_state=seed)
            outf.write('\n'.join(df_sampled["smiles"]) + '\n')

    print(f"subdir {subdir} DONE")


@app.command()
def main(
    input_dir: str = typer.Option(..., help="Path to the directory with tranches to sample from"),
    output_dir: str = typer.Option(..., help="Path to the directory where to save the sampled tranches"),
    sample_ratio: float = typer.Option(..., help="Ratio of the data to sample"),
    num_workers: int = typer.Option(8, help="Number of workers to use"),
    seed: int = typer.Option(42, help="Seed for the random number generator"),
) -> None:

    subdirs = sorted(glob.glob(input_dir + "/*"))
    print(f"subdirs: {subdirs}")

    with Pool(processes=num_workers) as p:
        p.starmap(sample_data, [(subdir, output_dir, sample_ratio, seed) for subdir in subdirs])

    print("all done")


if __name__ == "__main__":
    app()