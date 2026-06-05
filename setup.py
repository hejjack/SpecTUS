from setuptools import setup, find_packages

setup(
    name="spectus",
    version="0.1",
    packages=find_packages(where="spectus"),
    package_dir={"": "spectus"},
    # Optional dev extra for the MCES-distance split analysis (spectus/mces_split).
    # Kept out of the core deps so regular SpecTUS users don't pull the ILP stack:
    #   pip install -e ".[mces]"
    # Pinned to the versions that produced the published result (reproducibility).
    extras_require={
        "mces": [
            "myopic-mces==1.0.1",
            "rdkit==2026.3.2",
            "numpy==2.4.6",
            "scipy==1.17.1",
            "pandas==3.0.3",
            "pyarrow==24.0.0",
            "h5py==3.16.0",
            "joblib==1.5.3",
            "pulp==3.3.2",
            "networkx==3.6.1",
            "tqdm",
        ],
        "test": ["pytest"],
    },
)