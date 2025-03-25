# SpecTUS: Spectral Translator for Unknown Structures annotation from EI-MS spectra

**SpecTUS** is a transformer-based tool for reconstructing Gas Chromatography-Electron Ionization (GC-EI) mass spectra. The model reconstructs spectra in a de novo manner — directly translating the spectra into 2D molecular structures represented as SMILES strings. The model is pretrained on a large dataset of synthetic spectra and fine-tuned on a smaller dataset of experimental NIST20 spectra. The NIST20 is a proprietary dataset; therefore, we cannot share the final model, but the code for training the model is carefully documented and available for public use. If you own a license for the NIST20 dataset, you can train the model yourself or contact us to get it.

We make freely available the 17.2M synthetic spectra used for pretraining the model and the checkpoint of the pretrained model that
can be further finetuned on your own dataset [here](https://huggingface.co/MS-ML).

## 🎮 Demo 
You can run a demo inference of the final model hosted on our server via Jupyter Notebook in [this](https://github.com/ljocha/spectus-demo) repository.

## 📁 Repository Structure
```
SpecTUS/
├── config_runners/          # Shell scripts for model training, evaluation, and prediction, model comparison
├── configs/                 # YAML configuration files for model training, evaluation, prediction, model comparison
├── data/                    # Includes used NIST splits, example data, and synthetic data
├── forward/                 # Scripts for spectra generators (NEIMS, RASSP) training
├── notebooks/               # Jupyter notebooks for data preprocessing, pretraining, finetuning, hyperparameter search, model evaluation, and comparison of the models
├── predictions/             # Test predictions of all models (hyperparameter search) and database search methods (HSS, SSS, BDC) mentioned in the paper
```

## 🛠 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/SpecTUS.git
   cd SpecTUS
   ```
2. Set up the Python environment:
   ```bash
   conda env create -f trainSpectus_build.yaml
   ```

## 🚦 Usage
Data preprocessing, pretraining, finetuning, hyperparameter search, model evaluation, and comparison of the models are all described step by step in the [notebooks/](notebooks/).

An example notebook for [inference](notebooks/5_inference_on_open_data.ipynb) is available to help you get started reconstructing spectra from an `msp` file once you have your model trained.

## 📄 Citation
If you use **SpecTUS** in your research, please cite (OLD CITATION):
```
@misc{hájek2023denovoidentificationsmallmolecules,
      title={De-novo Identification of Small Molecules from Their GC-EI-MS Spectra},
      author={Adam Hájek and Michal Starý and Filip Jozefov and Helge Hecht and Elliott Price and Aleš Křenek},
      year={2023},
      eprint={2304.01634},
      archivePrefix={arXiv},
      primaryClass={physics.data-an},
      url={https://arxiv.org/abs/2304.01634},
}
```

