# AIDOVECL:
## AI-generated Dataset of Outpainted Vehicles for Eye-level Classification and Localization

## Installation Guide

To set up the environment for AIDOVECL, follow these steps:

First, ensure you have `conda` or `miniconda` installed. Then, create a new conda environment using the provided `aidovecl-env.yml` file.

```bash
conda env create -f aidovecl-env.yml
conda activate aidovecl
```

## Reproducing Figures

To produce the first and second figures for the draft, after activating `aidovecl` environment, execute the following command in the root directory of the repository:

```bash
python figs.py
```
This will generate the required figures and save them as PDF files in `figs/fig1` and `figs/fig2`, respectively.

## An Overview of the Package:
- `detec.py`: Detects vehicles and creates seed images.
- `outpaint.py`: Outpaints the seed images.
- `backdrop.py`: Generates background images.
- `utils.py`: Provides utilities for the above files.

For use cases, refer to 'fig1.py' and 'fig2.py'.


## Downloading and Extracting Dataset for Vehicle Classification and Localization

Download the dataset from the following source: [AIDOVECL Dataset](https://huggingface.co/datasets/amir-kazemi/aidovecl/tree/main).

After downloading, extract the zipped datasets to the `datasets` folder of the repository. The structure should look like this:
```bash
datasets/
    ├── real
    ├── outpainted
    └── augmented
```


