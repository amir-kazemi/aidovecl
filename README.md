# AIDOVECL:
## AI-generated Dataset of Outpainted Vehicles for Eye-level Classification and Localization

## Installation Guide

To set up the environment for AIDOVECL, follow these steps:

First, ensure you have `conda` or `miniconda` installed. Then, create a new conda environment using the provided `aidovecl-env.yml` file.

```bash
conda env create -f aidovecl-env.yml
conda activate aidovecl
```

Run the following command to add the environment to Jupyter:
```bash
python -m ipykernel install --user --name=aidovecl --display-name="Python (aidovecl)"
```

## Reproducing Figures

To produce the figures of the paper, after activating `aidovecl` environment, execute the jupyter notebooks.

**Note 1:** Please be patient during the initial run as the detection and inpainting models are being downloaded.

**Note 2:** Despite setting random seeds, 100% reproducibility is not guaranteed. For more information, refer to [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) and [PyTorch](https://pytorch.org/docs/stable/notes/randomness.html).

## An Overview of the Package:
- `detect.py`: Detects vehicles and creates seed images.
- `outpaint.py`: Outpaints the seed images.
- `backdrop.py`: Generates background images.
- `aidovecl.py`: Generates AIDOVECL, backgrounds, and augments real data with them.
- `yolo.py`: Trains and tests YOLO on datasets
- `utils.py`: Provides utilities for the above files.

For use cases, refer to jupyter notebooks.

## Downloading Datasets for Vehicle Classification and Localization

Download the dataset from the following source: [AIDOVECL Dataset](https://huggingface.co/datasets/amir-kazemi/aidovecl/tree/main).

After downloading, extract the zipped datasets to the `datasets` folder of the repository. The structure should look like this:
```bash
datasets/
    ├── real
        ├── bus
        ├── coupe
        ├── minibus
        ├── minivan
        ├── pickup
        ├── sedan
        ├── suv
        ├── truck                                   
        └── van
    ├── real_seeded_split
        ├── images
        ├── labels
        └── real_seeded_split.yaml
    └── augmented
        ├── images
        ├── labels
        └── augmented.yaml
```
The jupyter notebook `demo-fig-4.ipynb` demonstrates how to use `real` dataset to generate `real_seeded_split` and `augmented` datasets. It also showcases training and testing YOLO on them. Note that `real` dataset consists of collected images with no annotations, while`real_seeded_split` includes selected images for outpainting that are also annotated and split into train, val, and test folders.