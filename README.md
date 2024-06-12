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
- `detect.py`: Detects vehicles and creates seed images.
- `outpaint.py`: Outpaints the seed images.
- `backdrop.py`: Generates background images.
- `utils.py`: Provides utilities for the above files.

For use cases, refer to `figs/fig1.py` and `figs/fig1.py`.


## Downloading Datasets for Vehicle Classification and Localization

Download the dataset from the following source: [AIDOVECL Dataset](https://huggingface.co/datasets/amir-kazemi/aidovecl/tree/main).

After downloading, extract the zipped datasets to the `datasets` folder of the repository. The structure should look like this:
```bash
datasets/
    ├── real
    ├── outpainted
    └── augmented
```
## Training and Evaluating YOLO model on Datasets
The following lines train and evaluate YOLO on the `outpainted` dataset (a.k.a. AIDOVECL).
```bash
from ultralytics import YOLO
# load a pretrained model (recommended for training)
model = YOLO('yolov8n.pt')
# train the model (overwriting ./runs/detect/train/ if exists)
results = model.train(
    data='./datasets/outpainted/outpainted.yaml',
    epochs=1000,
    imgsz=512,
    patience=100,
    exist_ok=True
)
# load the best model
best_model = YOLO("./runs/detect/train/weights/best.pt")
# evaluate the model performance on test split
results = best_model.val(
    data='./datasets/outpainted/outpainted.yaml',
    split='test',
    imgsz=512,
    conf=0.5,
)
```



