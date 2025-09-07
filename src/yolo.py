from ultralytics import YOLO
import itertools
import os
import pandas as pd
from ultralytics import settings
import pickle
settings.reset()

# ── hyperparameters ─────────────────────────────────────────────────────────────
patience    = 20
batch_size  = 4
default_lr  = 0.01
img_size    = 512
seed        = 0
model_pt    = 'yolov8n.pt'
lrs         = {'YOLO': default_lr}

# ── case definitions ────────────────────────────────────────────────────────────
model_names = ['YOLOv8']
aug_settings = [
    {'mosaic_p': 0.0, 'mixup_p': 0.0},
    {'mosaic_p': 0.5, 'mixup_p': 0.0},
    {'mosaic_p': 0.0, 'mixup_p': 0.5},
]
datasets = ['real', 'augmented']


    
# ── core routines ───────────────────────────────────────────────────────────────
def train_case(model_name, train_dataset, mosaic_p, mixup_p):
    model = YOLO(model_pt)
    run_name = f"YOLO_{train_dataset}_mixup_{mixup_p:.2f}_mosaic_{mosaic_p:.2f}"

    model.train(
        data     = os.path.join("datasets", train_dataset, f"{train_dataset}.yaml"),
        epochs   = 1000,
        imgsz    = img_size,
        seed     = seed,
        mosaic   = mosaic_p,
        mixup    = mixup_p,
        patience = patience,
        batch    = batch_size,
        lr0      = lrs[model_name],
        name     = run_name
    )

def test_case(train_dataset, mosaic_p, mixup_p, test_dataset):
    
    ## .val method in YOLOv8 does not honor agnostic_nms user assignment which is a bug.
    # >>> ADDED: Locate DetectionValidator.postprocess (val path) and its NMS call site
    from ultralytics.utils import ops
    try:
        from ultralytics.models.yolo.detect.val import DetectionValidator as _DetVal  # YOLOv8 detect validator
    except Exception:
        from ultralytics.engine.validator import BaseValidator as _DetVal              # fallback layout

    class AgnosticVal(_DetVal):  # >>> ADDED
        def postprocess(self, preds):  # >>> ADDED
            # identical to upstream, except 'agnostic' now comes from agnostic_nms  # >>> ADDED
            return ops.non_max_suppression(  # >>> ADDED
                preds,
                self.args.conf,
                self.args.iou,
                labels=self.lb,
                multi_label=True,
                agnostic=getattr(self.args, "agnostic_nms", False),
                max_det=self.args.max_det,
            )
    
    # locate the best.pt from the corresponding train run
    run_name = f"YOLO_{train_dataset}_mixup_{mixup_p:.2f}_mosaic_{mosaic_p:.2f}"
    best_pt = os.path.join("runs", "detect", run_name, "weights", "best.pt")
    model   = YOLO(best_pt)

    test_name = f"test_YOLO_{train_dataset}_mixup_{mixup_p:.2f}_mosaic_{mosaic_p:.2f}_on_{test_dataset}"

    return model.val(
        data      = os.path.join("datasets", test_dataset, f"{test_dataset}.yaml"),
        split     = 'test',
        imgsz     = img_size,
        name      = test_name,
        save_json = True,
        half      = True,
        conf      = 0.25,
        iou       = 0.5,
        agnostic_nms=True,
        validator=AgnosticVal,   
    )

def finetune_all():
    """train every model × augmentation × dataset"""
    for model_name, aug, train_ds in itertools.product(model_names, aug_settings, datasets):
        train_case(model_name, train_ds, aug['mosaic_p'], aug['mixup_p'])

def test_all(write_to_csv=True):
    """Test each trained model on both real & augmented test sets, then summarize metrics."""
    # 1. Run through all combinations and collect results
    results = {}
    for aug in aug_settings:
        mosaic_p, mixup_p = aug['mosaic_p'], aug['mixup_p']
        for train_dataset in datasets:
            for test_dataset in datasets:
                key = f"test_YOLO_{train_dataset}_mixup_{mixup_p:.2f}_mosaic_{mosaic_p:.2f}_on_{test_dataset}"
                results[key] = test_case(train_dataset, mosaic_p, mixup_p, test_dataset)

    # 2. Define which metrics to extract
    
    metrics = [
        'metrics/precision(B)',
        'metrics/recall(B)',
        'metrics/mAP50(B)',
        'metrics/mAP50-95(B)',
        'fitness'
    ]

    # 3. Initialize data container
    data = {'Case': []}
    for metric in metrics:
        data[metric] = []

    # 4. Populate it from results
    for case_name, result in results.items():
        data['Case'].append(case_name)
        for metric in metrics:
            data[metric].append(result.results_dict[metric])

    # 5. Compute F1 scores
    f1_scores = []
    for prec, rec in zip(data['metrics/precision(B)'], data['metrics/recall(B)']):
        if prec + rec > 0:
            f1_scores.append(2 * (prec * rec) / (prec + rec))
        else:
            f1_scores.append(0.0)
    data['F1 Score'] = f1_scores

    # 6. Build DataFrame and rename for readability
    df = pd.DataFrame(data)
    df.columns = ['Case', 'Precision', 'Recall', 'mAP50', 'mAP50-95', 'Fitness', 'F1 Score']

    # 7. Print summary
    print(df)

    # 8. Export to LaTeX
    #latex_code = df.to_latex(index=False, float_format="{:0.3f}".format)
    #print(latex_code)

    if write_to_csv:
        # 9. Append results to CSV, matching existing columns
        csv_file = os.path.join("runs", "metrics.csv")


        # parse model, train/test, mixup & mosaic from "Case"
        parts = df['Case'].str.split('_')
        df['model_name']    = parts.str[1]
        df['train_dataset'] = parts.str[2]
        df['mixup']         = parts.str[4].astype(float)
        df['mosaic']        = parts.str[6].astype(float)
        df['test_dataset']  = parts.str[-1]

        # append or write to csv file
        if os.path.isfile(csv_file):
            existing_cols = pd.read_csv(csv_file, nrows=0).columns.tolist()
            df = df[existing_cols]
            df.to_csv(csv_file, mode='a', index=False, header=False)
        else:
            df.to_csv(csv_file, index=False)

        print(f"Appended {len(df)} rows to {csv_file}")
    
    
    # 10.Save
    #with open(os.path.join("runs", "yolo_test_results.pkl"), "wb") as f:
    #    pickle.dump(results, f)
    
    return results

# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    finetune_all()
    test_all()
