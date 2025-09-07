# -------------------- Standard Library --------------------
import os
import json
import itertools
import csv
# -------------------- Third-Party Libraries --------------------
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# TorchVision Models
from torchvision.models.detection import (
    fcos_resnet50_fpn,
    FCOS_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn,
    RetinaNet_ResNet50_FPN_Weights,
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.transforms.functional import to_tensor

# PyCOCO Tools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Ultralytics Augmentations
from ultralytics.data.augment import Mosaic, MixUp, Compose, LetterBox, Format
from ultralytics.utils.instance import Instances

# Progress Bar
from tqdm.auto import tqdm

# -------------------- Local Imports --------------------
from . import utils


class YOLO2TorchDataset(Dataset):
    """
    PyTorch Dataset for YOLOv8-format data with on-the-fly Mosaic & MixUp.
    Returns (image_tensor, target_dict) where target_dict contains 'boxes' and 'labels'.
    """
    def __init__(
        self,
        root: str,
        split: str = 'train',
        mosaic_p: float = 0.5,
        mixup_p: float = 0.5,
        final_size: int = 512
    ):
        self.img_dir = os.path.join(root, split, 'images')
        self.lbl_dir = os.path.join(root, split, 'labels')
        self.files = sorted(
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        )
        self.buffer = list(range(len(self.files)))
        self.final_size = final_size

        # Define transform pipeline
        mosaic    = Mosaic(self, imgsz=final_size, p=mosaic_p)
        letterbox = LetterBox(new_shape=(final_size, final_size), auto=False)
        #mixup     = MixUp(self, pre_transform=Compose([mosaic, letterbox]), p=mixup_p)
        mixup     = MixUp(self, p=mixup_p)
        fmt       = Format(
            bbox_format='xyxy',
            normalize=False,
            return_mask=False,
            return_keypoint=False
        )
        #self.transforms = Compose([mixup, fmt])
        self.transforms = Compose([mosaic, letterbox, mixup, fmt])

    def __len__(self):
        return len(self.files)

    def get_image_and_label(self, idx: int):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        ## img = Image.open(img_path).convert('RGB')
        ## orig_w, orig_h = img.size
        ## img = img.resize((self.final_size, self.final_size), resample=Image.BILINEAR)
        ## new_w, new_h = img.size
        ## img_np = np.array(img)[:, :, ::-1]
        
        img_np = cv2.imread(img_path, cv2.IMREAD_COLOR)                    # H×W×3 BGR
        orig_h, orig_w = img_np.shape[:2]
        img_np = cv2.resize(
            img_np,
            (self.final_size, self.final_size),
            interpolation=cv2.INTER_LINEAR
        )
        new_h, new_w = img_np.shape[:2]

        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        lbl_path = os.path.join(self.lbl_dir, lbl_name)
        boxes, labels = [], []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    cls, x_c, y_c, bw, bh = map(float, line.split())
                    # Rescale center-based YOLO boxes to resized image
                    x1 = (x_c - bw/2) * new_w
                    y1 = (y_c - bh/2) * new_h
                    x2 = (x_c + bw/2) * new_w
                    y2 = (y_c + bh/2) * new_h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls) + 1)

        
        # convert to numpy and force shape (N,4)
        bboxes = np.array(boxes, dtype=np.float32)
        if bboxes.size == 0:
            bboxes = bboxes.reshape(0, 4)
        segments = np.zeros((0, 0, 2), dtype=np.float32)
        inst = Instances(
            bboxes,
            segments,
            None,
            bbox_format='xyxy',
            normalized=False
        )
        data = {
            'img': img_np,
            'instances': inst,
            'cls': np.array(labels, dtype=np.int64),
            'resized_shape': img_np.shape[:2],
            'ori_shape': img_np.shape[:2],
            'im_file': img_path
        }
        return data

    def __getitem__(self, idx: int):
        data = self.get_image_and_label(idx)
        aug = self.transforms(data)
        img_tensor = aug['img'].float() / 255.0 
        boxes      = aug['bboxes']
        labels     = aug['cls']

        # …and then coerce them into FloatTensor and LongTensor:
        if not torch.is_tensor(boxes):
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = boxes.float()
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            labels = labels.long()
        
        target = {
            'boxes': boxes,
            'labels': labels
        }
        return img_tensor, target

def create_coco_json(data_root: str, split: str = 'test', num_classes: int = 10):
    
    if os.path.exists(os.path.join(data_root, split, 'annotations.json')): 
        return
    
    images_dir = os.path.join(data_root, split, 'images')
    labels_dir = os.path.join(data_root, split, 'labels')
    coco = {"images": [], "annotations": [], "categories": []}

    # 1) categories
    for cls_id in range(1, num_classes + 1):
        coco["categories"].append({
            "id": cls_id,
            "name": str(cls_id)
        })

    ann_id = 1
    # 2) images + annotations
    for img_id, fn in tqdm(enumerate(sorted(os.listdir(images_dir)), start=1)):
        if not fn.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        path = os.path.join(images_dir, fn)
        w, h = Image.open(path).size

        coco["images"].append({
            "id": img_id,
            "file_name": fn,
            "width": w,
            "height": h
        })

        lbl_path = os.path.join(labels_dir, os.path.splitext(fn)[0] + '.txt')
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    cls, x_c, y_c, bw, bh = map(float, line.split())
                    # convert to COCO bbox: [x_min, y_min, width, height]
                    x1    = (x_c - bw/2) * w
                    y1    = (y_c - bh/2) * h
                    box_w = bw * w
                    box_h = bh * h
                    coco["annotations"].append({
                        "id":            ann_id,
                        "image_id":      img_id,
                        "category_id":   int(cls) + 1,
                        "bbox":          [x1, y1, box_w, box_h],
                        "area":          box_w * box_h,
                        "iscrowd":       0
                    })
                    ann_id += 1

    out_file = os.path.join(data_root, split, 'annotations.json')
    with open(out_file, 'w') as fp:
        json.dump(coco, fp)
    print(f"→ Wrote {len(coco['images'])} images & {ann_id-1} annotations to\n  {out_file}")
    
    
def collate_fn(batch):
    # required to batch targets properly
    return tuple(zip(*batch))

def get_model(name: str, num_classes: int):
    """
    Build a detection model whose backbone is COCO-fine-tuned but with
    a fresh head for `num_classes`.
    """
    if name == 'FCOS':
        coco = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1)
        bk = coco.backbone.state_dict()
        model = fcos_resnet50_fpn(weights=None, num_classes=num_classes, weights_backbone=None)
        model.backbone.load_state_dict(bk)

    elif name == 'RetinaNet':
        coco = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)
        bk = coco.backbone.state_dict()
        model = retinanet_resnet50_fpn(weights=None, num_classes=num_classes, weights_backbone=None)
        model.backbone.load_state_dict(bk)

    elif name == 'SSD':
        coco = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
        bk = coco.backbone.state_dict()
        model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes, weights_backbone=None)
        model.backbone.load_state_dict(bk)

    elif name == 'FasterRCNN':
        coco = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        bk = coco.backbone.state_dict()
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes, weights_backbone=None)
        model.backbone.load_state_dict(bk)

    else:
        raise ValueError(f"Unknown model name: {name}")

    return model


def train(models: dict,
          data_root: str,
          num_classes: int,
          device: torch.device = None,
          epochs: int = 10,
          batch_size: int = 4,
          patience: int = 3,
          mosaic_p: float = 0,
          mixup_p:    float = 0):
    
    dataset = os.path.basename(os.path.normpath(data_root))
    #if dataset == 'real': patience *= 2
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = YOLO2TorchDataset(
        data_root, 'train',
        mosaic_p = mosaic_p,
        mixup_p  = mixup_p,
        final_size=512
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=4,        # spawn 4 loader workers
        pin_memory=True,      # speed host→GPU copies
        persistent_workers=True
    )

    val_ds = YOLO2TorchDataset(
        data_root, 'val',
        mosaic_p = 0.0,
        mixup_p  = 0.0,
        final_size=512
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=2,        # fewer workers for smaller eval set
        pin_memory=True,
        persistent_workers=True
    )

    os.makedirs('runs/torch/ckpt', exist_ok=True)

    for name in models:
        lr = models[name]
        print(f"\n⏳ Fine-tuning {name}...")
        model = get_model(name, num_classes).to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
        
        '''
        total_steps = epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=2,
            final_div_factor=10,
            anneal_strategy='cos'
        )
        '''
        
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',         # we want to reduce when val loss stops decreasing
            factor=0.9,         # lr ← lr * 0.1 when plateau
            patience=5,         # how many epochs to wait before reducing
            min_lr=1e-4,
            verbose=True
        )
        
        best_val_loss = float('inf')
        epochs_no_improve = 0

        # --- before your epochs loop ---
        scaler = torch.cuda.amp.GradScaler()
        for epoch in tqdm(range(1, epochs + 1), desc='Epochs'):


            # --- inside each epoch, training phase ---
            running_loss = 0.0
            for imgs, targets in train_loader:
                imgs    = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                # mixed‑precision forward
                with torch.cuda.amp.autocast():
                    loss_dict = model(imgs, targets)
                    loss      = sum(loss for loss in loss_dict.values())

                # scale, backward, step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                #scheduler.step()
                #current_lr = scheduler.get_last_lr()[0]

                running_loss += loss.item()
            avg_train = running_loss / len(train_loader)
            

            #scheduler.step()
            
            # --- validation ---
            val_loss = 0.0
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs    = [img.to(device)    for img in imgs]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(imgs, targets)
                    val_loss  += sum(loss for loss in loss_dict.values()).item()
            avg_val = val_loss / len(val_loader)
            

            print(f"[{name}] Epoch {epoch}/{epochs}  Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}")

            
            #scheduler.step(avg_val)
            # Manually log current learning rate
            #current_lr = optimizer.param_groups[0]['lr']
            #print(f"[{name}] Epoch {epoch}/{epochs}  Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f} Current LR: {current_lr}")
            
            # --- early stopping & checkpointing ---
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                epochs_no_improve = 0
                torch.save(
                    model.state_dict(),
                    f'runs/torch/ckpt/{name}_{dataset}_mixup_{mixup_p:.2f}_mosaic_{mosaic_p:.2f}.pth'
                )
                print(f"[{name}] 📌 New best model saved!")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"[{name}] ⏹ Early stopping after {epoch} epochs")
                    break

        print(f"[{name}] Training complete. Best val loss: {best_val_loss:.4f}")
        


def finetune_all():
    # Define models and learning rates
    model_names = ['FCOS', 'RetinaNet', 'SSD', 'FasterRCNN']
    lrs = {
        'FCOS': 2e-3,
        'RetinaNet': 4e-3,
        'SSD': 1e-3,
        'FasterRCNN': 8e-3,
    }

    # Define on-the-fly augmentation settings
    aug_settings = [
        {'mosaic_p': 0.0, 'mixup_p': 0.0},
        {'mosaic_p': 0.5, 'mixup_p': 0.0},
        {'mosaic_p': 0.0, 'mixup_p': 0.5}
    ]

    # Dataset names
    datasets = ['real', 'augmented']

    # Build all cases
    cases = {}

    for i, (model_name, aug, dataset) in enumerate(itertools.product(model_names, aug_settings, datasets)):
        lr = lrs[model_name]
        cases[f'case_{i}'] = {
            'models': {model_name: lr},
            'data_root': f'datasets/{dataset}',
            'mosaic_p': aug['mosaic_p'],
            'mixup_p': aug['mixup_p']
        }

    # Loop over cases to call train()
    for case_name, args in cases.items():
        #if list(args['models'].keys())[0] in ['FCOS', 'SSD', 'RetinaNet']: continue
        #if args['mosaic_p']==0 and args['mixup_p']==0: continue
        dataset = os.path.basename(os.path.normpath(args['data_root']))
        #if dataset == 'real': continue
        print(f"Running {case_name}...")
        print(args)
        utils.set_seeds(0)
        train(
            models=args['models'],
            data_root=args['data_root'],
            num_classes=10,
            epochs=200,
            batch_size=4,
            patience=20,
            mosaic_p=args['mosaic_p'],
            mixup_p=args['mixup_p']
        )

def test(model, data_root, device, fitness_weights=(0.0, 0.0, 0.1, 0.9)):
    """Evaluate a single model on a dataset (COCO-style)."""
    model.eval()
    gt_json = os.path.join(data_root, 'test', 'annotations.json')
    coco_gt = COCO(gt_json)

    # collect detections
    results = []
    for img_id in tqdm(coco_gt.getImgIds()):
        info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(data_root, 'test', 'images', info['file_name'])
        img = to_tensor(Image.open(img_path).convert('RGB')).to(device)
        with torch.no_grad():
            outputs = model([img])[0]

        boxes  = outputs['boxes'].detach().cpu().numpy()
        scores = outputs['scores'].detach().cpu().numpy()
        labels = outputs['labels'].detach().cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            results.append({
                'image_id':    img_id,
                'category_id': int(label),
                'bbox':        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                'score':       float(score)
            })


    coco_dt = coco_gt.loadRes(results)

    # --- mAP@.50 and mAP@.50–.95 ---
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    mAP5095, mAP50 = coco_eval.stats[0], coco_eval.stats[1]

    # --- Precision & Recall @ IoU=.50 ---
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.iouThrs = np.array([0.5])
    coco_eval.params.maxDets = [100]
    coco_eval.evaluate(); coco_eval.accumulate()
    precs = coco_eval.eval['precision'][0, :, :, 0, 0]
    valid = precs > -1
    precision = float(precs[valid].mean()) if valid.any() else 0.0
    recalls = coco_eval.eval['recall'][0, :, 0, 0]
    recall_vals = recalls[recalls > -1]
    recall = float(np.mean(recall_vals)) if recall_vals.size else 0.0

    # --- F1 & Fitness ---
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    fitness = fitness_weights[2] * mAP50 + fitness_weights[3] * mAP5095

    return {
        'Precision': precision,
        'Recall': recall,
        'mAP50': mAP50,
        'mAP50-95': mAP5095,
        'F1 Score': f1,
        'Fitness': fitness
    }

def test_all():
    
    # Generate COCO-style annotation JSONs for all test datasets
    create_coco_json('datasets/real', 'test', num_classes=10)
    create_coco_json('datasets/outpainted', 'test', num_classes=10)
    create_coco_json('datasets/augmented', 'test', num_classes=10)
    
    # now run evaluation script which reads test/annotations.json
    """Loop over checkpoints, run evaluation, and save metrics."""
    model_names = ['FCOS', 'RetinaNet', 'SSD', 'FasterRCNN']
    aug_settings = [
        {'mosaic_p': 0.0, 'mixup_p': 0.0},
        {'mosaic_p': 0.5, 'mixup_p': 0.0},
        {'mosaic_p': 0.0, 'mixup_p': 0.5},
    ]
    datasets = ['real', 'augmented']
    
    ckpts = {m: [] for m in model_names}
    for model_name, aug, dataset in itertools.product(model_names, aug_settings, datasets):
        path = (
            f"runs/torch/ckpt/"
            f"{model_name}_{dataset}"
            f"_mixup_{aug['mixup_p']:.2f}"
            f"_mosaic_{aug['mosaic_p']:.2f}.pth"
        )
        ckpts[model_name].append((path, dataset, aug))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10  # adjust as needed
    results = []

    for name, paths in ckpts.items():
        for path, train_ds, aug in paths:
            print(path)
            model = get_model(name, num_classes).to(device)  # assumes get_model is defined
            model.load_state_dict(torch.load(path, map_location=device))

            if hasattr(model, 'score_thresh'):
                model.score_thresh = 0.25
            elif name == 'FasterRCNN' and hasattr(model, 'rpn'):
                model.rpn.score_thresh = 0.25
            else:
                print(f'{name} does not have any score_thresh attribute')

            for test_ds in ['real', 'augmented']:
                metrics = test(model, f'datasets/{test_ds}', device)
                results.append({
                    'model_name':   name,
                    'train_dataset':train_ds,
                    'test_dataset': test_ds,
                    'mixup':        aug['mixup_p'],
                    'mosaic':       aug['mosaic_p'],
                    **metrics
                })

                print(f"\n>>> {name} {train_ds} {test_ds} {aug['mixup_p']} {aug['mosaic_p']} metrics:")
                for k, v in metrics.items():
                    print(f"  {k:>9s}: {v:.4f}")

    os.makedirs('runs', exist_ok=True)
    with open('runs/metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model_name','train_dataset','test_dataset',
            'mixup','mosaic',
            'Precision','Recall','mAP50','mAP50-95','F1 Score','Fitness'
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} rows to runs/metrics.csv")
    return results

if __name__ == "__main__":
    finetune_all()
    test_all()