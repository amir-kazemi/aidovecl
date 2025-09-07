
from __future__ import annotations

import json, os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil, glob
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional

# ---------------- font helper (minimal addition) ----------------

def _ensure_font_path(ttf_name: str = "DejaVuSans.ttf") -> str:
    """
    Return a usable TTF font path.
    Priority:
      1) arial.ttf if available on system
      2) repo-local config/<ttf_name> (downloaded if missing)
      3) fallback: returns empty string (caller should handle load_default)
    """
    # 1) Try system Arial first (keeps existing behavior if present)
    try:
        ImageFont.truetype("arial.ttf", 10)
        return "arial.ttf"
    except Exception:
        pass

    # 2) Use repository-local config folder
    repo_root = Path(__file__).resolve().parent.parent
    config_dir = repo_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    font_path = config_dir / ttf_name

    if not font_path.exists():
        # Download a permissively-licensed, widely-available font (DejaVuSans)
        url = "https://raw.githubusercontent.com/matplotlib/matplotlib/main/lib/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"
        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(font_path))
        except Exception:
            # If download fails for any reason, leave missing and let caller fallback
            return ""

    return str(font_path) if font_path.exists() else ""


def _get_font(size: int):
    """Load a TTF font at the given size, preferring Arial then local config TTF; fallback to default bitmap font."""
    # Try Arial with requested size
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        pass

    # Try ensured local TTF
    ttf_path = _ensure_font_path()
    if ttf_path:
        try:
            return ImageFont.truetype(ttf_path, size)
        except Exception:
            pass

    # Final fallback
    return ImageFont.load_default()


# ---------------- geometry ----------------

def _xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    if iw <= 0 or ih <= 0: return 0.0
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return float(inter / max(1e-9, area_a + area_b - inter))

def _scale_box(box, sx: float, sy: float):
    x1, y1, x2, y2 = box
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


# ---------------- loading ----------------

def _cat_map(ann): return {int(c["id"]): str(int(c["name"])-1) for c in ann["categories"]}

def load_gt(annotation_path: str):
    ann = json.load(open(annotation_path))
    categories = _cat_map(ann)
    images = {int(im["id"]): im for im in ann["images"]}
    gts = defaultdict(list)
    for ann in ann["annotations"]:
        if ann.get("iscrowd", 0) == 1: continue
        gts[int(ann["image_id"])] = gts[int(ann["image_id"])] + [{
            "bbox": _xywh_to_xyxy(ann["bbox"]),
            "category_id": int(ann["category_id"]),
        }]
    return images, categories, gts

def load_predictions(predictions_path: str, score_thr: float = 0.0):
    preds = json.load(open(predictions_path))
    by_img = defaultdict(list)
    for p in preds:
        if p.get("score", 0.0) >= score_thr:
            by_img[int(p["image_id"])].append({
                "bbox": _xywh_to_xyxy(p["bbox"]),
                "category_id": int(p["category_id"]),
                "score": float(p.get("score", 0.0)),
            })
    for lst in by_img.values():
        lst.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    return by_img

def maybe_align_category_ids(preds_map, gt_categories, override_offset=None):
    if override_offset is None:
        pred_ids = {int(p["category_id"]) for lst in preds_map.values() for p in lst}
        gt_ids = set(int(i) for i in gt_categories.keys())
        if pred_ids and gt_ids and min(pred_ids) == 0 and min(gt_ids) >= 1:
            shifted = {i + 1 for i in pred_ids}
            offset = 1 if shifted.issubset(gt_ids) else 0
        else: offset = 0
    else: offset = int(override_offset)
    if offset:
        for lst in preds_map.values():
            for p in lst:
                p["category_id"] = int(p["category_id"]) + offset
    return offset

# ---------------- matching ----------------

def match_image(preds, gts, iou_thr: float):
    tps, fps, fns, confusions = [], [], [], []
    matched_gts = set()
    for p in preds:
        best_iou, best_idx = 0.0, -1
        for gi, gt in enumerate(gts):
            if gi in matched_gts: continue
            ax1, ay1, ax2, ay2 = p["bbox"]
            bx1, by1, bx2, by2 = gt["bbox"]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            
            # intersection area
            inter = iw * ih
            # areas (clamped non-negative for safety)
            areaA = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
            areaB = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
            # union and IoU
            union = areaA + areaB - inter
            iou = 0.0 if inter <= 0 else inter / max(1e-9, union)
            
            if iou > best_iou: best_iou, best_idx = iou, gi
        if best_iou >= iou_thr and best_idx >= 0:
            gt = gts[best_idx]; matched_gts.add(best_idx)
            if int(gt["category_id"]) == int(p["category_id"]): tps.append((p, gt, best_iou))
            else: confusions.append((p, gt, best_iou))
        else: fps.append(p)
    for gi, gt in enumerate(gts):
        if gi not in matched_gts: fns.append(gt)
    return tps, fps, fns, confusions

# ---------------- drawing ----------------

RED = (255,0,0,255)
GREEN = (0,200,0,255)

def _draw_rect(draw, box, style="solid", width=3, color=None):
    x1,y1,x2,y2 = map(int, box)
    c = color or RED
    if style=="solid":
        draw.rectangle([x1,y1,x2,y2], outline=c, width=width)
    elif style=="dashed":
        step=10
        for x in range(x1,x2,step*2): draw.line([(x,y1),(min(x+step,x2),y1)], fill=c, width=width)
        for x in range(x1,x2,step*2): draw.line([(x,y2),(min(x+step,x2),y2)], fill=c, width=width)
        for y in range(y1,y2,step*2): draw.line([(x1,y),(x1,min(y+step,y2))], fill=c, width=width)
        for y in range(y1,y2,step*2): draw.line([(x2,y),(x2,min(y+step,y2))], fill=c, width=width)

STYLE_MAP={"GT":"solid","FP":"dashed","FN":"solid","PRED":"dashed"}  # FN uses GT box style

def _label_pred_number(draw, box, idx):
    # Proportional to image *width* (with a min of 12)
    w, _ = (draw.im.size if hasattr(draw, "im") and hasattr(draw, "im") else (0, 0))
    font_size = max(12, int(w * 0.02)) if w else 16
    font = _get_font(font_size)
    x1, y1, x2, y2 = map(int, box)
    pad = 2
    text = f"#{idx}"
    # robust text size
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)

    # keep label within image
    top = max(0, y1 - th - pad)
    draw.rectangle([x1, top, x1 + tw + pad*2, top + th + pad], fill=(255,255,255,200))
    draw.text((x1 + pad, top), text, fill=RED, font=font)

def _draw_legend(im, items, padding=6):
    # Legend OUTSIDE the image: create a bottom band and draw there (bottom-left)
    w, h = im.size
    font_size = max(15, int(w * 0.02)) if w else 16
    #print(font_size)
    font = _get_font(font_size)

    # measure legend
    maxw = 0
    line_heights = []
    for _, text in items:
        try:
            bbox = ImageDraw.Draw(im).textbbox((0,0), text, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except Exception:
            tw, th = font.getsize(text)
        maxw = max(maxw, tw)
        line_heights.append(th + 2)

    total_h = sum(line_heights) + padding * 2
    box_w = maxw + padding * 2

    # new canvas with extra bottom space
    new_im = Image.new(im.mode, (w, h + total_h), (255, 255, 255))
    new_im.paste(im, (0, 0))

    draw = ImageDraw.Draw(new_im, "RGBA")
    box_y1 = h
    # draw semi-opaque legend panel (full width not required; bottom-left block)
    draw.rectangle([0, box_y1, box_w, box_y1 + total_h], fill=(255,255,255,255), outline=(255,255,255,255))

    y = box_y1 + padding
    for (key, text), lh in zip(items, line_heights):
        draw.text((padding, y), text, fill=(0,0,0,255), font=font)
        y += lh

    return new_im

def _scale_examples(tps, fps, fns, confs, sx, sy):
    # scale bboxes for resized image
    tps_s = []
    for p, g, iou in tps:
        p2 = dict(p); g2 = dict(g)
        p2["bbox"] = _scale_box(p["bbox"], sx, sy)
        g2["bbox"] = _scale_box(g["bbox"], sx, sy)
        tps_s.append((p2, g2, iou))

    fps_s = []
    for p in fps:
        p2 = dict(p); p2["bbox"] = _scale_box(p["bbox"], sx, sy)
        fps_s.append(p2)

    fns_s = []
    for g in fns:
        g2 = dict(g); g2["bbox"] = _scale_box(g["bbox"], sx, sy)
        fns_s.append(g2)

    confs_s = []
    for p, g, iou in confs:
        p2 = dict(p); g2 = dict(g)
        p2["bbox"] = _scale_box(p["bbox"], sx, sy)
        g2["bbox"] = _scale_box(g["bbox"], sx, sy)
        confs_s.append((p2, g2, iou))

    return tps_s, fps_s, fns_s, confs_s

# ---------------- prediction image_id patching (least aggressive) ----------------

def _build_id_maps_from_images(images_dict):
    """
    Build maps from filename/stem -> image_id using GT images metadata.
    images_dict: {image_id(int): {"file_name": ..., ...}, ...}
    """
    id_by_filename = {im["file_name"]: int(img_id) for img_id, im in images_dict.items()}
    id_by_stem = {Path(k).stem: v for k, v in id_by_filename.items()}
    return id_by_filename, id_by_stem

def _maybe_patch_prediction_image_ids(preds_path: str, images_dict) -> Tuple[int, list]:
    """
    If predictions.json contains string image_ids (filenames/paths), replace them
    with the correct integer COCO IDs using GT 'images' metadata.
    - Backs up original to predictions.json.orig (if not already present)
    - Overwrites predictions.json in-place when fixes are applied
    Returns: (fixed_count, skipped_list)
    """
    try:
        preds = json.load(open(preds_path))
    except Exception:
        return 0, []

    id_by_filename, id_by_stem = _build_id_maps_from_images(images_dict)
    fixed, skipped = 0, []

    changed = False
    for p in preds:
        imgid = p.get("image_id")
        if isinstance(imgid, int):
            continue  # already correct
        base = os.path.basename(str(imgid))
        stem = Path(base).stem
        if base in id_by_filename:
            p["image_id"] = id_by_filename[base]; fixed += 1; changed = True
        elif stem in id_by_stem:
            p["image_id"] = id_by_stem[stem]; fixed += 1; changed = True
        else:
            skipped.append(imgid)

    if changed:
        bak = preds_path + ".orig"
        try:
            if not os.path.exists(bak):
                os.rename(preds_path, bak)
        except Exception:
            # Best-effort: if rename fails, continue to overwrite preds_path
            pass
        with open(preds_path, "w") as f:
            json.dump(preds, f, indent=2)

    return fixed, skipped

# ---------------- visualize ----------------

def visualize_image(image_path,tps,fps,fns,confs,categories,out_path,draw_mode="all"):
    """
    Styles/colors:
      - GT: solid
      - Any prediction (FP, TP/confusion pred): dashed, numbered #1, #2, ...
    """
    # 1) open and RESIZE image to fixed width=256 (keep aspect) BEFORE drawing
    im = Image.open(image_path).convert("RGB")
    w0, h0 = im.size
    target_w = 256
    if w0 != target_w:
        target_h = max(1, int(round(h0 * (target_w / float(w0)))))
        sx, sy = (target_w / float(w0), target_h / float(h0))
        im = im.resize((target_w, target_h), Image.LANCZOS)
        # scale all boxes to match resized image
        tps, fps, fns, confs = _scale_examples(tps, fps, fns, confs, sx, sy)
    else:
        target_h = h0  # unchanged

    draw = ImageDraw.Draw(im,"RGBA")
    legend=[]
    pred_counter = 0

    if draw_mode=="conf":
        for p,g,iou in confs:
            _draw_rect(draw,g["bbox"],STYLE_MAP["GT"], color=GREEN)
            legend.append(("GT",f"GT (solid): {categories.get(g['category_id'],g['category_id'])} (confused)"))
            pred_counter += 1
            _draw_rect(draw,p["bbox"],STYLE_MAP["PRED"], color=RED)
            _label_pred_number(draw, p["bbox"], pred_counter)
            legend.append(("PRED",f"Pred#{pred_counter}: {categories.get(p['category_id'],p['category_id'])} ({p.get('score',0):.2f}, IoU={iou:.2f})"))

    elif draw_mode=="fp":
        for g in fns:
            _draw_rect(draw,g["bbox"],STYLE_MAP["GT"], color=GREEN)
            legend.append(("GT",f"GT (solid): {categories.get(g['category_id'],g['category_id'])} (unmatched)"))
        for p,g,iou in tps:
            _draw_rect(draw,g["bbox"],STYLE_MAP["GT"], color=GREEN)
            legend.append(("GT",f"GT (solid): {categories.get(g['category_id'],g['category_id'])} (matched)"))
        for p,g,iou in confs:
            _draw_rect(draw,g["bbox"],STYLE_MAP["GT"], color=GREEN)
            legend.append(("GT",f"GT (solid): {categories.get(g['category_id'],g['category_id'])} (confused)"))

        for p in fps:
            pred_counter += 1
            _draw_rect(draw,p["bbox"],STYLE_MAP["FP"], color=RED)
            _label_pred_number(draw, p["bbox"], pred_counter)
            legend.append(("FP",f"Pred#{pred_counter} FP: {categories.get(p['category_id'],p['category_id'])} ({p.get('score',0):.2f})"))
        for p,g,iou in tps:
            pred_counter += 1
            _draw_rect(draw,p["bbox"],STYLE_MAP["PRED"], color=RED)
            _label_pred_number(draw, p["bbox"], pred_counter)
            legend.append(("PRED",f"Pred#{pred_counter} TP: {categories.get(p['category_id'],p['category_id'])} ({p.get('score',0):.2f}, IoU={iou:.2f})"))
        for p,g,iou in confs:
            pred_counter += 1
            _draw_rect(draw,p["bbox"],STYLE_MAP["PRED"], color=RED)
            _label_pred_number(draw, p["bbox"], pred_counter)
            legend.append(("PRED",f"Pred#{pred_counter} Conf: {categories.get(p['category_id'],p['category_id'])} ({p.get('score',0):.2f}, IoU={iou:.2f})"))

    elif draw_mode=="fn":
        for g in fns:
            _draw_rect(draw,g["bbox"],STYLE_MAP["GT"], color=GREEN)
            legend.append(("GT",f"FN GT (solid): {categories.get(g['category_id'],g['category_id'])}"))
        for p,g,iou in tps:
            pred_counter += 1
            _draw_rect(draw,p["bbox"],STYLE_MAP["PRED"], color=RED)
            _label_pred_number(draw, p["bbox"], pred_counter)
            legend.append(("PRED",f"Pred#{pred_counter} TP: {categories.get(p['category_id'],p['category_id'])} ({p.get('score',0):.2f}, IoU={iou:.2f})"))
        for p,g,iou in confs:
            pred_counter += 1
            _draw_rect(draw,p["bbox"],STYLE_MAP["PRED"], color=RED)
            _label_pred_number(draw, p["bbox"], pred_counter)
            legend.append(("PRED",f"Pred#{pred_counter} Conf: {categories.get(p['category_id'],p['category_id'])} ({p.get('score',0):.2f}, IoU={iou:.2f})"))
        for p in fps:
            pred_counter += 1
            _draw_rect(draw,p["bbox"],STYLE_MAP["FP"], color=RED)
            _label_pred_number(draw, p["bbox"], pred_counter)
            legend.append(("FP",f"Pred#{pred_counter} FP: {categories.get(p['category_id'],p['category_id'])} ({p.get('score',0):.2f})"))

    else:
        for g in fns:
            _draw_rect(draw,g["bbox"],STYLE_MAP["GT"], color=GREEN)
            legend.append(("GT",f"GT (solid): {categories.get(g['category_id'],g['category_id'])} (unmatched)"))
        for p,g,iou in tps+confs:
            _draw_rect(draw,g["bbox"],STYLE_MAP["GT"], color=GREEN)
            legend.append(("GT",f"GT (solid): {categories.get(g['category_id'],g['category_id'])} (matched)"))
        for p in fps:
            pred_counter += 1
            _draw_rect(draw,p["bbox"],STYLE_MAP["FP"], color=RED)
            _label_pred_number(draw, p["bbox"], pred_counter)
            legend.append(("FP",f"Pred#{pred_counter} FP: {categories.get(p['category_id'],p['category_id'])} ({p.get('score',0):.2f})"))
        for p,g,iou in tps+confs:
            pred_counter += 1
            _draw_rect(draw,p["bbox"],STYLE_MAP["PRED"], color=RED)
            _label_pred_number(draw, p["bbox"], pred_counter)
            legend.append(("PRED",f"Pred#{pred_counter}: {categories.get(p['category_id'],p['category_id'])} ({p.get('score',0):.2f}, IoU={iou:.2f})"))

    # Legend placed OUTSIDE (bottom band)
    im = _draw_legend(im,legend)
    Path(out_path).parent.mkdir(parents=True,exist_ok=True)
    im.save(out_path)

# ---------------- helper for FP case selection ----------------

def _is_fp_case(preds, gts, tps, fps, confs):
    if not gts and (preds and len(preds) > 0):
        return True
    if gts and fps:
        gt_classes = {int(gt["category_id"]) for gt in gts}
        for p in fps:
            if int(p["category_id"]) in gt_classes:
                return True
    return False


def _select_diverse(bag, k, mode):
    """
    Round-robin selection across classes up to k items.
    mode in {"conf","fp","fn"} controls how we compute class IDs per item.
    Each bag item is (img_id, im_meta, tps, fps, fns, confs).
    """
    # build mapping: class_id -> list of items
    by_cls = defaultdict(list)
    for item in bag:
        _, _, tps, fps, fns, confs = item
        if mode == "conf":
            cls_ids = [int(g["category_id"]) for _, g, _ in confs] or []
        elif mode == "fp":
            cls_ids = [int(p["category_id"]) for p in fps] or []
        elif mode == "fn":
            cls_ids = [int(g["category_id"]) for g in fns] or []
        else:
            cls_ids = []
        # if no relevant boxes for this mode, skip from diversity grouping
        # (keeps behavior stable: such items won’t be prioritized for diversity)
        for cid in set(cls_ids):
            by_cls[cid].append(item)

    # If we couldn't derive any class buckets (e.g., empty), fall back
    if not by_cls:
        return bag[:k]

    # Round-robin draw
    cls_keys = list(by_cls.keys())
    out, idx = [], 0
    # optional: shuffle within each class to vary picks between runs
    for lst in by_cls.values(): 
        # stable order is fine; comment out the next line if you want determinism
        # rng.shuffle(lst)  # requires rng in scope; skip to keep minimal
        pass
    while len(out) < k and any(by_cls.values()):
        cid = cls_keys[idx % len(cls_keys)]
        if by_cls[cid]:
            out.append(by_cls[cid].pop(0))
        idx += 1
    return out[:k]

# ---------------- export ----------------

def export_samples(results_dict,datasets_root,iou_thr=0.45,score_thr=0.25,
                                per_case_k=200,prefer_diverse_confusions=True,
                                prefer_diverse_fns_fps=True,pred_to_gt_id_offset=None,seed=0):

    for path in glob.glob("runs/detect/test*/mismatches*"):
        shutil.rmtree(path)
    rng=np.random.default_rng(seed)
    for case_name in results_dict.keys():
        run_dir=os.path.join("runs","detect",case_name)
        preds_path=os.path.join(run_dir,"predictions.json")
        test_dataset=case_name.split("_on_")[-1]
        if test_dataset != "real": continue
        gt_ann=os.path.join(datasets_root,test_dataset,"test","annotations.json")
        img_root=os.path.join(datasets_root,test_dataset,"test","images")
        if not (os.path.isfile(preds_path) and os.path.isfile(gt_ann)): continue

        # Load GT first to get images metadata (needed for in-place prediction patching).
        images,categories,gts_map=load_gt(gt_ann)

        # --- NEW: least-aggressive in-place patch of predictions.json image_id ---
        # If predictions contain non-integer image_id (filenames/paths), fix them using GT images.
        _maybe_patch_prediction_image_ids(preds_path, images)

        # Now load predictions (assumes image_id are integers)
        preds_map=load_predictions(preds_path,score_thr)
        maybe_align_category_ids(preds_map,categories,pred_to_gt_id_offset)

        conf_images,fp_images,fn_images=[],[],[]
        for img_id,im_meta in images.items():
            tps,fps,fns,confs=match_image(preds_map.get(img_id,[]),gts_map.get(img_id,[]),iou_thr)
            if confs: conf_images.append((img_id,im_meta,tps,fps,fns,confs))
            preds_here = preds_map.get(img_id, [])
            gts_here = gts_map.get(img_id, [])
            if _is_fp_case(preds_here, gts_here, tps, fps, confs):
                fp_images.append((img_id,im_meta,tps,fps,fns,confs))
            if fns: fn_images.append((img_id,im_meta,tps,fps,fns,confs))

        out_root=os.path.join(run_dir,"mismatches")
        def _render(name, bag, mode):
            save = os.path.join(out_root, name); os.makedirs(save, exist_ok=True)

            # >>> NEW: apply diversity if requested
            if (mode == "conf" and prefer_diverse_confusions) or \
               (mode in ("fp","fn") and prefer_diverse_fns_fps):
                bag_to_draw = _select_diverse(bag, per_case_k, mode)
            else:
                bag_to_draw = bag[:per_case_k]

            for img_id, im_meta, tps, fps, fns, confs in bag_to_draw:
                in_path = os.path.join(img_root, im_meta["file_name"])
                base = Path(im_meta["file_name"]).stem
                out_path = os.path.join(save, f"{base}.png")
                visualize_image(in_path, tps, fps, fns, confs, categories, out_path, draw_mode=mode)
                
        _render("confusions",conf_images,"conf")
        _render("false_positives",fp_images,"fp")
        _render("false_negatives",fn_images,"fn")



class MismatchComparison:
    """
    Build 2×2 comparison panels of 'resolved', 'emergent', and 'persistent' mismatches
    between two directories of images. Includes global row/column labels aligned to tiles.
    """

    def __init__(
        self,
        tile_w: int = 256,
        gap: int = 5,
        pad_color=(255, 255, 255),
        fontsize: int = 10,
        pad_frac: float = 0.012,
        row_pad_frac: float = 0.002,
    ):
        self.tile_w = tile_w
        self.gap = gap
        self.pad_color = pad_color
        self.fontsize = fontsize
        self.pad_frac = pad_frac
        self.row_pad_frac = row_pad_frac
        plt.rcParams['axes.titlesize'] = 10  # keep your title size

    # ----------------------------- I/O & sampling -----------------------------

    @staticmethod
    def _index_images_by_stem(folder: str,
                              exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")) -> Dict[str, str]:
        folder = Path(folder)
        out: Dict[str, str] = {}
        for ext in exts:
            for p in folder.glob(f"*{ext}"):
                out.setdefault(p.stem, str(p))
        return out

    @staticmethod
    def _sample(stems: List[str], k: int, seed: Optional[int] = None) -> List[str]:
        rng = random.Random(seed)  # local RNG; deterministic if seed is int
        if len(stems) <= k:
            return stems.copy()
        return rng.sample(stems, k)

    # ------------------------------- grid utils -------------------------------

    def _concat_grid(self, paths: List[str], rows: int, cols: int) -> np.ndarray:
        """
        Build a rows×cols grid (keeps aspect ratio per tile).
        If self.tile_w is set, images are resized by width; rows are padded to row max height.
        """
        pad_color = self.pad_color
        gap = self.gap
        tile_w = self.tile_w

        need = rows * cols
        if not paths:
            return np.ones((10, 10, 3), dtype=np.uint8) * np.array(pad_color, dtype=np.uint8)

        imgs = []
        for p in paths[:need]:
            im = Image.open(p).convert("RGB")
            if tile_w is not None and im.width != tile_w:
                new_h = max(1, round(im.height * (tile_w / im.width)))
                im = im.resize((tile_w, new_h), Image.BILINEAR)
            imgs.append(im)

        base_w = (tile_w if tile_w is not None else (imgs[0].width if imgs else 1))
        while len(imgs) < need:  # placeholders if fewer than needed
            imgs.append(Image.new("RGB", (base_w, 1), pad_color))

        row_arrays = []
        for r in range(rows):
            row_imgs = imgs[r*cols:(r+1)*cols]
            row_h = max(im.height for im in row_imgs)
            padded = []
            for im in row_imgs:
                if im.height == row_h:
                    arr = np.asarray(im)
                else:
                    canvas = Image.new("RGB", (base_w, row_h), pad_color)
                    canvas.paste(im, (0, 0))
                    arr = np.asarray(canvas)
                padded.append(arr)
            spacer = np.ones((row_h, gap, 3), dtype=np.uint8) * np.array(pad_color, dtype=np.uint8)
            row = np.hstack([np.hstack([padded[j], spacer]) if j < cols-1 else padded[j] for j in range(cols)])
            row_arrays.append(row)

        spacer_v = np.ones((gap, row_arrays[0].shape[1], 3), dtype=np.uint8) * np.array(pad_color, dtype=np.uint8)
        return np.vstack([np.vstack([row_arrays[i], spacer_v]) if i < rows-1 else row_arrays[i] for i in range(rows)])

    @staticmethod
    def _pad_to(arr: np.ndarray, H: int, W: int, color=(255, 255, 255)) -> np.ndarray:
        """Pad an array to exactly H×W with given color (top-left anchored)."""
        h, w = arr.shape[:2]
        if h == H and w == W:
            return arr
        out = np.full((H, W, 3), np.array(color, np.uint8), dtype=np.uint8)
        out[:h, :w] = arr
        return out

    def _row_tops_from_paths(self, paths: List[str], m: int, n: int) -> List[int]:
        """Pixel y (from top) of each row's top edge for a grid built like _concat_grid."""
        tile_w = self.tile_w
        gap = self.gap
        need = m * n

        imgs = []
        for p in paths[:need]:
            im = Image.open(p).convert("RGB")
            if tile_w is not None and im.width != tile_w:
                new_h = max(1, round(im.height * (tile_w / im.width)))
                im = im.resize((tile_w, new_h), Image.BILINEAR)
            imgs.append(im)

        base_w = (tile_w if tile_w is not None else (imgs[0].width if imgs else 1))
        while len(imgs) < need:
            imgs.append(Image.new("RGB", (base_w, 1), (255, 255, 255)))

        row_heights = [max(im.height for im in imgs[r*n:(r+1)*n]) for r in range(m)]

        y_tops, y = [], 0
        for r in range(m):
            y_tops.append(y)
            y += row_heights[r] + (gap if r < m-1 else 0)
        return y_tops

    # ---------------------------- annotation helpers --------------------------

    def _add_global_grid_labels(self, fig, axes, m, n, *,
                                top_left_paths=None, bottom_left_paths=None,
                                comp_top=None, comp_bottom=None):
        """
        Add row numbers (left) aligned with top-left corners of leftmost tiles,
        and column letters (a..f) centered under the bottom row across both top/bottom panels.
        """
        fontsize = self.fontsize
        pad_frac = self.pad_frac
        row_pad_frac = self.row_pad_frac

        (ax11, ax12), (ax21, ax22) = axes
        pos11, pos12 = ax11.get_position(), ax12.get_position()
        pos21, pos22 = ax21.get_position(), ax22.get_position()

        # ----- Row numbers (left side), aligned to left panels' row tops -----
        left_x = min(pos11.x0, pos21.x0) - pad_frac

        H_top = comp_top.shape[0]
        H_bot = comp_bottom.shape[0]

        ytops_top = self._row_tops_from_paths(top_left_paths, m, n)
        ytops_bot = self._row_tops_from_paths(bottom_left_paths, m, n)

        for i, ypix in enumerate(ytops_top):  # rows 1..m
            y_fig = pos11.y1 - (ypix / H_top) * pos11.height
            fig.text(left_x, y_fig - row_pad_frac, f"{i+1}", ha='right', va='top', fontsize=fontsize)

        for i, ypix in enumerate(ytops_bot):  # rows m+1..2m
            y_fig = pos21.y1 - (ypix / H_bot) * pos21.height
            fig.text(left_x, y_fig - row_pad_frac, f"{m + i + 1}", ha='right', va='top', fontsize=fontsize)

        # ----- Column letters under bottom row across both columns of panels -----
        bottom_y = min(pos21.y0, pos22.y0) - pad_frac
        for j in range(2 * n):
            pos = pos21 if j < n else pos22  # use bottom-row axes for horizontal reference
            j_local = j if j < n else (j - n)
            x = pos.x0 + (j_local + 0.5) * (pos.width / n)
            fig.text(x, bottom_y, chr(ord('a') + j), ha='center', va='top', fontsize=fontsize)

    # --------------------------------- driver ---------------------------------

    def show_mismatch_comparison(
        self,
        A_dir: str,
        B_dir: str,
        m: int,
        n: int,
        *,
        seed: int = 1337,
        output_dir: str = "./figs/fig7-9",
        output_name: str = "comparing_mismatches.pdf",
        figsize=(12, 10),
        space=(0, 0),
        titles: Optional[Dict[str, str]] = None,
        dpi: int = 300,
    ) -> None:
        """
        Build the figure and save as PDF.
        titles: optional dict with keys 'resolved', 'emergent', 'persistA', 'persistB'.
        """
        # 1) Index & set ops
        A_idx = self._index_images_by_stem(A_dir)
        B_idx = self._index_images_by_stem(B_dir)

        A_stems, B_stems = set(A_idx.keys()), set(B_idx.keys())
        resolved   = sorted(A_stems - B_stems)     # A only
        emergent   = sorted(B_stems - A_stems)     # B only
        persistent = sorted(A_stems & B_stems)     # A ∩ B

        # 2) Sample
        k = m * n
        resolved_k = self._sample(resolved,   k, seed)
        emergent_k = self._sample(emergent,   k, seed + 1)
        persist_k  = self._sample(persistent, k, seed + 2)

        resolved_paths       = [A_idx[s] for s in resolved_k]
        emergent_paths       = [B_idx[s] for s in emergent_k]
        persistent_A_paths   = [A_idx[s] for s in persist_k]
        persistent_B_paths   = [B_idx[s] for s in persist_k]

        # 3) Grids
        comp_resolved   = self._concat_grid(resolved_paths,     m, n)
        comp_emergent   = self._concat_grid(emergent_paths,     m, n)
        comp_persist_A  = self._concat_grid(persistent_A_paths, m, n)
        comp_persist_B  = self._concat_grid(persistent_B_paths, m, n)

        # 4) Pad to common size
        H = max(x.shape[0] for x in [comp_resolved, comp_emergent, comp_persist_A, comp_persist_B])
        W = max(x.shape[1] for x in [comp_resolved, comp_emergent, comp_persist_A, comp_persist_B])
        comp_resolved  = self._pad_to(comp_resolved,  H, W, self.pad_color)
        comp_emergent  = self._pad_to(comp_emergent,  H, W, self.pad_color)
        comp_persist_A = self._pad_to(comp_persist_A, H, W, self.pad_color)
        comp_persist_B = self._pad_to(comp_persist_B, H, W, self.pad_color)

        # 5) Plot
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        (ax11, ax12), (ax21, ax22) = axes

        ax11.imshow(comp_resolved);  ax11.axis("off")
        ax12.imshow(comp_emergent);  ax12.axis("off")
        ax21.imshow(comp_persist_A); ax21.axis("off")
        ax22.imshow(comp_persist_B); ax22.axis("off")

        # Titles
        if titles is None:
            ax11.set_title(f"Resolved {len(resolved_k)} of {len(resolved)}")
            ax12.set_title(f"Emergent {len(emergent_k)} of {len(emergent)}")
            ax21.set_title(f"Persistent (trained on Real) {len(persist_k)} of {len(persistent)}")
            ax22.set_title(f"Persistent (trained on Aug) {len(persist_k)} of {len(persistent)}")
        else:
            ax11.set_title(titles.get("resolved", "Resolved"))
            ax12.set_title(titles.get("emergent", "Emergent"))
            ax21.set_title(titles.get("persistA", "Persistent (trained on Real)"))
            ax22.set_title(titles.get("persistB", "Persistent (trained on Aug)"))

        plt.subplots_adjust(wspace=space[0], hspace=space[1])

        # 6) Global labels (rows/columns)
        self._add_global_grid_labels(
            fig, axes, m, n,
            top_left_paths=resolved_paths,        # top-left panel’s source list
            bottom_left_paths=persistent_A_paths, # bottom-left panel’s source list
            comp_top=comp_resolved, comp_bottom=comp_persist_A
        )

        # 7) Save
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, output_name)
        plt.savefig(out_path, format="pdf", bbox_inches='tight', dpi=dpi)
        
class ConfusionGridPlotter:
    """
    Plot a grid of (optionally normalized) confusion matrices from a results dict.

    Parameters
    ----------
    font_size : int
        Global default font size.
    axes_title_size : int
        Font size for axes titles.
    axes_label_size : int
        Font size for x/y axis labels.
    xtick_size : int
        Font size for x-tick labels.
    ytick_size : int
        Font size for y-tick labels.
    legend_size : int
        Font size for legends (kept here for parity with your rcParams block).
    """

    def __init__(self,
                 font_size: int = 16,
                 axes_title_size: int = 20,
                 axes_label_size: int = 18,
                 xtick_size: int = 14,
                 ytick_size: int = 14,
                 legend_size: int = 18):
        self.rc = dict(font_size=font_size,
                       axes_titlesize=axes_title_size,
                       axes_labelsize=axes_label_size,
                       xtick_labelsize=xtick_size,
                       ytick_labelsize=ytick_size,
                       legend_fontsize=legend_size)
        self._apply_rcparams()

    def _apply_rcparams(self):
        plt.rcParams['font.size'] = self.rc['font_size']
        plt.rcParams['axes.titlesize'] = self.rc['axes_titlesize']
        plt.rcParams['axes.labelsize'] = self.rc['axes_labelsize']
        plt.rcParams['xtick.labelsize'] = self.rc['xtick_labelsize']
        plt.rcParams['ytick.labelsize'] = self.rc['ytick_labelsize']
        plt.rcParams['legend.fontsize'] = self.rc['legend_fontsize']

    @staticmethod
    def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
        # Normalize columns with safe divide (avoid /0)
        col_sums = matrix.sum(axis=0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            norm = np.divide(matrix, col_sums, where=(col_sums != 0))
        norm[np.isnan(norm)] = 0.0
        return norm

    @staticmethod
    def _title_from_key(s: str) -> str:
        """
        Replicates your lambda logic more defensively.
        Expects keys like: YOLO_{train}_{mixup=...}_{mosaic=...}_{test}
        """
        parts = s.split('_')
        # Train domain text
        trained_on = ' on Real '
        if len(parts) > 2 and parts[2].lower().startswith('aug'):
            trained_on = ' on Aug. '
        # Pull mixup/mosaic if present
        kv = []
        for k, v in zip(parts, parts[1:]):
            if k in {'mixup', 'mosaic'}:
                kv.append(f'{k}={v}')
        kv_text = f" ({', '.join(kv)})" if kv else ""
        return f"Trained{trained_on}{kv_text}"

    def _plot_single(self,
                     ax,
                     matrix_data,
                     class_names: dict,
                     normalize: bool = True,
                     show_numbers: bool = True,
                     xlabel: bool = True,
                     ylabel: bool = True):
        # Prepare labels and matrix
        matrix = np.array(matrix_data, dtype=float)
        if normalize:
            matrix = self._normalize_columns(matrix)

        # Labels by index with fallback
        n = matrix.shape[0]
        labels = [class_names.get(i, 'background') for i in range(n)]

        im = ax.matshow(matrix, cmap=cm.Blues)

        # Ticks & tick labels
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=30)
        ax.set_yticklabels(labels)

        # Annotate cells
        vmax = matrix.max() if matrix.size else 1.0
        for i in range(n):
            for j in range(n):
                if normalize:
                    text = f'{matrix[i, j]:.2f}'
                    if show_numbers:
                        text += f'\n({int(np.array(matrix_data)[i, j])})'
                else:
                    text = f'{int(np.array(matrix_data)[i, j])}'
                ax.text(j, i, text,
                        va='center', ha='center',
                        color='black' if matrix[i, j] < 0.5 * vmax else 'white')

        if xlabel:
            ax.set_xlabel('True labels')
        else:
            ax.set_xlabel(None)
        if ylabel:
            ax.set_ylabel('Predicted labels')
        else:
            ax.set_ylabel(None)

        # Optional colorbar per subplot (commented to match your original)
        # plt.colorbar(im, ax=ax)

    def plot_all(self,
                 results: dict,
                 nrows: int = 3,
                 ncols: int = 2,
                 figsize=(16, 24),
                 filter_suffix: str = "real",
                 normalize: bool = True,
                 show_numbers: bool = True,
                 save_path: Optional[str] = None):
        """
        Build and render the full grid.

        Parameters
        ----------
        results : dict
            Mapping of case_name -> result object with:
              - result.confusion_matrix.matrix
              - result.names (dict: idx -> label)
        nrows, ncols : int
            Grid layout.
        figsize : tuple
            Figure size.
        filter_suffix : str
            Only plot cases whose key ends with this suffix (e.g., 'real').
        normalize : bool
            Column-normalize the confusion matrices.
        show_numbers : bool
            If True, show raw counts alongside normalized values.
        save_path : str | None
            If provided, saves the figure to this path (dirs created if needed).

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()

        i = 0
        for case_name, result in results.items():
            if not case_name.endswith(filter_suffix):
                continue
            if i >= len(axes):
                break  # stop if more cases than subplots

            ax = axes[i]
            conf_matrix = result.confusion_matrix
            matrix_data = conf_matrix.matrix
            class_names = result.names  # dict: idx -> label

            row = i // ncols
            xlabel = (row == nrows - 1)
            ylabel = (i % ncols == 0)

            self._plot_single(ax,
                              matrix_data,
                              class_names,
                              normalize=normalize,
                              show_numbers=show_numbers,
                              xlabel=xlabel,
                              ylabel=ylabel)

            ax.set_title(self._title_from_key(case_name))
            i += 1

        # Hide any unused axes
        for j in range(i, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, format=os.path.splitext(save_path)[1][1:] or "pdf",
                        bbox_inches='tight', dpi=300)
        return fig