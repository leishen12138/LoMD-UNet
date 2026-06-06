"""
eval.py  —  多分类语义分割评估脚本
支持数据集：binary / suim / dut
指标：mIoU, mRecall, mPrecision, mF1-score, mPA（均为逐类计算后取均值）

用法示例：
  python eval.py \
      --dataset suim \
      --pred_path ./results/suim \
      --gt_path   /data/SUIM/test/masks
"""

import os
import argparse
import numpy as np
import cv2

from dataset import DATASET_CONFIGS


# ─── 参数 ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser("SAM2-UNet Evaluation")
parser.add_argument("--dataset", type=str, required=True,
                    choices=["binary", "suim", "dut"],
                    help="数据集类型，决定类别数与调色板")
parser.add_argument("--pred_path", type=str, required=True,
                    help="预测结果目录（PNG 图像）")
parser.add_argument("--gt_path", type=str, required=True,
                    help="GT mask 目录")
args = parser.parse_args()


# ─── 数据集配置 ────────────────────────────────────────────────────────────────

dataset_name = args.dataset.lower()

if dataset_name in DATASET_CONFIGS:
    cfg = DATASET_CONFIGS[dataset_name]
    num_classes = cfg['num_classes']
    class_names = cfg['classes']
    palette = np.array(cfg['palette'], dtype=np.uint8)   # (C, 3)
    is_binary = False
    print(f"[Info] 多分类模式 | dataset={dataset_name} | "
          f"num_classes={num_classes} | classes={class_names}")
else:
    num_classes = 2          # 前景 / 背景
    class_names = ('Background', 'Foreground')
    palette = None
    is_binary = True
    print("[Info] 二值分割模式 | threshold=127")


# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def rgb_to_classmap(img_bgr: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    将 BGR 图像（OpenCV 读入）按调色板转为类别索引图 (H, W) int32。
    对每个像素取 L1 最近调色板颜色。
    palette: (C, 3) RGB uint8
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.int16)  # (H,W,3)
    pal     = palette.astype(np.int16)                                     # (C, 3)
    # (H, W, 1, 3) - (1, 1, C, 3)  →  (H, W, C)
    diff    = np.abs(img_rgb[:, :, None, :] - pal[None, None, :, :]).sum(axis=3)
    return diff.argmin(axis=2).astype(np.int32)


def binary_to_classmap(img_gray: np.ndarray, threshold: int = 127) -> np.ndarray:
    """灰度图 → 二值类别图（0=背景，1=前景）"""
    return (img_gray > threshold).astype(np.int32)


def update_confusion_matrix(conf_mat: np.ndarray,
                             pred_cls: np.ndarray,
                             gt_cls:   np.ndarray,
                             num_classes: int) -> None:
    """原地累加混淆矩阵。行=GT，列=Pred。"""
    mask = (gt_cls >= 0) & (gt_cls < num_classes)
    conf_mat += np.bincount(
        num_classes * gt_cls[mask].astype(np.int64) + pred_cls[mask].astype(np.int64),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)


def compute_metrics(conf_mat: np.ndarray):
    """
    从混淆矩阵计算逐类及均值指标。
    conf_mat[i, j] = 真实类别 i 被预测为类别 j 的像素数。

    返回字典：
        per_class_iou, per_class_recall, per_class_precision, per_class_f1,
        per_class_pa
        mIoU, mRecall, mPrecision, mF1, mPA
    """
    # TP[c] = conf_mat[c, c]
    TP = np.diag(conf_mat).astype(np.float64)
    # FP[c] = 该列之和 - TP（预测为 c 但实际不是 c）
    FP = conf_mat.sum(axis=0).astype(np.float64) - TP
    # FN[c] = 该行之和 - TP（实际是 c 但未预测为 c）
    FN = conf_mat.sum(axis=1).astype(np.float64) - TP

    eps = 1e-8

    iou = TP / (TP + FP + FN + eps)
    recall = TP / (TP + FN + eps)          # = Sensitivity
    precision = TP / (TP + FP + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    # PA[c] = TP[c] / (TP[c] + FN[c])，即该类被正确分类的像素比例
    # 注：PA 与 Recall 共享同一公式（逐类像素准确率 = 该类召回率）
    pa = TP / (TP + FN + eps)

    # 只对 GT 中出现过的类别取均值（避免空类别拉低分数）
    present = conf_mat.sum(axis=1) > 0        # 该类在 GT 中出现过

    return {
        "per_class_iou":       iou,
        "per_class_recall":    recall,
        "per_class_precision": precision,
        "per_class_f1":        f1,
        "per_class_pa":        pa,
        "present":             present,
        "mIoU":       iou[present].mean(),
        "mRecall":    recall[present].mean(),
        "mPrecision": precision[present].mean(),
        "mF1":        f1[present].mean(),
        "mPA":        pa[present].mean(),
    }


# ─── 主评估循环 ────────────────────────────────────────────────────────────────

conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
mask_names = sorted(os.listdir(args.gt_path))
total = len(mask_names)

for i, mask_name in enumerate(mask_names):
    gt_path = os.path.join(args.gt_path,   mask_name)
    pred_name = os.path.splitext(mask_name)[0] + ".png"
    pred_path = os.path.join(args.pred_path, pred_name)

    if not os.path.exists(pred_path):
        print(f"[Warning] 预测文件不存在，跳过: {pred_path}")
        continue

    gt_img = cv2.imread(gt_path)
    pred_img = cv2.imread(pred_path)

    if gt_img is None or pred_img is None:
        print(f"[Warning] 图像读取失败，跳过: {mask_name}")
        continue

    # 统一到 GT 尺寸
    h, w = gt_img.shape[:2]
    if pred_img.shape[:2] != (h, w):
        pred_img = cv2.resize(pred_img, (w, h), interpolation=cv2.INTER_NEAREST)

    # 转为类别索引图
    if is_binary:
        gt_cls = binary_to_classmap(cv2.cvtColor(gt_img,   cv2.COLOR_BGR2GRAY))
        pred_cls = binary_to_classmap(cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY))
    else:
        gt_cls = rgb_to_classmap(gt_img,   palette)
        pred_cls = rgb_to_classmap(pred_img, palette)

    update_confusion_matrix(conf_mat, pred_cls, gt_cls, num_classes)
    print(f"[{i+1:4d}/{total}] {mask_name}")


# ─── 计算并打印结果 ────────────────────────────────────────────────────────────

results = compute_metrics(conf_mat)

col_w = max(len(n) for n in class_names) + 2   # 列宽自适应类名长度

print("\n" + "=" * 60)
print(f"  数据集: {args.dataset.upper()}    样本数: {total}")
print("=" * 60)

# 表头
print(f"\n{'类别':<{col_w}}  {'IoU':>8}  {'Recall':>8}  {'Precision':>10}  {'F1':>8}  {'PA':>8}")
print("-" * (col_w + 53))

for c, name in enumerate(class_names):
    marker = "" if results["present"][c] else "  ← (GT中未出现)"
    print(f"{name:<{col_w}}  "
          f"{results['per_class_iou'][c]:>8.4f}  "
          f"{results['per_class_recall'][c]:>8.4f}  "
          f"{results['per_class_precision'][c]:>10.4f}  "
          f"{results['per_class_f1'][c]:>8.4f}  "
          f"{results['per_class_pa'][c]:>8.4f}"
          f"{marker}")

print("-" * (col_w + 53))
print(f"{'Mean (present)':<{col_w}}  "
      f"{results['mIoU']:>8.4f}  "
      f"{results['mRecall']:>8.4f}  "
      f"{results['mPrecision']:>10.4f}  "
      f"{results['mF1']:>8.4f}  "
      f"{results['mPA']:>8.4f}")
print("=" * 60)

print(f"\n  mIoU       : {results['mIoU']:.4f}")
print(f"  mRecall    : {results['mRecall']:.4f}")
print(f"  mPrecision : {results['mPrecision']:.4f}")
print(f"  mF1-score  : {results['mF1']:.4f}")
print(f"  mPA        : {results['mPA']:.4f}")
print("=" * 60 + "\n")