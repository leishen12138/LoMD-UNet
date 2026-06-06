import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from LoMDUNet import LoMDUNet
from dataset import TestDataset, DATASET_CONFIGS
import torchvision.transforms.functional as TF


# ─── 参数 ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser("LoMD-UNet Test")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to the model checkpoint")
parser.add_argument("--test_image_path", type=str, required=True,
                    help="path to the test image directory")
parser.add_argument("--test_gt_path", type=str, required=True,
                    help="path to the test ground-truth mask directory")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to save predicted masks")
parser.add_argument("--dataset", type=str, default="binary",
                    choices=["binary", "suim", "dut"],
                    help="dataset type (determines number of classes and colour palette)")
args = parser.parse_args()


def add_gaussian_noise(img, sigma=0):

    if sigma <= 0:
        return img

    noise = torch.randn_like(img) * (sigma / 255.0)

    noisy_img = img + noise

    return noisy_img

def add_gaussian_blur(img, kernel_size=0):

    """
    img: tensor (B,C,H,W)
    kernel_size: odd number, e.g. 3,5,7
    """

    if kernel_size <= 1:
        return img

    blurred_img = TF.gaussian_blur(
        img,
        kernel_size=[kernel_size, kernel_size]
    )

    return blurred_img

def add_color_distortion(img, red_scale=1.0):

    """
    img: tensor (B,C,H,W)

    red_scale:
        1.0 = clean
        0.8 = mild
        0.6 = medium
        0.4 = strong
    """

    distorted_img = img.clone()

    # RGB
    # channel 0 = R
    distorted_img[:, 0, :, :] *= red_scale

    return distorted_img


# ─── 配置 ──────────────────────────────────────────────────────────────────────

dataset_name = args.dataset.lower()
if dataset_name in DATASET_CONFIGS:
    cfg = DATASET_CONFIGS[dataset_name]
    num_classes = cfg['num_classes']
    palette = np.array(cfg['palette'], dtype=np.uint8)   # (C, 3)
    print(f"[Info] Multi-class mode | dataset={dataset_name} | "
          f"classes={cfg['classes']} | num_classes={num_classes}")
else:
    num_classes = 1
    palette = None
    print("[Info] Binary segmentation mode")


# ─── 模型 & 数据集 ─────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 352,
                          dataset_name=dataset_name)
model = LoMDUNet(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
model.eval()

os.makedirs(args.save_path, exist_ok=True)


# ─── 推理循环 ──────────────────────────────────────────────────────────────────

for i in range(test_loader.size):
    with torch.no_grad():
        image, gt, name = test_loader.load_data()
        image = image.to(device)                              # (1, 3, H, W)
        # image = add_gaussian_noise(image, sigma=35)
        # image = add_gaussian_blur(image, kernel_size=9)
        image = add_color_distortion(
            image,
            red_scale=0.2
        )

        res, _, _ = model(image)                              # (1, C, h, w)

        # 将输出上采样到 GT 原始尺寸
        gt_h, gt_w = gt.shape[:2]
        res = F.interpolate(res, size=(gt_h, gt_w),
                            mode='bilinear', align_corners=False)

        if num_classes == 1:
            # ── 二值模式：与原始逻辑完全一致 ──────────────────────────────
            res = res.sigmoid().data.cpu().numpy().squeeze()  # (H, W) float
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res * 255).astype(np.uint8)                # 灰度图
            save_name = name[:-4] + ".png"
            imageio.imsave(os.path.join(args.save_path, save_name), res)

        else:
            # ── 多分类模式 ────────────────────────────────────────────────
            # pred_cls: (H, W) int，每像素的预测类别索引
            pred_cls = res.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.int32)

            # 将类别索引映射为 RGB 颜色图（与训练时的 palette 对应）
            rgb_out = palette[pred_cls]                        # (H, W, 3) uint8

            save_name = name[:-4] + ".png"
            imageio.imsave(os.path.join(args.save_path, save_name), rgb_out)

        print(f"Saved  {save_name}")
