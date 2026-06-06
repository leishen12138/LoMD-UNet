import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset, DATASET_CONFIGS
from LoMDUNet import LoMDUNet


# ─── Argument Parsing ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser("LoMD-UNet")
parser.add_argument("--hiera_path", type=str, required=True,
                    help="path to the SAM2 pretrained hiera checkpoint")
parser.add_argument("--train_image_path", type=str, required=True,
                    help="path to the training images directory")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the training mask directory")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to store checkpoints")
parser.add_argument("--dataset", type=str, default="binary",
                    choices=["binary", "suim", "dut"],
                    help="dataset type: 'binary' (original), 'suim', or 'dut'")
parser.add_argument("--epoch", type=int, default=20,
                    help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.001,
                    help="initial learning rate")
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--weight_decay", type=float, default=5e-4)
args = parser.parse_args()


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def structure_loss_multiclass(pred, mask, num_classes):
    """
    pred: (N, C, H, W) float logits
    mask: (N, H, W) long  — class indices
    """
    # 把 long mask 转成 one-hot float，形状 (N, C, H, W)
    mask_onehot = F.one_hot(mask, num_classes=num_classes)   # (N, H, W, C)
    mask_onehot = mask_onehot.permute(0, 3, 1, 2).float()    # (N, C, H, W)

    loss = 0.0
    for c in range(num_classes):
        p = pred[:, c:c+1, :, :]          # (N, 1, H, W)
        m = mask_onehot[:, c:c+1, :, :]   # (N, 1, H, W)
        loss += structure_loss(p, m)
    return loss / num_classes

# ─── Training ─────────────────────────────────────────────────────────────────

def main(args):
    # Resolve number of classes from dataset choice
    dataset_name = args.dataset.lower()
    if dataset_name in DATASET_CONFIGS:
        num_classes = DATASET_CONFIGS[dataset_name]['num_classes']
        print(f"[Info] Multi-class mode: dataset={dataset_name}, "
              f"classes={DATASET_CONFIGS[dataset_name]['classes']}, "
              f"num_classes={num_classes}")
    else:
        num_classes = 1
        print("[Info] Binary segmentation mode")

    # Dataset & DataLoader
    dataset = FullDataset(
        args.train_image_path,
        args.train_mask_path,
        size=352,
        mode='train',
        dataset_name=dataset_name,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    device = torch.device("cuda")
    model = LoMDUNet(args.hiera_path, num_classes=num_classes)
    model.to(device)

    # Freeze encoder weights except LoRA parameters
    for name, param in model.named_parameters():
        if 'encoder' in name and 'lora' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optim = opt.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)

    os.makedirs(args.save_path, exist_ok=True)

    for epoch in range(args.epoch):
        model.train()
        for i, batch in enumerate(dataloader):
            x      = batch['image'].to(device)   # (N, 3, H, W)
            target = batch['label'].to(device)   # (N, H, W) long  OR  (N, 1, H, W) float

            optim.zero_grad()
            pred0, pred1, pred2 = model(x)
            t0 = _resize_target(target, pred0)
            t1 = _resize_target(target, pred1)
            t2 = _resize_target(target, pred2)
            loss0 = structure_loss_multiclass(pred0, t0, num_classes)
            loss1 = structure_loss_multiclass(pred1, t1, num_classes)
            loss2 = structure_loss_multiclass(pred2, t2, num_classes)
            loss = loss0 + loss1 + loss2
            loss.backward()
            optim.step()

            if i % 50 == 0:
                print(f"epoch:{epoch + 1}-{i + 1}: loss:{loss.item():.4f}  "
                      f"(l0={loss0.item():.4f}, l1={loss1.item():.4f}, "
                      f"l2={loss2.item():.4f})")

        scheduler.step()

        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epoch:
            ckpt_path = os.path.join(
                args.save_path, f'SAM2-UNet-{dataset_name}-{epoch + 1}.pth'
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f'[Saving Snapshot:] {ckpt_path}')


def _resize_target(target, pred):
    """
    Resize a long-type target mask (N, H, W) to match pred's spatial size
    (N, C, H', W') using nearest-neighbour interpolation.
    A no-op when sizes already match (the common case for the main head).
    """
    H_pred, W_pred = pred.shape[2], pred.shape[3]
    H_tgt, W_tgt   = target.shape[1], target.shape[2]
    if H_pred == H_tgt and W_pred == W_tgt:
        return target
    # F.interpolate needs float; cast back to long afterwards
    target_f = target.unsqueeze(1).float()          # (N, 1, H, W)
    target_f = F.interpolate(target_f, size=(H_pred, W_pred), mode='nearest')
    return target_f.squeeze(1).long()               # (N, H', W')


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main(args)
