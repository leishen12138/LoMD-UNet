import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
# from mamba_ssm import Mamba2
import loralib as lora


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DynamicFeatureRefiner(nn.Module):
    """动态特征融合细化器 (Dynamic Feature Fusion Refiner)"""
    def __init__(self, channel):
        super(DynamicFeatureRefiner, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channel // 4, channel, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_att = self.channel_attention(x)
        spatial_att = self.spatial_attention(x)
        refined_x = x * channel_att * spatial_att
        return refined_x


class LoRA_Adapter(nn.Module):
    def __init__(self, blk, r=8, alpha=16):
        super(LoRA_Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            lora.Linear(dim, 32, r=r, lora_alpha=alpha),
            nn.GELU(),
            lora.Linear(32, dim, r=r, lora_alpha=alpha),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        prompted_x = x + prompt
        return self.block(prompted_x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net


class MambaDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MambaDecoderBlock, self).__init__()

        # 局部特征提取（替代 SSM 的卷积分支）
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

        # 门控线性变换（替代 SSM 的序列扫描）
        # 输入 x → 分为 value 分支和 gate 分支，element-wise 相乘
        self.gate_proj = nn.Linear(in_channels, in_channels * 2, bias=False)
        self.out_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.norm = nn.LayerNorm(in_channels)

        # 最终通道映射
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W)
        residual = x

        # 局部卷积分支
        x = self.dw_conv(x)  # (B, C, H, W)

        # 序列建模分支
        B, C, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, L, C)
        x_seq = self.norm(x_seq)

        # 门控：将投影结果分成 value 和 gate 两半
        gate_out = self.gate_proj(x_seq)  # (B, L, 2C)
        value, gate = gate_out.chunk(2, dim=-1)  # each (B, L, C)
        x_seq = value * torch.sigmoid(gate)  # 门控激活

        x_seq = self.out_proj(x_seq)  # (B, L, C)
        x_out = x_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # 残差连接
        x_out = x_out + residual

        return self.conv(x_out)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class SAM2UNet(nn.Module):
    """
    SAM2-UNet with support for:
      - Binary segmentation (num_classes=1): output is a single logit map,
        trained with BCE + IoU structure loss.
      - Multi-class segmentation (num_classes>1): output is a (N, C, H, W)
        logit map, trained with cross-entropy loss.

    Args:
        checkpoint_path (str | None): Path to SAM2 pretrained checkpoint.
        num_classes (int): Number of output classes. Use 1 for binary.
    """
    def __init__(self, checkpoint_path=None, num_classes=1) -> None:
        super(SAM2UNet, self).__init__()
        self.num_classes = num_classes

        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        # blocks = []
        # for block in self.encoder.blocks:
        #     blocks.append(
        #         Adapter(block)
        #     )
        # self.encoder.blocks = nn.Sequential(
        #     *blocks
        # )

        # Replace encoder blocks with LoRA adapters
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(LoRA_Adapter(block))
        self.encoder.blocks = nn.Sequential(*blocks)

        # Dynamic feature refiners
        self.refiner1 = DynamicFeatureRefiner(144)
        self.refiner2 = DynamicFeatureRefiner(288)
        self.refiner3 = DynamicFeatureRefiner(576)
        self.refiner4 = DynamicFeatureRefiner(1152)

        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 64)
        self.rfb3 = RFB_modified(576, 64)
        self.rfb4 = RFB_modified(1152, 64)

        # Decoder with Mamba blocks
        self.up1 = Up(128, 64)
        self.mamba1 = MambaDecoderBlock(64, 64)
        self.up2 = Up(128, 64)
        self.mamba2 = MambaDecoderBlock(64, 64)
        self.up3 = Up(128, 64)
        self.mamba3 = MambaDecoderBlock(64, 64)

        # Output heads: output num_classes channels
        # For binary (num_classes=1) this is identical to the original design.
        self.side1 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.side2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.head  = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)

        # Dynamic feature refinement
        x1 = self.refiner1(x1)
        x2 = self.refiner2(x2)
        x3 = self.refiner3(x3)
        x4 = self.refiner4(x4)

        x1 = self.rfb1(x1)
        x2 = self.rfb2(x2)
        x3 = self.rfb3(x3)
        x4 = self.rfb4(x4)

        # Decoder path with Mamba blocks
        x = self.up1(x4, x3)
        x = self.mamba1(x)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear', align_corners=False)

        x = self.up2(x, x2)
        x = self.mamba2(x)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear', align_corners=False)

        x = self.up3(x, x1)
        x = self.mamba3(x)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear', align_corners=False)

        # out:  (N, num_classes, H, W)
        # out1: (N, num_classes, H, W)  – deep supervision, stride-16
        # out2: (N, num_classes, H, W)  – deep supervision, stride-8
        return out, out1, out2


if __name__ == "__main__":
    with torch.no_grad():
        # Binary test
        model = SAM2UNet(num_classes=1).cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        out, out1, out2 = model(x)
        print("binary:", out.shape, out1.shape, out2.shape)

        # Multi-class test (SUIM: 6 classes)
        model_mc = SAM2UNet(num_classes=6).cuda()
        out, out1, out2 = model_mc(x)
        print("multi-class:", out.shape, out1.shape, out2.shape)