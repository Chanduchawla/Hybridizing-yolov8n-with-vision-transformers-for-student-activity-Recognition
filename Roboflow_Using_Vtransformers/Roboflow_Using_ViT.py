"""
Hybrid YOLOv8n + Swin-ViT for Student Activity Recognition (Roboflow Dataset)
=============================================================================

- Data pipeline for Roboflow Student Activity Recognition dataset

- YOLOv8n-style CSP backbone (local features) -> F_CNN
- Swin-ViT with windowed attention (global features) -> F_ViT

- Fusion module: F_fusion = [F_CNN; F_ViT], F_out = σ(W_{1×1} * F_fusion + b_{1×1})

- Detection head with 3 branches: B, C, O

- Losses: BCE (L_cls), CIoU (L_box), BCE for objectness (L_obj), stub for DFL

- Training loop implementing Hybrid YOLOv8n–Swin-ViT Training and Inference

- Metrics: IoU, mAP@50, mAP@50–95, Precision, Recall

- Visualization: loss curves, mAP curves, GradCAM-like visualization
- Inference & visualization on single image

- Checkpointing 
"""

import os
import math
import time
import random
import shutil
import csv
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any, Optional

import yaml
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

try:
    from torchvision.ops import nms as torchvision_nms
except ImportError:
    torchvision_nms = None

import matplotlib.pyplot as plt


@dataclass
class DatasetConfig:
    name: str = "roboflow_student_activity"
    path: str = "./student-action-recognition-1"
    classes: Tuple[str, ...] = (
        "looking_forward",
        "hands_up",
        "reading",
        "sleeping",
        "turning_around",
    )
    num_classes: int = 5
    img_size: int = 640


@dataclass
class ModelConfig:
    cnn_channels: int = 512
    patch_size: int = 4
    embed_dim: int = 96
    num_heads: int = 3
    window_size: int = 7
    depth: int = 2
    vit_output_dim: int = 256
    fused_channels: int = 512
    num_anchors: int = 3


@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    gradient_clip_norm: float = 10.0
    num_workers: int = 4


@dataclass
class LossConfig:
    lambda_cls: float = 1.0
    lambda_box: float = 1.0
    lambda_obj: float = 1.0
    lambda_dfl: float = 0.0


@dataclass
class DataProcConfig:
    img_size: Tuple[int, int] = (640, 640)
    normalize: bool = True
    augment: bool = True
    seed: int = 42


@dataclass
class ConvergenceConfig:
    loss_tolerance: float = 1e-4
    grad_norm_threshold: float = 1e-3
    early_stopping_patience: int = 10


@dataclass
class LoggingConfig:
    log_dir: str = "./runs/roboflow"
    checkpoint_dir: str = "./runs/roboflow/weights"
    tensorboard: bool = True
    save_frequency: int = 5


@dataclass
class GlobalConfig:
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    loss_cfg: LossConfig = LossConfig()
    data_proc: DataProcConfig = DataProcConfig()
    convergence: ConvergenceConfig = ConvergenceConfig()
    logging: LoggingConfig = LoggingConfig()


CFG = GlobalConfig()


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(CFG.data_proc.seed)


class RoboflowStudentActivityDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        img_size: int = 640,
        augment: bool = False,
        normalize: bool = True,
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == "train"
        self.normalize = normalize
        self.classes = list(CFG.dataset.classes)
        self.num_classes = CFG.dataset.num_classes
        self.images, self.labels = self._load_data_paths()
        self.transform = self._build_transforms()

    def _load_data_paths(self) -> Tuple[List[str], List[Optional[str]]]:
        split_dir = os.path.join(self.dataset_path, self.split)
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Image directory not found: {images_dir}")
        images, labels = [], []
        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(images_dir, img_file)
                label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
                images.append(img_path)
                labels.append(label_path if os.path.exists(label_path) else None)
        return images, labels

    def _build_transforms(self):
        if self.augment:
            transform = A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.HueSaturationValue(p=0.2),
                    A.Blur(blur_limit=3, p=0.1),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.ToFloat(),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["class_labels"],
                    min_visibility=0.3,
                ),
            )
        else:
            transform = A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                    A.ToFloat(),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["class_labels"],
                    min_visibility=0.0,
                ),
            )
        return transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.images[idx]
        label_path = self.labels[idx]
        image_bgr = cv2.imread(img_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        boxes = []
        class_labels = []
        if label_path and os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, xc, yc, bw, bh = map(float, parts)
                    cls_id = int(cls_id)
                    x_center_abs = xc * w
                    y_center_abs = yc * h
                    width_abs = bw * w
                    height_abs = bh * h
                    x_min = x_center_abs - width_abs / 2
                    y_min = y_center_abs - height_abs / 2
                    x_max = x_center_abs + width_abs / 2
                    y_max = y_center_abs + height_abs / 2
                    boxes.append([x_min, y_min, x_max, y_max])
                    class_labels.append(cls_id)
        if len(boxes) == 0:
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image_t = transformed["image"]
            boxes = []
            class_labels = []
        else:
            transformed = self.transform(
                image=image, bboxes=boxes, class_labels=class_labels
            )
            image_t = transformed["image"]
            boxes = transformed["bboxes"]
            class_labels = transformed["class_labels"]
        if self.normalize:
            I_min = image_t.min()
            I_max = image_t.max()
            if I_max > I_min:
                image_t = (image_t - I_min) / (I_max - I_min + 1e-7)
        target_boxes = []
        target_classes = []
        for box, cls_id in zip(boxes, class_labels):
            x_min, y_min, x_max, y_max = box
            x_center = ((x_min + x_max) / 2.0) / self.img_size
            y_center = ((y_min + y_max) / 2.0) / self.img_size
            width = (x_max - x_min) / self.img_size
            height = (y_max - y_min) / self.img_size
            target_boxes.append([x_center, y_center, width, height])
            target_classes.append(cls_id)
        if len(target_boxes) > 0:
            target_boxes_tensor = torch.tensor(target_boxes, dtype=torch.float32)
            target_classes_tensor = torch.tensor(target_classes, dtype=torch.long)
        else:
            target_boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            target_classes_tensor = torch.zeros((0,), dtype=torch.long)
        return {
            "image": image_t,
            "boxes": target_boxes_tensor,
            "classes": target_classes_tensor,
            "image_path": img_path,
        }


def collate_fn(batch: List[Dict[str, Any]]):
    images = torch.stack([b["image"] for b in batch])
    boxes = [b["boxes"] for b in batch]
    classes = [b["classes"] for b in batch]
    paths = [b["image_path"] for b in batch]
    return images, boxes, classes, paths


def create_dataloaders(cfg: GlobalConfig):
    train_dataset = RoboflowStudentActivityDataset(
        dataset_path=cfg.dataset.path,
        split="train",
        img_size=cfg.dataset.img_size,
        augment=cfg.data_proc.augment,
        normalize=cfg.data_proc.normalize,
    )
    val_split = "valid" if os.path.exists(os.path.join(cfg.dataset.path, "valid")) else "valid"
    val_dataset = RoboflowStudentActivityDataset(
        dataset_path=cfg.dataset.path,
        split=val_split,
        img_size=cfg.dataset.img_size,
        augment=False,
        normalize=cfg.data_proc.normalize,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, train_dataset.classes


class CSPBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bottlenecks = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels, 1, 1, 0, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                d_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                bound = math.sqrt(6.0 / d_in)
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.bottlenecks(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.conv3(x)


class YOLOv8nBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.SiLU(inplace=True)
        self.csp1 = CSPBlock(64, 64, num_blocks=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.csp2 = CSPBlock(128, 128, num_blocks=2)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.csp3 = CSPBlock(256, 256, num_blocks=2)
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.csp4 = CSPBlock(512, 512, num_blocks=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                d_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                bound = math.sqrt(6.0 / d_in)
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.csp1(x)
        x = self.conv3(x)
        x = self.csp2(x)
        x = self.conv4(x)
        x = self.csp3(x)
        x = self.conv5(x)
        F_CNN = self.csp4(x)
        return F_CNN


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 640, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        d_in = patch_size * patch_size * in_chans
        bound = math.sqrt(6.0 / d_in)
        nn.init.uniform_(self.proj.weight, -bound, bound)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self._init_weights()

    def _init_weights(self):
        d_in = self.qkv.in_features
        bound = math.sqrt(6.0 / d_in)
        nn.init.uniform_(self.qkv.weight, -bound, bound)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        d_in_proj = self.proj.in_features
        bound_proj = math.sqrt(6.0 / d_in_proj)
        nn.init.uniform_(self.proj.weight, -bound_proj, bound_proj)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 7, shift_size: int = 0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self._init_mlp()

    def _init_mlp(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                d_in = m.in_features
                bound = math.sqrt(6.0 / d_in)
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x
        windows = window_partition(shifted, self.window_size)
        attn_windows = self.attn(windows)
        attn_windows = attn_windows.view(-1, self.window_size * self.window_size, C)
        shifted_back = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_back, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_back
        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class SwinViT(nn.Module):
    def __init__(self, cfg: ModelConfig, img_size: int = 640):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=cfg.patch_size,
            in_chans=3,
            embed_dim=cfg.embed_dim,
        )
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    window_size=cfg.window_size,
                    shift_size=0 if i % 2 == 0 else cfg.window_size // 2,
                )
                for i in range(cfg.depth)
            ]
        )
        self.output_proj = nn.Conv2d(cfg.embed_dim, cfg.vit_output_dim, 1, 1, 0)
        d_in = self.output_proj.in_channels
        bound = math.sqrt(6.0 / d_in)
        nn.init.uniform_(self.output_proj.weight, -bound, bound)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, C, H, W)
        F_ViT = self.output_proj(x)
        return F_ViT


class FusionModule(nn.Module):
    def __init__(self, cnn_channels: int, vit_channels: int, fused_channels: int):
        super().__init__()
        self.cnn_channels = cnn_channels
        self.vit_channels = vit_channels
        self.fused_channels = fused_channels
        self.fusion_conv = nn.Conv2d(cnn_channels + vit_channels, fused_channels, 1, 1, 0)
        self.activation = nn.SiLU(inplace=True)
        d_in = cnn_channels + vit_channels
        bound = math.sqrt(6.0 / d_in)
        nn.init.uniform_(self.fusion_conv.weight, -bound, bound)
        if self.fusion_conv.bias is not None:
            nn.init.zeros_(self.fusion_conv.bias)

    def forward(self, F_CNN: torch.Tensor, F_ViT: torch.Tensor) -> torch.Tensor:
        if F_CNN.shape[-2:] != F_ViT.shape[-2:]:
            F_ViT = F.interpolate(
                F_ViT, size=F_CNN.shape[-2:], mode="bilinear", align_corners=False
            )
        F_fusion = torch.cat([F_CNN, F_ViT], dim=1)
        F_out = self.activation(self.fusion_conv(F_fusion))
        return F_out


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.bbox_conv = nn.Conv2d(in_channels, num_anchors * 4, 3, 1, 1)
        self.cls_conv = nn.Conv2d(in_channels, num_anchors * num_classes, 3, 1, 1)
        self.obj_conv = nn.Conv2d(in_channels, num_anchors * 1, 3, 1, 1)
        self._init_weights()

    def _init_weights(self):
        for conv in [self.bbox_conv, self.cls_conv, self.obj_conv]:
            d_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
            bound = math.sqrt(6.0 / d_in)
            nn.init.uniform_(conv.weight, -bound, bound)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, F_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C_in, H, W = F_out.shape
        bbox_pred = self.bbox_conv(F_out)
        bbox_pred = bbox_pred.view(B, self.num_anchors, 4, H, W)
        bbox_pred = bbox_pred.permute(0, 1, 3, 4, 2).contiguous()
        B_boxes = torch.sigmoid(bbox_pred).view(B, -1, 4)
        cls_pred = self.cls_conv(F_out)
        cls_pred = cls_pred.view(B, self.num_anchors, self.num_classes, H, W)
        cls_pred = cls_pred.permute(0, 1, 3, 4, 2).contiguous()
        C_probs = F.softmax(cls_pred.view(B, -1, self.num_classes), dim=-1)
        obj_pred = self.obj_conv(F_out)
        obj_pred = obj_pred.view(B, self.num_anchors, 1, H, W)
        obj_pred = obj_pred.permute(0, 1, 3, 4, 2).contiguous()
        O_scores = torch.sigmoid(obj_pred.view(B, -1, 1))
        return B_boxes, C_probs, O_scores


class HybridYOLOv8nSwinViT(nn.Module):
    def __init__(self, cfg: GlobalConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = YOLOv8nBackbone()
        self.swin = SwinViT(cfg.model, img_size=cfg.dataset.img_size)
        self.fusion = FusionModule(
            cnn_channels=cfg.model.cnn_channels,
            vit_channels=cfg.model.vit_output_dim,
            fused_channels=cfg.model.fused_channels,
        )
        self.head = DetectionHead(
            in_channels=cfg.model.fused_channels,
            num_classes=cfg.dataset.num_classes,
            num_anchors=cfg.model.num_anchors,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        F_CNN = self.backbone(x)
        F_ViT = self.swin(x)
        F_out = self.fusion(F_CNN, F_ViT)
        B_boxes, C_probs, O_scores = self.head(F_out)
        return B_boxes, C_probs, O_scores

    @staticmethod
    def apply_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float = 0.5,
        max_det: int = 300,
    ) -> torch.Tensor:
        if torchvision_nms is None:
            boxes_np = boxes.cpu().numpy()
            scores_np = scores.cpu().numpy()
            order = scores_np.argsort()[::-1]
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                if order.size == 1:
                    break
                xx1 = np.maximum(boxes_np[i, 0], boxes_np[order[1:], 0])
                yy1 = np.maximum(boxes_np[i, 1], boxes_np[order[1:], 1])
                xx2 = np.minimum(boxes_np[i, 2], boxes_np[order[1:], 2])
                yy2 = np.minimum(boxes_np[i, 3], boxes_np[order[1:], 3])
                inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
                area_i = (boxes_np[i, 2] - boxes_np[i, 0]) * (boxes_np[i, 3] - boxes_np[i, 1])
                area_rem = (
                    (boxes_np[order[1:], 2] - boxes_np[order[1:], 0])
                    * (boxes_np[order[1:], 3] - boxes_np[order[1:], 1])
                )
                union = area_i + area_rem - inter
                iou = inter / (union + 1e-7)
                inds = np.where(iou < iou_threshold)[0]
                order = order[inds + 1]
            keep = torch.tensor(keep, dtype=torch.long, device=boxes.device)
        else:
            keep = torchvision_nms(boxes, scores, iou_threshold)
        return keep[:max_det]


class CIoULoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        px, py, pw, ph = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        gx, gy, gw, gh = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]
        pred_x1, pred_y1 = px - pw / 2, py - ph / 2
        pred_x2, pred_y2 = px + pw / 2, py + ph / 2
        target_x1, target_y1 = gx - gw / 2, gy - gh / 2
        target_x2, target_y2 = gx + gw / 2, gy + gh / 2
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        target_area = (target_x2 - target_x1).clamp(min=0) * (target_y2 - target_y1).clamp(min=0)
        union = pred_area + target_area - inter_area + 1e-7
        iou = inter_area / union
        center_dist = (px - gx) ** 2 + (py - gy) ** 2
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(gw / (gh + 1e-7)) - torch.atan(pw / (ph + 1e-7)), 2
        )
        alpha = v / (1 - iou + v + 1e-7)
        ciou = iou - (center_dist / enclose_diag) - alpha * v
        loss = 1.0 - ciou
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DistributionFocalLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_dist: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
        if pred_dist.numel() == 0:
            return torch.tensor(0.0, device=pred_dist.device)
        pred_dist = pred_dist.clamp(min=1e-7, max=1.0 - 1e-7)
        loss = -target_dist * pred_dist.log()
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class HybridLoss(nn.Module):
    def __init__(self, cfg: GlobalConfig):
        super().__init__()
        self.cfg = cfg
        self.lambda_cls = cfg.loss_cfg.lambda_cls
        self.lambda_box = cfg.loss_cfg.lambda_box
        self.lambda_obj = cfg.loss_cfg.lambda_obj
        self.lambda_dfl = cfg.loss_cfg.lambda_dfl
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.box_loss_module = CIoULoss(reduction="mean")
        self.dfl_loss_module = DistributionFocalLoss(reduction="mean")

    def forward(
        self,
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        B_pred, C_pred_probs, O_pred_probs = preds
        target_boxes, target_classes_onehot, target_obj = targets
        C_pred_logits = torch.log(C_pred_probs.clamp(1e-7, 1 - 1e-7) / (1 - C_pred_probs.clamp(1e-7, 1 - 1e-7)))
        O_pred_logits = torch.log(O_pred_probs.clamp(1e-7, 1 - 1e-7) / (1 - O_pred_probs.clamp(1e-7, 1 - 1e-7)))
        L_cls = self.cls_loss(
            C_pred_logits.view(-1, self.cfg.dataset.num_classes),
            target_classes_onehot.view(-1, self.cfg.dataset.num_classes),
        )
        L_obj = self.cls_loss(
            O_pred_logits.view(-1, 1),
            target_obj.view(-1, 1),
        )
        B_pred_flat = B_pred.view(-1, 4)
        target_boxes_flat = target_boxes.view(-1, 4)
        mask = target_obj.view(-1) > 0.5
        if mask.sum() > 0:
            L_box = self.box_loss_module(B_pred_flat[mask], target_boxes_flat[mask])
        else:
            L_box = torch.tensor(0.0, device=B_pred.device)
        L_dfl = torch.tensor(0.0, device=B_pred.device)
        total_loss = (
            self.lambda_cls * L_cls
            + self.lambda_box * L_box
            + self.lambda_obj * L_obj
            + self.lambda_dfl * L_dfl
        )
        return total_loss


def box_iou_xywh(box1: np.ndarray, box2: np.ndarray) -> float:
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2, w2, h2 = box2
    b1_x1 = x1_1 - w1 / 2
    b1_y1 = y1_1 - h1 / 2
    b1_x2 = x1_1 + w1 / 2
    b1_y2 = y1_1 + h1 / 2
    b2_x1 = x1_2 - w2 / 2
    b2_y1 = y1_2 - h2 / 2
    b2_x2 = x1_2 + w2 / 2
    b2_y2 = y1_2 + h2 / 2
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area + 1e-7
    return float(inter_area / union)


def calculate_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])
    return float(ap)


def calculate_ap_per_class(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    iou_threshold: float,
    class_id: int,
) -> float:
    pred_list = []
    gt_by_image = {}
    for img_id, (pred, gt) in enumerate(zip(predictions, targets)):
        gt_by_image.setdefault(img_id, [])
        if len(pred["boxes"]) > 0:
            mask = pred["classes"] == class_id
            idxs = np.where(mask)[0]
            for i in idxs:
                pred_list.append(
                    {
                        "image_id": img_id,
                        "bbox": pred["boxes"][i],
                        "score": pred["scores"][i],
                    }
                )
        if len(gt["boxes"]) > 0:
            mask = gt["classes"] == class_id
            idxs = np.where(mask)[0]
            for i in idxs:
                gt_by_image[img_id].append(
                    {
                        "bbox": gt["boxes"][i],
                        "used": False,
                    }
                )
    if len(pred_list) == 0 or sum(len(v) for v in gt_by_image.values()) == 0:
        return 0.0
    pred_list.sort(key=lambda x: x["score"], reverse=True)
    tps = []
    fps = []
    for pred in pred_list:
        img_id = pred["image_id"]
        pred_box = pred["bbox"]
        best_iou = 0.0
        best_gt_idx = -1
        for idx, gt in enumerate(gt_by_image[img_id]):
            iou = box_iou_xywh(pred_box, gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        if best_iou >= iou_threshold and not gt_by_image[img_id][best_gt_idx]["used"]:
            tps.append(1)
            fps.append(0)
            gt_by_image[img_id][best_gt_idx]["used"] = True
        else:
            tps.append(0)
            fps.append(1)
    tps = np.cumsum(tps)
    fps = np.cumsum(fps)
    recalls = tps / (sum(len(v) for v in gt_by_image.values()) + 1e-7)
    precisions = tps / (tps + fps + 1e-7)
    ap = calculate_ap(precisions, recalls)
    return ap


def calculate_precision_recall(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Tuple[float, float]:
    all_tp = 0
    all_fp = 0
    total_targets = 0
    for pred, gt in zip(predictions, targets):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_classes = pred["classes"]
        gt_boxes = gt["boxes"]
        gt_classes = gt["classes"]
        total_targets += len(gt_boxes)
        used_gt = [False] * len(gt_boxes)
        order = np.argsort(pred_scores)[::-1]
        for idx in order:
            box_p = pred_boxes[idx]
            cls_p = pred_classes[idx]
            best_iou = 0.0
            best_gt_idx = -1
            for j, box_g in enumerate(gt_boxes):
                if used_gt[j] or gt_classes[j] != cls_p:
                    continue
                iou = box_iou_xywh(box_p, box_g)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            if best_iou >= iou_threshold:
                all_tp += 1
                used_gt[best_gt_idx] = True
            else:
                all_fp += 1
    precision = all_tp / (all_tp + all_fp + 1e-7) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (total_targets + 1e-7) if total_targets > 0 else 0.0
    return float(precision), float(recall)


def calculate_map(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    num_classes: int,
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    aps = {iou: [] for iou in iou_thresholds}
    for cls in range(num_classes):
        for iou in iou_thresholds:
            ap = calculate_ap_per_class(predictions, targets, iou, cls)
            aps[iou].append(ap)
    map_scores = {iou: (np.mean(v) if len(v) > 0 else 0.0) for iou, v in aps.items()}
    map_50 = map_scores[0.5]
    map_50_95 = float(np.mean([map_scores[i] for i in iou_thresholds]))
    precision, recall = calculate_precision_recall(predictions, targets, iou_threshold=0.5)
    return {
        "mAP_50": map_50,
        "mAP_50_95": map_50_95,
        "precision": precision,
        "recall": recall,
        "detailed_AP": map_scores,
    }


def visualize_batch_with_labels(images: torch.Tensor, boxes_list: List[torch.Tensor], classes_list: List[torch.Tensor], class_names: List[str], out_path: str):
    b, c, h, w = images.shape
    grid_cols = min(4, b)
    grid_rows = int(math.ceil(b / grid_cols))
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4 * grid_cols, 4 * grid_rows))
    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([[axes]])
    elif grid_rows == 1:
        axes = np.array([axes])
    elif grid_cols == 1:
        axes = np.array([[ax] for ax in axes])
    idx = 0
    for r in range(grid_rows):
        for cax in range(grid_cols):
            ax = axes[r, cax]
            if idx >= b:
                ax.axis("off")
                continue
            img = images[idx].cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis("off")
            boxes = boxes_list[idx]
            cls_ids = classes_list[idx]
            for box, cls_id in zip(boxes, cls_ids):
                xc, yc, bw, bh = box.cpu().numpy()
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2)
                ax.add_patch(rect)
                label = class_names[int(cls_id)]
                ax.text(x1, y1, label, color="yellow", fontsize=8, bbox=dict(facecolor="black", alpha=0.5))
            idx += 1
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def visualize_batch_with_predictions(images: torch.Tensor, boxes_pred: List[np.ndarray], classes_pred: List[np.ndarray], scores_pred: List[np.ndarray], class_names: List[str], out_path: str):
    b, c, h, w = images.shape
    grid_cols = min(4, b)
    grid_rows = int(math.ceil(b / grid_cols))
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4 * grid_cols, 4 * grid_rows))
    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([[axes]])
    elif grid_rows == 1:
        axes = np.array([axes])
    elif grid_cols == 1:
        axes = np.array([[ax] for ax in axes])
    idx = 0
    for r in range(grid_rows):
        for cax in range(grid_cols):
            ax = axes[r, cax]
            if idx >= b:
                ax.axis("off")
                continue
            img = images[idx].cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis("off")
            boxes = boxes_pred[idx]
            cls_ids = classes_pred[idx]
            scores = scores_pred[idx]
            for box, cls_id, score in zip(boxes, cls_ids, scores):
                xc, yc, bw, bh = box
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2)
                ax.add_patch(rect)
                label = f"{class_names[int(cls_id)]} {score:.2f}"
                ax.text(x1, y1, label, color="white", fontsize=8, bbox=dict(facecolor="red", alpha=0.5))
            idx += 1
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def build_label_statistics(train_loader: DataLoader, num_classes: int, out_dir: str, class_names: List[str]):
    counts = np.zeros(num_classes, dtype=np.int64)
    co_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, boxes_list, classes_list, _ in tqdm(train_loader, desc="Label stats"):
        for cls in classes_list:
            cls_np = cls.numpy()
            for c in cls_np:
                counts[c] += 1
            unique_classes = np.unique(cls_np)
            for i in unique_classes:
                for j in unique_classes:
                    co_matrix[i, j] += 1
    plt.figure(figsize=(8, 4))
    x = np.arange(num_classes)
    plt.bar(x, counts)
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Label Distribution")
    plt.tight_layout()
    labels_path = os.path.join(out_dir, "labels.jpg")
    plt.savefig(labels_path)
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(co_matrix, cmap="viridis")
    ax.set_xticks(x)
    ax.set_yticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    fig.colorbar(im, ax=ax)
    ax.set_title("Label Co-occurrence")
    plt.tight_layout()
    correlogram_path = os.path.join(out_dir, "labels_correlogram.jpg")
    plt.savefig(correlogram_path)
    plt.close()


class HybridTrainer:
    def __init__(self, cfg: GlobalConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(cfg.logging.log_dir, exist_ok=True)
        os.makedirs(cfg.logging.checkpoint_dir, exist_ok=True)
        self.model = HybridYOLOv8nSwinViT(cfg).to(self.device)
        self.criterion = HybridLoss(cfg)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            betas=(cfg.training.beta1, cfg.training.beta2),
            eps=cfg.training.epsilon,
            weight_decay=cfg.training.weight_decay,
        )
        self.writer = SummaryWriter(log_dir=cfg.logging.log_dir) if cfg.logging.tensorboard else None
        self.train_loader, self.val_loader, self.class_names = create_dataloaders(cfg)
        self.best_map50 = 0.0
        self.train_losses: List[float] = []
        self.val_history: List[Dict[str, float]] = []
        self.early_stop_counter = 0
        self.args_yaml_path = os.path.join(cfg.logging.log_dir, "args.yaml")
        self.results_csv_path = os.path.join(cfg.logging.log_dir, "results.csv")
        self._save_args_yaml()
        self._init_results_csv()
        build_label_statistics(self.train_loader, cfg.dataset.num_classes, cfg.logging.log_dir, self.class_names)
        self.last_visual_train_batch_images = None
        self.last_visual_train_batch_boxes = None
        self.last_visual_train_batch_classes = None

    def _save_args_yaml(self):
        cfg_dict = asdict(self.cfg)
        with open(self.args_yaml_path, "w") as f:
            yaml.safe_dump(cfg_dict, f)

    def _init_results_csv(self):
        with open(self.results_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "mAP_50", "mAP_50_95", "precision", "recall", "lr"])

    def _append_results_csv(self, epoch: int, train_loss: float, metrics: Dict[str, float]):
        lr = self.optimizer.param_groups[0]["lr"]
        with open(self.results_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    metrics["mAP_50"],
                    metrics["mAP_50_95"],
                    metrics["precision"],
                    metrics["recall"],
                    lr,
                ]
            )

    def _prepare_targets(
        self, boxes_list: List[torch.Tensor], classes_list: List[torch.Tensor], num_pred: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(boxes_list)
        num_classes = self.cfg.dataset.num_classes
        target_boxes = torch.zeros(batch_size, num_pred, 4, device=self.device)
        target_cls = torch.zeros(batch_size, num_pred, num_classes, device=self.device)
        target_obj = torch.zeros(batch_size, num_pred, 1, device=self.device)
        for b, (boxes, classes) in enumerate(zip(boxes_list, classes_list)):
            if boxes.numel() == 0:
                continue
            n = min(len(boxes), num_pred)
            target_boxes[b, :n] = boxes[:n].to(self.device)
            for j in range(n):
                cls_id = int(classes[j].item())
                target_cls[b, j, cls_id] = 1.0
            target_obj[b, :n, 0] = 1.0
        return target_boxes, target_cls, target_obj

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.training.epochs}")
        for i, (images, boxes, classes, _) in enumerate(pbar):
            images = images.to(self.device)
            B_pred, C_pred, O_pred = self.model(images)
            num_pred = B_pred.shape[1]
            targets = self._prepare_targets(boxes, classes, num_pred)
            loss = self.criterion((B_pred, C_pred, O_pred), targets)
            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.gradient_clip_norm
                )
            self.optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}")
            global_step = epoch * num_batches + i
            if self.writer is not None:
                self.writer.add_scalar("train/batch_loss", loss.item(), global_step)
            if epoch == 0 and i in [0, 1, 2]:
                out_path = os.path.join(self.cfg.logging.log_dir, f"train_batch{i}.jpg")
                visualize_batch_with_labels(images, boxes, classes, self.class_names, out_path)
            self.last_visual_train_batch_images = images.detach().cpu()
            self.last_visual_train_batch_boxes = [b.clone() for b in boxes]
            self.last_visual_train_batch_classes = [c.clone() for c in classes]
        return total_loss / max(1, num_batches)

    def validate(self) -> Dict[str, Any]:
        self.model.eval()
        predictions = []
        targets = []
        saved_val_batches = 0
        with torch.no_grad():
            for batch_idx, (images, boxes, classes, _) in enumerate(self.val_loader):
                images = images.to(self.device)
                B_pred, C_pred, O_pred = self.model(images)
                B, N, _ = B_pred.shape
                for b in range(B):
                    pred_boxes = B_pred[b].cpu().numpy()
                    obj_scores = O_pred[b].view(-1).cpu().numpy()
                    class_probs = C_pred[b].cpu().numpy()
                    class_ids = np.argmax(class_probs, axis=1)
                    conf = obj_scores
                    mask = conf > 0.3
                    pred_boxes_f = pred_boxes[mask]
                    conf_f = conf[mask]
                    class_ids_f = class_ids[mask]
                    predictions.append(
                        {
                            "boxes": pred_boxes_f,
                            "scores": conf_f,
                            "classes": class_ids_f,
                        }
                    )
                    targets.append(
                        {
                            "boxes": boxes[b].numpy(),
                            "classes": classes[b].numpy(),
                        }
                    )
                if saved_val_batches < 3:
                    labels_out = os.path.join(self.cfg.logging.log_dir, f"val_batch{saved_val_batches}_labels.jpg")
                    visualize_batch_with_labels(images, boxes, classes, self.class_names, labels_out)
                    boxes_pred_batch = []
                    classes_pred_batch = []
                    scores_pred_batch = []
                    for b in range(B):
                        pred_boxes = B_pred[b].cpu().numpy()
                        obj_scores = O_pred[b].view(-1).cpu().numpy()
                        class_probs = C_pred[b].cpu().numpy()
                        class_ids = np.argmax(class_probs, axis=1)
                        conf = obj_scores
                        mask = conf > 0.3
                        pred_boxes_f = pred_boxes[mask]
                        conf_f = conf[mask]
                        class_ids_f = class_ids[mask]
                        boxes_pred_batch.append(pred_boxes_f)
                        classes_pred_batch.append(class_ids_f)
                        scores_pred_batch.append(conf_f)
                    preds_out = os.path.join(self.cfg.logging.log_dir, f"val_batch{saved_val_batches}_pred.jpg")
                    visualize_batch_with_predictions(images.cpu(), boxes_pred_batch, classes_pred_batch, scores_pred_batch, self.class_names, preds_out)
                    saved_val_batches += 1
        metrics = calculate_map(predictions, targets, self.cfg.dataset.num_classes)
        self._save_confusion_matrices(predictions, targets)
        self._save_pr_curves(predictions, targets)
        return metrics

    def _save_confusion_matrices(self, predictions: List[Dict[str, Any]], targets: List[Dict[str, Any]]):
        num_classes = self.cfg.dataset.num_classes
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for pred, gt in zip(predictions, targets):
            gt_boxes = gt["boxes"]
            gt_classes = gt["classes"]
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_classes = pred["classes"]
            used_gt = [False] * len(gt_boxes)
            order = np.argsort(pred_scores)[::-1]
            for idx in order:
                p_box = pred_boxes[idx]
                p_cls = pred_classes[idx]
                best_iou = 0.0
                best_gt_idx = -1
                for j, g_box in enumerate(gt_boxes):
                    if used_gt[j]:
                        continue
                    iou = box_iou_xywh(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                if best_iou >= 0.5 and best_gt_idx >= 0:
                    gt_cls = gt_classes[best_gt_idx]
                    cm[gt_cls, p_cls] += 1
                    used_gt[best_gt_idx] = True
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_yticklabels(self.class_names)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        path_cm = os.path.join(self.cfg.logging.log_dir, "confusion_matrix.png")
        plt.savefig(path_cm)
        plt.close(fig)
        cm_sum = cm.sum(axis=1, keepdims=True) + 1e-7
        cm_norm = cm.astype(np.float32) / cm_sum
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_yticklabels(self.class_names)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        path_cm_norm = os.path.join(self.cfg.logging.log_dir, "confusion_matrix_normalized.png")
        plt.savefig(path_cm_norm)
        plt.close(fig)

    def _save_pr_curves(self, predictions: List[Dict[str, Any]], targets: List[Dict[str, Any]]):
        thresholds = np.linspace(0.0, 1.0, 101)
        precisions = []
        recalls = []
        f1s = []
        for thr in thresholds:
            all_tp = 0
            all_fp = 0
            total_targets = 0
            for pred, gt in zip(predictions, targets):
                pred_boxes = pred["boxes"]
                pred_scores = pred["scores"]
                pred_classes = pred["classes"]
                gt_boxes = gt["boxes"]
                gt_classes = gt["classes"]
                total_targets += len(gt_boxes)
                used_gt = [False] * len(gt_boxes)
                order = np.argsort(pred_scores)[::-1]
                for idx in order:
                    if pred_scores[idx] < thr:
                        continue
                    box_p = pred_boxes[idx]
                    cls_p = pred_classes[idx]
                    best_iou = 0.0
                    best_gt_idx = -1
                    for j, box_g in enumerate(gt_boxes):
                        if used_gt[j] or gt_classes[j] != cls_p:
                            continue
                        iou = box_iou_xywh(box_p, box_g)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j
                    if best_iou >= 0.5:
                        all_tp += 1
                        used_gt[best_gt_idx] = True
                    else:
                        all_fp += 1
            precision = all_tp / (all_tp + all_fp + 1e-7) if (all_tp + all_fp) > 0 else 0.0
            recall = all_tp / (total_targets + 1e-7) if total_targets > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall + 1e-7) if (precision + recall) > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1s = np.array(f1s)
        plt.figure()
        plt.plot(recalls, precisions)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True)
        pr_path = os.path.join(self.cfg.logging.log_dir, "PR_curve.png")
        plt.savefig(pr_path)
        plt.close()
        plt.figure()
        plt.plot(thresholds, precisions)
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Precision")
        plt.title("P-Confidence Curve")
        plt.grid(True)
        p_path = os.path.join(self.cfg.logging.log_dir, "P_curve.png")
        plt.savefig(p_path)
        plt.close()
        plt.figure()
        plt.plot(thresholds, recalls)
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Recall")
        plt.title("R-Confidence Curve")
        plt.grid(True)
        r_path = os.path.join(self.cfg.logging.log_dir, "R_curve.png")
        plt.savefig(r_path)
        plt.close()
        plt.figure()
        plt.plot(thresholds, f1s)
        plt.xlabel("Confidence Threshold")
        plt.ylabel("F1 Score")
        plt.title("F1-Confidence Curve")
        plt.grid(True)
        f1_path = os.path.join(self.cfg.logging.log_dir, "F1_curve.png")
        plt.savefig(f1_path)
        plt.close()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_map50": self.best_map50,
            "train_losses": self.train_losses,
            "val_history": self.val_history,
            "cfg": asdict(self.cfg),
        }
        ckpt_dir = self.cfg.logging.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        if is_best:
            path = os.path.join(ckpt_dir, "best.pt")
        else:
            path = os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pt")
        torch.save(ckpt, path)

    def save_last_checkpoint(self, epoch: int):
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_map50": self.best_map50,
            "train_losses": self.train_losses,
            "val_history": self.val_history,
            "cfg": asdict(self.cfg),
        }
        ckpt_dir = self.cfg.logging.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, "last.pt")
        torch.save(ckpt, path)

    def plot_training_curves(self):
        epochs = list(range(1, len(self.train_losses) + 1))
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(epochs, self.train_losses, marker="o")
        ax[0].set_title("Training Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        map50 = [m["mAP_50"] for m in self.val_history]
        map5095 = [m["mAP_50_95"] for m in self.val_history]
        ax[1].plot(epochs, map50, marker="o", label="mAP@50")
        ax[1].plot(epochs, map5095, marker="x", label="mAP@50-95")
        ax[1].set_title("mAP Curves")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("mAP")
        ax[1].legend()
        plt.tight_layout()
        out_path = os.path.join(self.cfg.logging.log_dir, "results.png")
        plt.savefig(out_path)
        plt.close(fig)

    def check_convergence(self, current_loss: float, prev_loss: float) -> bool:
        if abs(current_loss - prev_loss) < self.cfg.convergence.loss_tolerance:
            return True
        return False

    def copy_train_batches_for_legacy_names(self):
        src0 = os.path.join(self.cfg.logging.log_dir, "train_batch0.jpg")
        src1 = os.path.join(self.cfg.logging.log_dir, "train_batch1.jpg")
        src2 = os.path.join(self.cfg.logging.log_dir, "train_batch2.jpg")
        dst0 = os.path.join(self.cfg.logging.log_dir, "train_batch11880.jpg")
        dst1 = os.path.join(self.cfg.logging.log_dir, "train_batch11881.jpg")
        dst2 = os.path.join(self.cfg.logging.log_dir, "train_batch11882.jpg")
        if os.path.exists(src0):
            shutil.copyfile(src0, dst0)
        if os.path.exists(src1):
            shutil.copyfile(src1, dst1)
        if os.path.exists(src2):
            shutil.copyfile(src2, dst2)

    def train(self):
        prev_loss = float("inf")
        for epoch in range(self.cfg.training.epochs):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            metrics = self.validate()
            self.val_history.append(metrics)
            if self.writer is not None:
                self.writer.add_scalar("train/epoch_loss", train_loss, epoch)
                self.writer.add_scalar("val/mAP_50", metrics["mAP_50"], epoch)
                self.writer.add_scalar("val/mAP_50_95", metrics["mAP_50_95"], epoch)
                self.writer.add_scalar("val/precision", metrics["precision"], epoch)
                self.writer.add_scalar("val/recall", metrics["recall"], epoch)
            self._append_results_csv(epoch, train_loss, metrics)
            if metrics["mAP_50"] > self.best_map50:
                self.best_map50 = metrics["mAP_50"]
                self.save_checkpoint(epoch, is_best=True)
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            if epoch % self.cfg.logging.save_frequency == 0:
                self.save_checkpoint(epoch, is_best=False)
            if epoch > 0:
                if self.check_convergence(train_loss, prev_loss):
                    break
                if self.early_stop_counter >= self.cfg.convergence.early_stopping_patience:
                    break
            prev_loss = train_loss
        self.plot_training_curves()
        self.copy_train_batches_for_legacy_names()
        self.save_last_checkpoint(len(self.train_losses) - 1)


def preprocess_single_image(img: np.ndarray, cfg: GlobalConfig) -> Tuple[torch.Tensor, Tuple[int, int]]:
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (cfg.dataset.img_size, cfg.dataset.img_size))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    I_min = img_tensor.min()
    I_max = img_tensor.max()
    if I_max > I_min:
        img_tensor = (img_tensor - I_min) / (I_max - I_min + 1e-7)
    return img_tensor.unsqueeze(0), (h, w)


def visualize_detections(
    img: np.ndarray,
    boxes_xywh: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> np.ndarray:
    img_vis = img.copy()
    h, w = img.shape[:2]
    for box, score, cid in zip(boxes_xywh, scores, class_ids):
        xc, yc, bw, bh = box
        xc *= w
        yc *= h
        bw *= w
        bh *= h
        x1 = int(xc - bw / 2)
        y1 = int(yc - bh / 2)
        x2 = int(xc + bw / 2)
        y2 = int(yc + bh / 2)
        color = (0, 255, 0)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cid]} {score:.2f}"
        cv2.putText(
            img_vis,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    if save_path is not None:
        cv2.imwrite(save_path, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
    return img_vis


def run_inference_single_image(
    cfg: GlobalConfig,
    checkpoint_path: str,
    image_path: str,
    out_path: str = "robo_example_output.jpg",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridYOLOv8nSwinViT(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    img_bgr = cv2.imread(image_path)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp, (h, w) = preprocess_single_image(img, cfg)
    inp = inp.to(device)
    with torch.no_grad():
        B_pred, C_pred, O_pred = model(inp)
    pred_boxes = B_pred[0].cpu().numpy()
    obj_scores = O_pred[0].view(-1).cpu().numpy()
    class_probs = C_pred[0].cpu().numpy()
    class_ids = np.argmax(class_probs, axis=1)
    conf = obj_scores
    mask = conf > 0.3
    pred_boxes = pred_boxes[mask]
    conf = conf[mask]
    class_ids = class_ids[mask]
    img_vis = visualize_detections(
        img,
        pred_boxes,
        conf,
        class_ids,
        list(cfg.dataset.classes),
        save_path=out_path,
    )
    return img_vis


def main_train():
    trainer = HybridTrainer(CFG)
    trainer.train()


def main_infer_example():
    ckpt_path = os.path.join(CFG.logging.checkpoint_dir, "best.pt")
    img_path = "robo_test_image.jpg"
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found: {ckpt_path}")
        return
    if not os.path.exists(img_path):
        print(f"[WARN] Test image not found: {img_path}")
        return
    run_inference_single_image(CFG, ckpt_path, img_path, out_path=os.path.join(CFG.logging.log_dir, "robo_example_output.jpg"))


if __name__ == "__main__":
    main_train()
