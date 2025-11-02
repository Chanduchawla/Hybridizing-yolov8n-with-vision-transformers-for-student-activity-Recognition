import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import nms
from PIL import Image
from roboflow import Roboflow

rf = Roboflow(api_key="0XuHi2VgGqme2HRZpjjp")
project = rf.workspace("handrecognizer").project("student-action-recognition")
version = project.version(1)
dataset = version.download("yolov8")

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=640, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans * (patch_size ** 2), embed_dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        x = x.view(B, C * self.patch_size * self.patch_size, self.grid_size, self.grid_size)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)
        
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
    
    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows
    
    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
        
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        x = x.view(B, H, W, C)
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class SwinTransformerStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2
            )
            for i in range(depth)
        ])
        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None
            
    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2
        return x, H, W

class SwinTransformerBackbone(nn.Module):
    def __init__(self, img_size=640, patch_size=4, in_chans=3, 
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.stages = nn.ModuleList()
        for i_stage in range(self.num_layers):
            stage = SwinTransformerStage(
                dim=int(embed_dim * 2 ** i_stage),
                depth=depths[i_stage],
                num_heads=num_heads[i_stage],
                window_size=window_size,
                downsample=PatchMerging if i_stage < self.num_layers - 1 else None
            )
            self.stages.append(stage)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
    def forward(self, x):
        x = self.patch_embed(x)
        H, W = self.patch_embed.grid_size, self.patch_embed.grid_size
        features = []
        for stage in self.stages:
            x, H, W = stage(x, H, W)
            B, L, C = x.shape
            x_reshaped = x.transpose(1, 2).view(B, C, H, W)
            features.append(x_reshaped)
        return features

class FeatureFusionModule(nn.Module):
    def __init__(self, cnn_channels, vit_channels, out_channels):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(cnn_channels + vit_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, f_cnn, f_vit):
        if f_cnn.shape[2:] != f_vit.shape[2:]:
            f_vit = F.interpolate(f_vit, size=f_cnn.shape[2:], mode='bilinear', align_corners=False)
        f_fusion = torch.cat([f_cnn, f_vit], dim=1)
        f_out = self.fusion_conv(f_fusion)
        f_out = f_out + self.refine(f_out)
        return f_out

class HybridYOLOv8nSwinViT(nn.Module):
    def __init__(self, yolo_model_path='yolov8n.pt', num_classes=5, img_size=640):
        super().__init__()
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_backbone = self.yolo_model.model.model[:10]
        self.swin_backbone = SwinTransformerBackbone(
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7
        )
        self.fusion_modules = nn.ModuleList([
            FeatureFusionModule(cnn_channels=256, vit_channels=192, out_channels=256),
            FeatureFusionModule(cnn_channels=512, vit_channels=384, out_channels=512),
            FeatureFusionModule(cnn_channels=512, vit_channels=768, out_channels=512)
        ])
        self.yolo_neck = self.yolo_model.model.model[10:15]
        self.yolo_head = self.yolo_model.model.model[15:]
        self.num_classes = num_classes
        
    def forward(self, x):
        cnn_features = []
        for i, module in enumerate(self.yolo_backbone):
            x_cnn = module(x if i == 0 else cnn_features[-1])
            cnn_features.append(x_cnn)
        
        vit_features = self.swin_backbone(x)
        fused_features = []
        fusion_indices = [6, 8, 9]
        
        for idx, (fusion_idx, fusion_module) in enumerate(zip(fusion_indices, self.fusion_modules)):
            f_cnn = cnn_features[fusion_idx]
            f_vit = vit_features[idx + 1]
            f_fused = fusion_module(f_cnn, f_vit)
            fused_features.append(f_fused)
        
        neck_out = fused_features[-1]
        for module in self.yolo_neck:
            neck_out = module(neck_out)
        
        predictions = []
        for module in self.yolo_head:
            pred = module(neck_out)
            predictions.append(pred)
        
        return predictions

class RoboflowDataset(Dataset):
    def __init__(self, dataset_path, img_size=640, split='train', transform=None):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.split = split
        self.transform = transform
        
        self.images_dir = os.path.join(dataset_path, split, 'images')
        self.labels_dir = os.path.join(dataset_path, split, 'labels')
        
        self.image_files = [f for f in os.listdir(self.images_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        label_path = os.path.join(self.labels_dir, 
                                 os.path.splitext(self.image_files[idx])[0] + '.txt')
        
        boxes, labels = self.load_annotations(label_path)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            image = transforms.Resize((self.img_size, self.img_size))(image)
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)
        }
        
        return image, target
    
    def load_annotations(self, label_path):
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        x1 = (x_center - width/2) * self.img_size
                        y1 = (y_center - height/2) * self.img_size
                        x2 = (x_center + width/2) * self.img_size
                        y2 = (y_center + height/2) * self.img_size
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(class_id))
        
        return boxes, labels

class MetricCalculator:
    def __init__(self, num_classes, iou_thresholds=None):
        self.num_classes = num_classes
        if iou_thresholds is None:
            self.iou_thresholds = np.linspace(0.5, 0.95, 10)
        
    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def calculate_ap(self, precisions, recalls):
        recalls = np.concatenate(([0.], recalls, [1.]))
        precisions = np.concatenate(([0.], precisions, [0.]))
        
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
        
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        return ap
    
    def calculate_metrics_per_class(self, predictions, targets, class_id, iou_threshold=0.5):
        pred_boxes = []
        pred_scores = []
        gt_boxes = []
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            if len(pred['boxes']) > 0 and len(target['boxes']) > 0:
                for j, (box, score, label) in enumerate(zip(pred['boxes'], pred['scores'], pred['labels'])):
                    if label == class_id:
                        pred_boxes.append(box.cpu().numpy())
                        pred_scores.append(score.cpu().numpy())
                
                for j, (box, label) in enumerate(zip(target['boxes'], target['labels'])):
                    if label == class_id:
                        gt_boxes.append(box.cpu().numpy())
        
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return 0.0, 0.0, 0.0
        
        pred_boxes = np.array(pred_boxes)
        pred_scores = np.array(pred_scores)
        gt_boxes = np.array(gt_boxes)
        
        sorted_indices = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
        
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        gt_matched = np.zeros(len(gt_boxes))
        
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue
                
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp[i] = 1
                gt_matched[best_gt_idx] = 1
            else:
                fp[i] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / (len(gt_boxes) + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        ap = self.calculate_ap(precisions, recalls)
        
        precision = precisions[-1] if len(precisions) > 0 else 0.0
        recall = recalls[-1] if len(recalls) > 0 else 0.0
        
        return precision, recall, ap
    
    def calculate_map(self, predictions, targets):
        map50 = 0.0
        map50_95 = 0.0
        class_aps_50 = []
        class_aps_50_95 = []
        
        for class_id in range(self.num_classes):
            ap50 = 0.0
            aps = []
            
            for iou_threshold in self.iou_thresholds:
                precision, recall, ap = self.calculate_metrics_per_class(
                    predictions, targets, class_id, iou_threshold)
                aps.append(ap)
                if iou_threshold == 0.5:
                    ap50 = ap
            
            map50 += ap50
            map50_95 += np.mean(aps) if len(aps) > 0 else 0.0
            class_aps_50.append(ap50)
            class_aps_50_95.append(np.mean(aps) if len(aps) > 0 else 0.0)
        
        map50 /= self.num_classes
        map50_95 /= self.num_classes
        
        return map50, map50_95, class_aps_50, class_aps_50_95

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=0.05
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['epochs'],
            eta_min=1e-6
        )
        self.metric_calculator = MetricCalculator(num_classes=config['num_classes'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {
            'precision': [], 'recall': [], 'mAP50': [], 'mAP50_95': []
        }
        
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config["epochs"]}')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            self.optimizer.zero_grad()
            
            predictions = self.model(images)
            loss = self.calculate_loss(predictions, targets)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def calculate_loss(self, predictions, targets):
        return torch.tensor(0.1, requires_grad=True).to(self.device)
    
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                predictions = self.model(images)
                
                loss = self.calculate_loss(predictions, targets)
                val_loss += loss.item()
                
                detections = self.process_predictions(predictions)
                all_predictions.extend(detections)
                all_targets.extend(targets)
        
        avg_val_loss = val_loss / len(self.val_loader)
        self.val_losses.append(avg_val_loss)
        
        metrics = self.compute_metrics(all_predictions, all_targets)
        return avg_val_loss, metrics
    
    def process_predictions(self, predictions):
        detections = []
        for pred in predictions:
            if len(pred) > 0:
                boxes = pred[..., :4]
                scores = pred[..., 4:5]
                labels = pred[..., 5:].argmax(dim=-1)
                
                if len(boxes) > 0:
                    keep = nms(boxes, scores.squeeze(), 0.5)
                    detections.append({
                        'boxes': boxes[keep],
                        'scores': scores[keep],
                        'labels': labels[keep]
                    })
                else:
                    detections.append({
                        'boxes': torch.zeros((0, 4)),
                        'scores': torch.zeros(0),
                        'labels': torch.zeros(0, dtype=torch.int64)
                    })
            else:
                detections.append({
                    'boxes': torch.zeros((0, 4)),
                    'scores': torch.zeros(0),
                    'labels': torch.zeros(0, dtype=torch.int64)
                })
        return detections
    
    def compute_metrics(self, predictions, targets):
        map50, map50_95, class_aps_50, class_aps_50_95 = self.metric_calculator.calculate_map(predictions, targets)
        
        overall_precision = 0.0
        overall_recall = 0.0
        count = 0
        
        for class_id in range(self.config['num_classes']):
            precision, recall, _ = self.metric_calculator.calculate_metrics_per_class(
                predictions, targets, class_id, 0.5)
            overall_precision += precision
            overall_recall += recall
            count += 1
        
        overall_precision /= count if count > 0 else 1
        overall_recall /= count if count > 0 else 1
        
        metrics = {
            'precision': overall_precision,
            'recall': overall_recall,
            'mAP50': map50,
            'mAP50_95': map50_95,
            'class_aps_50': class_aps_50,
            'class_aps_50_95': class_aps_50_95
        }
        
        return metrics
    
    def train(self):
        best_map = 0
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss, metrics = self.validate(epoch)
            
            self.scheduler.step()
            
            self.metrics_history['precision'].append(metrics['precision'])
            self.metrics_history['recall'].append(metrics['recall'])
            self.metrics_history['mAP50'].append(metrics['mAP50'])
            self.metrics_history['mAP50_95'].append(metrics['mAP50_95'])
            
            if metrics['mAP50'] > best_map:
                best_map = metrics['mAP50']
                self.save_checkpoint(epoch, 'best_roboflow_model.pth')
        
        self.plot_training_curves()
        return self.metrics_history
    
    def save_checkpoint(self, epoch, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics_history': self.metrics_history
        }
        torch.save(checkpoint, filename)
        
    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.train_losses) + 1)
        
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, self.metrics_history['precision'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, self.metrics_history['mAP50'], 'c-', label='mAP@50', linewidth=2)
        axes[1, 0].plot(epochs, self.metrics_history['mAP50_95'], 'm-', label='mAP@50-95', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(epochs, self.metrics_history['precision'], label='Precision', linewidth=2)
        axes[1, 1].plot(epochs, self.metrics_history['recall'], label='Recall', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('roboflow_training_curves.png', dpi=300, bbox_inches='tight')

def create_dataset_yaml(dataset_path, output_path='roboflow_dataset.yaml'):
    dataset_config = {
        'path': dataset_path,
        'train': 'train',
        'val': 'valid',
        'test': 'test',
        'names': {
            0: 'Looking Forward',
            1: 'Raising Hand', 
            2: 'Reading',
            3: 'Sleeping',
            4: 'Turning Around'
        },
        'nc': 5
    }
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    return output_path

def main():
    config = {
        'dataset_path': dataset.location,
        'num_classes': 5,
        'img_size': 640,
        'batch_size': 16,
        'epochs': 50,
        'learning_rate': 0.001
    }
    
    yaml_path = create_dataset_yaml(config['dataset_path'])
    
    model = HybridYOLOv8nSwinViT(num_classes=config['num_classes'], img_size=config['img_size'])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = RoboflowDataset(config['dataset_path'], split='train', transform=transform)
    val_dataset = RoboflowDataset(config['dataset_path'], split='valid', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    trainer = Trainer(model, train_loader, val_loader, config)
    metrics_history = trainer.train()

if __name__ == "__main__":
    main()
