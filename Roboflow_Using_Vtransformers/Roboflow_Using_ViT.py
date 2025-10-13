import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



!nvidia-smi

import os
HOME = os.getcwd()
print(HOME)

!pip install ultralytics==8.2.103 -q

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

from IPython.display import display, Image

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}
!yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source='https://media.roboflow.com/notebooks/examples/dog.jpeg' save=True

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}
Image(filename='runs/detect/predict/dog.jpeg', height=600)

# Commented out IPython magic to ensure Python compatibility.
!mkdir -p {HOME}/datasets
# %cd {HOME}/datasets
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="0XuHi2VgGqme2HRZpjjp")
project = rf.workspace("handrecognizer").project("student-action-recognition")
version = project.version(1)
dataset = version.download("yolov8")

!pip install ultralytics transformers torchvision roboflow opencv-python matplotlib -q

import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from ultralytics import YOLO
from roboflow import Roboflow

# Load the YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # Change to 'best.pt' after training


rf = Roboflow(api_key="0XuHi2VgGqme2HRZpjjp")
project = rf.workspace("handrecognizer").project("student-action-recognition")
version = project.version(1)
dataset = version.download("yolov8")  # Download YOLO-formatted dataset

# Define dataset path
DATASET_PATH = dataset.location
print("Dataset downloaded at:", DATASET_PATH)

import os
import torch
import timm
from ultralytics import YOLO
from roboflow import Roboflow
from IPython.display import Image, display

class AGLA(nn.Module):
    def __init__(self, in_channels):
        super(AGLA, self).__init__()
        self.global_att = nn.AdaptiveAvgPool2d(1)
        self.local_att = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        global_feat = self.global_att(x)
        local_feat = self.local_att(x)
        attention = self.sigmoid(global_feat + local_feat)
        return x * attention

class VAYOLO(nn.Module):
    def __init__(self, base_model, transformer, in_channels=768):
        super(VAYOLO, self).__init__()
        self.backbone = base_model.model.model[:6]  # YOLOv8 backbone layers
        self.neck = base_model.model.model[6:-2]    # YOLO Neck layers
        self.head = base_model.model.model[-2:]     # YOLO Head layers
        self.transformer = transformer
        self.agla = AGLA(in_channels)  # Adaptive Global-Local Attention
        self.conv1x1 = nn.Conv2d(in_channels, 256, kernel_size=1)  # Reduce dimensions

    def forward(self, x):
        # Pass through YOLO backbone first
        x = self.backbone(x)

        # Pass through ViT
        transformer_feat = self.transformer(x.flatten(2).transpose(1, 2))
        transformer_feat = transformer_feat.transpose(1, 2).reshape(x.shape)

        # Apply AGLA and dimensional reduction
        fused_feat = self.agla(transformer_feat)
        fused_feat = self.conv1x1(fused_feat)

        # Continue through YOLO neck and head
        x = self.neck(fused_feat)
        x = self.head(x)
        return x


# %cd {HOME}
# Load YOLOv8 model
model = YOLO('yolov8n.pt')


vit_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)

# Freeze the Vision Transformer layers
for param in vit_model.parameters():
    param.requires_grad = False

# Replace YOLO's default backbone with Swin Transformer
model.model.model[-2] = vit_model  # Inject Transformer into YOLO

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}
model.train(data=f'{dataset.location}/data.yaml', epochs=50, imgsz=800, plots=True)

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}
Image(filename=f'{HOME}/runs/detect/train2/confusion_matrix.png', width=600)

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}
Image(filename=f'{HOME}/runs/detect/train2/results.png', width=600)

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}
Image(filename=f'{HOME}/runs/detect/train2/val_batch0_pred.jpg', width=600)

#
# %cd {HOME}
Image(filename=f'{HOME}/runs/detect/train2/val_batch1_pred.jpg', width=600)


# %cd {HOME}
Image(filename=f'{HOME}/runs/detect/train2/val_batch2_pred.jpg', width=600)

# %cd {HOME}
Image(filename=f'{HOME}/runs/detect/train2/confusion_matrix_normalized.png', width=600)

import pandas as pd
print(pd.read_csv("/kaggle/working/runs/detect/train2/results.csv"))

from ultralytics import YOLO

# Load your trained models
best_model = YOLO("/kaggle/working/runs/detect/train2/weights/best.pt")  # Load best model (recommended for testing)
last_model = YOLO("/kaggle/working/runs/detect/train2/weights/last.pt")  # Load last trained model (for comparison)

import os


weights_path = "/kaggle/working/runs/detect/train2/weights/best.pt"
print("File exists:", os.path.exists(weights_path))


print("Files in the directory:", os.listdir("/kaggle/working/runs/detect/train2/weights/"))

from ultralytics import YOLO


best_model_path = "/kaggle/working/runs/detect/train2/weights/best.pt"
model = YOLO(best_model_path)

import os

train_runs_path = "/kaggle/working/runs/detect/"
print("Available training runs:", os.listdir(train_runs_path))

last_model_path = "/kaggle/working/runs/detect/train2/weights/last.pt"
model = YOLO(last_model_path)

image_path = "/kaggle/working/datasets/Student-action-recognition-1/test/images/around07_rgb_1_frame30_png_jpg.rf.923f36c4d145167bded81064e7b6c0f3.jpg"

model.predict(source=image_path, save=True, conf=0.25)

from ultralytics import YOLO

model = YOLO("/kaggle/working/runs/detect/train2/weights/best.pt")  # Use best.pt


results = model.predict(source="/kaggle/working/datasets/Student-action-recognition-1/test/images/around06_rgb_1_frame119_png_jpg.rf.e23a1c5aaae1ea4c6415b8a4297355f7.jpg", save=True, conf=0.25)


for result in results:
    print("Predictions:")
    for box in result.boxes:
        class_id = int(box.cls[0])  # Class ID
        confidence = box.conf[0].item()  # Confidence score
        print(f"Class ID: {class_id}, Confidence: {confidence:.2f}")

class_names = model.names


for result in results:
    print("Predictions:")
    for box in result.boxes:
        class_id = int(box.cls[0])  # Get class ID
        confidence = box.conf[0].item()  # Confidence score
        print(f"Class: {class_names[class_id]}, Confidence: {confidence:.2f}")
