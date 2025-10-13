import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

pip install torch torchvision ultralytics

import yaml

# Define the dataset structure
dataset_yaml = {
    'path': '/kaggle/input/scb-dataset/5k_HRW_yolo_Dataset_jpg',  # Base dataset path
    'train': 'images/train',  # Train images path
    'val': 'images/val',      # Validation images path
    'names': {                # Class labels
        0: 'hand-raising',
        1: 'read',
        2: 'write'
    }
}

# File path to save the YAML file
yaml_file_path = '/kaggle/working/dataset.yaml'

# Write the YAML file
with open(yaml_file_path, 'w') as file:
    yaml.dump(dataset_yaml, file, default_flow_style=False)

print(f"YAML file created at {yaml_file_path}")

from ultralytics import YOLO
import torch
from torch import nn
from torchvision.models import swin_transformer

# Custom backbone using Swin Transformer
class SwinBackbone(nn.Module):
    def __init__(self):
        super(SwinBackbone, self).__init__()
        self.swin = swin_transformer.swin_t()
        # Choose the appropriate Swin Transformer variant (t, s, b, l)

    def forward(self, x):
        x = self.swin(x).features
        # Access the feature maps from Swin Transformer
        return x

# Custom YOLOv8n model with Swin as the backbone
class YOLOv8n_Swin(nn.Module):
    def __init__(self):
        super(YOLOv8n_Swin, self).__init__()
        self.yolo = YOLO('yolov8n.pt')  # Load the YOLOv8n model
        self.yolo.model.backbone = SwinBackbone()  # Replace backbone with Swin

    def forward(self, x):
        return self.yolo.model(x)  # Use the modified model for predictions

# Initialize the YOLO model
model = YOLO('yolov8n.pt')  # Load the base YOLOv8n model

# Define the dataset YAML path
yaml_path = '/kaggle/working/dataset.yaml'

# Train the YOLOv8 model
model.train(
    data=yaml_path,     # Path to the dataset YAML file
    epochs=50,          # Number of epochs
    imgsz=640,          # Image size
    batch=16,           # Batch size
    workers=4           # Number of workers
)

from IPython.display import Image, display

# Path to the confusion matrix image
confusion_matrix_path = '/kaggle/working/runs/detect/train4/confusion_matrix.png'

# Display the image
display(Image(filename=confusion_matrix_path))

from IPython.display import Image, display

# Path to the confusion matrix image
confusion_matrix_path = '/kaggle/working/runs/detect/train4/results.png'

# Display the image
display(Image(filename=confusion_matrix_path))

import pandas as pd
dt=pd.read_csv("/kaggle/working/runs/detect/train4/results.csv")
dt

print(dt)

from IPython.display import Image, display

# Path to the confusion matrix image
confusion_matrix_path = '/kaggle/working/runs/detect/train4/val_batch0_pred.jpg'

# Display the image
display(Image(filename=confusion_matrix_path))

from IPython.display import Image, display

# Path to the confusion matrix image
confusion_matrix_path = '/kaggle/working/runs/detect/train4/val_batch1_pred.jpg'

# Display the image
display(Image(filename=confusion_matrix_path))

from IPython.display import Image, display

# Path to the confusion matrix image
confusion_matrix_path = '/kaggle/working/runs/detect/train4/val_batch2_pred.jpg'

# Display the image
display(Image(filename=confusion_matrix_path))
