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

pip install ultralytics

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

# Load the YOLOv8n model
model = YOLO('yolov8n.pt')  # Pretrained YOLOv8n weights

# Train the model
model.train(
    data='/kaggle/working/dataset.yaml',  # Path to the dataset YAML file
    epochs=50,        # Number of epochs
    imgsz=640,        # Image size
    batch=16,         # Batch size
    workers=4,        # Number of data loader workers
    device=0          # Use GPU (0) if available, otherwise CPU
)

from IPython.display import Image, display

# Path to the confusion matrix image
confusion_matrix_path = '/kaggle/working/runs/detect/train/confusion_matrix.png'

# Display the image
display(Image(filename=confusion_matrix_path))

from IPython.display import Image, display

# Path to the confusion matrix image
confusion_matrix_path = '/kaggle/working/runs/detect/train/results.png'

# Display the image
display(Image(filename=confusion_matrix_path))

import pandas as pd
dt=pd.read_csv("/kaggle/working/runs/detect/train/results.csv")
dt

from IPython.display import Image, display

# Path to the confusion matrix image
confusion_matrix_path = '/kaggle/working/runs/detect/train/val_batch0_pred.jpg'

# Display the image
display(Image(filename=confusion_matrix_path))

from IPython.display import Image, display

# Path to the confusion matrix image
confusion_matrix_path = '/kaggle/working/runs/detect/train/val_batch1_pred.jpg'

# Display the image
display(Image(filename=confusion_matrix_path))

from IPython.display import Image, display

# Path to the confusion matrix image
confusion_matrix_path = '/kaggle/working/runs/detect/train/val_batch2_pred.jpg'

# Display the image
display(Image(filename=confusion_matrix_path))
