Hybridizing YOLOv8n with Vision Transformers for Superior Real-Time Student Activity Detection

1. Description
This repository contains the implementation of the research work entitled: “Hybridizing YOLOv8n with Vision Transformers for Superior Real-Time Student Activity Detection.”
The base implementation uses YOLOv8n for real-time object detection. To improve contextual understanding, a Swin Vision Transformer (ViT) module is integrated.
This hybrid model has been tested on:
	•	SCB Dataset (Reading, Writing, Raising Hand)
	•	Roboflow Student Activity Dataset (Looking Forward, Raising Hands, Reading, Sleeping, Turning Around)

2. System Requirements
	•	Operating System: Windows 10/11, Ubuntu 20.04+, or macOS 12+
	•	Python: 3.8 – 3.11
	•	GPU (Recommended): NVIDIA GPU with CUDA 11.6+
	•	RAM: Minimum 8 GB (16 GB recommended for training)
	•	Storage: At least 10 GB free (for datasets and model checkpoints)

3. Required Libraries
Install dependencies using:
pip install -r requirements.txt
Contents of requirements.txt:
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=1.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0

4. Usage Instructions
	1	Clone or download the project. git clone <repo_link>
	2	cd YOLOv8n-ViT-Hybrid
	3	
	4	Set up the dataset. Organize your dataset as: dataset/
	5	  ├── images/train
	6	  ├── images/val
	7	  ├── labels/train
	8	  ├── labels/val
	9	
	10	Train the YOLOv8n model. python scb_using_yolo.ipynb
	11	python Roboflow_using_Yolo.ipynb
	12	
	13	Train the Vision Transformer model. python scb_using_VisionTransformers.ipynb
	14	python Roboflow_using_ViT.ipynb
	15	
	16	Hybrid Inference.
	◦	The hybrid architecture combines YOLOv8n with Swin-ViT.
	◦	Run detection on new images: python detect.py --weights runs/train/hybrid/weights/best.pt --source test_images/
	◦	
	17	Output. The program outputs:
	◦	Bounding box predictions with class labels
	◦	Performance metrics (Precision, Recall, F1-score, mAP@50, mAP@50-95)
	◦	Confusion matrix and training graphs

5. Reference
If you use this code in your research, please cite:
Chevala Chandu. Hybridizing YOLOv8n with Vision Transformers for Superior Real-Time Student Activity Detection. Arabian Journal for Science and Engineering.

6. Contact
For queries or clarifications, please contact: Chevala Chandu 📧 Email: chanduchawla3820@gmail.com

