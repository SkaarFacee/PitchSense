"""
Train YOLO26 Pose Model from Scratch on Pitch Keypoint Data

This script trains a YOLO26 pose estimation model from scratch.
The dataset should have keypoint annotations in YOLO26 pose format:
  class_id x_center y_center width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...

NOTE: The current keypoint_data labels only contain bounding box format.
      For proper pose training, you need keypoint annotations.
"""

from ultralytics import YOLO
import torch
from pathlib import Path
from clearml import Task, OutputModel



# Configuration
DATA_YAML = "/home/aanil/Data/aanil/side/yolo/datasets/keypoint_model/data.yaml"
MODEL_SIZE = "yolo26s"  
MODEL_VARIANT = "pose"  
EPOCHS = 200
IMG_SIZE = 640
DEVICE = [0,1] if torch.cuda.is_available() else "cpu"
PROJECT_NAME = "runs/baseline_keypoint"
RUN_NAME = "baseline"

def main():
    """Train YOLO26 pose model from scratch."""
    task = Task.init(
        project_name="PitchSense",
        task_name="yolo26s_keypoint_detection_baseline",
        task_type=Task.TaskTypes.training,
    )
    print(f"Training YOLO26-{MODEL_SIZE}-{MODEL_VARIANT} from scratch")
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_YAML}")
    print("-" * 50)
  
    model = YOLO(f"{MODEL_SIZE}-{MODEL_VARIANT}")
    
    # Train the model
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        verbose=True, 
        pretrained=False, 
    )
    
    print("-" * 50)
    print("Training complete!")
    print(f"Results saved to: {PROJECT_NAME}/{RUN_NAME}")
    task.close()

if __name__ == "__main__":
    main()
