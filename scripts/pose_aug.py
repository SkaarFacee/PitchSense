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
EPOCHS = 500
IMG_SIZE = 640
DEVICE = [0, 1] if torch.cuda.is_available() else "cpu"
PROJECT_NAME = "/home/aanil/Data/aanil/side/yolo/outputs/yolo26_s_keypoint"
RUN_NAME = "baseline_aug_resize"


# Simple augmentation settings
AUGMENTATION = {
    # Color / lighting augmentation
    "hsv_h": 0.015,
    "hsv_s": 0.4,
    "hsv_v": 0.3,

    # Geometric augmentation
    "degrees": 5.0,
    "translate": 0.10,
    "scale": 0.30,
    "shear": 2.0,
    "perspective": 0.0005,

    # Flips
    "fliplr": 0.5,
    "flipud": 0.0,

    # YOLO-style augmentation
    "mosaic": 0.5,
    "mixup": 0.05,
    "copy_paste": 0.0,

    # Turn off mosaic near the end for more stable final training
    "close_mosaic": 20,

    # Randomly resize batches around IMG_SIZE
    # 0.25 means roughly 0.75x to 1.25x of IMG_SIZE
    "multi_scale": 0.25,
}


def main():
    """Train YOLO26 pose model from scratch."""
    task = Task.init(
        project_name="PitchSense",
        task_name="yolo26s_keypoint_detection_baseline_aug_resize",
        task_type=Task.TaskTypes.training,
    )

    print(f"Training YOLO26-{MODEL_SIZE}-{MODEL_VARIANT} from scratch")
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_YAML}")
    print(f"Image size: {IMG_SIZE}")
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
        patience=50,     # early stopping
        # data augmentation + random resizing
        **AUGMENTATION,
    )

    print("-" * 50)
    print("Training complete!")
    print(f"Results saved to: {PROJECT_NAME}/{RUN_NAME}")

    task.close()


if __name__ == "__main__":
    main()