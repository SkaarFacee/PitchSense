import pathlib
import shutil
from typing import Dict, List, Tuple

import torch
import yaml
from clearml import Task, OutputModel
from torch.utils.data import random_split
from ultralytics import YOLO

from datasetLoader import PitchSenseDataset


BASE_PATH = "/home/aanil/Data/aanil/side/yolo/datasets/Soccernet/tracking"
OUTPUT_ROOT = pathlib.Path("/home/aanil/Data/aanil/side/yolo/outputs/baseline_200epochs")
SAVE_DIR = OUTPUT_ROOT / "saved_models"

MODEL_NAME = "yolo26n.yaml"  
VAL_RATIO = 0.15
SEED = 42
EPOCHS = 200
IMGSZ = 1280
BATCH = 32  
DEVICE = [0,1] if torch.cuda.is_available() else "cpu"



class PATHS:
    train_path = pathlib.Path(f"{BASE_PATH}/train")
    test_path = pathlib.Path(f"{BASE_PATH}/test")


def yolo_box_from_xywh(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    box_w = w / img_w
    box_h = h / img_h
    return x_center, y_center, box_w, box_h


def clamp_box(cx: float, cy: float, bw: float, bh: float) -> Tuple[float, float, float, float]:
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 1e-6), 1.0)
    bh = min(max(bh, 1e-6), 1.0)
    return cx, cy, bw, bh


def get_image_size(sample: dict) -> Tuple[int, int]:
    config = sample["config"]
    img_w = config.get("imwidth")
    img_h = config.get("imheight")

    if img_w is None or img_h is None:
        raise ValueError(f"Missing image size in config for sample: {sample['img_path']}")

    return int(img_w), int(img_h)


def normalize_class_name(name: str) -> str:
    return str(name).strip().lower()


def build_class_mapping(dataset) -> Tuple[Dict[str, int], List[str]]:
    class_names = set()

    for sample in dataset:
        gt_df = sample["gt"]
        if gt_df is None or gt_df.empty:
            continue

        for value in gt_df["name"].dropna().unique():
            class_names.add(normalize_class_name(value))

    if not class_names:
        raise ValueError("No class names found in dataset ground truth.")

    sorted_names = sorted(class_names)
    class_map = {name: idx for idx, name in enumerate(sorted_names)}
    return class_map, sorted_names


def make_unique_stem(img_path: pathlib.Path) -> str:
    match_name = img_path.parent.parent.name
    return f"{match_name}_{img_path.stem}"


def export_split(samples, split_name: str, out_root: pathlib.Path, class_map: Dict[str, int]) -> None:
    images_dir = out_root / "images" / split_name
    labels_dir = out_root / "labels" / split_name

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    written_images = 0
    written_labels = 0

    for sample in samples:
        img_path = pathlib.Path(sample["img_path"])
        gt_df = sample["gt"]

        if not img_path.exists():
            print(f"[WARN] Missing image, skipping: {img_path}")
            continue

        img_w, img_h = get_image_size(sample)
        stem = make_unique_stem(img_path)

        dst_img_path = images_dir / f"{stem}{img_path.suffix}"
        dst_label_path = labels_dir / f"{stem}.txt"

        shutil.copy2(img_path, dst_img_path)
        written_images += 1

        label_lines = []
        if gt_df is not None and not gt_df.empty:
            for _, row in gt_df.iterrows():
                raw_name = row.get("name", None)
                if raw_name is None:
                    continue

                class_name = normalize_class_name(raw_name)
                if class_name not in class_map:
                    continue

                x = float(row["x"])
                y = float(row["y"])
                w = float(row["w"])
                h = float(row["h"])

                if w <= 0 or h <= 0:
                    continue

                class_id = class_map[class_name]
                cx, cy, bw, bh = yolo_box_from_xywh(x, y, w, h, img_w, img_h)
                cx, cy, bw, bh = clamp_box(cx, cy, bw, bh)

                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        written_labels += 1

    print(f"[{split_name}] exported images: {written_images}, labels: {written_labels}")


def write_dataset_yaml(out_root: pathlib.Path, class_names: List[str]) -> pathlib.Path:
    data = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }

    yaml_path = out_root / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return yaml_path


def copy_best_weights(save_dir: pathlib.Path, run_name: str) -> pathlib.Path | None:
    best_weights = save_dir / run_name / "weights" / "best.pt"
    final_model_path = save_dir / "yolo26n_best.pt"

    if best_weights.exists():
        shutil.copy2(best_weights, final_model_path)
        print(f"Saved best model to: {final_model_path}")
        return final_model_path

    print("[WARN] best.pt not found after training.")
    return None


def main() -> None:
    task = Task.init(
        project_name="PitchSense",
        task_name="yolo26n_baseline_scratch",
        task_type=Task.TaskTypes.training,
    )

    task.connect(
        {
            "base_path": BASE_PATH,
            "output_root": str(OUTPUT_ROOT),
            "save_dir": str(SAVE_DIR),
            "model_name": MODEL_NAME,
            "val_ratio": VAL_RATIO,
            "seed": SEED,
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "device": DEVICE,
            "train_path": str(PATHS.train_path),
            "test_path": str(PATHS.test_path),
        }
    )

    logger = task.get_logger()

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Training device:", DEVICE)

    logger.report_text(f"CUDA available: {torch.cuda.is_available()}")
    logger.report_text(f"CUDA device count: {torch.cuda.device_count()}")
    logger.report_text(f"Training device: {DEVICE}")

    train_root = PATHS.train_path
    test_root = PATHS.test_path

    dataset = PitchSenseDataset([train_root, test_root])

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    print(f"Dataset samples: {len(dataset)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    logger.report_scalar("dataset", "total_samples", value=len(dataset), iteration=0)
    logger.report_scalar("dataset", "train_samples", value=len(train_dataset), iteration=0)
    logger.report_scalar("dataset", "val_samples", value=len(val_dataset), iteration=0)

    class_map, class_names = build_class_mapping(dataset)
    print("Class mapping:", class_map)
    logger.report_text(f"Class mapping: {class_map}")

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    export_split(train_dataset, "train", OUTPUT_ROOT, class_map)
    export_split(val_dataset, "val", OUTPUT_ROOT, class_map)

    yaml_path = write_dataset_yaml(OUTPUT_ROOT, class_names)
    print(f"YOLO dataset yaml written to: {yaml_path}")
    task.upload_artifact("dataset_yaml", artifact_object=str(yaml_path))

    run_name = "yolo26n_baseline_scratch"

    model = YOLO(MODEL_NAME)
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=str(SAVE_DIR),
        name=run_name,
        pretrained=False,
        verbose=True,
    )

    final_model_path = copy_best_weights(SAVE_DIR, run_name)

    if final_model_path and final_model_path.exists():
        output_model = OutputModel(task=task, name="yolo26n_best")
        output_model.update_weights(weights_filename=str(final_model_path))
        task.upload_artifact("best_model", artifact_object=str(final_model_path))

    task.upload_artifact("output_root", artifact_object=str(OUTPUT_ROOT))
    print("Training complete.")
    task.close()


if __name__ == "__main__":
    main()