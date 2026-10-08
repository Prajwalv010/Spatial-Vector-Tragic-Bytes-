"""Pothole Model Training & Fine-Tuning Pipeline.

Trains / fine-tunes a YOLO model specifically for ground hazard (pothole) detection,
avoiding false positives on indoor floors, shadows, keyboards, fabrics, and furniture.

Usage:
    # 1. Download or prepare your dataset (YOLO format with data.yaml)
    #    E.g. from Roboflow Universe: "Pothole Detection"
    #
    # 2. Run training:
    python scripts/train_pothole_detector.py --data path/to/data.yaml --epochs 30 --batch 8

    # 3. Quick test inference on a test image/folder:
    python scripts/train_pothole_detector.py --test path/to/image.jpg
"""

import argparse
import os
import shutil
import sys
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train / fine-tune YOLO pothole detector")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data.yaml dataset config file",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolov8n.pt",
        help="Base YOLO model weights (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('cpu', '0', etc.). Autodetects CUDA if omitted.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="pothole_yolov8.pt",
        help="Destination name for best weights in Nirman-Hackathon/ (default: pothole_yolov8.pt)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Run inference test on a single image or folder using trained model",
    )
    return parser.parse_args()


def train(args):
    from ultralytics import YOLO

    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"[*] Training Device: {device.upper()}")
    if device == "cpu":
        print("[!] Note: Training on CPU is functional but slower than GPU.")

    if not args.data or not os.path.isfile(args.data):
        print(f"[!] Error: Dataset configuration file not found at: {args.data}")
        print("\nTo set up a pothole dataset in YOLO format:")
        print("1. Download a dataset from Roboflow Universe (e.g. search 'Pothole Detection')")
        print("   Export format: YOLOv8")
        print("2. The downloaded folder contains 'data.yaml', 'train/images', 'valid/images'")
        print("3. TIP: Include 'negative samples' (indoor frames of bedrooms, desks, mousepads, keyboards)")
        print("   with EMPTY label text files (.txt). This explicitly trains the model NOT to fire on indoor items!")
        sys.exit(1)

    print(f"[*] Loading base model: {args.base_model}")
    model = YOLO(args.base_model)

    print(f"[*] Starting training for {args.epochs} epochs (imgsz={args.imgsz}, batch={args.batch})...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=2,
        plots=True,
    )

    # Find best.pt from run
    best_weights = getattr(model.trainer, "best", None)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_weights = os.path.join(repo_root, args.output_name)

    if best_weights and os.path.isfile(best_weights):
        shutil.copy2(best_weights, target_weights)
        print(f"[+] Successfully trained and saved best weights to: {target_weights}")
    else:
        print("[!] Warning: Could not locate best.pt automatically. Check runs/detect/train/weights/best.pt")

    print("[*] Training completed.")


def run_test(image_path: str, model_path: str = "pothole_yolov8.pt"):
    from ultralytics import YOLO
    import cv2

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights = os.path.join(repo_root, model_path)
    if not os.path.isfile(weights):
        weights = model_path

    if not os.path.isfile(weights):
        print(f"[!] Model weights not found at {weights}")
        return

    print(f"[*] Loading model from {weights}")
    model = YOLO(weights)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Could not read image at {image_path}")
        return

    results = model(img, conf=0.35, verbose=False)
    boxes = results[0].boxes
    print(f"\n[*] Test Results for {image_path}:")
    print(f"    Detections found: {len(boxes) if boxes is not None else 0}")
    if boxes is not None:
        for idx, box in enumerate(boxes):
            coords = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            print(f"    - Detection #{idx+1}: Conf={conf:.1%}, Box=[{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]")


def main():
    args = parse_args()
    if args.test:
        run_test(args.test, args.output_name)
    else:
        train(args)


if __name__ == "__main__":
    main()
