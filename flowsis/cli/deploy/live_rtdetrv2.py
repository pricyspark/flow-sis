import cv2
import sys
import argparse
import torch
import time
from pathlib import Path
from flowsis.rtdetrv2 import RTDetrV2
from flowsis.utils import (
    AugmentationPipeline,
    TransformDataset,
    build_autocast_context,
    build_grad_scaler,
    get_device,
    load_training_state,
    resolve_resume_checkpoint,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with trained RT-DETRv2 object detector.")
    parser.add_argument("--model_path", type=str, default="outputs/rtdetrv2/final")
    parser.add_argument("--video_source", type=str, default="live")
    return parser.parse_args()


def main():
    device = get_device()
    args = parse_args()
    
    model = RTDetrV2(args.model_path, device=device)
    model.eval()
    
    cap_device = 0 if args.video_source == "live" else args.video_source
    cap = cv2.VideoCapture(cap_device)
    
    while True:
        ret, frame_bgr = cap.read()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        start = time.perf_counter()
        inference = model.infer(frame_rgb)
        end = time.perf_counter()
        print(inference.detections)
        print(end - start)
        
        cv2.imshow('asdf', frame_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
