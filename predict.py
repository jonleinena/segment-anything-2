# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import cv2
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from typing import List, Dict
import matplotlib.pyplot as plt
from cog import BasePredictor, Input, Path, BaseModel

from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

DEVICE = "cuda"
MODEL_CACHE = "checkpoints"
BASE_URL = f"https://weights.replicate.delivery/default/sam-2/{MODEL_CACHE}/"

def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")

class Output(BaseModel):
    video_path: Path
    individual_masks: List[Path]

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
        model_files = [
            "sam2_hiera_large.pt",
        ]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        model_cfg = "sam2_hiera_l.yaml"
        sam2_checkpoint = f"{MODEL_CACHE}/sam2_hiera_large.pt"

        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

        # Enable bfloat16 and TF32 for better performance
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def predict(
        self,
        video: Path = Input(description="Input video file"),
        prompts: Dict[str, List[List[int]]] = Input(
            description="Dictionary of prompts for each object to track. Format: {'object_id': [[x1, y1, label], [x2, y2, label], ...]}. Label 1 for positive, 0 for negative."
        ),
    ) -> Output:
        """Run a single prediction on the model"""
        # Extract frames from video
        video_dir = self.extract_frames(video)

        # Initialize inference state
        inference_state = self.predictor.init_state(video_path=video_dir)

        # Process user prompts
        for obj_id, obj_prompts in prompts.items():
            points = np.array([[p[0], p[1]] for p in obj_prompts], dtype=np.float32)
            labels = np.array([p[2] for p in obj_prompts], dtype=np.int32)
            self.predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,  # Assume all prompts are on the first frame
                obj_id=int(obj_id),
                points=points,
                labels=labels,
            )

        # Propagate prompts and generate masks for each frame
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Create output video with masks
        output_video_path = self.create_output_video(video_dir, video_segments)

        # Save individual masks
        individual_mask_paths = self.save_individual_masks(video_segments)

        return Output(video_path=output_video_path, individual_masks=individual_mask_paths)

    def extract_frames(self, video_path: Path) -> str:
        output_dir = "/tmp/video_frames"
        os.makedirs(output_dir, exist_ok=True)
        
        command = [
            "ffmpeg", "-i", str(video_path), "-q:v", "2", 
            "-start_number", "0", f"{output_dir}/%05d.jpg"
        ]
        subprocess.run(command, check=True)
        
        return output_dir

    def create_output_video(self, video_dir: str, video_segments: dict) -> Path:
        output_video_path = Path("/tmp/output_video.mp4")
        frame_names = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        first_frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
        height, width = first_frame.shape[:2]
        out = cv2.VideoWriter(str(output_video_path), fourcc, 30, (width, height))

        for frame_idx, frame_name in enumerate(frame_names):
            frame_path = os.path.join(video_dir, frame_name)
            frame = cv2.imread(frame_path)
            
            if frame_idx in video_segments:
                for obj_id, mask in video_segments[frame_idx].items():
                    color_mask = np.concatenate([np.random.random(3), [0.35]])
                    frame[mask] = frame[mask] * (1 - color_mask[3]) + color_mask[:3] * 255 * color_mask[3]

            out.write(frame)

        out.release()
        return output_video_path

    def save_individual_masks(self, video_segments: dict) -> List[Path]:
        mask_paths = []
        os.makedirs("/tmp/individual_masks", exist_ok=True)

        for frame_idx, masks in video_segments.items():
            for obj_id, mask in masks.items():
                mask_image = mask.astype(np.uint8) * 255
                mask_path = Path(f"/tmp/individual_masks/frame_{frame_idx}_obj_{obj_id}.png")
                Image.fromarray(mask_image).save(mask_path)
                mask_paths.append(mask_path)

        return mask_paths
