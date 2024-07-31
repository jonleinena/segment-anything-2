from cog import BasePredictor, Input, Path
import torch
import numpy as np
import cv2
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import os
import tempfile
import shutil

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        sam2_checkpoint = "sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    def predict(
        self,
        video: Path = Input(description="Input video file"),
        points: str = Input(description="Comma-separated list of x,y coordinates for positive points"),
        negative_points: str = Input(description="Comma-separated list of x,y coordinates for negative points", default=""),
        object_id: int = Input(description="Unique ID for the object to be tracked", default=1)
    ) -> Path:
        """Run video segmentation prediction on the model"""
        # Create a temporary directory to store video frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract frames from the video
            cap = cv2.VideoCapture(str(video))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            for i in range(frame_count):
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(os.path.join(temp_dir, f"{i:05d}.jpg"), frame)
                else:
                    break
            cap.release()

            # Initialize inference state
            inference_state = self.predictor.init_state(video_path=temp_dir)

            # Process input points
            pos_points = np.array([list(map(float, p.split(','))) for p in points.split()], dtype=np.float32)
            neg_points = np.array([list(map(float, p.split(','))) for p in negative_points.split()] if negative_points else [], dtype=np.float32)
            
            # Combine points and labels
            all_points = np.vstack([pos_points, neg_points]) if len(neg_points) > 0 else pos_points
            labels = np.array([1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32)

            # Add points and get initial mask
            _, _, _ = self.predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=object_id,
                points=all_points,
                labels=labels,
            )

            # Propagate the mask through the video
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # Create output video with segmentation overlay
            output_path = Path("output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

            for i in range(frame_count):
                frame = cv2.imread(os.path.join(temp_dir, f"{i:05d}.jpg"))
                if i in video_segments and object_id in video_segments[i]:
                    mask = video_segments[i][object_id]
                    frame[mask] = frame[mask] * 0.5 + np.array([0, 0, 255], dtype=np.uint8) * 0.5
                out.write(frame)

            out.release()

        return output_path