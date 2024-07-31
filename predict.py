# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import io
import time
import torch
import mimetypes
import subprocess
import numpy as np
from tqdm import tqdm
from PIL import Image
import supervision as sv
from typing import Iterator, Tuple, List, Dict
import matplotlib.pyplot as plt
from cog import BasePredictor, Input, Path
from contextlib import contextmanager
import shutil
import tempfile
from typing import Any

mimetypes.add_type("image/webp", ".webp")

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

class Predictor(BasePredictor):
    def setup(self) -> None:
        global build_sam2

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            print("sam2 not found. Installing...")
            os.system("pip install --no-build-isolation -e .")
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
        model_files = ["sam2_hiera_tiny.pt"]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        model_cfg = "sam2_hiera_t.yaml"
        sam2_checkpoint = f"{MODEL_CACHE}/sam2_hiera_tiny.pt"

        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))

        # Enable bfloat16 and TF32 for better performance
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.mask_annotator = sv.MaskAnnotator()
        self.box_annotator = sv.BoxAnnotator()

    def parse_inputs(
        self,
        click_coordinates: str,
        click_labels: str,
        click_object_ids: str,
    ) -> tuple:
        click_coordinates = click_coordinates.replace(" ", "")
        click_labels = click_labels.replace(" ", "")
        click_object_ids = click_object_ids.replace(" ", "")

        # Parse click coordinates
        click_list = [
            list(map(int, click.replace(" ", "").split(",")))
            for click in click_coordinates.strip("[]").split("],[")
        ]
        num_clicks = len(click_list)
        # Handle click labels
        click_labels_list = list(map(int, click_labels.split(",")))
        click_labels_list = click_labels_list * (
            num_clicks // len(click_labels_list) + 1
        )
        click_labels_list = click_labels_list[:num_clicks]
        # Handle click object IDs
        if click_object_ids:
            object_ids_list = click_object_ids.split(",")
        else:
            object_ids_list = [f"object_{i}" for i in range(1, num_clicks + 1)]
        object_ids_list = object_ids_list * (num_clicks // len(object_ids_list) + 1)
        object_ids_list = object_ids_list[:num_clicks]

        # Map string labels to unique integer IDs
        label_to_id = {}
        id_counter = 1
        object_ids_int_list = []
        for label in object_ids_list:
            if label not in label_to_id:
                label_to_id[label] = id_counter
                id_counter += 1
            object_ids_int_list.append(label_to_id[label])

        return (click_list, click_labels_list, object_ids_int_list)

    def save_image(self, image: Image.Image, path: Path, format: str, quality: int):
        save_params = {"format": format.upper()}
        if format.lower() != "png":
            save_params["quality"] = quality
            save_params["optimize"] = True
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, **save_params)

    @contextmanager
    def temporary_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def predict(
        self,
        input_image: Path = Input(description="Input image file path"),
        # Segmentation inputs
        click_coordinates: str = Input(
            description="Click coordinates as '[x,y],[x,y],...'. Determines number of clicks."
        ),
        click_labels: str = Input(
            description="Click types (1=foreground, 0=background) as '1,1,0,1'. Auto-extends if shorter than coordinates.",
            default="1",
        ),
        click_object_ids: str = Input(
            description="Object labels for clicks as 'person,dog,cat'. Auto-generates if missing or incomplete.",
            default="",
        ),
        output_quality: int = Input(
            description="JPG/WebP compression quality (0-100, ignored for PNG and video)",
            default=80,
            ge=0,
            le=100,
        ),
       
    ) -> Dict[str, Any]:
        # Parse inputs
        click_list, click_labels_list, object_ids_int_list = (
            self.parse_inputs(
                click_coordinates, click_labels, click_object_ids
            )
        )

        # Create output directory
        output_dir = Path("predict_outputs")
        output_dir.mkdir(exist_ok=True)

        # Load and process the image
        image = np.array(Image.open(input_image).convert("RGB"))
        self.predictor.set_image(image)

        # Process clicks and generate masks
        masks = []
        for click, click_type, obj_id in zip(click_list, click_labels_list, object_ids_int_list):
            points = np.array([click], dtype=np.float32)
            labels = np.array([click_type], dtype=np.int32)
            mask, _, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False,
            )
            print(mask[0].shape)
            masks.append(mask[0])
        print(masks)

        # Generate and save the result
        annotated_image = self.process_image(image, masks, object_ids_int_list)
        output_path = output_dir / f"output.webp"

        pil_image = Image.fromarray(annotated_image)
        pil_image.save(output_path, format="WEBP", quality=output_quality)

        # Extract contours and normalize coordinates using Supervision
        normalized_contours = self.extract_contours_using_supervision(masks, (image.shape[0], image.shape[1]))

        return {
            "output_image": output_path,
            "contours": normalized_contours
        }

    def process_image(self, image, masks, tracker_ids):
        masks = np.stack(masks).astype(bool)
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks,
            class_id=np.array(tracker_ids),
        )

        annotated_image = image.copy()
        annotated_image = self.mask_annotator.annotate(
            scene=annotated_image, detections=detections
        )

        return annotated_image

    def extract_contours_using_supervision(self, masks: List[np.ndarray], image_shape: Tuple[int, int]) -> List[Dict[str, any]]:
        """Extracts normalized contour coordinates from masks using Supervision's polygon utilities.

        Args:
            masks (List[np.ndarray]): List of binary masks.
            image_shape (Tuple[int, int]): Shape of the image as (height, width).

        Returns:
            List[Dict[str, any]]: List containing object IDs and their corresponding normalized contours.
        """
        contours_info = []
        height, width = image_shape

        for idx, mask in enumerate(masks):
            # Convert mask to boolean
            mask_bool = mask.astype(bool)
            # Use Supervision's mask_to_polygon to get polygons
            polygons = sv.mask_to_polygons(mask_bool)
            normalized_contours = []
            for polygon in polygons:
                # Normalize the polygon coordinates
                normalized_polygon = polygon.tolist()
                normalized_polygon = [
                    [point[0] / width, point[1] / height] for point in normalized_polygon
                ]
                normalized_contours.append(normalized_polygon)
            contours_info.append({
                "object_id": idx + 1,
                "contours": normalized_contours
            })

        return contours_info