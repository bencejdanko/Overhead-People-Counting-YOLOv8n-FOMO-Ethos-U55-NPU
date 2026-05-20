import numpy as np
import cv2
import torch
from transformers import pipeline

class SamAnnotator:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.sam_generator = pipeline(
            "mask-generation", 
            model="facebook/sam-vit-base", 
            device=self.device
        )

    @staticmethod
    def _mask_to_centroid(mask, orig_w, orig_h, min_area=0):
        binary_mask = np.asarray(mask).astype(np.uint8)
        area = int(binary_mask.sum())
        if area <= min_area:
            return None

        moments = cv2.moments(binary_mask * 255)
        if moments["m00"] == 0:
            return None

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return (cx / orig_w, cy / orig_h)

    @staticmethod
    def _iter_sam_masks(outputs):
        if isinstance(outputs, dict):
            for mask in outputs.get("masks", []):
                yield mask
            return

        for output in outputs:
            if isinstance(output, dict):
                if "segmentation" in output:
                    yield output["segmentation"]
                elif "masks" in output:
                    for mask in output["masks"]:
                        yield mask

    def get_true_centroids(self, pil_img, example_objects=None):
        orig_w, orig_h = pil_img.size
        true_centroids = []
        
        # Guided box prompt path
        if example_objects and 'bbox' in example_objects and len(example_objects['bbox']) > 0:
            for i, category in enumerate(example_objects['category']):
                if category == 0:  # COCO person index
                    x_min, y_min, w, h = example_objects['bbox'][i]
                    box_prompt = [int(x_min), int(y_min), int(x_min + w), int(y_min + h)]
                    
                    outputs = self.sam_generator(pil_img, input_boxes=[[box_prompt]])
                    for mask in self._iter_sam_masks(outputs):
                        centroid = self._mask_to_centroid(mask, orig_w, orig_h)
                        if centroid is not None:
                            true_centroids.append(centroid)
                            break
                        
        # Global fallback path
        if len(true_centroids) == 0:
            outputs = self.sam_generator(pil_img)
            for mask in self._iter_sam_masks(outputs):
                centroid = self._mask_to_centroid(mask, orig_w, orig_h, min_area=200)
                if centroid is not None:
                    true_centroids.append(centroid)
                        
        return true_centroids
