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
                    binary_mask = outputs[0]['masks'][0].astype(np.uint8) * 255
                    
                    moments = cv2.moments(binary_mask)
                    if moments["m00"] != 0:
                        cx = moments["m10"] / moments["m00"]
                        cy = moments["m01"] / moments["m00"]
                        true_centroids.append((cx / orig_w, cy / orig_h))
                        
        # Global fallback path
        if len(true_centroids) == 0:
            outputs = self.sam_generator(pil_img)
            for output in outputs:
                if output['area'] > 200:
                    binary_mask = output['segmentation'].astype(np.uint8) * 255
                    moments = cv2.moments(binary_mask)
                    if moments["m00"] != 0:
                        cx = moments["m10"] / moments["m00"]
                        cy = moments["m01"] / moments["m00"]
                        true_centroids.append((cx / orig_w, cy / orig_h))
                        
        return true_centroids