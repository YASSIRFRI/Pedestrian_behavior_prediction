"""Detection utilities for the behavior classifier."""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Union
import cv2

from ..constants import MIN_AREA_RATIO, MIN_RESOLUTION, PROXIMITY_THRESHOLD


class DetectionUtils:
    """Utilities for object detection and bounding box processing."""
    
    def __init__(self, verbose: bool = True):
        """Initialize detection utilities."""
        self.verbose = verbose

    def detect_contextual_objects(
        self, 
        image: Image.Image, 
        grounding_dino, 
        gd_processor, 
        device: str, 
        threshold: float = 0.3
    ) -> Dict:
        """Detect person, dog, stroller, bicycle, and other relevant objects."""
        # Define contextual objects that matter for behavior classification
        contextual_text = "person. dog. stroller. baby stroller. bicycle. bike. scooter. wheelchair."

        if self.verbose:
            print(f"   🎯 Detecting objects: {contextual_text}")
            print(f"   🎯 Detection threshold: {threshold}")

        inputs = gd_processor(images=image, text=contextual_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = grounding_dino(**inputs)

        image_width, image_height = image.size
        target_sizes = torch.tensor([[image_height, image_width]]).to(device)

        results = gd_processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            box_threshold=threshold,
            text_threshold=threshold,
            target_sizes=target_sizes
        )[0]

        if self.verbose:
            print(f"   📊 Raw detections: {len(results['boxes'])}")
            for i, (box, score, label) in enumerate(zip(results['boxes'], results['scores'], results['labels'])):
                print(f"      {i+1}. {label}: {score:.3f} at {box.cpu().numpy().astype(int).tolist()}")

        return results

    def extract_person_detections(self, all_detections: Dict) -> Tuple[List[np.ndarray], List[float]]:
        """Extract person detections from all detections."""
        person_boxes = []
        person_scores = []
        other_objects = []

        for i, label in enumerate(all_detections["labels"]):
            if "person" in label.lower():
                person_boxes.append(all_detections["boxes"][i].cpu().numpy())
                person_scores.append(all_detections["scores"][i].cpu().numpy())
            else:
                other_objects.append({
                    'label': label,
                    'box': all_detections["boxes"][i].cpu().numpy(),
                    'score': all_detections["scores"][i].cpu().numpy()
                })

        if self.verbose:
            print(f"   ✅ Found {len(person_boxes)} person detection(s)")
            print(f"   🎯 Found {len(other_objects)} other object(s):")
            for obj in other_objects:
                print(f"      - {obj['label']}: {obj['score']:.3f}")

        return person_boxes, person_scores

    def filter_small_detections(
        self, 
        boxes: List[np.ndarray], 
        scores: List[float], 
        image_width: int, 
        image_height: int,
        min_area_ratio: float = MIN_AREA_RATIO, 
        min_resolution: int = MIN_RESOLUTION
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Filter out detections that are too small to provide meaningful behavior classification."""
        if self.verbose:
            print(f"   🔍 Filtering small detections:")
            print(f"      Min area ratio: {min_area_ratio}")
            print(f"      Min resolution: {min_resolution}px")

        filtered_boxes = []
        filtered_scores = []
        filtered_count = 0

        total_image_area = image_width * image_height

        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height

            # Calculate area ratio
            area_ratio = box_area / total_image_area

            # Check minimum resolution (smallest dimension)
            min_dimension = min(box_width, box_height)

            # Check if detection meets criteria
            if area_ratio >= min_area_ratio and min_dimension >= min_resolution:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                if self.verbose:
                    print(f"      ✅ Kept detection {i+1}: {box_width:.0f}x{box_height:.0f} "
                          f"(area ratio: {area_ratio:.4f}, min dim: {min_dimension:.0f})")
            else:
                filtered_count += 1
                if self.verbose:
                    print(f"      ❌ Filtered detection {i+1}: {box_width:.0f}x{box_height:.0f} "
                          f"(area ratio: {area_ratio:.4f}, min dim: {min_dimension:.0f})")

        if self.verbose:
            print(f"   📊 Filtering results: {len(filtered_boxes)} kept, {filtered_count} filtered")

        return filtered_boxes, filtered_scores

    def smart_expand_box(
        self, 
        box: np.ndarray, 
        image_width: int, 
        image_height: int, 
        expansion_factor: float = 1.5
    ) -> np.ndarray:
        """Smart expansion that avoids extending upward and focuses on ground-level context."""
        x1, y1, x2, y2 = box

        if self.verbose:
            print(f"      🔍 Smart box expansion:")
            print(f"         Original: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

        width = x2 - x1
        height = y2 - y1

        left_expansion = width * 0.4 
        right_expansion = width * 0.4 
        top_expansion = height * 0.0 
        bottom_expansion = height * 0.4

        new_x1 = max(0, x1 - left_expansion)
        new_x2 = min(image_width, x2 + right_expansion)
        new_y1 = max(0, y1 - top_expansion)
        new_y2 = min(image_height, y2 + bottom_expansion)

        if self.verbose:
            print(f"         Expanded: [{new_x1:.0f}, {new_y1:.0f}, {new_x2:.0f}, {new_y2:.0f}]")
            expansion_area = (new_x2-new_x1) * (new_y2-new_y1)
            original_area = width * height
            print(f"         Area expansion: {expansion_area/original_area:.1f}x")

        return np.array([new_x1, new_y1, new_x2, new_y2])

    def merge_contextual_patches(
        self, 
        person_boxes: List[np.ndarray], 
        all_detections: Dict, 
        image_width: int, 
        image_height: int
    ) -> List[np.ndarray]:
        """Merge person bounding boxes with nearby contextual objects."""
        if self.verbose:
            print(f"   🔗 Merging contextual patches:")

        merged_patches = []

        for i, person_box in enumerate(person_boxes):
            if self.verbose:
                print(f"      Processing person {i+1}:")
                print(f"         Person box: {person_box.astype(int).tolist()}")

            # Start with the person box
            patch_boxes = [person_box]
            contextual_objects = []

            # Find nearby contextual objects
            for j, detection_box in enumerate(all_detections['boxes']):
                label = all_detections['labels'][j].lower()

                # Skip if it's another person
                if 'person' in label:
                    continue

                # Check if contextual object is close to person
                if self._boxes_are_related(person_box, detection_box.cpu().numpy(), PROXIMITY_THRESHOLD):
                    patch_boxes.append(detection_box.cpu().numpy())
                    contextual_objects.append(label)
                    if self.verbose:
                        print(f"         Added contextual object: {label}")

            # Merge all related boxes into one patch
            merged_patch = self._merge_boxes(patch_boxes, image_width, image_height)
            merged_patches.append(merged_patch)

            if self.verbose:
                print(f"         Merged patch: {merged_patch.astype(int).tolist()}")
                print(f"         Contextual objects: {contextual_objects if contextual_objects else 'None'}")

        return merged_patches

    def _boxes_are_related(
        self, 
        box1: np.ndarray, 
        box2: np.ndarray, 
        proximity_threshold: float = PROXIMITY_THRESHOLD
    ) -> bool:
        """Check if two boxes are spatially related."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate centers
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)

        # Calculate distance relative to box sizes
        box1_size = max(x2_1 - x1_1, y2_1 - y1_1)
        box2_size = max(x2_2 - x1_2, y2_2 - y1_2)
        avg_size = (box1_size + box2_size) / 2

        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        relative_distance = distance / avg_size

        return relative_distance < proximity_threshold

    def _merge_boxes(
        self, 
        boxes: List[np.ndarray], 
        image_width: int, 
        image_height: int
    ) -> np.ndarray:
        """Merge multiple boxes into one encompassing box."""
        if len(boxes) == 1:
            return boxes[0]

        # Find the encompassing box
        min_x = min(box[0] for box in boxes)
        min_y = min(box[1] for box in boxes)
        max_x = max(box[2] for box in boxes)
        max_y = max(box[3] for box in boxes)

        # Ensure bounds
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(image_width, max_x)
        max_y = min(image_height, max_y)

        return np.array([min_x, min_y, max_x, max_y])