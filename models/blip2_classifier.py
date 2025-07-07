"""BLIP-2 based behavior classifier implementation."""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict

from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    AutoProcessor, AutoModelForZeroShotObjectDetection,
    pipeline
)

from ..constants import (
    BEHAVIOR_CLASSES, DEFAULT_BLIP2_MODEL, GROUNDING_DINO_MODEL,
    ZERO_SHOT_MODEL, DEFAULT_DETECTION_THRESHOLD, MIN_AREA_RATIO,
    MIN_RESOLUTION, EXPANSION_FACTOR, PROXIMITY_THRESHOLD,
    BEHAVIOR_LABEL_MAPPING
)
from ..utils.detection import DetectionUtils
from ..utils.visualization import VisualizationUtils


class BLIP2BehaviorClassifier:
    """BLIP-2 based behavior classifier for detecting and classifying human behaviors in images and videos."""
    
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose: bool = True,
        save_outputs: bool = True,
        output_dir: str = "./outputs",
        blip2_model: str = DEFAULT_BLIP2_MODEL
    ):
        """
        Initialize BLIP-2 Behavior Classifier.

        Args:
            device: Device to run models on ('cuda' or 'cpu')
            verbose: Enable detailed logging
            save_outputs: Save comprehensive visualizations and debug info
            output_dir: Directory to save outputs
            blip2_model: BLIP-2 model to use
        """
        self.device = device
        self.blip2_model_name = blip2_model
        self.model_name = f"BLIP2-{blip2_model.split('/')[-1]}"
        self.verbose = verbose
        self.save_outputs = save_outputs
        self.output_dir = output_dir

        # Initialize utilities
        self.detection_utils = DetectionUtils(verbose=verbose)
        self.viz_utils = VisualizationUtils(verbose=verbose)

        # Create output directory structure
        if self.save_outputs:
            self._setup_output_directories()

        # Initialize processing counters
        self._reset_counters()

        if self.verbose:
            self._print_initialization_info()

        # Load models
        self._load_models()

        if self.verbose:
            self._print_completion_info()

    def _setup_output_directories(self) -> None:
        """Create organized output directory structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.output_dir, f"blip2_session_{timestamp}")

        self.dirs = {
            'comprehensive': os.path.join(self.session_dir, 'comprehensive_visualizations'),
            'data': os.path.join(self.session_dir, 'data'),
            'debug': os.path.join(self.session_dir, 'debug_info')
        }

        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        if self.verbose:
            print(f"📁 Created output directories:")
            for name, path in self.dirs.items():
                print(f"   📂 {name}: {path}")

    def _reset_counters(self) -> None:
        """Reset processing counters for new session."""
        self.counters = {
            'images_processed': 0,
            'persons_detected': 0,
            'persons_filtered': 0,
            'visualizations_saved': 0,
            'captions_generated': 0,
            'classifications_made': 0
        }

    def _print_initialization_info(self) -> None:
        """Print initialization information."""
        print(f"🚀 {'='*70}")
        print(f"🚀 INITIALIZING BLIP-2 BEHAVIOR CLASSIFIER")
        print(f"🚀 {'='*70}")
        print(f"🔧 Device: {self.device}")
        print(f"🤖 BLIP-2 Model: {self.blip2_model_name}")
        print(f"📝 Verbose mode: {self.verbose}")
        print(f"💾 Save outputs: {self.save_outputs}")
        if self.save_outputs:
            print(f"📁 Output directory: {self.output_dir}")

    def _print_completion_info(self) -> None:
        """Print completion information."""
        print(f"✅ {'='*70}")
        print(f"✅ INITIALIZATION COMPLETE!")
        print(f"✅ Behavior classes: {BEHAVIOR_CLASSES}")
        print(f"✅ {'='*70}\n")

    def _load_models(self) -> None:
        """Load all models with detailed progress tracking."""
        if self.verbose:
            print(f"🔄 Loading models on {self.device}...")

        # Load BLIP-2 model
        self._load_blip2_model()
        
        # Load zero-shot text classifier
        self._load_zero_shot_classifier()
        
        # Load Grounding DINO for detection
        self._load_grounding_dino()

        # Define behavior labels for classification
        self.behavior_labels = [
            "walking", "running", "riding a bicycle", "walking a dog",
            "riding a scooter", "standing", "sitting", "pushing a"
        ]

    def _load_blip2_model(self) -> None:
        """Load BLIP-2 model."""
        if self.verbose:
            print(f"📷 Loading BLIP-2 model: {self.blip2_model_name}...")
        
        try:
            self.blip2_processor = Blip2Processor.from_pretrained(self.blip2_model_name)

            if self.device == 'cuda':
                if self.verbose:
                    print(f"   🔧 Loading BLIP-2 with float16 precision for GPU...")
                self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                    self.blip2_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                if self.verbose:
                    print(f"   🔧 Loading BLIP-2 with float32 precision for CPU...")
                self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                    self.blip2_model_name,
                    torch_dtype=torch.float32
                )
                self.blip2_model.to(self.device)

            self.blip2_model.eval()
            
            if self.verbose:
                print(f"   ✅ BLIP-2 model loaded successfully")
                total_params = sum(p.numel() for p in self.blip2_model.parameters())
                trainable_params = sum(p.numel() for p in self.blip2_model.parameters() if p.requires_grad)
                print(f"   📊 Total parameters: {total_params:,}")
                print(f"   📊 Trainable parameters: {trainable_params:,}")

        except Exception as e:
            print(f"   ❌ Error loading BLIP-2 model: {e}")
            raise

    def _load_zero_shot_classifier(self) -> None:
        """Load zero-shot text classifier."""
        if self.verbose:
            print(f"🔤 Loading zero-shot text classifier...")
        
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=ZERO_SHOT_MODEL,
                device=0 if self.device == 'cuda' else -1
            )
            if self.verbose:
                print(f"   ✅ Zero-shot classifier loaded successfully")
        except Exception as e:
            print(f"   ❌ Error loading zero-shot classifier: {e}")
            raise

    def _load_grounding_dino(self) -> None:
        """Load Grounding DINO detector."""
        if self.verbose:
            print(f"🎯 Loading Grounding DINO detector...")
        
        try:
            self.gd_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
            self.grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(
                GROUNDING_DINO_MODEL
            ).to(self.device)
            self.grounding_dino.eval()
            if self.verbose:
                print(f"   ✅ Grounding DINO loaded successfully")
        except Exception as e:
            print(f"   ❌ Error loading Grounding DINO: {e}")
            raise

    def generate_blip2_caption(
        self, 
        image: Union[Image.Image, np.ndarray], 
        prompt: Optional[str] = None
    ) -> str:
        """Generate caption for an image using BLIP-2."""
        if self.verbose:
            print(f"      📝 Generating BLIP-2 caption...")
            if prompt:
                print(f"         Using prompt: '{prompt}'")

        try:
            # Prepare inputs
            if prompt:
                inputs = self.blip2_processor(images=image, text=prompt, return_tensors="pt")
            else:
                inputs = self.blip2_processor(images=image, return_tensors="pt")

            # Move to device and appropriate dtype
            if self.device == 'cuda':
                inputs = {k: v.to(self.device, torch.float16 if v.dtype == torch.float32 else v.dtype)
                         for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate caption
            with torch.no_grad():
                generated_ids = self.blip2_model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=5,
                    temperature=1.0,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    do_sample=True,
                    early_stopping=True
                )

            # Decode the generated caption
            caption = self.blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            # Clean up the caption (remove prompt if it was included)
            if prompt and caption.lower().startswith(prompt.lower()):
                caption = caption[len(prompt):].strip()

            self.counters['captions_generated'] += 1

            if self.verbose:
                print(f"         Generated caption: '{caption}'")
                print(f"         Caption length: {len(caption)} characters")

            return caption

        except Exception as e:
            if self.verbose:
                print(f"         ❌ Error generating caption: {e}")
            return "Error generating caption"

    def generate_enhanced_caption(self, image: Union[Image.Image, np.ndarray]) -> Tuple[str, Dict[str, str]]:
        """Generate multiple captions with different prompts for better behavior understanding."""
        if self.verbose:
            print(f"      🔍 Generating enhanced captions with multiple prompts...")

        prompts = [
            "",
            "A person is",
            "The person on is X on the sidewalk. What is X?",
        ]

        captions = {}
        for i, prompt in enumerate(prompts):
            if self.verbose:
                prompt_name = "unconditional" if prompt == "" else f"prompt_{i}"
                print(f"         Generating {prompt_name} caption...")

            caption = self.generate_blip2_caption(image, prompt if prompt else None)
            captions[f"prompt_{i}" if prompt else "unconditional"] = caption

        main_caption = captions["unconditional"]

        if self.verbose:
            print(f"      📋 Caption variations:")
            for prompt_type, caption in captions.items():
                print(f"         {prompt_type}: '{caption}'")

        return main_caption, captions

    def classify_caption(self, caption: str) -> Tuple[np.ndarray, Dict]:
        """Classify caption using zero-shot classification."""
        if self.verbose:
            print(f"      🔤 Classifying caption...")

        result = self.classifier(
            caption,
            candidate_labels=self.behavior_labels,
            hypothesis_template="This image shows a person {}."
        )

        self.counters['classifications_made'] += 1

        if self.verbose:
            print(f"         Top 3 classifications:")
            for i, (label, score) in enumerate(zip(result['labels'][:3], result['scores'][:3])):
                print(f"            {i+1}. {label}: {score:.4f}")

        # Convert to behavior scores
        behavior_scores = []
        for behavior in BEHAVIOR_CLASSES:
            # Map our behavior classes to the labels used in classification
            label = BEHAVIOR_LABEL_MAPPING.get(behavior, behavior)

            # Find the score for this behavior
            if label in result['labels']:
                idx = result['labels'].index(label)
                score = result['scores'][idx]
            else:
                score = 0.0
            behavior_scores.append(score)

        return np.array(behavior_scores), result

    def detect_and_classify(
        self,
        image: Union[Image.Image, np.ndarray],
        threshold: float = DEFAULT_DETECTION_THRESHOLD,
        show_crops: bool = True,
        save_visualizations: Optional[bool] = None,
        image_name: str = ""
    ) -> List[Dict]:
        """
        Enhanced detection and classification with BLIP-2.

        Args:
            image: Input image (PIL or numpy array)
            threshold: Detection threshold
            show_crops: Whether to display visualizations
            save_visualizations: Whether to save visualizations (None = use show_crops value)
            image_name: Name of the image for labeling

        Returns:
            List of detection dictionaries containing behavior predictions
        """
        # Handle save_visualizations parameter
        if save_visualizations is None:
            save_visualizations = show_crops

        # Store show_crops setting for visualization method
        self._show_plots = show_crops

        # Update counters
        self.counters['images_processed'] += 1

        # Convert image format if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image

        image_width, image_height = pil_image.size

        if self.verbose:
            self._print_processing_header(image_name, image_width, image_height)

        # Stage 1: Enhanced Contextual Detection
        all_detections = self.detection_utils.detect_contextual_objects(
            pil_image, self.grounding_dino, self.gd_processor, self.device, threshold
        )

        if len(all_detections["boxes"]) == 0:
            if self.verbose:
                print(f"   ⚠️  No objects detected with threshold {threshold}")
            return []

        # Process detections
        person_boxes, person_scores = self.detection_utils.extract_person_detections(all_detections)
        
        if len(person_boxes) == 0:
            if self.verbose:
                print(f"   ⚠️  No persons detected")
            return []

        # Filter small detections
        person_boxes, person_scores = self.detection_utils.filter_small_detections(
            person_boxes, person_scores, image_width, image_height
        )

        if len(person_boxes) == 0:
            if self.verbose:
                print(f"   ⚠️  All person detections filtered out (too small)")
            return []

        self.counters['persons_detected'] += len(person_boxes)

        # Create contextual patches
        merged_patches = self.detection_utils.merge_contextual_patches(
            person_boxes, all_detections, image_width, image_height
        )

        detections = []

        # Process each person detection
        for idx, (original_box, patch_box, score) in enumerate(zip(person_boxes, merged_patches, person_scores)):
            if self.verbose:
                print(f"\n   👤 {'='*50}")
                print(f"   👤 PROCESSING PERSON {idx+1}/{len(person_boxes)} WITH BLIP-2")
                print(f"   👤 {'='*50}")

            # Use smart expansion on the merged patch
            smart_expanded_box = self.detection_utils.smart_expand_box(
                patch_box, image_width, image_height, EXPANSION_FACTOR
            )

            # Crop with smart expanded box
            x1, y1, x2, y2 = smart_expanded_box
            person_crop = pil_image.crop((x1, y1, x2, y2))

            # Generate enhanced captions using BLIP-2
            main_caption, caption_variations = self.generate_enhanced_caption(person_crop)

            # Classify the main caption
            behavior_scores, classification_result = self.classify_caption(main_caption)

            # Get best matching behavior
            best_idx = np.argmax(behavior_scores)
            best_behavior = BEHAVIOR_CLASSES[best_idx]
            best_score = behavior_scores[best_idx]
            final_confidence = best_score * score

            if self.verbose:
                print(f"      🎯 Final Results:")
                print(f"         Predicted behavior: {best_behavior}")
                print(f"         Classification score: {best_score:.3f}")
                print(f"         Detection score: {score:.3f}")
                print(f"         Final confidence: {final_confidence:.3f}")

            # Create comprehensive visualization for this person
            if save_visualizations:
                viz_path = self.viz_utils.create_comprehensive_visualization(
                    pil_image, original_box, smart_expanded_box, person_crop,
                    main_caption, caption_variations, best_behavior, behavior_scores, 
                    final_confidence, idx+1, image_name, self.dirs.get('comprehensive', self.output_dir),
                    self.blip2_model_name, self._show_plots
                )
                if viz_path:
                    self.counters['visualizations_saved'] += 1

            # Store detection results
            detections.append({
                'box': original_box,
                'patch_box': patch_box,
                'expanded_box': smart_expanded_box,
                'behavior': best_behavior,
                'confidence': final_confidence,
                'caption': main_caption,
                'caption_variations': caption_variations,
                'classification_scores': behavior_scores.tolist(),
                'classification_result': {
                    'labels': classification_result['labels'],
                    'scores': classification_result['scores']
                }
            })

        # Save debug information
        if self.save_outputs:
            self._save_debug_info(image_name, detections)

        # Print final summary
        if self.verbose:
            self._print_detection_summary(detections, image_name)

        return detections

    def _print_processing_header(self, image_name: str, width: int, height: int) -> None:
        """Print processing header information."""
        print(f"\n🎬 {'='*80}")
        print(f"🎬 PROCESSING IMAGE WITH BLIP-2: {image_name if image_name else 'UNNAMED'}")
        print(f"🎬 {'='*80}")
        print(f"📐 Image dimensions: {width} x {height} pixels")
        print(f"🤖 BLIP-2 Model: {self.blip2_model_name}")
        print(f"📊 Processing session stats:")
        for key, value in self.counters.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")

    def _save_debug_info(self, image_name: str, detections: List[Dict]) -> None:
        """Save detailed debug information."""
        if not self.save_outputs:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_data = {
                'image_name': image_name,
                'timestamp': timestamp,
                'blip2_model': self.blip2_model_name,
                'processing_counters': self.counters.copy(),
                'detections': []
            }

            for i, detection in enumerate(detections):
                debug_detection = {
                    'person_index': i + 1,
                    'bounding_box': detection['box'].tolist(),
                    'patch_box': detection['patch_box'].tolist(),
                    'expanded_box': detection['expanded_box'].tolist(),
                    'predicted_behavior': detection['behavior'],
                    'confidence': float(detection['confidence']),
                    'main_caption': detection['caption'],
                    'caption_variations': detection.get('caption_variations', {}),
                    'behavior_scores': detection['classification_scores'],
                    'full_classification': {
                        'labels': detection['classification_result']['labels'],
                        'scores': [float(s) for s in detection['classification_result']['scores']]
                    }
                }
                debug_data['detections'].append(debug_detection)

            # Save debug info
            safe_image_name = image_name.replace('.', '_').replace(' ', '_') if image_name else "unknown"
            filename = f"blip2_debug_{safe_image_name}_{timestamp}.json"
            filepath = os.path.join(self.dirs['debug'], filename)

            with open(filepath, 'w') as f:
                json.dump(debug_data, f, indent=2)

            if self.verbose:
                print(f"   🐛 Saved debug info: {filename}")

        except Exception as e:
            if self.verbose:
                print(f"   ❌ Error saving debug info: {e}")

    def _print_detection_summary(self, detections: List[Dict], image_name: str = "") -> None:
        """Print detailed summary of detections."""
        print(f"\n📊 {'='*60}")
        print(f"📊 BLIP-2 DETECTION SUMMARY {f'- {image_name}' if image_name else ''}")
        print(f"📊 {'='*60}")

        if not detections:
            print(f"   ⚠️  No detections found")
            return

        print(f"   👥 Total persons detected: {len(detections)}")
        print(f"   🤖 BLIP-2 Model: {self.blip2_model_name}")
        
        behavior_counts = {}
        for detection in detections:
            behavior = detection['behavior']
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1

        print(f"   🏷️  Behavior distribution:")
        for behavior, count in sorted(behavior_counts.items()):
            print(f"      {behavior}: {count} person(s)")

        confidences = [d['confidence'] for d in detections]
        print(f"   📈 Average confidence: {np.mean(confidences):.3f}")
        print(f"   🔝 Highest confidence: {max(confidences):.3f}")
        print(f"   📉 Lowest confidence: {min(confidences):.3f}")

        print(f"\n✅ {'='*80}")
        print(f"✅ BLIP-2 PROCESSING COMPLETE FOR IMAGE: {image_name if image_name else 'UNNAMED'}")
        print(f"✅ Total persons analyzed: {len(detections)}")
        print(f"✅ Comprehensive visualizations created: {len(detections)}")
        print(f"✅ BLIP-2 Model used: {self.blip2_model_name}")
        print(f"✅ {'='*80}\n")