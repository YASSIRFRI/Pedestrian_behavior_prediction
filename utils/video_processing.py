"""Video processing utilities for the behavior classifier."""

import os
import cv2
import json
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Optional
import re

from ..constants import DEFAULT_TARGET_FPS, SUPPORTED_VIDEO_FORMATS
from .visualization import VisualizationUtils


class VideoProcessor:
    """Handles video processing and batch operations."""
    
    def __init__(self, verbose: bool = True):
        """Initialize video processor."""
        self.verbose = verbose
        self.viz_utils = VisualizationUtils(verbose=verbose)

    def extract_behavior_from_filename(self, filename: str) -> str:
        """Extract behavior ground truth from filename."""
        behaviors_map = {
            'walk': 'walking',
            'run': 'running',
            'bike': 'biking',
            'dog': 'dog walking',
            'scoot': 'scootering',
            'stand': 'standing',
            'sit': 'sitting',
            'stroll': 'stroller'
        }

        filename_lower = filename.lower()
        for key, behavior in behaviors_map.items():
            if key in filename_lower:
                return behavior
        return 'unknown'

    def process_video(
        self,
        video_path: str,
        classifier,
        output_folder: str,
        target_fps: int = DEFAULT_TARGET_FPS,
        save_video: bool = True
    ) -> Dict:
        """Process a single video and save results."""
        video_name = os.path.basename(video_path)
        model_name = classifier.model_name
        
        if self.verbose:
            print(f"\nProcessing {video_name} with {model_name}...")

        # Extract ground truth
        ground_truth = self.extract_behavior_from_filename(video_name)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, original_fps // target_fps)

        # Output video setup
        output_video_path = None
        out = None
        if save_video:
            output_name = f"{model_name}_{video_name}"
            output_video_path = os.path.join(output_folder, output_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (width, height))

        # Process frames
        frame_results = []
        behavior_counts = defaultdict(int)

        pbar = tqdm(total=total_frames, desc=f"{model_name} - {video_name}")

        frame_count = 0
        last_detections = []

        # Set classifier to non-verbose for video processing
        original_verbose = classifier.verbose
        classifier.verbose = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every frame_skip frames
                if frame_count % frame_skip == 0:
                    last_detections = classifier.detect_and_classify(
                        frame, 
                        show_crops=False,
                        save_visualizations=False
                    )
                    frame_results.append(last_detections)

                    for det in last_detections:
                        behavior_counts[det['behavior']] += 1

                # Draw detections on current frame
                if save_video and out is not None:
                    frame_with_boxes = frame.copy()

                    # Model-specific colors
                    color = (0, 255, 0) if "OWL" in model_name else (255, 0, 0)

                    for det in last_detections:
                        frame_with_boxes = self.viz_utils.draw_bbox(
                            frame_with_boxes,
                            det['box'],
                            det['behavior'],
                            det['confidence'],
                            color
                        )

                    # Add info overlay
                    cv2.putText(frame_with_boxes, f"Model: {model_name}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame_with_boxes, f"GT: {ground_truth}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame_with_boxes, f"FPS: {target_fps} (reduced from {original_fps})", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Write frame only if we processed it
                    if frame_count % frame_skip == 0:
                        out.write(frame_with_boxes)

                frame_count += 1
                pbar.update(1)

        finally:
            # Restore classifier verbose setting
            classifier.verbose = original_verbose
            pbar.close()
            cap.release()
            if out is not None:
                out.release()

        # Results summary
        results_summary = {
            'video_name': video_name,
            'ground_truth': ground_truth,
            'model': model_name,
            'total_frames': total_frames,
            'frames_processed': len(frame_results),
            'original_fps': original_fps,
            'output_fps': target_fps,
            'frame_skip': frame_skip,
            'behavior_counts': dict(behavior_counts),
            'most_common_behavior': max(behavior_counts, key=behavior_counts.get) if behavior_counts else None,
            'output_path': output_video_path,
            'detections': frame_results
        }

        if self.verbose:
            print(f"✓ Video processed: {len(frame_results)} frames analyzed at {target_fps} FPS")

        return results_summary

    def process_video_folder(
        self,
        video_folder: str,
        classifier,
        output_folder: str,
        target_fps: int = DEFAULT_TARGET_FPS,
        save_videos: bool = True
    ) -> List[Dict]:
        """Process all videos in a folder."""
        # Find video files
        video_files = []
        for ext in SUPPORTED_VIDEO_FORMATS:
            video_files.extend([f for f in os.listdir(video_folder) if f.lower().endswith(ext)])
        
        if not video_files:
            raise ValueError(f"No video files found in {video_folder}")
        
        if self.verbose:
            print(f"Found {len(video_files)} videos to process")

        all_results = []
        model_name = classifier.model_name

        if self.verbose:
            print(f"\n{'='*50}")
            print(f"Processing with {model_name} at {target_fps} FPS")
            print(f"{'='*50}")

        # Create model-specific output folder
        model_output_folder = os.path.join(output_folder, f'{model_name.lower()}_results_{target_fps}fps')
        os.makedirs(model_output_folder, exist_ok=True)

        # Process each video
        for video_file in video_files:
            video_path = os.path.join(video_folder, video_file)
            try:
                results = self.process_video(
                    video_path, 
                    classifier, 
                    model_output_folder,
                    target_fps=target_fps,
                    save_video=save_videos
                )
                all_results.append(results)
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error processing {video_file}: {e}")
                continue

        if self.verbose:
            print(f"\n{'='*50}")
            print(f"All videos processed successfully!")
            print(f"Output saved to: {model_output_folder}")
            print(f"{'='*50}")

        return all_results

    def save_results(self, results: List[Dict], output_path: str) -> None:
        """Save processing results to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            
            # Process detections to make them JSON serializable
            if 'detections' in serializable_result:
                serializable_detections = []
                for frame_detections in serializable_result['detections']:
                    frame_data = []
                    for det in frame_detections:
                        det_copy = det.copy()
                        # Convert numpy arrays to lists
                        if 'box' in det_copy:
                            det_copy['box'] = det_copy['box'].tolist()
                        if 'patch_box' in det_copy:
                            det_copy['patch_box'] = det_copy['patch_box'].tolist()
                        if 'expanded_box' in det_copy:
                            det_copy['expanded_box'] = det_copy['expanded_box'].tolist()
                        frame_data.append(det_copy)
                    serializable_detections.append(frame_data)
                serializable_result['detections'] = serializable_detections
            
            serializable_results.append(serializable_result)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        if self.verbose:
            print(f"Results saved to: {output_path}")

    def create_summary_report(self, results: List[Dict], output_folder: str) -> str:
        """Create a summary report of all processing results."""
        if not results:
            return "No results to summarize"

        # Calculate statistics
        total_videos = len(results)
        total_frames = sum(r['frames_processed'] for r in results)
        total_detections = sum(
            len(frame_det) 
            for r in results 
            for frame_det in r.get('detections', [])
        )

        # Behavior distribution
        behavior_counts = defaultdict(int)
        for result in results:
            for behavior, count in result.get('behavior_counts', {}).items():
                behavior_counts[behavior] += count

        # Ground truth vs prediction accuracy (simplified)
        correct_predictions = 0
        total_predictions = 0
        for result in results:
            gt = result.get('ground_truth', 'unknown')
            predicted = result.get('most_common_behavior', 'unknown')
            if gt != 'unknown' and predicted != 'unknown':
                total_predictions += 1
                if gt == predicted:
                    correct_predictions += 1

        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0

        # Create report
        report = f"""
BEHAVIOR CLASSIFIER PROCESSING REPORT
=====================================

SUMMARY STATISTICS:
- Videos processed: {total_videos}
- Total frames analyzed: {total_frames}
- Total detections: {total_detections}
- Average detections per frame: {total_detections/total_frames:.2f}

BEHAVIOR DISTRIBUTION:
"""
        for behavior, count in sorted(behavior_counts.items()):
            percentage = (count / sum(behavior_counts.values()) * 100) if behavior_counts else 0
            report += f"- {behavior}: {count} ({percentage:.1f}%)\n"

        report += f"""
ACCURACY:
- Ground truth accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})

DETAILED RESULTS:
"""
        for i, result in enumerate(results, 1):
            report += f"""
Video {i}: {result['video_name']}
- Ground truth: {result.get('ground_truth', 'unknown')}
- Most common behavior: {result.get('most_common_behavior', 'unknown')}
- Frames processed: {result['frames_processed']}
- Total detections: {sum(len(fd) for fd in result.get('detections', []))}
"""

        # Save report
        report_path = os.path.join(output_folder, 'processing_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)

        if self.verbose:
            print(f"Summary report saved to: {report_path}")
            print("\nSUMMARY:")
            print(f"- Processed {total_videos} videos")
            print(f"- Analyzed {total_frames} frames")
            print(f"- Found {total_detections} detections")
            print(f"- Accuracy: {accuracy:.1f}%")

        return report_path