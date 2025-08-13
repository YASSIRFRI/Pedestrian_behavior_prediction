"""Main entry point for the behavior classifier package."""

import os
import argparse
from pathlib import Path
from typing import Union, List, Dict

from models.blip2_classifier import BLIP2BehaviorClassifier
from utils.video_processing import VideoProcessor
from constants import (
    DEFAULT_BLIP2_MODEL, AVAILABLE_BLIP2_MODELS,
    DEFAULT_DETECTION_THRESHOLD, DEFAULT_TARGET_FPS,
    SUPPORTED_VIDEO_FORMATS, SUPPORTED_IMAGE_FORMATS
)


def classify_image(
    image_path: Union[str, Path],
    output_dir: str = "./outputs",
    model_name: str = DEFAULT_BLIP2_MODEL,
    threshold: float = DEFAULT_DETECTION_THRESHOLD,
    device: str = "auto",
    verbose: bool = True,
    save_visualizations: bool = True
) -> List[Dict]:
    """
    Classify behavior in a single image.
    
    Args:
        image_path: Path to the image file
        output_dir: Directory to save outputs
        model_name: BLIP-2 model to use
        threshold: Detection threshold
        device: Device to use ('auto', 'cuda', or 'cpu')
        verbose: Enable verbose output
        save_visualizations: Save visualization outputs
        
    Returns:
        List of detection dictionaries
    """
    import torch
    from PIL import Image
    
    # Auto-detect device if needed
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize classifier
    classifier = BLIP2BehaviorClassifier(
        device=device,
        verbose=verbose,
        save_outputs=save_visualizations,
        output_dir=output_dir,
        blip2_model=model_name
    )
    
    # Load and process image
    image = Image.open(image_path)
    image_name = Path(image_path).name
    
    # Classify
    detections = classifier.detect_and_classify(
        image=image,
        threshold=threshold,
        show_crops=False,
        save_visualizations=save_visualizations,
        image_name=image_name
    )
    
    return detections


def classify_video(
    video_path: Union[str, Path],
    output_dir: str = "./outputs",
    model_name: str = DEFAULT_BLIP2_MODEL,
    threshold: float = DEFAULT_DETECTION_THRESHOLD,
    fps: int = DEFAULT_TARGET_FPS,
    device: str = "auto",
    verbose: bool = True,
    save_video: bool = True
) -> Dict:
    """
    Classify behavior in a single video.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save outputs
        model_name: BLIP-2 model to use
        threshold: Detection threshold
        fps: Target FPS for processing
        device: Device to use ('auto', 'cuda', or 'cpu')
        verbose: Enable verbose output
        save_video: Save annotated video
        
    Returns:
        Results dictionary
    """
    import torch
    
    # Auto-detect device if needed
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize classifier
    classifier = BLIP2BehaviorClassifier(
        device=device,
        verbose=verbose,
        save_outputs=True,
        output_dir=output_dir,
        blip2_model=model_name
    )
    
    # Initialize video processor
    processor = VideoProcessor(verbose=verbose)
    
    # Process video
    results = processor.process_video(
        video_path=str(video_path),
        classifier=classifier,
        output_folder=output_dir,
        target_fps=fps,
        save_video=save_video
    )
    
    return results


def batch_process_videos(
    video_folder: Union[str, Path],
    output_dir: str = "./outputs",
    model_name: str = DEFAULT_BLIP2_MODEL,
    threshold: float = DEFAULT_DETECTION_THRESHOLD,
    fps: int = DEFAULT_TARGET_FPS,
    device: str = "auto",
    verbose: bool = True,
    save_videos: bool = True,
    create_report: bool = True
) -> List[Dict]:
    """
    Process all videos in a folder.
    
    Args:
        video_folder: Path to folder containing videos
        output_dir: Directory to save outputs
        model_name: BLIP-2 model to use
        threshold: Detection threshold
        fps: Target FPS for processing
        device: Device to use ('auto', 'cuda', or 'cpu')
        verbose: Enable verbose output
        save_videos: Save annotated videos
        create_report: Create summary report
        
    Returns:
        List of results dictionaries
    """
    import torch
    
    # Auto-detect device if needed
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize classifier
    classifier = BLIP2BehaviorClassifier(
        device=device,
        verbose=verbose,
        save_outputs=True,
        output_dir=output_dir,
        blip2_model=model_name
    )
    
    # Initialize video processor
    processor = VideoProcessor(verbose=verbose)
    
    # Process all videos
    all_results = processor.process_video_folder(
        video_folder=str(video_folder),
        classifier=classifier,
        output_folder=output_dir,
        target_fps=fps,
        save_videos=save_videos
    )
    
    # Save results
    results_file = os.path.join(output_dir, 'batch_results.json')
    processor.save_results(all_results, results_file)
    
    # Create summary report
    if create_report:
        processor.create_summary_report(all_results, output_dir)
    
    return all_results


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="BLIP-2 Behavior Classifier")
    parser.add_argument("input", help="Input image, video, or folder path")
    parser.add_argument("--output", "-o", default="./outputs", help="Output directory")
    parser.add_argument("--model", "-m", default=DEFAULT_BLIP2_MODEL,
                       choices=AVAILABLE_BLIP2_MODELS, help="BLIP-2 model to use")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_DETECTION_THRESHOLD,
                       help="Detection threshold")
    parser.add_argument("--fps", "-f", type=int, default=DEFAULT_TARGET_FPS,
                       help="Target FPS for video processing")
    parser.add_argument("--device", "-d", default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--no-visualizations", action="store_true", 
                       help="Disable visualization outputs")
    parser.add_argument("--no-videos", action="store_true",
                       help="Don't save annotated videos")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    try:
        if input_path.is_file():
            # Single file processing
            if input_path.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                print(f"Processing image: {input_path}")
                detections = classify_image(
                    image_path=input_path,
                    output_dir=args.output,
                    model_name=args.model,
                    threshold=args.threshold,
                    device=args.device,
                    verbose=args.verbose,
                    save_visualizations=not args.no_visualizations
                )
                print(f"Found {len(detections)} person(s)")
                
            elif input_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
                print(f"Processing video: {input_path}")
                results = classify_video(
                    video_path=input_path,
                    output_dir=args.output,
                    model_name=args.model,
                    threshold=args.threshold,
                    fps=args.fps,
                    device=args.device,
                    verbose=args.verbose,
                    save_video=not args.no_videos
                )
                print(f"Processed {results['frames_processed']} frames")
                
            else:
                print(f"Unsupported file format: {input_path.suffix}")
                return
                
        elif input_path.is_dir():
            # Batch processing
            print(f"Batch processing folder: {input_path}")
            results = batch_process_videos(
                video_folder=input_path,
                output_dir=args.output,
                model_name=args.model,
                threshold=args.threshold,
                fps=args.fps,
                device=args.device,
                verbose=args.verbose,
                save_videos=not args.no_videos,
                create_report=True
            )
            print(f"Processed {len(results)} videos")
            
        else:
            print(f"Input path does not exist: {input_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()