"""Command line interface for the behavior classifier."""

import os
import click
import torch
from pathlib import Path

from .models.blip2_classifier import BLIP2BehaviorClassifier
from .utils.video_processing import VideoProcessor
from .constants import (
    DEFAULT_BLIP2_MODEL, AVAILABLE_BLIP2_MODELS, 
    DEFAULT_DETECTION_THRESHOLD, DEFAULT_TARGET_FPS,
    SUPPORTED_VIDEO_FORMATS, SUPPORTED_IMAGE_FORMATS
)


@click.group()
@click.version_option()
def main():
    """BLIP-2 Behavior Classifier - AI-powered behavior detection in videos and images."""
    pass


@main.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='./outputs', help='Output directory for results')
@click.option('--model', '-m', default=DEFAULT_BLIP2_MODEL, 
              type=click.Choice(AVAILABLE_BLIP2_MODELS), 
              help='BLIP-2 model to use')
@click.option('--threshold', '-t', default=DEFAULT_DETECTION_THRESHOLD, 
              help='Detection threshold (0.0-1.0)')
@click.option('--device', '-d', default='auto', 
              type=click.Choice(['auto', 'cuda', 'cpu']),
              help='Device to use for inference')
@click.option('--verbose/--quiet', '-v/-q', default=True, 
              help='Enable/disable verbose output')
@click.option('--save-visualizations/--no-visualizations', default=True,
              help='Save comprehensive visualizations')
def classify_image(input_path, output, model, threshold, device, verbose, save_visualizations):
    """Classify behavior in a single image."""
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check if input is a supported image format
    input_path = Path(input_path)
    if input_path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
        click.echo(f"Error: Unsupported image format. Supported: {SUPPORTED_IMAGE_FORMATS}")
        return
    
    try:
        # Initialize classifier
        if verbose:
            click.echo(f"Initializing BLIP-2 classifier with model: {model}")
        
        classifier = BLIP2BehaviorClassifier(
            device=device,
            verbose=verbose,
            save_outputs=save_visualizations,
            output_dir=output,
            blip2_model=model
        )
        
        # Load and process image
        from PIL import Image
        image = Image.open(input_path)
        
        if verbose:
            click.echo(f"Processing image: {input_path.name}")
        
        # Classify image
        detections = classifier.detect_and_classify(
            image=image,
            threshold=threshold,
            show_crops=False,
            save_visualizations=save_visualizations,
            image_name=input_path.name
        )
        
        # Print results
        if detections:
            click.echo(f"\n✅ Found {len(detections)} person(s):")
            for i, detection in enumerate(detections, 1):
                click.echo(f"  Person {i}: {detection['behavior']} (confidence: {detection['confidence']:.3f})")
        else:
            click.echo("❌ No persons detected in the image")
            
        if save_visualizations:
            click.echo(f"\n📁 Results saved to: {output}")
        
    except Exception as e:
        click.echo(f"❌ Error processing image: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


@main.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='./outputs', help='Output directory for results')
@click.option('--model', '-m', default=DEFAULT_BLIP2_MODEL,
              type=click.Choice(AVAILABLE_BLIP2_MODELS),
              help='BLIP-2 model to use')
@click.option('--threshold', '-t', default=DEFAULT_DETECTION_THRESHOLD,
              help='Detection threshold (0.0-1.0)')
@click.option('--fps', '-f', default=DEFAULT_TARGET_FPS,
              help='Target FPS for processing (to reduce computational load)')
@click.option('--device', '-d', default='auto',
              type=click.Choice(['auto', 'cuda', 'cpu']),
              help='Device to use for inference')
@click.option('--verbose/--quiet', '-v/-q', default=True,
              help='Enable/disable verbose output')
@click.option('--save-video/--no-video', default=True,
              help='Save output video with annotations')
def classify_video(video_path, output, model, threshold, fps, device, verbose, save_video):
    """Classify behavior in a single video."""
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check if input is a supported video format
    video_path = Path(video_path)
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        click.echo(f"Error: Unsupported video format. Supported: {SUPPORTED_VIDEO_FORMATS}")
        return
    
    try:
        # Initialize classifier
        if verbose:
            click.echo(f"Initializing BLIP-2 classifier with model: {model}")
        
        classifier = BLIP2BehaviorClassifier(
            device=device,
            verbose=verbose,
            save_outputs=True,
            output_dir=output,
            blip2_model=model
        )
        
        # Initialize video processor
        processor = VideoProcessor(verbose=verbose)
        
        if verbose:
            click.echo(f"Processing video: {video_path.name}")
        
        # Process video
        results = processor.process_video(
            video_path=str(video_path),
            classifier=classifier,
            output_folder=output,
            target_fps=fps,
            save_video=save_video
        )
        
        # Print results
        click.echo(f"\n✅ Video processing complete!")
        click.echo(f"  Frames processed: {results['frames_processed']}")
        click.echo(f"  Most common behavior: {results['most_common_behavior']}")
        click.echo(f"  Behavior counts: {results['behavior_counts']}")
        
        if save_video:
            click.echo(f"  Output video: {results['output_path']}")
        
        # Save results
        results_file = os.path.join(output, f"{video_path.stem}_results.json")
        processor.save_results([results], results_file)
        click.echo(f"  Results saved to: {results_file}")
        
    except Exception as e:
        click.echo(f"❌ Error processing video: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


@main.command()
@click.argument('video_folder', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--output', '-o', default='./outputs', help='Output directory for results')
@click.option('--model', '-m', default=DEFAULT_BLIP2_MODEL,
              type=click.Choice(AVAILABLE_BLIP2_MODELS),
              help='BLIP-2 model to use')
@click.option('--threshold', '-t', default=DEFAULT_DETECTION_THRESHOLD,
              help='Detection threshold (0.0-1.0)')
@click.option('--fps', '-f', default=DEFAULT_TARGET_FPS,
              help='Target FPS for processing (to reduce computational load)')
@click.option('--device', '-d', default='auto',
              type=click.Choice(['auto', 'cuda', 'cpu']),
              help='Device to use for inference')
@click.option('--verbose/--quiet', '-v/-q', default=True,
              help='Enable/disable verbose output')
@click.option('--save-videos/--no-videos', default=True,
              help='Save output videos with annotations')
@click.option('--create-report/--no-report', default=True,
              help='Create summary report')
def batch_process(video_folder, output, model, threshold, fps, device, verbose, save_videos, create_report):
    """Process all videos in a folder."""
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Initialize classifier
        if verbose:
            click.echo(f"Initializing BLIP-2 classifier with model: {model}")
        
        classifier = BLIP2BehaviorClassifier(
            device=device,
            verbose=verbose,
            save_outputs=True,
            output_dir=output,
            blip2_model=model
        )
        
        # Initialize video processor
        processor = VideoProcessor(verbose=verbose)
        
        if verbose:
            click.echo(f"Processing videos in folder: {video_folder}")
        
        # Process all videos
        all_results = processor.process_video_folder(
            video_folder=video_folder,
            classifier=classifier,
            output_folder=output,
            target_fps=fps,
            save_videos=save_videos
        )
        
        if not all_results:
            click.echo("❌ No videos were successfully processed")
            return
        
        # Save results
        results_file = os.path.join(output, 'batch_results.json')
        processor.save_results(all_results, results_file)
        
        # Create summary report
        if create_report:
            report_path = processor.create_summary_report(all_results, output)
            click.echo(f"\n📊 Summary report: {report_path}")
        
        # Print final summary
        click.echo(f"\n✅ Batch processing complete!")
        click.echo(f"  Videos processed: {len(all_results)}")
        click.echo(f"  Results saved to: {results_file}")
        
        # Calculate overall statistics
        total_detections = sum(
            len(frame_det) 
            for result in all_results 
            for frame_det in result.get('detections', [])
        )
        click.echo(f"  Total detections: {total_detections}")
        
    except Exception as e:
        click.echo(f"❌ Error in batch processing: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


@main.command()
@click.option('--model', '-m', default=DEFAULT_BLIP2_MODEL,
              type=click.Choice(AVAILABLE_BLIP2_MODELS),
              help='BLIP-2 model to test')
@click.option('--device', '-d', default='auto',
              type=click.Choice(['auto', 'cuda', 'cpu']),
              help='Device to use for testing')
def test_setup(model, device):
    """Test the setup by initializing the classifier."""
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    click.echo("🧪 Testing behavior classifier setup...")
    click.echo(f"  Device: {device}")
    click.echo(f"  Model: {model}")
    
    try:
        # Initialize classifier
        classifier = BLIP2BehaviorClassifier(
            device=device,
            verbose=False,
            save_outputs=False,
            blip2_model=model
        )
        
        click.echo("✅ Setup test successful!")
        click.echo("  All models loaded correctly")
        click.echo("  Ready to process images and videos")
        
        # Clean up
        del classifier
        if device == 'cuda':
            torch.cuda.empty_cache()
        
    except Exception as e:
        click.echo(f"❌ Setup test failed: {e}")
        click.echo("  Please check your installation and try again")


if __name__ == '__main__':
    main()