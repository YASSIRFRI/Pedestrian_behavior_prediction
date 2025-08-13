import cv2
import os
import glob
from pathlib import Path

def video_to_frames(input_dir, output_dir):
    """
    Convert videos to frames and save them in organized folders.
    
    Args:
        input_dir (str): Directory containing video files
        output_dir (str): Directory where frame folders will be created
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Common video file extensions
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm', '*.m4v']
    
    # Get all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
        video_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video file(s)")
    
    for video_path in video_files:
        # Get video filename without extension
        video_name = Path(video_path).stem
        
        # Create output folder for this video
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        print(f"\nProcessing: {video_name}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Total frames: {total_frames}, FPS: {fps:.2f}")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Create frame filename (zero-padded for proper sorting)
            frame_filename = f"frame_{frame_count:06d}.jpg"
            frame_path = os.path.join(video_output_dir, frame_filename)
            
            # Save frame
            cv2.imwrite(frame_path, frame)
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        print(f"Completed: {frame_count} frames extracted to {video_output_dir}")

def main():
    # Set your paths here
    input_directory = r"D:\sidewalk_test_data\sequences"
    output_directory = r"D:\sidewalk_test_data\images"
    
    # Check if input directory exists
    if not os.path.exists(input_directory):
        print(f"Error: Input directory does not exist: {input_directory}")
        return
    
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    
    # Convert videos to frames
    video_to_frames(input_directory, output_directory)
    
    print("\nAll videos processed successfully!")

if __name__ == "__main__":
    main()