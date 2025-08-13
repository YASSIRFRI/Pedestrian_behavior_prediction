import os
import pandas as pd
from PIL import Image
import csv
import re

def build_dataset_csv(dataset_folder, output_csv="dataset_description.csv"):
    """
    Reads images and annotations to build a comprehensive CSV description.
    
    Args:
        dataset_folder (str): Path to the folder containing images and annotation CSVs
        output_csv (str): Output CSV filename
    """
    
    # List to store all the data
    dataset_data = []
    
    # Get all image files in the directory (with _r.jpg pattern)
    image_files = [f for f in os.listdir(dataset_folder) if f.endswith('_r.jpg')]
    
    # Dictionary to cache annotation data to avoid reading the same file multiple times
    annotation_cache = {}
    
    print(f"Processing {len(image_files)} image files...")
    
    for image_file in image_files:
        try:
            # Extract frame_id and person_id from filename
            # Pattern: {frame_id}_{person_id}_r.jpg
            filename_parts = image_file.replace('_r.jpg', '').split('_')
            
            if len(filename_parts) != 2:
                print(f"Skipping {image_file}: unexpected filename format")
                continue
                
            frame_id = filename_parts[0]
            person_id = int(filename_parts[1])
            
            # Load image to get dimensions
            image_path = os.path.join(dataset_folder, image_file)
            with Image.open(image_path) as img:
                width, height = img.size
                total_pixels = width * height
            
            # Load annotation data
            annotation_file = f"{frame_id}_annotation.csv"
            annotation_path = os.path.join(dataset_folder, annotation_file)
            
            behavior_class = "unknown"  # Default value
            
            if annotation_file not in annotation_cache:
                if os.path.exists(annotation_path):
                    try:
                        # Read the annotation file as plain text (comma-separated behaviors)
                        with open(annotation_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            behaviors = [behavior.strip() for behavior in content.split(',')]
                            annotation_cache[annotation_file] = behaviors
                    except Exception as e:
                        print(f"Error reading {annotation_file}: {e}")
                        annotation_cache[annotation_file] = None
                else:
                    print(f"Annotation file not found: {annotation_file}")
                    annotation_cache[annotation_file] = None
            
            # Get behavior class from annotation
            if annotation_cache[annotation_file] is not None:
                behaviors = annotation_cache[annotation_file]
                
                # Get behavior for this person (1-indexed person_id to 0-indexed list)
                if person_id <= len(behaviors) and person_id > 0:
                    behavior_class = behaviors[person_id - 1]
                else:
                    behavior_class = "out_of_range"
            
            # Add to dataset
            dataset_data.append({
                'frame_id': frame_id,
                'person_id': person_id,
                'behavior_class': behavior_class,
                'width': width,
                'height': height,
                'total_pixels': total_pixels,
                'image_filename': image_file
            })
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(dataset_data)
    
    # Sort by frame_id and person_id for better organization
    df = df.sort_values(['frame_id', 'person_id'])
    
    # Save to CSV
    output_path = os.path.join(dataset_folder, output_csv)
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset CSV created successfully: {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Unique frames: {df['frame_id'].nunique()}")
    print(f"Unique persons: {df['person_id'].nunique()}")
    print(f"Behavior classes found: {df['behavior_class'].unique()}")
    
    return df

def inspect_annotation_structure(dataset_folder, sample_annotation_file=None):
    """
    Helper function to inspect the structure of annotation CSV files.
    This helps understand how to properly extract behavior classes.
    """
    if sample_annotation_file is None:
        # Find the first annotation file
        annotation_files = [f for f in os.listdir(dataset_folder) if f.endswith('_annotation.csv')]
        if not annotation_files:
            print("No annotation files found!")
            return
        sample_annotation_file = annotation_files[0]
    
    annotation_path = os.path.join(dataset_folder, sample_annotation_file)
    
    if os.path.exists(annotation_path):
        print(f"\nInspecting annotation file: {sample_annotation_file}")
        try:
            # Read as plain text first
            with open(annotation_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                print(f"Raw content: '{content}'")
                
                behaviors = [behavior.strip() for behavior in content.split(',')]
                print(f"Parsed behaviors: {behaviors}")
                print(f"Number of behaviors: {len(behaviors)}")
                
                for i, behavior in enumerate(behaviors):
                    print(f"  Person {i+1}: {behavior}")
                    
        except Exception as e:
            print(f"Error reading annotation file: {e}")
    else:
        print(f"Annotation file not found: {annotation_path}")

if __name__ == "__main__":
    # Set your dataset folder path here
    dataset_folder = "D:\cropped_persons\dataset"  # Update this path
    
    # First, inspect annotation structure to understand the format
    print("=== Inspecting Annotation Structure ===")
    inspect_annotation_structure(dataset_folder)
    
    # Build the dataset CSV
    print("\n=== Building Dataset CSV ===")
    df = build_dataset_csv(dataset_folder)
    
    # Display summary statistics
    print("\n=== Summary Statistics ===")
    print(df.describe())
    
    # Show sample of the final dataset
    print("\n=== Sample Data ===")
    print(df.head(10))