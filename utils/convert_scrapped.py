import os
import csv
import random
from PIL import Image
import glob
from pathlib import Path

def generate_unique_id(existing_ids):
    """Generate a unique random 8-digit ID"""
    while True:
        new_id = random.randint(100000000, 999999999)
        if new_id not in existing_ids:
            existing_ids.add(new_id)
            return new_id

def convert_images_and_create_csvs():
    # Source folder path
    source_folder = r"D:\cropped_persons\dataset\scrapped"
    
    # Create output directories
    output_images_folder = os.path.join(source_folder, "converted_images")
    output_csv_folder = os.path.join(source_folder, "csv_files")
    
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_csv_folder, exist_ok=True)
    
    # Supported image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png','*.avif', '*.webp']
    
    # Get all image files
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_folder, extension), recursive=False))
        image_files.extend(glob.glob(os.path.join(source_folder, extension.upper()), recursive=False))
    
    if not image_files:
        print(f"No image files found in {source_folder}")
        return
    
    print(f"Found {len(image_files)} image files to process")
    
    # Keep track of used IDs
    used_ids = set()
    processed_count = 0
    
    for image_path in image_files:
        try:
            # Generate unique ID
            unique_id = generate_unique_id(used_ids)
            
            # Open and convert image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (for formats like PNG with transparency)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as JPG with unique ID
                output_image_path = os.path.join(output_images_folder, f"{unique_id}.jpg")
                img.save(output_image_path, "JPEG", quality=95, optimize=True)
            
            # Create CSV file with the same ID
            csv_file_path = os.path.join(output_csv_folder, f"{unique_id}.csv")
            
            # Get original file info
            original_filename = os.path.basename(image_path)
            file_size = os.path.getsize(image_path)
            
            # Create CSV with basic information
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['id', 'original_filename', 'original_path', 'file_size_bytes', 'converted_filename']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header
                writer.writeheader()
                
                # Write data
                writer.writerow({
                    'id': unique_id,
                    'original_filename': original_filename,
                    'original_path': image_path,
                    'file_size_bytes': file_size,
                    'converted_filename': f"{unique_id}.jpg"
                })
            
            processed_count += 1
            print(f"Processed: {original_filename} -> {unique_id}.jpg (CSV created)")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Converted images saved to: {output_images_folder}")
    print(f"CSV files saved to: {output_csv_folder}")

if __name__ == "__main__":
    random.seed()
    
    print("Starting image conversion and CSV generation...")
    convert_images_and_create_csvs()
    print("Done!")