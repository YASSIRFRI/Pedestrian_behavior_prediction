import os
import csv
from PIL import Image
import glob

def generate_image_dataset_csv(folder_path, output_csv_path):
    """
    Generate a CSV file with information about all images in the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing images
        output_csv_path (str): Path where the output CSV file will be saved
    """
    
    # Common image file extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.gif', '*.webp']
    
    # Find all image files in the directory
    image_files = []
    for extension in image_extensions:
        pattern = os.path.join(folder_path, extension)
        image_files.extend(glob.glob(pattern, recursive=False))
        # Also check uppercase extensions
        pattern_upper = os.path.join(folder_path, extension.upper())
        image_files.extend(glob.glob(pattern_upper, recursive=False))
    
    # Remove duplicates and sort
    image_files = list(set(image_files))
    
    print(f"Found {len(image_files)} image files in {folder_path}")
    
    # Prepare data for CSV
    csv_data = []
    
    for image_path in image_files:
        try:
            # Extract filename without extension for frame_id
            filename = os.path.basename(image_path)
            frame_id = os.path.splitext(filename)[0]
            
            # Get image dimensions
            with Image.open(image_path) as img:
                width, height = img.size
                total_pixels = width * height
            
            # Create row data
            row_data = {
                'frame_id': frame_id,
                'old_name': filename,
                'url': image_path,
                'height': height,
                'width': width,
                'total_pixels': total_pixels
            }
            
            csv_data.append(row_data)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    # Sort by frame_id lexicographically (ascending)
    csv_data.sort(key=lambda x: x['frame_id'])
    
    # Write to CSV file
    fieldnames = ['frame_id', 'old_name', 'url', 'height', 'width', 'total_pixels']
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"CSV file created successfully: {output_csv_path}")
    print(f"Total rows written: {len(csv_data)}")
    
    return csv_data

def main():
    # Configuration
    folder_path = r"D:\cropped_persons\dataset\test_dataset"
    output_csv_path = "dataset_description_complete.csv"
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist!")
        return
    
    # Generate the CSV
    try:
        data = generate_image_dataset_csv(folder_path, output_csv_path)
        
        # Display some statistics
        if data:
            print("\n--- Statistics ---")
            print(f"First frame_id: {data[0]['frame_id']}")
            print(f"Last frame_id: {data[-1]['frame_id']}")
            
            # Image dimension statistics
            heights = [row['height'] for row in data]
            widths = [row['width'] for row in data]
            
            print(f"Height range: {min(heights)} - {max(heights)} pixels")
            print(f"Width range: {min(widths)} - {max(widths)} pixels")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()