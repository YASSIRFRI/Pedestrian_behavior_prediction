import os
import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
import re

def extract_image_id(filename):
    """Extract image ID from filename (removes extension)"""
    return os.path.splitext(filename)[0]

def parse_xml_annotation(xml_path):
    """Parse XML annotation file and extract relevant information"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract filename
    filename = root.find('filename').text
    image_id = extract_image_id(filename)
    
    # Extract image dimensions
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Count objects by class/behavior
    behavior_counts = defaultdict(int)
    objects = root.findall('object')
    
    for obj in objects:
        behavior_name = obj.find('name').text
        behavior_counts[behavior_name] += 1
    
    return {
        'image_id': image_id,
        'filename': filename,
        'width': width,
        'height': height,
        'behavior_counts': dict(behavior_counts)
    }

def create_annotation_csv(dataset_dir, output_csv='annotations.csv'):
    """
    Process all XML files in dataset directory and create annotation CSV
    
    Args:
        dataset_dir (str): Path to dataset directory containing XML and image files
        output_csv (str): Output CSV filename
    """
    
    # Find all XML files in the directory
    xml_files = [f for f in os.listdir(dataset_dir) if f.endswith('.xml')]
    
    if not xml_files:
        print("No XML files found in the specified directory!")
        return
    
    print(f"Found {len(xml_files)} XML files to process...")
    
    # Process all XML files
    data = []
    all_behaviors = set()
    
    for xml_file in xml_files:
        xml_path = os.path.join(dataset_dir, xml_file)
        try:
            annotation_data = parse_xml_annotation(xml_path)
            data.append(annotation_data)
            all_behaviors.update(annotation_data['behavior_counts'].keys())
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue
    
    if not data:
        print("No valid annotation data found!")
        return
    
    print(f"Found {len(all_behaviors)} unique behaviors: {sorted(all_behaviors)}")
    
    # Create DataFrame
    rows = []
    for item in data:
        row = {
            'image_id': item['image_id'],
            'filename': item['filename'],
            'width': item['width'],
            'height': item['height']
        }
        
        # Add behavior counts (0 if behavior not present in this image)
        for behavior in sorted(all_behaviors):
            row[behavior] = item['behavior_counts'].get(behavior, 0)
        
        # Add empty source column
        row['source'] = ''
        
        rows.append(row)
    
    # Create DataFrame and sort by image_id
    df = pd.DataFrame(rows)
    
    # Sort by image_id (handle both numeric and string IDs)
    try:
        # Try to sort numerically if possible
        df['sort_key'] = df['image_id'].apply(lambda x: int(re.findall(r'\d+', str(x))[0]) if re.findall(r'\d+', str(x)) else float('inf'))
        df = df.sort_values('sort_key').drop('sort_key', axis=1)
    except:
        # Fall back to string sorting
        df = df.sort_values('image_id')
    
    # Reorder columns: image_id, filename, behaviors (sorted), width, height, source
    behavior_columns = sorted(all_behaviors)
    column_order = ['image_id', 'filename'] + behavior_columns + ['width', 'height', 'source']
    df = df[column_order]
    
    # Save to CSV
    output_path = os.path.join(dataset_dir, output_csv)
    df.to_csv(output_path, index=False)
    
    print(f"\nAnnotation CSV created successfully!")
    print(f"Output file: {output_path}")
    print(f"Total images processed: {len(df)}")
    print(f"Columns created: {list(df.columns)}")
    
    # Display first few rows as preview
    print("\nPreview of first 5 rows:")
    print(df.head().to_string(index=False))
    
    return df

# Example usage
if __name__ == "__main__":
    # Replace with your dataset directory path
    dataset_directory = r"D:\cropped_persons\dataset\test_dataset"  # Change this to your actual path
    
    # Check if directory exists
    if not os.path.exists(dataset_directory):
        print(f"Directory '{dataset_directory}' not found!")
        print("Please update the 'dataset_directory' variable with your actual dataset path.")
    else:
        # Create the annotation CSV
        df = create_annotation_csv(dataset_directory, 'annotations.csv')