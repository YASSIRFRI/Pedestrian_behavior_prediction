import os
import xml.etree.ElementTree as ET
from pathlib import Path

def has_objects(xml_file_path):
    """
    Check if XML annotation file contains any object tags
    Returns True if objects exist, False otherwise
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Look for <object> tags
        objects = root.findall('object')
        return len(objects) > 0
    
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file_path}: {e}")
        return True  # Keep file if we can't parse it (safer approach)
    except Exception as e:
        print(f"Error reading file {xml_file_path}: {e}")
        return True

def delete_empty_annotations(dataset_path, dry_run=True):
    """
    Delete XML files with no objects and their corresponding JPG images
    
    Args:
        dataset_path (str): Path to the dataset directory
        dry_run (bool): If True, only show what would be deleted without actually deleting
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Directory {dataset_path} does not exist!")
        return
    
    # Find all XML files
    xml_files = list(dataset_path.glob("*.xml"))
    
    if not xml_files:
        print("No XML files found in the directory!")
        return
    
    print(f"Found {len(xml_files)} XML files to check...")
    
    files_to_delete = []
    
    # Check each XML file
    for xml_file in xml_files:
        if not has_objects(xml_file):
            # Find corresponding image file
            image_name = xml_file.stem + ".jpg"
            image_path = dataset_path / image_name
            
            files_to_delete.append({
                'xml': xml_file,
                'image': image_path,
                'image_exists': image_path.exists()
            })
    
    if not files_to_delete:
        print("No XML files without objects found!")
        return
    
    print(f"\nFound {len(files_to_delete)} XML files with no objects:")
    print("-" * 50)
    
    for item in files_to_delete:
        print(f"XML: {item['xml'].name}")
        if item['image_exists']:
            print(f"  -> Corresponding image: {item['image'].name} (EXISTS)")
        else:
            print(f"  -> Corresponding image: {item['image'].name} (NOT FOUND)")
        print()
    
    if dry_run:
        print("DRY RUN MODE - No files were deleted.")
        print("Set dry_run=False to actually delete the files.")
        return
    
    response = input(f"Are you sure you want to delete {len(files_to_delete)} XML files and their images? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Deletion cancelled.")
        return
    
    # Delete files
    deleted_count = 0
    for item in files_to_delete:
        try:
            # Delete XML file
            item['xml'].unlink()
            print(f"Deleted: {item['xml'].name}")
            
            # Delete corresponding image if it exists
            if item['image_exists']:
                item['image'].unlink()
                print(f"Deleted: {item['image'].name}")
            
            deleted_count += 1
            
        except Exception as e:
            print(f"Error deleting files for {item['xml'].name}: {e}")
    
    print(f"\nSuccessfully processed {deleted_count} file pairs.")

# Main execution
if __name__ == "__main__":
    # Set your dataset path
    DATASET_PATH = r"D:\cropped_persons\dataset\test_dataset"
    
    print("=== Empty Annotation File Cleaner ===")
    print(f"Dataset path: {DATASET_PATH}")
    print()
    
    # First run in dry-run mode to see what would be deleted
    print("Running in DRY RUN mode first...")
    delete_empty_annotations(DATASET_PATH, dry_run=True)
    
    print("\n" + "="*50)
    
    # Ask if user wants to proceed with actual deletion
    proceed = input("Do you want to proceed with the actual deletion? (yes/no): ")
    
    if proceed.lower() == 'yes':
        delete_empty_annotations(DATASET_PATH, dry_run=False)
    else:
        print("Operation cancelled.")