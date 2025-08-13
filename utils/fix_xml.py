import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse
from pathlib import Path

def fix_xml_file(xml_path, default_label="object"):
    """
    Fix a single XML file by ensuring all <name> tags have non-empty values
    """
    try:
        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        fixed_objects = 0
        total_objects = 0
        
        # Find all object elements
        for obj in root.findall('object'):
            total_objects += 1
            name_elem = obj.find('name')
            
            if name_elem is not None:
                # Check if name is empty, None, or whitespace only
                if not name_elem.text or not name_elem.text.strip():
                    name_elem.text = default_label
                    fixed_objects += 1
            else:
                # Create name element if it doesn't exist
                name_elem = ET.SubElement(obj, 'name')
                name_elem.text = default_label
                fixed_objects += 1
        
        if fixed_objects > 0:
            # Save the fixed XML with proper formatting
            rough_string = ET.tostring(root, 'unicode')
            reparsed = minidom.parseString(rough_string)
            
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(reparsed.toprettyxml(indent="  "))
            
            print(f"✅ Fixed {xml_path}: {fixed_objects}/{total_objects} objects updated")
            return True, fixed_objects, total_objects
        else:
            print(f"✓ {xml_path}: No fixes needed ({total_objects} objects)")
            return False, 0, total_objects
            
    except Exception as e:
        print(f"❌ Error processing {xml_path}: {e}")
        return False, 0, 0

def curate_xml_directory(xml_dir_path, default_label="object", backup=True):
    """
    Process all XML files in a directory
    """
    xml_dir = Path(xml_dir_path)
    
    if not xml_dir.exists():
        print(f"❌ Directory not found: {xml_dir}")
        return
    
    if not xml_dir.is_dir():
        print(f"❌ Path is not a directory: {xml_dir}")
        return
    
    # Find all XML files
    xml_files = list(xml_dir.glob("*.xml"))
    
    if not xml_files:
        print(f"❌ No XML files found in: {xml_dir}")
        return
    
    print(f"🔍 Found {len(xml_files)} XML files in: {xml_dir}")
    print(f"🏷️ Default label for empty names: '{default_label}'")
    
    if backup:
        backup_dir = xml_dir / "xml_backup"
        backup_dir.mkdir(exist_ok=True)
        print(f"📁 Backup directory: {backup_dir}")
    
    total_files_fixed = 0
    total_objects_fixed = 0
    total_objects_processed = 0
    
    for xml_file in xml_files:
        try:
            # Create backup if requested
            if backup:
                backup_path = backup_dir / xml_file.name
                with open(xml_file, 'r', encoding='utf-8') as src:
                    with open(backup_path, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
            
            # Fix the XML file
            was_fixed, objects_fixed, total_objects = fix_xml_file(xml_file, default_label)
            
            if was_fixed:
                total_files_fixed += 1
            
            total_objects_fixed += objects_fixed
            total_objects_processed += total_objects
            
        except Exception as e:
            print(f"❌ Error processing {xml_file}: {e}")
    
    print(f"\n🎉 CURATION COMPLETE!")
    print(f"📊 Summary:")
    print(f"   📁 Total XML files: {len(xml_files)}")
    print(f"   ✅ Files fixed: {total_files_fixed}")
    print(f"   👥 Total objects processed: {total_objects_processed}")
    print(f"   🔧 Objects fixed: {total_objects_fixed}")
    print(f"   🏷️ Default label used: '{default_label}'")
    
    if backup:
        print(f"   💾 Backups saved in: {backup_dir}")
    
    print(f"\n✅ XML files are now ready for LabelImg!")
    print(f"💡 You can now open the folder in LabelImg without errors")

def main():
    """
    Main function with command line interface
    """
    parser = argparse.ArgumentParser(
        description="Fix XML annotation files for LabelImg compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python xml_curator.py annotations/
  python xml_curator.py annotations/ --label "unlabeled"
  python xml_curator.py annotations/ --no-backup
        """
    )
    
    parser.add_argument(
        'xml_directory',
        help='Path to directory containing XML annotation files'
    )
    
    parser.add_argument(
        '--label', '-l',
        default='object',
        help='Default label for empty name tags (default: "object")'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup files'
    )
    
    args = parser.parse_args()
    
    print("🔧 XML ANNOTATION CURATOR FOR LABELIMG")
    print("=" * 50)
    
    curate_xml_directory(
        xml_dir_path=args.xml_directory,
        default_label=args.label,
        backup=not args.no_backup
    )

if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        print("🔧 XML ANNOTATION CURATOR FOR LABELIMG")
        print("=" * 50)
        print("This script fixes XML files so they work with LabelImg")
        print()
        
        xml_dir = input("📁 Enter path to XML files directory: ").strip()
        if not xml_dir:
            print("❌ No directory specified. Exiting.")
            xml_dir=r"D:\cropped_persons\dataset\test_dataset"
            #exit(1)
        
        default_label = input("🏷️ Enter default label for empty names (default: 'object'): ").strip()
        if not default_label:
            default_label = "object"
        
        backup_choice = input("💾 Create backup files? (y/n, default: y): ").strip().lower()
        create_backup = backup_choice != 'n'
        
        print()
        curate_xml_directory(xml_dir, default_label, create_backup)
    else:
        main()