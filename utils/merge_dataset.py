import os
import shutil
from pathlib import Path

# Define base and target folders
base_folder = r"D:\cropped_persons"
output_folder = os.path.join(base_folder, "dataset")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop over all frame folders
for frame_name in os.listdir(base_folder):
    frame_path = os.path.join(base_folder, frame_name)
    
    if os.path.isdir(frame_path) and frame_name.startswith("frame_"):
        frame_id = frame_name.split("_")[1]

        # Handle raw_crops
        raw_crops_path = os.path.join(frame_path, "raw_crops")
        if os.path.exists(raw_crops_path):
            for file in os.listdir(raw_crops_path):
                if file.endswith(".jpg"):
                    person_id = file.split("_")[1]  # Extract person ID
                    new_name = f"{frame_id}_{person_id}_r.jpg"
                    shutil.copy(
                        os.path.join(raw_crops_path, file),
                        os.path.join(output_folder, new_name)
                    )

        # Handle expanded_crops
        expanded_crops_path = os.path.join(frame_path, "expanded_crops")
        if os.path.exists(expanded_crops_path):
            for file in os.listdir(expanded_crops_path):
                if file.endswith(".jpg"):
                    person_id = file.split("_")[1]  # Extract person ID
                    new_name = f"{frame_id}_{person_id}_x.jpg"
                    shutil.copy(
                        os.path.join(expanded_crops_path, file),
                        os.path.join(output_folder, new_name)
                    )

        # Handle annotation file
        annotation_path = os.path.join(frame_path, "annotation.txt")
        if os.path.exists(annotation_path):
            new_annotation_name = f"{frame_id}_annotation.csv"
            shutil.copy(
                annotation_path,
                os.path.join(output_folder, new_annotation_name)
            )

print("✅ Merge complete. Files are in:", output_folder)
