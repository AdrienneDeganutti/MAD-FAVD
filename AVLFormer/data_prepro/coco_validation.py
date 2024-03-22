import json
import os

json_file = '/home/adrienne/MAD-FAVD/AVLFormer/datasets/metadata/train.caption_coco_format.json'
output_folder = '/home/adrienne/MAD-FAVD/AVLFormer/datasets/coco/train_coco.json'

with open(json_file, 'r') as original_file:
    original_data = json.load(original_file)

#annotations = original_data['annotations']

# Prepare the list to hold all annotation data
annotations_list = []
images_list = []

section = original_data['annotations']

# Initialize the counter for the 'id' field
id_counter = 0
section_counter = 0

for annotation in section:
    # Extract the annotation ID and the associated caption
    annotation_id = section[section_counter]['image_id']
    caption = section[section_counter]['caption']

    try:
        numeric_id = int(annotation_id)
    except ValueError:
        print("Invalid ID: not a number!")
        
    if numeric_id in range(353729, 353816):
        print(f'Skipping:', numeric_id)
    elif numeric_id in range(132522, 132530):
        print(f'Skipping:', numeric_id)
    elif numeric_id in range(163935, 163940):
        print(f'Skipping:', numeric_id)
    elif numeric_id in range(249770, 249772):
        print(f'Skipping:', numeric_id)
    elif numeric_id in range(304135, 304159):
        print(f'Skipping:', numeric_id)
    elif numeric_id in range(307977, 307979):
        print(f'Skipping:', numeric_id)
    elif numeric_id in range(333149, 333160):
        print(f'Skipping:', numeric_id)
    elif numeric_id in range(358371, 358377):
        print(f'Skipping:', numeric_id)
    
    else:
        # Append the data in the specified format
        annotations_list.append({
            "image_id": annotation_id,
            "caption": caption,
            "id": id_counter
        })

        images_list.append({
            "id": annotation_id,
            "file_name": annotation_id
        })

        # Increment the counter for the next entry
        id_counter += 1

        print(f'Finished annotation:', annotation_id)
    
    section_counter += 1

# Create the final JSON structure
final_json = {"annotations": annotations_list, "images": images_list}

# Save the JSON file
with open(output_folder, 'w') as f:
    json.dump(final_json, f, indent=4)
