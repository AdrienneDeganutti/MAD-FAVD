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

for annotation in section:
    # Extract the annotation ID and the associated caption
    annotation_id = section[id_counter]['image_id']
    caption = section[id_counter]['caption']

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

# Create the final JSON structure
final_json = {"annotations": annotations_list, "images": images_list}

# Save the JSON file
with open(json_file, 'w') as f:
    json.dump(final_json, f, indent=4)
