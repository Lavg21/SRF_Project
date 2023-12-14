import os
import json
import shutil

train_folder = r'D:\Facultate\UTCN\An IV\Semestrul 1\Sisteme de recunoastere a formelor\Proiect\script\train'

# Load the JSON data
json_file_path = os.path.join(train_folder, '_annotations.coco.json')
with open(json_file_path) as json_file:
    data = json.load(json_file)

# Get the list of categories and their IDs
categories = {category['id']: category['name'] for category in data['categories']}

# Create folders for each category
output_folder = 'output_folder'
for category_id, category_name in categories.items():
    folder_path = os.path.join(output_folder, category_name)
    os.makedirs(folder_path, exist_ok=True)

# Move images to their corresponding folders
for image_info in data['images']:
    image_id = image_info['id']

    # Find the category for the image filename, default to None if not found
    category_id = next((category['id'] for category in data['categories'] if category['name'].lower() in image_info['file_name'].lower()), None)

    # If a matching category is found, proceed with moving the file
    if category_id is not None:
        category_name = categories[category_id]
        source_path = os.path.join(train_folder, image_info['file_name'])
        destination_path = os.path.join(output_folder, category_name, image_info['file_name'])
        shutil.copy(source_path, destination_path)

print("Images organized into folders based on categories.")
