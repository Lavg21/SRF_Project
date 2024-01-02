# import os
# import json
# import shutil
#
# def assign_categories_and_move(folder_path, json_path, output_folder):
#     # Citirea informațiilor din fișierul JSON
#     with open(json_path, 'r') as json_file:
#         data = json.load(json_file)
#
#     # Crearea unui dicționar pentru a asocia id-urile de imagini cu categoriile date
#     image_categories = {img['id']: data['categories'][ann['category_id']]['name']
#                         for img in data.get('images', [])
#                         for ann in data.get('annotations', []) if ann['image_id'] == img['id']}
#
#     # Crearea unor subfoldere corespunzătoare categoriilor date
#     for category in set(image_categories.values()):
#         category_folder = os.path.join(output_folder, category)
#         os.makedirs(category_folder, exist_ok=True)
#
#     # Mutarea imaginilor în subfolderele corespunzătoare
#     for image in data.get('images', []):
#         image_id = image['id']
#         filename = image['file_name']
#         filepath = os.path.join(folder_path, filename)
#
#         # Verificare dacă fișierul este imagine și este prezent în dicționarul de categorii
#         if os.path.isfile(filepath) and image_id in image_categories:
#             category = image_categories[image_id]
#             destination_folder = os.path.join(output_folder, category)
#             destination_path = os.path.join(destination_folder, filename)
#
#             # Mutarea fișierului în subfolderul corespunzător
#             shutil.move(filepath, destination_path)
#             print(f'{filename}: {category}')
#
# if __name__ == "__main__":
#     folder_path = r"D:\Facultate\UTCN\An IV\Semestrul 1\Sisteme de recunoastere a formelor\Proiect\script\train"
#     json_path = r"D:\Facultate\UTCN\An IV\Semestrul 1\Sisteme de recunoastere a formelor\Proiect\script\train\_annotations.coco.json"
#     output_folder = r"D:\Facultate\UTCN\An IV\Semestrul 1\Sisteme de recunoastere a formelor\Proiect\script\output_folder"
#     assign_categories_and_move(folder_path, json_path, output_folder)

# import os
# import json
# import shutil
#
# def assign_categories_and_move(folder_path, json_path, output_folder):
#     # Citirea informațiilor din fișierul JSON
#     with open(json_path, 'r') as json_file:
#         data = json.load(json_file)
#
#     # Crearea unui dicționar pentru a asocia id-urile de imagini cu categoriile date
#     image_categories = {img['id']: data['categories'][ann['category_id']]['name']
#                         for img in data.get('images', [])
#                         for ann in data.get('annotations', []) if ann['image_id'] == img['id']}
#
#     # Crearea unor subfoldere corespunzătoare categoriilor date
#     for category in set(image_categories.values()):
#         category_folder = os.path.join(output_folder, category)
#         os.makedirs(category_folder, exist_ok=True)
#
#     # Mutarea imaginilor în subfolderele corespunzătoare
#     for image in data.get('images', []):
#         image_id = image['id']
#         filename = image['file_name']
#         filepath = os.path.join(folder_path, filename)
#
#         # Verificare dacă fișierul este imagine și este prezent în dicționarul de categorii
#         if os.path.isfile(filepath) and image_id in image_categories:
#             category = image_categories[image_id]
#             destination_folder = os.path.join(output_folder, category)
#             destination_path = os.path.join(destination_folder, filename)
#
#             # Mutarea fișierului în subfolderul corespunzător
#             shutil.move(filepath, destination_path)
#             print(f'{filename}: {category}')
#
# if __name__ == "__main__":
#     folder_path = r"D:\Facultate\UTCN\An IV\Semestrul 1\Sisteme de recunoastere a formelor\Proiect\script\valid"
#     json_path = r"D:\Facultate\UTCN\An IV\Semestrul 1\Sisteme de recunoastere a formelor\Proiect\script\valid\_annotations.coco.json"
#     output_folder = r"D:\Facultate\UTCN\An IV\Semestrul 1\Sisteme de recunoastere a formelor\Proiect\script\output_folder_test"
#     assign_categories_and_move(folder_path, json_path, output_folder)

import os

def rename_files_in_folders(folder_path, extension='.jpg'):
    # Iteratem prin subfoldere
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)

        # Verificam sa fie director
        if os.path.isdir(subfolder_path):
            # Initializam counter-ul
            counter = 0

            # Iteratem prin fisiere
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)

                # Verifcam daca e fisier
                if os.path.isfile(file_path):
                    # Generam noul nume
                    new_filename = f"{counter:06d}{extension}"

                    # Construim path-ul
                    new_file_path = os.path.join(subfolder_path, new_filename)

                    # Redenumim
                    os.rename(file_path, new_file_path)
                    print(f'Renamed: {filename} -> {new_filename}')

                    # Incrementam counterul pt urmatorul fisier
                    counter += 1

if __name__ == "__main__":
    train_folder_path = r"D:\Facultate\UTCN\An IV\Semestrul 1\Sisteme de recunoastere a formelor\Proiect\script\output_folder_test"
    rename_files_in_folders(train_folder_path)
