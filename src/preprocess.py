# iterate through directory
import os
import allin1

# IF USING GOOGLE COLAB, UNCOMMENT THIS BLOCK
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)


directory_path = "PATH_TO_SUBSET_DATASET"
out_dir = "auto_preprocess"
for entry_name in os.listdir(directory_path):
    # assert that the entry name is not a csv
    if entry_name.split(".")[-1] == "mp3": # ensures that file suffix is an mp3
      full_path = os.path.join(directory_path, entry_name)
      allin1.analyze(full_path, out_dir=f'{out_dir}/struct', demix_dir=f'{out_dir}/separated', keep_byproducts=True, overwrite=False)