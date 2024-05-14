from glob import glob
import os
import re

folder = "C:\\"

# Get all .bmp files in the specified directory and its subdirectories
file_list = glob(folder + "*.bmp")

# Sort files based on the numerical part extracted from the filename
file_list = sorted(file_list, key=lambda x: int(re.findall("(\d+)", x)[0]))
print(file_list)

# Variables for managing batches and folders
counter = 0  # Folder counter
count2 = 0  # Picture counter within each folder
new_folder = None  # Current folder where files are being moved

for file in file_list:
    # Create a new folder after every 25 files
    if count2 == 0:
        new_folder = f"{folder}{counter:03d}\\"
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        counter += 1

    # Move file to the new folder
    sample = file.split("\\")[-1]
    os.replace(file, new_folder + sample)

    # Increment the file count, reset if it reaches 25
    count2 += 1
    if count2 == 25:
        count2 = 0

