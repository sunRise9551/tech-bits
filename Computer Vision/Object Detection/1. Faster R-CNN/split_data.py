import os
import random

"""
.xml files under: "./Dataset/./Annotations"

1. Split according to val_rate
2. Generate "train.txt" "val.txt"

"""

# File path for Folder that holds al .xml files
files_path = "../Dataset/VOCdevkit/VOC2012/Annotations"

# Check if given Folder exists
if not os.path.exists(files_path):
    print("Folder does not exists")
    exit(1)


# Set split validation rate 0 to 1
val_rate = 0.5

files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
files_num = len(files_name)

val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))

train_files = []
val_files = []

for index, file_name in enumerate(files_name):
    if index in val_index:
        val_files.append(file_name)
    else:
        train_files.append(file_name)


try:
    train_f = open("train.txt", "x")
    eval_f = open("val.txt", "x")
    train_f.write("\n".join(train_files))
    eval_f.write("\n".join(val_files))
except FileExistsError as e:
    print(e)
    exit(1)



# for file in file_path:
#