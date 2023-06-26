import os
import os.path as osp
from random import shuffle

ratio = [0.9, 0.1]  # [train, val] totally equal 1

def main():
    txt_path = "/home/hyl/home/DL_Homework/SAR-Vehicle/DL_dataset/train.txt"
    imgs_labels = []

    # Read original txt
    file = open(txt_path, "r")
    for line in file:
        line = line.strip("\n")
        imgs_labels.append(line)
    file.close()

    shuffle(imgs_labels)

    train = imgs_labels[:int(len(imgs_labels) * ratio[0])]
    val = imgs_labels[int(len(imgs_labels) * ratio[0]):]

    # Rewrite train.txt
    file = open(txt_path, "w")
    for line in train:
        file.write(line + "\n")
    file.close()

    # Write val.txt
    val_path = txt_path.replace("train", "val")
    file = open(val_path, "w")
    for line in val:
        file.write(line + "\n")
    file.close()

    print("Finish!")

if __name__ == "__main__":
    main()