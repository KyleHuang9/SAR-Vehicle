import os
import os.path as osp
from random import shuffle

ratio = [0.8, 0.2]  # [train, val] totally equal 1

def get_paths():
    txt = "/home/hyl/home/DL_Homework/SAR-Vehicle/DL_dataset/test.txt"
    file = open(txt, "r")
    imgs_path = []

    for line in file:
        line = line.strip("\n")
        line = line.split(" ")
        imgs_path.append(line[0])
    return imgs_path

def transdata():
    txt_path = "/home/hyl/home/DL_Homework/SAR-Vehicle/DL_dataset/train_.txt"
    out_train_path = "/home/hyl/home/DL_Homework/SAR-Vehicle/DL_dataset/train.txt"
    out_val_path = "/home/hyl/home/DL_Homework/SAR-Vehicle/DL_dataset/val.txt"
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
    file = open(out_train_path, "w")
    for line in train:
        file.write(line + "\n")
    file.close()

    # Write val.txt
    file = open(out_val_path, "w")
    for line in val:
        file.write(line + "\n")
    file.close()

    print("Finish!")

def main():
    transdata()

if __name__ == "__main__":
    main()