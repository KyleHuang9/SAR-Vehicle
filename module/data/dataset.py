import os
import os.path as osp
import random
import time
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from .data_augment import (
    augment_hsv,
    letterbox,
    random_affine,
)
from module.utils.event import LOGGER
from module.utils.show import show

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mp4", "mov", "avi", "mkv"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])
VID_FORMATS.extend([f.upper() for f in VID_FORMATS])

class TrainValDataset(Dataset):
    def __init__(
        self,
        dir,
        img_size=(32, 32),
        batch_size=32,
        augment=False,
        hyp=None,
        rank=-1,
        task="train",
    ):
        assert task.lower() in ("train", "val", "test"), f"Not supported task: {task}"
        t1 = time.time()
        self.__dict__.update(locals())
        self.main_process = self.rank in (-1, 0)
        self.task = task
        self.img_paths, self.labels = self.get_imgs_labels(self.dir)
        t2 = time.time()
        if self.main_process:
            LOGGER.info(f"%.1fs for dataset initialization." % (t2 - t1))

    def __len__(self):
        """Get the length of dataset"""
        return len(self.img_paths)

    def __getitem__(self, index):
        if self.augment and random.random() < self.hyp["augment"]:
            img, label = self.get_augument(index)

        else:
            if self.hyp and "test_load_size" in self.hyp:
                img, (h0, w0), (h, w) = self.load_image(index, self.hyp["test_load_size"])
            else:
                img, (h0, w0), (h, w) = self.load_image(index)
            
            label = self.labels[index]

        # show(img, label)
        
        h, w, c = img.shape
        label = torch.tensor(label)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img[None, 0]
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), label

    def load_image(self, index, force_load_size=None):
        path = self.img_paths[index]
        try:
            im = cv2.imread(path)
            assert im is not None, f"opencv cannot read image correctly or {path} not exists"
        except:
            im = cv2.cvtColor(np.asarray(Image.open(path)), cv2.COLOR_RGB2BGR)
            assert im is not None, f"Image Not Found {path}, workdir: {os.getcwd()}"

        h0, w0 = im.shape[:2]  # origin shape
        return im, (h0, w0), im.shape[:2]

    def get_imgs_labels(self, dir):
        assert osp.exists(dir), f"{dir} is an invalid directory path!"
        txt = osp.join(dir, self.task + ".txt")
        assert osp.exists(txt), f"{txt} is an invalid path!"
        file = open(txt, "r")
        imgs_path = []
        labels = []

        for line in file:
            line = line.strip("\n")
            line = line.split(" ")
            imgs_path.append(line[0])
            labels.append(int(line[1]))
        return imgs_path, labels

    def get_augument(self, index):
        img, _, (h, w) = self.load_image(index)
        label_per_img = self.labels[index]

        shape = (self.img_size)  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)

        img = random_affine(img,
                            degrees=self.hyp['degrees'],
                            translate=self.hyp['translate'],
                            scale=self.hyp['scale'],
                            shear=self.hyp['shear'],
                            new_shape=shape)

        img, label = self.general_augment(img, label_per_img)
        return img, label

    def general_augment(self, img, labels):
        """Gets images and labels after general augment
        This function applies hsv, random ud-flip and random lr-flips augments.
        """

        # HSV color-space
        augment_hsv(
            img,
            hgain=self.hyp["hsv_h"],
            sgain=self.hyp["hsv_s"],
            vgain=self.hyp["hsv_v"],
        )

        # Flip up-down
        if random.random() < self.hyp["flipud"]:
            img = np.flipud(img)

        # Flip left-right
        if random.random() < self.hyp["fliplr"]:
            img = np.fliplr(img)

        return img, labels