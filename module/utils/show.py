import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

def show(img, label):
    # assert isinstance(label, int), "lable is not int!"
    idx = random.randint(0, 10000)
    print("img_shape: ", img.shape)
    print("label" + str(idx) + ": ", label)
    cv2.putText(img, str(label), (0, 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=1)
    cv2.namedWindow("img" + str(idx) + "_" + str(img.shape[0]) + "x" + str(img.shape[1]), cv2.WINDOW_NORMAL)
    cv2.imshow("img" + str(idx) + "_" + str(img.shape[0]) + "x" + str(img.shape[1]), img)
    cv2.waitKey()