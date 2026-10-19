import cv2
import numpy as np
from PIL import Image
import os


def generate_boundary(mask_path, save_path):

    # 读取mask
    mask = np.array(Image.open(mask_path))

    # 二值化
    binary = (mask > 0).astype(np.uint8)

    # 3x3 kernel
    kernel = np.ones((3, 3), np.uint8)

    # dilation
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # erosion
    eroded = cv2.erode(binary, kernel, iterations=1)

    # morphological gradient
    boundary = dilated - eroded

    # 保存
    boundary_img = (boundary * 255).astype(np.uint8)

    Image.fromarray(boundary_img).save(save_path)

    print(f"[✔] Saved boundary mask: {save_path}")


if __name__ == "__main__":

    mask_path = "data/SUIM/annotations/training/d_r_103_.png"

    save_path = "boundary_test.png"

    generate_boundary(mask_path, save_path)
