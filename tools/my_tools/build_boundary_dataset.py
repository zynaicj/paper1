import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def generate_boundary(mask_path):

    mask = np.array(Image.open(mask_path))

    # 二值化
    binary = (mask > 0).astype(np.uint8)

    # kernel
    kernel = np.ones((3, 3), np.uint8)

    # morphology gradient
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)

    boundary = dilated - eroded

    boundary = (boundary * 255).astype(np.uint8)

    return boundary


def process_split(split, split_file):

    mask_dir = "data/SUIM/Segmentation"
    save_dir = "data/SUIM/boundary"

    os.makedirs(save_dir, exist_ok=True)

    with open(f"data/SUIM/splits/{split_file}") as f:
        mask_ids = [line.strip() for line in f]

    for mask_id in tqdm(mask_ids):
        mask_path = os.path.join(mask_dir, f"{mask_id}.png")
        boundary = generate_boundary(mask_path)
        save_path = os.path.join(save_dir, f"{mask_id}.png")
        Image.fromarray(boundary).save(save_path)

    print(f"[✔] Finished: {split}")


if __name__ == "__main__":

    process_split("training", "train.txt")

    process_split("validation", "val.txt")

    print("\n[✔] Boundary dataset build complete!")
