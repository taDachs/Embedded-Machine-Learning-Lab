#!/usr/bin/env python3

import os
import shutil

data_path = "../data/tiktok_dancing/segmentation_full_body_tik_tok_2615_img/segmentation_full_body_tik_tok_2615_img/"
out_path = "../data/tiktok_dancing/"

split = 0.7

src_images_dir = os.path.join(data_path, "images")
src_masks_dir = os.path.join(data_path, "masks")

images_dir = os.path.join(out_path, "images")
masks_dir = os.path.join(out_path, "masks")

os.makedirs(os.path.join(images_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(images_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(masks_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(masks_dir, "val"), exist_ok=True)

files = os.listdir(src_images_dir)
images = sorted([file for file in files if file.endswith(".png")])

num_test = int(len(images) * split)

train_images = images[:num_test]
val_images = images[num_test:]

for image in train_images:
    img_src = os.path.join(src_images_dir, image)
    mask_src = os.path.join(src_masks_dir, image)
    img_dst = os.path.join(images_dir, "train", image)
    mask_dst = os.path.join(masks_dir, "train", image)
    shutil.copyfile(img_src, img_dst)
    shutil.copyfile(mask_src, mask_dst)

for image in val_images:
    img_src = os.path.join(src_images_dir, image)
    mask_src = os.path.join(src_masks_dir, image)
    img_dst = os.path.join(images_dir, "val", image)
    mask_dst = os.path.join(masks_dir, "val", image)
    shutil.copyfile(img_src, img_dst)
    shutil.copyfile(mask_src, mask_dst)

