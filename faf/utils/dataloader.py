import torch
import os
import json
import pandas as pd
import numpy as np
from PIL import Image

import torchvision
from torchvision import transforms as tf

CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def class_to_num(class_str):
    for idx, string in enumerate(CLASSES):
        if string == class_str:
            return idx


def num_to_class(number):
    for idx, string in enumerate(CLASSES):
        if idx == number:
            return string
    return "none"


class VOCTransform:
    def __init__(self, train=True, only_person=False):
        self.only_person = only_person
        self.train = train
        if train:
            self.augmentation = tf.RandomApply([tf.ColorJitter(0.2, 0.2, 0.2, 0.2)])

    def __call__(self, image, target):
        num_bboxes = 10
        width, height = 320, 320

        img_width, img_height = image.size

        scale = min(width / img_width, height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)

        diff_width, diff_height = width - new_width, height - new_height
        image = tf.functional.resize(image, size=(new_height, new_width))
        image = tf.functional.pad(
            image,
            padding=(
                diff_width // 2,
                diff_height // 2,
                diff_width // 2 + diff_width % 2,
                diff_height // 2 + diff_height % 2,
            ),
        )
        target = target["annotation"]["object"]

        target_vectors = []
        for item in target:
            x0 = int(item["bndbox"]["xmin"]) * scale + diff_width // 2
            w = (int(item["bndbox"]["xmax"]) - int(item["bndbox"]["xmin"])) * scale
            y0 = int(item["bndbox"]["ymin"]) * scale + diff_height // 2
            h = (int(item["bndbox"]["ymax"]) - int(item["bndbox"]["ymin"])) * scale

            target_vector = [
                (x0 + w / 2) / width,
                (y0 + h / 2) / height,
                w / width,
                h / height,
                1.0,
                class_to_num(item["name"]),
            ]

            if self.only_person:
                if target_vector[5] == class_to_num("person"):
                    target_vector[5] = 0.0
                    target_vectors.append(target_vector)
            else:
                target_vectors.append(target_vector)

        target_vectors = list(sorted(target_vectors, key=lambda x: x[2] * x[3]))
        target_vectors = torch.tensor(target_vectors)
        if target_vectors.shape[0] < num_bboxes:
            zeros = torch.zeros((num_bboxes - target_vectors.shape[0], 6))
            zeros[:, -1] = -1
            target_vectors = torch.cat([target_vectors, zeros], 0)
        elif target_vectors.shape[0] > num_bboxes:
            target_vectors = target_vectors[:num_bboxes]

        if self.train:
            return (
                self.augmentation(tf.functional.to_tensor(image)),
                target_vectors,
            )
        else:
            return tf.functional.to_tensor(image), target_vectors


def VOCDataLoader(train=True, batch_size=32, shuffle=False, path=None):
    if path is None:
        path = "data/"
    if train:
        image_set = "train"
    else:
        image_set = "val"

    if not os.path.exists(
        os.path.join(path, "VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg")
    ):
        dataset = torchvision.datasets.VOCDetection(
            path, year="2012", image_set=image_set, download=True
        )

    dataset = torchvision.datasets.VOCDetection(
        path,
        year="2012",
        image_set=image_set,
        download=False,
        transforms=VOCTransform(train=train),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def VOCDataLoaderPerson(train=True, batch_size=32, shuffle=False, path=None):
    if path is None:
        path = "data/"
    if train:
        image_set = "train"
    else:
        image_set = "val"

    if not os.path.exists(
        os.path.join(path, "VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg")
    ):
        dataset = torchvision.datasets.VOCDetection(
            path, year="2012", image_set=image_set, download=True
        )

    dataset = torchvision.datasets.VOCDetection(
        path,
        year="2012",
        image_set=image_set,
        download=False,
        transforms=VOCTransform(train=train, only_person=True),
    )
    with open(os.path.join(path, "person_indices.json"), "r") as fd:
        indices = list(json.load(fd)[image_set])
    dataset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def voc_only_person_dataset(train: bool, path: str) -> torch.utils.data.Dataset:
    if train:
        image_set = "train"
    else:
        image_set = "val"

    if not os.path.exists(
        os.path.join(path, "VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg")
    ):
        dataset = torchvision.datasets.VOCDetection(
            path, year="2012", image_set=image_set, download=True
        )

    dataset = torchvision.datasets.VOCDetection(
        path,
        year="2012",
        image_set=image_set,
        download=False,
        transforms=VOCTransform(train=train, only_person=True),
    )
    with open(os.path.join(path, "person_indices.json"), "r") as fd:
        indices = list(json.load(fd)[image_set])
    dataset = torch.utils.data.Subset(dataset, indices)
    return dataset


def tiktok_dancing_dataset(train: bool, path: str) -> torch.utils.data.Dataset:
    return TikTokDancingDataset(
        csv_file=os.path.join(path, "tiktok_dancing", "df.csv"),
        root_dir=os.path.join(
            path,
            "tiktok_dancing",
            "segmentation_full_body_tik_tok_2615_img",
            "segmentation_full_body_tik_tok_2615_img",
        ),
        transform=VOCTransform(train=train, only_person=True),
    )


class TikTokDancingDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.ds_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ds_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.ds_frame.iloc[idx, 1])
        image = Image.open(img_name).convert("RGB")
        mask_name = os.path.join(self.root_dir, self.ds_frame.iloc[idx, 2])
        mask = Image.open(mask_name)
        mask = torch.tensor(np.array(mask))
        rows, cols = torch.nonzero(mask[..., 0], as_tuple=True)

        # Find the corners of the bounding box
        x_min = torch.min(cols)
        y_min = torch.min(rows)
        x_max = torch.max(cols)
        y_max = torch.max(rows)

        targets = {
            "annotation": {
                "object": [
                    {
                        "bndbox": {
                            "xmin": x_min,
                            "xmax": x_max,
                            "ymin": y_min,
                            "ymax": y_max,
                        },
                        "name": "person",
                    }
                ]
            }
        }

        if self.transform:
            image, targets = self.transform(image, targets)

        return image, targets
