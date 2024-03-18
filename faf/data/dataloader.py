import torch
import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import random

from .augmentation import Augmentation

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
    def __init__(self, augmentation: Augmentation, only_person=False):
        self.only_person = only_person
        if augmentation:
            self.augmentation = augmentation
        else:
            self.augmentation = None

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

        if self.augmentation is not None:
            # In person-only mode:
            # a[4] == 1 , a[5] == 0 if person
            # a[4] == 0 , a[5] == -1 otherwise

            image, target_vectors = self.augmentation(
                width, height, np.array(image), target_vectors
            )

        return tf.functional.to_tensor(image), target_vectors


def VOCDataLoader(augment=False, train=True, batch_size=32, shuffle=False, path=None):
    if path is None:
        path = "data/"
    if train:
        image_set = "train"
    else:
        image_set = "val"

    if not os.path.exists(os.path.join(path, "VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg")):
        dataset = torchvision.datasets.VOCDetection(
            path, year="2012", image_set=image_set, download=True
        )

    if augment:
        augmentation = Augmentation(crop_p=0)
    else:
        augmentation = None

    dataset = torchvision.datasets.VOCDetection(
        path,
        year="2012",
        image_set=image_set,
        download=False,
        transforms=VOCTransform(augmentation=augmentation),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def VOCDataLoaderPerson(augment=False, train=True, batch_size=32, shuffle=False, path=None):
    if path is None:
        path = "data/"

    if train:
        image_set = "train"
    else:
        image_set = "val"

    if not os.path.exists(os.path.join(path, "VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg")):
        dataset = torchvision.datasets.VOCDetection(
            path, year="2012", image_set=image_set, download=True
        )

    if augment:
        augmentation = Augmentation(crop_p=0)
    else:
        augmentation = None

    dataset = torchvision.datasets.VOCDetection(
        path,
        year="2012",
        image_set=image_set,
        download=False,
        transforms=VOCTransform(augmentation=augmentation, only_person=True),
    )

    with open(os.path.join(path, "person_indices.json"), "r") as fd:
        indices = list(json.load(fd)[image_set])

    dataset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class HumanDataset(torch.utils.data.Dataset):
    def __init__(self, subset_name, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        assert subset_name in ["train", "val"]

        self.image_dir = os.path.join(root_dir, "images", subset_name)
        self.label_dir = os.path.join(root_dir, "labels", subset_name)

        self.sample_list = sorted([name.split(".")[0] for name in os.listdir(self.image_dir)])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.sample_list[idx] + ".jpg")
        image = Image.open(img_name).convert("RGB")
        w, h = image.size
        label_name = os.path.join(self.label_dir, self.sample_list[idx] + ".txt")

        with open(label_name, "r") as f:
            lines = f.readlines()

        objects = []
        for line in lines:
            line = line.strip().split(" ")[1:]  # first one is label --> always human
            cx = w * float(line[0])
            cy = h * float(line[1])
            wb = w * float(line[2])
            hb = h * float(line[3])

            bbox = {
                "xmin": cx - wb / 2,
                "ymin": cy - hb / 2,
                "xmax": cx + wb / 2,
                "ymax": cy + hb / 2,
            }
            objects.append({"bndbox": bbox, "name": "person"})

        targets = {"annotation": {"object": objects}}

        if self.transform:
            image, targets = self.transform(image, targets)

        return image, targets


def HumanDatasetDataLoaderPerson(
    augment=False, train=True, batch_size=32, shuffle=False, path=None
):
    subset_name = "train" if train else "val"
    if path is None:
        path = "data/"

    if augment:
        augmentation = Augmentation(crop_p=0)
    else:
        augmentation = None

    dataset = HumanDataset(
        subset_name=subset_name,
        root_dir=os.path.join(
            path,
            "human_dataset",
        ),
        transform=VOCTransform(augmentation=augmentation, only_person=True),
    )

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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


def TikTokDataLoaderPerson(augment=False, train=True, batch_size=32, shuffle=False, path=None):
    if path is None:
        path = "data/"

    if augment:
        augmentation = Augmentation(crop_p=0)
    else:
        augmentation = None

    dataset = TikTokDancingDataset(
        csv_file=os.path.join(path, "tiktok_dancing", "df.csv"),
        root_dir=os.path.join(
            path,
            "tiktok_dancing",
            "segmentation_full_body_tik_tok_2615_img",
            "segmentation_full_body_tik_tok_2615_img",
        ),
        transform=VOCTransform(augmentation=augmentation, only_person=True),
    )

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def FullDataLoaderPerson(augment=False, train=True, batch_size=32, shuffle=False, path=None):
    if path is None:
        path = "data/"

    if train:
        image_set = "train"
    else:
        image_set = "val"

    if augment:
        augmentation = Augmentation(crop_p=0)
    else:
        augmentation = None

    tiktok_dataset = TikTokDancingDataset(
        csv_file=os.path.join(path, "tiktok_dancing", "df.csv"),
        root_dir=os.path.join(
            path,
            "tiktok_dancing",
            "segmentation_full_body_tik_tok_2615_img",
            "segmentation_full_body_tik_tok_2615_img",
        ),
        transform=VOCTransform(augmentation=augmentation, only_person=True),
    )

    if not os.path.exists(os.path.join(path, "VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg")):
        voc_dataset = torchvision.datasets.VOCDetection(
            path, year="2012", image_set=image_set, download=True
        )
    voc_dataset = torchvision.datasets.VOCDetection(
        path,
        year="2012",
        image_set=image_set,
        download=False,
        transforms=VOCTransform(augmentation=augmentation, only_person=True),
    )

    with open(os.path.join(path, "person_indices.json"), "r") as fd:
        indices = list(json.load(fd)[image_set])

    voc_dataset = torch.utils.data.Subset(voc_dataset, indices)

    human_dataset = HumanDataset(
        subset_name=image_set,
        root_dir=os.path.join(
            path,
            "human_dataset",
        ),
        transform=VOCTransform(augmentation=augmentation, only_person=True),
    )

    full_dataset = torch.utils.data.ConcatDataset([tiktok_dataset, voc_dataset, human_dataset])

    # shuffle once to break up the datasets
    indices = list(range(len(full_dataset)))
    random.seed(420)
    random.shuffle(indices)
    full_dataset = torch.utils.data.Subset(full_dataset, indices)

    return torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle)


def get_dataset_by_name(
    name,
    augment=False,
    train=True,
    path=None,
):
    if path is None:
        path = "data/"

    if train:
        image_set = "train"
    else:
        image_set = "val"

    if augment:
        augmentation = Augmentation(crop_p=0, v_flip_p=0)
    else:
        augmentation = None

    name = name.lower()
    assert name in ["full", "human", "voc", "tiktok"]

    tiktok_dataset = TikTokDancingDataset(
        csv_file=os.path.join(path, "tiktok_dancing", "df.csv"),
        root_dir=os.path.join(
            path,
            "tiktok_dancing",
            "segmentation_full_body_tik_tok_2615_img",
            "segmentation_full_body_tik_tok_2615_img",
        ),
        transform=VOCTransform(augmentation=augmentation, only_person=True),
    )

    if not os.path.exists(os.path.join(path, "VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg")):
        voc_dataset = torchvision.datasets.VOCDetection(
            path, year="2012", image_set=image_set, download=True
        )
    voc_dataset = torchvision.datasets.VOCDetection(
        path,
        year="2012",
        image_set=image_set,
        download=False,
        transforms=VOCTransform(augmentation=augmentation, only_person=True),
    )

    with open(os.path.join(path, "person_indices.json"), "r") as fd:
        indices = list(json.load(fd)[image_set])

    voc_dataset = torch.utils.data.Subset(voc_dataset, indices)

    human_dataset = HumanDataset(
        subset_name=image_set,
        root_dir=os.path.join(
            path,
            "human_dataset",
        ),
        transform=VOCTransform(augmentation=augmentation, only_person=True),
    )

    full_dataset = torch.utils.data.ConcatDataset([tiktok_dataset, voc_dataset, human_dataset])
    indices = list(range(len(full_dataset)))
    random.seed(420)
    random.shuffle(indices)
    full_dataset = torch.utils.data.Subset(full_dataset, indices)

    if name == "tiktok":
        return tiktok_dataset
    elif name == "voc":
        return voc_dataset
    elif name == "human":
        return human_dataset
    elif name == "full":
        return full_dataset
    else:
        raise ValueError(name)
