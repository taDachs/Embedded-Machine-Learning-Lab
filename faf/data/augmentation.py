import random

import torch
import albumentations as A


class Augmentation:
    def __init__(
        self,
        crop_p=0.5,
        rotate_p=0.3,
        h_flip_p=0.5,
        v_flip_p=0.5,
        contrast_p=0.2,
        noise_p=0.0,
        noise_intensity=0.5,
        jitter_p=0.0,
        seed=42,
    ):
        self.crop_p = crop_p
        self.rotate_p = rotate_p
        self.h_flip_p = h_flip_p
        self.v_flip_p = v_flip_p
        self.contrast_p = contrast_p
        self.noise_p = noise_p
        self.noise_intensity = noise_intensity
        self.jitter_p = jitter_p
        self.seed = seed
        random.seed(self.seed)

    def __call__(
        self, width: int, height: int, image: torch.Tensor, targets: torch.Tensor
    ):
        transform = A.Compose(
            [
                A.RandomSizedBBoxSafeCrop(height=height, width=width, erosion_rate=0.5, always_apply=False, p=self.crop_p),
                A.RandomRotate90(p=self.rotate_p),
                A.HorizontalFlip(p=self.h_flip_p),
                A.VerticalFlip(p=self.v_flip_p),
                A.RandomBrightnessContrast(p=self.contrast_p),
                A.ISONoise(intensity=(self.noise_intensity, self.noise_intensity)),
                A.ColorJitter(p=self.jitter_p),
            ],
            bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.1),
        )

        num_boxes = len(targets)
        # Keep only bounding boxes that are non-zero
        first_zero_row = 0
        for t in targets:
            if torch.all(t[:4] != 0):
                first_zero_row += 1
        targets = targets[:first_zero_row]

        transformed = transform(image=image, bboxes=targets)
        image = transformed["image"]

        bboxes = torch.Tensor()
        if transformed["bboxes"]:
            bboxes = torch.stack([torch.Tensor(i) for i in transformed["bboxes"]])

        padding = num_boxes - len(bboxes)
        pad_tensor = torch.zeros((padding, 6))
        pad_tensor[..., -1] = -1
        bboxes = torch.cat((bboxes, pad_tensor))
        return image, bboxes
