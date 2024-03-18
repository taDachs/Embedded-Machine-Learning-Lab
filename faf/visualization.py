import pandas as pd
import cv2
import torch
import numpy as np
from typing import List
from .utils.viz import num_to_class
import seaborn as sns
import matplotlib.pyplot as plt

from faf.data.dataloader import (
    VOCDataLoaderPerson,
    FullDataLoaderPerson,
    HumanDatasetDataLoaderPerson,
)
from faf.utils.yolo import nms, filter_boxes, filter_boxes_separate


def draw_bbox_opencv(
    image,
    bbox,
    class_label,
    confidence,
    color=(255, 0, 0),
    thickness=2,
    font_scale=0.5,
    font_thickness=1,
):
    """
    Draw a bounding box with a class label and confidence on an image using OpenCV.

    Parameters:
    - image: Image as a numpy array of shape (H, W, C).
    - bbox: Bounding box coordinates as (x_min, y_min, x_max, y_max).
    - class_label: String of the class label to annotate.
    - confidence: Confidence score as a float.
    - color: Color for the bounding box and text (B, G, R).
    - thickness: Thickness of the bounding box lines.
    - font_scale: Scale of the font used for the text.
    - font_thickness: Thickness of the font.
    """
    x_min, y_min, x_max, y_max = bbox

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    label = f"{class_label}: {confidence:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )

    cv2.rectangle(
        image,
        (x_min, y_min - int(1.3 * text_height)),
        (x_min + text_width, y_min),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (x_min, y_min - int(0.3 * text_height)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        font_thickness,
    )


def visualize_result(
    image: torch.Tensor,
    output: List[torch.tensor] = None,
    target: torch.Tensor = None,
) -> torch.Tensor():
    pad = 0
    image = image.numpy()[0, :].transpose(1, 2, 0)
    image_size = image.shape[:2]
    image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
    image = (image * 255).astype(np.uint8)

    if output:
        bboxes = torch.stack(output, dim=0)
        for i in range(bboxes.shape[1]):

            if bboxes[0, i, -1] >= 0:
                x_min = (
                    int(bboxes[0, i, 0] * image_size[0] - bboxes[0, i, 2] * image_size[0] / 2) + pad
                )
                y_min = (
                    int(bboxes[0, i, 1] * image_size[1] - bboxes[0, i, 3] * image_size[1] / 2) + pad
                )
                x_max = pad + int(x_min + bboxes[0, i, 2] * image_size[0])
                y_max = pad + int(y_min + bboxes[0, i, 3] * image_size[1])

                draw_bbox_opencv(
                    image,
                    (x_min, y_min, x_max, y_max),
                    num_to_class(bboxes[0, i, 5]),
                    bboxes[0, i, 4],
                    color=(255, 0, 0),
                )

    if target is not None:
        for i in range(target.shape[1]):
            if target[0, i, -1] >= 0:
                x_min = (
                    int(target[0, i, 0] * image_size[0] - target[0, i, 2] * image_size[0] / 2) + pad
                )
                y_min = (
                    int(target[0, i, 1] * image_size[1] - target[0, i, 3] * image_size[1] / 2) + pad
                )
                x_max = pad + int(x_min + target[0, i, 2] * image_size[0])
                y_max = pad + int(y_min + target[0, i, 3] * image_size[1])

                draw_bbox_opencv(
                    image,
                    (x_min, y_min, x_max, y_max),
                    num_to_class(target[0, i, 5]),
                    target[0, i, 4],
                    color=(0, 255, 0),
                )

    return image


def plot_time_against_step(
    df: pd.DataFrame,
    ax=None,
    title=None,
):
    if ax is None:
        return_ax = True
        fig, ax = plt.subplots()
    else:
        return_ax = False

    if title is None:
        title = "Time vs Step"

    df = df.copy()

    df["mean_times"] = df["times"].apply(np.mean)
    df["variance_times"] = df["times"].apply(np.var)

    sns.lineplot(data=df, x=df.index, y="mean_times", label="Mean of Times", ax=ax)

    sns.scatterplot(data=df, x=df.index, y="mean_times", color="red", ax=ax)

    ax.fill_between(
        df.index,
        df["mean_times"] - df["variance_times"],
        df["mean_times"] + df["variance_times"],
        color="gray",
        alpha=0.2,
        label="Variance",
    )

    for index, row in df.iterrows():
        ax.text(
            x=index,
            y=row["mean_times"],
            s=row["step"],
            color="blue",
            fontweight="bold",
            fontsize=9,
            ha="right",
            va="bottom",
        )

    ax.set_title("Time vs Step")
    ax.set_xlabel("Index")
    ax.set_ylabel("Mean Time")

    ax.set_ylim(0, np.max(df["mean_times"]) * 1.5)

    ax.legend()
    plt.tight_layout()

    if return_ax:
        return fig, ax


def plot_average_precision_against_step(
    df: pd.DataFrame,
    ax=None,
    title=None,
):
    if ax is None:
        return_ax = True
        fig, ax = plt.subplots()
    else:
        return_ax = False

    if title is None:
        title = "Time vs Step"

    sns.lineplot(data=df, x=df.index, y="average_precision", label="Average Precision", ax=ax)

    sns.scatterplot(data=df, x=df.index, y="average_precision", color="red", ax=ax)

    for index, row in df.iterrows():
        ax.text(
            x=index,
            y=row["average_precision"],
            s=row["step"],
            color="blue",
            fontweight="bold",
            fontsize=9,
            ha="right",
            va="bottom",
        )

    ax.set_title("Average Precision vs Step")
    ax.set_xlabel("Index")
    ax.set_ylabel("Average Precision")

    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    if return_ax:
        return fig, ax


def plot_size_against_step(
    df: pd.DataFrame,
    ax=None,
    title=None,
):
    if ax is None:
        return_ax = True
        fig, ax = plt.subplots()
    else:
        return_ax = False

    if title is None:
        title = "Time vs Step"

    sns.lineplot(data=df, x=df.index, y="size", label="Size (MACS)", ax=ax)

    sns.scatterplot(data=df, x=df.index, y="size", color="red", ax=ax)

    for index, row in df.iterrows():
        ax.text(
            x=index,
            y=row["size"],
            s=row["step"],
            color="blue",
            fontweight="bold",
            fontsize=9,
            ha="right",
            va="bottom",
        )

    ax.set_title("Size (MACS) vs Step")
    ax.set_xlabel("Index")
    ax.set_ylabel("Size (MACS) Precision")

    ax.set_ylim(0, 2.5e9)

    plt.tight_layout()

    if return_ax:
        return fig, ax


def plot_average_precision_against_time(
    df: pd.DataFrame,
    ax=None,
    title=None,
):
    if ax is None:
        return_ax = True
        fig, ax = plt.subplots()
    else:
        return_ax = False

    if title is None:
        title = "Average Precision vs Step"

    df = df.copy()

    df["mean_times"] = df["times"].apply(np.mean)

    sns.lineplot(data=df, x="mean_times", y="average_precision", label="Average Precision", ax=ax)

    sns.scatterplot(data=df, x="mean_times", y="average_precision", color="red", ax=ax)

    for index, row in df.iterrows():
        ax.text(
            x=row["mean_times"],
            y=row["average_precision"],
            s=row["step"],
            color="blue",
            fontweight="bold",
            fontsize=9,
            ha="right",
            va="bottom",
        )

    ax.set_title("Average Precision vs Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Average Precision")

    ax.invert_xaxis()

    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    if return_ax:
        return fig, ax


def generate_samples(net, draw_gt=True, num=None):
    # loader = VOCDataLoaderPerson(augment=False, batch_size=1, shuffle=True)
    loader = FullDataLoaderPerson(augment=True, batch_size=1, shuffle=True)
    # loader = HumanDatasetDataLoaderPerson(augment=True, train=False, batch_size=1, shuffle=True)

    images = []

    for i, (inputs, targets) in enumerate(loader):
        if i >= num:
            break
        outputs = net(inputs)

        # filter boxes based on confidence score (class_score*confidence)
        # outputs = filter_boxes(outputs, 0.1)
        outputs = filter_boxes_separate(outputs, 0.5, 0.3)

        # filter boxes based on overlap
        outputs = nms(outputs, 0.25)

        if draw_gt:
            image = visualize_result(inputs, outputs, targets)
        else:
            image = visualize_result(inputs, outputs)

        images.append(image)

    return images


def vis_single_image(net, inputs, image):
    outputs = net(inputs[None, ...])
    outputs = filter_boxes_separate(outputs, 0.5, 0.3)
    outputs = nms(outputs, 0.25)
    image = visualize_result(image[None, ...], outputs)
    return image


def draw_boxes(image, targets):
    image_size = image.shape[:2]
    pad = 0

    for target in targets:
        if target[-1] >= 0:
            x_min = int(target[0] * image_size[0] - target[2] * image_size[0] / 2) + pad
            y_min = int(target[1] * image_size[1] - target[3] * image_size[1] / 2) + pad
            x_max = pad + int(x_min + target[2] * image_size[0])
            y_max = pad + int(y_min + target[3] * image_size[1])

            draw_bbox_opencv(
                image,
                (x_min, y_min, x_max, y_max),
                "person",
                target[4],
                color=(255, 0, 0),
            )
