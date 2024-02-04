#!/usr/bin/env python3
import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
from faf.tinyyolov2 import TinyYoloV2

from faf.visualization import (
    plot_time_against_step,
    plot_average_precision_against_step,
    plot_size_against_step,
    generate_samples,
    plot_average_precision_against_time,
)

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("results/test/results.csv", converters={"times": ast.literal_eval})

fig, ax = plot_time_against_step(df)
fig.savefig(os.path.join("figures", "time_against_step.png"))
fig, ax = plot_average_precision_against_step(df)
fig.savefig(os.path.join("figures", "average_precision_against_step.png"))
fig, ax = plot_size_against_step(df)
fig.savefig(os.path.join("figures", "size_against_step.png"))
fig, ax = plot_average_precision_against_time(df)
fig.savefig(os.path.join("figures", "average_precision_against_time.png"))

net = TinyYoloV2.from_saved_state_dict("weights/test/final.pt")

images = generate_samples(net, num=10, draw_gt=False)

fig, axs = plt.subplots(2, 5, figsize=(10, 5))

for i, image in enumerate(images):
    axs[i % 2, i // 2].imshow(image)
    axs[i % 2, i // 2].axis("off")

plt.tight_layout()
fig.savefig(os.path.join("figures", "detections.png"))

# plt.show()
