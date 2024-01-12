#!/usr/bin/env python3
# coding: utf-8

# # Embedded ML Lab - Challenge (testing yolo example)
#
# This is an example of inference with the VOC data set and tinyyolov2. There are pretrained weights (`voc_pretrained.pt`) stored that can be loaded into the model.


import torch

from faf.utils.dataloader import VOCDataLoader


import tqdm

from faf.tinyyolov2 import TinyYoloV2
from faf.utils.yolo import nms, filter_boxes
from faf.utils.viz import display_result


loader = VOCDataLoader(train=False, batch_size=1)


# make an instance with 20 classes as output
net = TinyYoloV2(num_classes=20)

# load pretrained weights
sd = torch.load("weights/voc_pretrained.pt")
net.load_state_dict(sd)

# put network in evaluation mode
net.eval()


for idx, (input, target) in tqdm.tqdm(enumerate(loader), total=len(loader)):

    # input is a 1 x 3 x 320 x 320 image
    output = net(input)
    "output is of a tensor of size 32 x 125 x 10 x 10"
    # output is a 32 x 125 x 10 x 10 tensor

    # filter boxes based on confidence score (class_score*confidence)
    output = filter_boxes(output, 0.1)

    # filter boxes based on overlap
    output = nms(output, 0.25)

    display_result(input, output, target, file_path="yolo_prediction.png")
