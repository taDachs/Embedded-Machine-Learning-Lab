#!/usr/bin/env python3
# coding: utf-8

# # Embedded ML Lab - Challenge (Camera example)
#
# This is an example notebook for the camera usage


from faf.utils.camera import CameraServer
import time
import cv2
from faf.tinyyolov2 import TinyYoloV2
import torch
from faf.inference import to_onnx

from torchvision.transforms import ToTensor

from faf.visualization import vis_single_image

device = torch.device("cuda")
net = TinyYoloV2.from_saved_state_dict("weights/for_larger_data/final.pt")
net.to(device)
net.eval()

now = time.time()

net.to(torch.device("cpu"))
net = to_onnx(net, "weights/for_larger_data")


# Define a callback function (your detection pipeline)
# Make sure to first load all your pipeline code and only at the end init the camera


def callback(image):
    global now

    fps = f"{int(1/(time.time() - now))}"
    now = time.time()
    image = image[0:320, 0:320, :]
    frame = ToTensor()(image)
    frame_gpu = frame.to(device)
    image = vis_single_image(net, frame_gpu, frame)
    cv2.putText(
        image,
        "fps=" + fps,
        (2, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (100, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return image


# Initialize the camera with the callback
cam = CameraServer(callback, webcam=True)


# The camera stream can be started with cam.start()
# The callback gets asynchronously called (can be stopped with cam.stop())
cam.start()

while True:
    try:
        pass
    except:
        cam.stop()
        cam.release()
        break
