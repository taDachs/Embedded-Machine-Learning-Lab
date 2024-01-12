#!/usr/bin/env python3
# coding: utf-8

# # Embedded ML Lab - Challenge (Camera example)
#
# This is an example notebook for the camera usage


from faf.utils.camera import CameraDisplay
import time
import cv2

now = time.time()


# Define a callback function (your detection pipeline)
# Make sure to first load all your pipeline code and only at the end init the camera


def callback(image):
    global now

    fps = f"{int(1/(time.time() - now))}"
    now = time.time()
    image = image[0:320, 0:320, :]
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
cam = CameraDisplay(callback)


# The camera stream can be started with cam.start()
# The callback gets asynchronously called (can be stopped with cam.stop())
cam.start()


# The camera should always be stopped and released for a new camera is instantiated (calling CameraDisplay(callback) again)
cam.stop()
cam.release()
