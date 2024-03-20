import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import ipywidgets
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg
from faf.utils.camera import Camera
import time
import torch
from faf.utils.yolo import nms, filter_boxes
from faf.visualization import draw_boxes
import cv2


class CameraTensorrtDisplay:
    def __init__(self, engine_path, lazy_camera_init: bool = False, crop_to_fill=False):
        self.lazy_camera_init = lazy_camera_init
        self.engine_path = engine_path
        if not self.lazy_camera_init:
            self.initialize_camera()
        else:
            self.camera = None
        self.image_widget = ipywidgets.Image(format="jpeg")
        self.image_widget.value = bgr8_to_jpeg(np.zeros((320, 320, 3), dtype=np.uint8))
        display(self.image_widget)

        self._processing_frame = False
        self.fps = None
        self.crop_to_fill = crop_to_fill

    def initialize_camera(self):
        print("Initializing camera...")
        self.camera = Camera(
            width=640,
            height=360,
            capture_width=1280,
            capture_height=720,
            capture_fps=30,
        )

    def release(self):
        if self.camera is not None:
            self.camera.running = False
            if self.camera.cap is not None:
                self.camera.cap.release()
            print("Camera released")
            return

    def run(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt_runtime = trt.Runtime(TRT_LOGGER)

        device = cuda.Device(0)
        self.cuda_context = device.make_context()
        print("initialized context")

        # Load your serialized TensorRT engine (model)
        with open(self.engine_path, "rb") as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        print("initialized model")

        # Allocate buffers for input and output
        context = engine.create_execution_context()
        input_shape = context.get_binding_shape(0)
        output_shape = context.get_binding_shape(1)
        dtype = trt.nptype(engine.get_binding_dtype(0))

        # Allocate host and device buffers
        d_input = cuda.mem_alloc(int(np.prod(input_shape) * 4))
        d_output = cuda.mem_alloc(int(np.prod(output_shape) * 4))
        h_output = np.empty(output_shape, dtype=dtype)
        bindings = [int(d_input), int(d_output)]
        stream = cuda.Stream()

        print("initialized buffers")

        if self.camera is None:
            self.initialize_camera()

        self.camera.running = True
        self._processing_frame = False
        now = time.time()

        while True:
            if not self._processing_frame:
                self._processing_frame = True
                re, image = self.camera.cap.read()
                if not re:
                    print("error capturing frame")
                    continue

                if self.crop_to_fill:
                    image = image[0:320, 0:320, :]
                else:
                    width = 640
                    height = 360
                    scale = min(320/width, 320/height)

                    image = cv2.resize(image, fx=scale, fy=scale, dsize=None)

                    pad_height = max(0, 320 - scale*height)
                    top_pad = int(pad_height // 2)
                    bottom_pad = int(pad_height - top_pad)

                    # Pad the image
                    image = cv2.copyMakeBorder(image, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

                frame = np.array(image)

                frame = (frame.T / 255.0).astype(np.float32)
                frame = np.expand_dims(frame, axis=0)
                input_data = np.ascontiguousarray(frame)

                # Transfer input data to the GPU.
                cuda.memcpy_htod_async(d_input, input_data, stream)
                # Execute the model.
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()

                fps = f"{int(1/(time.time() - now))}"
                now = time.time()

                outputs = torch.tensor(h_output)
                outputs = filter_boxes(outputs, 0.3)
                outputs = nms(outputs, 0.25)
                outputs = [np.array(output) for output in outputs]
                targets = np.stack(outputs, axis=0)[0]

                draw_boxes(image, targets)

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

                self.image_widget.value = bgr8_to_jpeg(image)
                self._processing_frame = False

    def stop(self):
        self.camera.running = False
        self.cuda_context.pop()
        del self.cuda.context
        self.release()

    def __del__(self):
        self.stop()
