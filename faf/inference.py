import os
import torch
import onnx
import onnxruntime
from faf.tinyyolov2 import TinyYoloV2
import numpy as np
import logging
import tensorrt as trt

try:
    import pycuda.driver as cuda
except:
    logging.warning("failed to load pycuda")


def to_onnx(net: TinyYoloV2, weights_path: str, cuda=False):
    net.eval()
    x = torch.randn(1, 3, 320, 320, requires_grad=True)

    out_path = os.path.join(weights_path, "inference.onnx")

    # Export the model
    torch.onnx.export(
        net,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        out_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },  # variable length axes
    )

    return InferenceModel(out_path, cuda)


class InferenceModel:
    def __init__(self, path: str, cuda=False):
        if cuda:
            providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device()})]
        else:
            providers = ["CPUExecutionProvider"]
        self.ort_session = onnxruntime.InferenceSession(path, providers=providers)

    def __call__(self, x):
        x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
        ort_inputs = {self.ort_session.get_inputs()[0].name: x}
        ort_outs = self.ort_session.run(None, ort_inputs)
        y = ort_outs[0]

        return torch.from_numpy(y)

    def to(self, _):
        pass

    def eval(self):
        pass

    def train(self):
        pass


class TorchModel:
    def __init__(self, path, device):
        self.net = TinyYoloV2.from_saved_state_dict(path)
        self.device = device
        self.net.to(device)

    def __call__(self, x):
        frame = np.array(x)
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)
        frame = frame.transpose(0, 3, 1, 2)
        frame = frame.astype(np.float32)

        frame = torch.tensor(frame)
        frame = frame.to(self.device)

        outputs = self.net(frame)[0]

        return outputs


class OnnxModel:
    def __init__(self, path, device):
        self.device = device
        if device == "cpu":
            self.ort_session = onnxruntime.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
        else:
            providers = onnxruntime.get_available_providers()
            assert "CUDAExecutionProvider" in providers, "CUDAExecution Provider is not available."

            session_options = onnxruntime.SessionOptions()
            cuda_provider_options = {
                "device_id": "0",
            }

            self.ort_session = onnxruntime.InferenceSession(
                path,
                sess_options=session_options,
                providers=["CUDAExecutionProvider"],
                provider_options=[cuda_provider_options],
            )
        self.io_binding = self.ort_session.io_binding()

    def __call__(self, x):
        frame = np.array(x)
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)
        frame = frame.transpose(0, 3, 1, 2)
        frame = frame.astype(np.float32)

        # Create a tensor from the frame for binding
        frame_tensor = onnxruntime.OrtValue.ortvalue_from_numpy(frame, self.device)
        output_tensor = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
            [1, 5, 10, 10, 6], np.float32, self.device
        )

        # Bind the input tensor to the input name
        self.io_binding.bind_ortvalue_input("input", frame_tensor)
        self.io_binding.bind_ortvalue_output("output", output_tensor)
        self.ort_session.run_with_iobinding(self.io_binding)

        # Retrieve the outputs
        ort_outs = self.io_binding.copy_outputs_to_cpu()[0]

        outputs = torch.tensor(ort_outs)[0]
        return outputs


class TensorRTModel:
    def __init__(self, path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(TRT_LOGGER)

        self.device = cuda.Device(0)
        self.cuda_context = self.device.make_context()

        with open(path, "rb") as f:
            engine_data = f.read()
            self.engine = self.trt_runtime.deserialize_cuda_engine(engine_data)

        self.context = self.engine.create_execution_context()
        input_shape = self.context.get_binding_shape(0)
        output_shape = self.context.get_binding_shape(1)
        dtype = trt.nptype(self.engine.get_binding_dtype(0))

        # Allocate host and device buffers
        self.d_input = cuda.mem_alloc(int(np.prod(input_shape) * 4))  # 4 bytes per float
        self.d_output = cuda.mem_alloc(int(np.prod(output_shape) * 4))  # 4 bytes per float
        self.h_output = np.empty(output_shape, dtype=dtype)
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

    def __call__(self, x):
        frame = np.array(x)
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)
        frame = frame.transpose(0, 3, 1, 2)
        frame = frame.astype(np.float32)

        input_data = np.ascontiguousarray(frame)

        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        outputs = torch.tensor(self.h_output)
        return outputs

    def __del__(self):
        del self.engine
        self.cuda_context.pop()
        del self.cuda_context


def get_model_by_file(path, device_name: str = "cpu"):
    ending = os.path.basename(path).split(".")[-1]
    assert device_name in ["cpu", "cuda"]

    if ending == "pt":
        logging.info(f"using torch model for {path}")
        return TorchModel(path, torch.device(device_name))
    elif ending == "onnx":
        logging.info(f"using onnx runtime for {path}")
        return OnnxModel(path, device_name)
    elif ending == "trt" or ending == "engine":
        logging.info(f"using tensorrt for {path}")
        raise NotImplementedError()
    else:
        raise ValueError(ending)


if __name__ == "__main__":
    net = TinyYoloV2.from_saved_state_dict("weights/test/final.pt")
    to_onnx(net, "weights/test")
    net = InferenceModel("weights/test/inference.onnx")

    x = torch.randn(1, 3, 320, 320, requires_grad=True)
    print(net(x).shape)
    print(type(net(x)))
