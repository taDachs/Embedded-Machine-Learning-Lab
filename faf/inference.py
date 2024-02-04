import os
import torch
import onnx
import onnxruntime
from faf.tinyyolov2 import TinyYoloV2
import numpy as np


def to_onnx(net: TinyYoloV2, weights_path: str):
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

    return InferenceModel(out_path)


class InferenceModel:
    def __init__(self, path: str):
        self.ort_session = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )

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


if __name__ == "__main__":
    net = TinyYoloV2.from_saved_state_dict("weights/test/final.pt")
    to_onnx(net, "weights/test")
    net = InferenceModel("weights/test/inference.onnx")

    x = torch.randn(1, 3, 320, 320, requires_grad=True)
    print(net(x).shape)
    print(type(net(x)))
