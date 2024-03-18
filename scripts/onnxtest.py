import torch
import onnx
import onnxruntime
from faf.tinyyolov2 import TinyYoloV2
import numpy as np

net = TinyYoloV2.from_saved_state_dict("./weights/test/final.pt")


net.eval()
x = torch.randn(1, 3, 320, 320, requires_grad=True)
torch_out = net(x, yolo=True)

# Export the model
torch.onnx.export(
    net,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    "tinyyolo_final.onnx",  # where to save the model (can be a file or file-like object)
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


onnx_model = onnx.load("tinyyolo_final.onnx")
onnx.checker.check_model(onnx_model)


providers = [
    (
        "CUDAExecutionProvider",
        {
            "device_id": torch.cuda.current_device(),
            "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
        },
    )
]
# providers = ["CPUExecutionProvider"]

ort_session = onnxruntime.InferenceSession("tinyyolo_final.onnx", providers=providers)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
