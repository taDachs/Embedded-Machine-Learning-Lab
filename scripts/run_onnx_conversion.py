#!/usr/bin/env python3

import os
from faf.cli import add_default_params, setup_logging
from faf.tinyyolov2 import TinyYoloV2
import logging
import torch
import argparse


def main():
    setup_logging()

    parser = argparse.ArgumentParser("run_onnx_conversion")

    add_default_params(parser)

    args = parser.parse_args()

    logging.info(f"Loading model from {args.input_model}")
    net = TinyYoloV2.from_saved_state_dict(args.input_model)

    net.eval()
    x = torch.randn(1, 3, 320, 320, requires_grad=True)

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    logging.info(f"Saving model to {args.output_model}")

    # Export the model
    torch.onnx.export(
        net,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        args.output_model,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        keep_initializers_as_inputs=True,
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        # dynamic_axes={
        #     "input": {0: "batch_size"},
        #     "output": {0: "batch_size"},
        # },  # variable length axes
    )


if __name__ == "__main__":
    main()
