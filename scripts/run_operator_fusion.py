#!/usr/bin/env python3

import os
from faf.fuse_conv_bn import operator_fusion
from faf.cli import add_default_params, setup_logging
from faf.tinyyolov2 import TinyYoloV2
import logging
import torch
import argparse


def main():
    setup_logging()

    parser = argparse.ArgumentParser("run_operator_fusion")

    add_default_params(parser)

    args = parser.parse_args()

    logging.info(f"Loading model from {args.input_model}")
    net = TinyYoloV2.from_saved_state_dict(args.input_model)

    new_net = operator_fusion(net)

    logging.info(f"Saving model to {args.output_model}")
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save(new_net.state_dict(), args.output_model)


if __name__ == "__main__":
    main()
