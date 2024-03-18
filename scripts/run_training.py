#!/usr/bin/env python3

import os
from faf.training import train
from faf.data.dataloader import get_dataset_by_name
from faf.cli import add_default_params, setup_logging
from faf.tinyyolov2 import TinyYoloV2
import logging
import torch
import argparse


def main():
    setup_logging()

    parser = argparse.ArgumentParser("run_training")

    add_default_params(parser)

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--only-last",
        action="store_true",
        help="if set, freezes all layers except the last",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    ds = get_dataset_by_name(args.dataset, augment=not args.no_augment, train=True)

    logging.info(f"Loading model from {args.input_model}")
    net = TinyYoloV2.from_saved_state_dict(args.input_model)

    train(net, ds, args.epochs, args.learning_rate, args.batch_size, device, args.only_last)

    logging.info(f"Saving model to {args.output_model}")
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save(net.state_dict(), args.output_model)


if __name__ == "__main__":
    main()
