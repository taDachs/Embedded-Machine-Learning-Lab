#!/usr/bin/env python3

import os
from faf.cli import setup_logging
from faf.data.dataloader import get_dataset_by_name
from faf.inference import get_model_by_file
import logging
import torch
import argparse
from faf.metrics import test_precision_inference
import json
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    setup_logging()

    parser = argparse.ArgumentParser("test_ap")

    parser.add_argument(
        "--input-models",
        type=str,
        required=True,
        nargs="+",
        help="paths to input models",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--data", type=str, default="./data", help="path to data dir")
    parser.add_argument("--device", type=str, default="cpu", help="device to run on (cuda or cpu)")
    parser.add_argument("--out-file", type=str, required=True, help="saves results to file as json")
    parser.add_argument(
        "--dataset",
        type=str,
        default="full",
        choices=["voc", "human", "tiktok", "full"],
        help="dataset used for training between pruning steps",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()
    ds = get_dataset_by_name(args.dataset, False, False, args.data)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    results = {}
    for input_model in args.input_models:
        logging.info(f"Loading model from {input_model}")
        net = get_model_by_file(input_model, args.device)
        ap, _, _ = test_precision_inference(
            net=net,
            testloader=loader,
            filter_threshold=args.score_thresh,
            nms_threshold=args.nms_thresh,
            num_batches=args.num_batches,
        )
        results[os.path.basename(input_model).split(".")[0]] = ap

    logging.info(f"saving results to {args.out_file}")
    with open(args.out_file, "w+") as f:
        json.dump(results, f)

    keys = list(results.keys())
    vals = [results[k] for k in keys]
    sns.barplot(x=keys, y=vals)
    plt.savefig("results.png")


if __name__ == "__main__":
    main()
