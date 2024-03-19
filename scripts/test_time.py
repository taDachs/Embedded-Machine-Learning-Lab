#!/usr/bin/env python3

import os
from faf.cli import setup_logging
from faf.data.dataloader import get_dataset_by_name
import logging
import torch
import argparse
from faf.inference import get_model_by_file
from timeit import default_timer as timer
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import json


def main():
    setup_logging()

    parser = argparse.ArgumentParser("test_time")

    parser.add_argument(
        "--input-models",
        type=str,
        required=True,
        nargs="+",
        help="paths to input models",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu", help="device to run on (cuda or cpu)")
    parser.add_argument("--out-file", type=str, required=True, help="saves results to file as json")

    args = parser.parse_args()

    results = {}
    for input_model in args.input_models:
        logging.info(f"Loading model from {input_model}")
        net = get_model_by_file(input_model, args.device)
        test_data = np.zeros((320, 320, 3), dtype=np.uint8)

        for i in tqdm(range(args.num_warmup), desc="warump"):
            net(test_data)

        start = timer()
        for i in tqdm(range(args.runs), desc="testing"):
            net(test_data)
        end = timer()

        results[os.path.basename(input_model).split(".")[0]] = (end - start) / args.runs

        del net

    logging.info(f"saving results to {args.out_file}")
    with open(args.out_file, "w+") as f:
        json.dump(results, f)

    keys = list(results.keys())
    vals = [results[k] for k in keys]
    sns.barplot(x=keys, y=vals)
    plt.savefig("results.png")


if __name__ == "__main__":
    main()
