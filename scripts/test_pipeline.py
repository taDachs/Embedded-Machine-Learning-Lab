#!/usr/bin/env python3
from faf.tinyyolov2 import TinyYoloV2
import logging
import torch
from faf.experiment import Experiment
from faf.evaluation import eval_model
from faf.inference import to_onnx
import os
import pandas as pd

import argparse

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s %(levelname)-2s] %(message)s",
        datefmt="%m-%d %H:%M",
    )

    parser = argparse.ArgumentParser("run_pipeline")

    parser.add_argument("--data", type=str, required=True, help="path to data folder")
    parser.add_argument("--config", type=str, required=True, help="path to config")
    parser.add_argument("--results", type=str, required=True, help="path to results")
    parser.add_argument("--weights", type=str, required=True, help="path to weights")

    parser.add_argument(
        "--train-device", type=str, default="cpu", help="device used for training"
    )
    parser.add_argument(
        "--eval-device", type=str, default="cpu", help="device used for evaluation"
    )

    parser.add_argument(
        "--no-eval", action="store_true", help="if set, doesn't run eval"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="if set, doesn't run step and loads weights from previous run",
    )

    args = parser.parse_args()

    train_device = torch.device(args.train_device)
    eval_device = torch.device(args.eval_device)

    basename = os.path.basename(args.config).split(".")[0]

    weights_dir = os.path.join(args.weights, basename)
    results_dir = os.path.join(args.results, basename)
    results_table_path = os.path.join(results_dir, "results.csv")

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(args.data, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    experiment = Experiment.from_config(args.config)

    logging.info(f"Loading original weights from {experiment.start_weights_path}")
    net = TinyYoloV2(20)
    net.to(train_device)
    net.load_state_dict(torch.load(experiment.start_weights_path))

    if not args.no_eval:
        results = eval_model(
            net,
            args.data,
            device=eval_device,
            batch_size=64,
            num_test_batches=10,
            iterations=100,
        )
        headers = ["step"] + list(results.keys())
        df = pd.DataFrame(columns=headers)
        results["step"] = "base"
        df = df.append(results, ignore_index=True)
        logging.info(f"Average Precision: {results['average_precision']}")

    for i, step in enumerate(experiment.steps):
        step_weights_path = os.path.join(weights_dir, f"step_{i}_{step.name}.pt")
        if args.no_cache or not os.path.exists(step_weights_path):
            logging.info(f"Performing Step {i}: {step.name}")
            step.set_device(train_device)
            step.set_data_path(args.data)
            net.to(train_device)
            net = step.run(net)

            logging.info(f"Saving weights to {step_weights_path}")
            torch.save(net.state_dict(), step_weights_path)
        else:
            logging.info(f"Loading weights from {step_weights_path}")
            net = TinyYoloV2.from_saved_state_dict(step_weights_path)

        if not args.no_eval:
            logging.info("Evaluating Step")
            net.to(eval_device)
            results = eval_model(
                net,
                args.data,
                device=eval_device,
                batch_size=64,
                num_test_batches=10,
                iterations=100,
            )
            results["step"] = step.name
            df = df.append(results, ignore_index=True)
            logging.info(f"Average Precision: {results['average_precision']}")

    if args.no_cache or not os.path.exists(step_weights_path):
        final_weights_path = os.path.join(weights_dir, "final.pt")
        logging.info(f"Saving final weights to {final_weights_path}")
        torch.save(net.state_dict(), final_weights_path)

    # onnx test
    net.to(torch.device("cpu"))
    inference = to_onnx(net, weights_dir)
    if not args.no_eval:
        results = eval_model(
            inference,
            args.data,
            device=torch.device("cpu"),
            batch_size=64,
            num_test_batches=10,
            iterations=100,
            test_size=False,
        )
        results["step"] = "onnx_runtime"
        df = df.append(results, ignore_index=True)
        logging.info(f"Average Precision: {results['average_precision']}")

    if not args.no_eval:
        logging.info(f"Saving results to {results_table_path}")
        df.to_csv(results_table_path, index=False)
