#!/usr/bin/env python3

import os
from faf.pruning import iterative_prune
from faf.data.dataloader import get_dataset_by_name
from faf.cli import add_default_params, setup_logging
from faf.tinyyolov2 import TinyYoloV2
import logging
import torch
import argparse


def main():
    setup_logging()

    parser = argparse.ArgumentParser("run_pruning")

    add_default_params(parser)

    parser.add_argument(
        "--prune-ratio",
        type=float,
        default=0.05,
        help="relative amount by which the model is pruned each iteration",
    )
    parser.add_argument(
        "--target-ap",
        type=float,
        required=True,
        help="if the average precision falls below that value the pruning is stopped",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num-eval-batches",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    train_ds = get_dataset_by_name(args.dataset, augment=not args.no_augment, train=True)
    test_ds = get_dataset_by_name(args.dataset, augment=False, train=False)

    logging.info(f"Loading model from {args.input_model}")
    net = TinyYoloV2.from_saved_state_dict(args.input_model)

    new_net = iterative_prune(
        net=net,
        train_ds=train_ds,
        test_ds=test_ds,
        prune_ratio=args.prune_ratio,
        target_acc=args.target_ap,
        num_eval_batches=args.num_eval_batches,
        num_train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.train_batch_size,
        data_path=args.data,
        device=device,
    )

    logging.info(f"Saving model to {args.output_model}")
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save(new_net.state_dict(), args.output_model)


if __name__ == "__main__":
    main()
