import argparse
import logging


def add_default_params(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-model", type=str, required=True, help="path to input model")
    parser.add_argument("--output-model", type=str, required=True, help="path to output model")
    parser.add_argument("--data", type=str, default="./data", help="path to data dir")
    parser.add_argument("--device", type=str, default="cpu", help="device to run on (cuda or cpu)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="full",
        choices=["voc", "human", "tiktok", "full"],
        help="dataset used for training between pruning steps",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="if set, disable augmentations",
    )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s %(levelname)-2s] %(message)s",
        datefmt="%m-%d %H:%M",
    )

    pil_logger = logging.getLogger("PIL")
    pil_logger.setLevel(logging.INFO)
