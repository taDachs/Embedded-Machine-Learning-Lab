import logging
import time
import json
import os

import torch
import tqdm
from torch.utils.data import DataLoader

from faf.tinyyolov2 import TinyYoloV2
from faf.utils.ap import ap, precision_recall_levels
from faf.utils.dataloader import voc_only_person_dataset
from faf.utils.loss import YoloLoss
from faf.utils.yolo import filter_boxes, nms
from datetime import datetime


def train_epoch(
    net: TinyYoloV2,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.modules.loss._Loss,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    net.to(device)
    net.train()
    epoch_loss = []
    pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
    pbar.set_description(f"[TRAINING] Loss: {0:.4f}")
    for idx, (input, target) in pbar:
        input = input.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        # Yolo head is implemented in the loss for training, therefore yolo=False
        output = net(input, yolo=False)
        loss, _ = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss)

        pbar.set_description(f"[TRAINING] Loss: {loss:.4f}")

    return torch.mean(torch.stack(epoch_loss))


def eval_epoch(
    net: TinyYoloV2,
    loader: DataLoader,
    device: torch.device,
    filter_threshold: float = 0.0,
    nms_threshold: float = 0.5,
    single_batch: bool = False,
):
    net.to(device)
    net.eval()
    results = {}
    results["precision"] = []
    results["recall"] = []
    results["time"] = []
    with torch.no_grad():
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
        pbar.set_description("[EVALUATION]")
        for idx, (inputs, targets) in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            start = time.time()
            outputs = net(inputs, yolo=True)
            end = time.time()
            elapsed = end - start
            results["time"].append(elapsed / inputs.shape[0])  # divide by batch size

            # The right threshold values can be adjusted for the target application
            outputs = filter_boxes(outputs, filter_threshold)
            outputs = nms(outputs, nms_threshold)

            for output, target in zip(outputs, targets):
                precision, recall = precision_recall_levels(target, output)
                results["precision"].append(precision)
                results["recall"].append(recall)

            if single_batch:
                break

    results["average_precision"] = ap(results["precision"], results["recall"])
    return results


def perform_stripping_step(
    weights_path: str,
    device: torch.device,
    learning_rate: float = 0.001,
    batch_size: int = 128,
    data_path: str = "data/",
    num_epochs=15,
):
    # strip classes
    net = TinyYoloV2(num_classes=1)
    sd = torch.load(weights_path)
    net.load_state_dict({k: v for k, v in sd.items() if "9" not in k}, strict=False)

    # fine tune model
    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, net.parameters()),
        lr=learning_rate,
    )
    criterion = YoloLoss(anchors=net.anchors)
    ds = voc_only_person_dataset(train=True, path=data_path)
    ds_test = voc_only_person_dataset(train=False, path=data_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    results = []
    loss = []
    for i in range(num_epochs):
        logging.info(f"Epoch {i+1}/{num_epochs}")
        epoch_loss = train_epoch(net, optimizer, criterion, loader, device)
        loss.append(epoch_loss)

        epoch_result = eval_epoch(net, loader_test, device, single_batch=True)
        results.append(epoch_result)
    return net, results


if __name__ == "__main__":
    net, results = perform_stripping_step(
        "weights/voc_pretrained.pt", torch.device("cpu"), num_epochs=15
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("weights", exist_ok=True)
    torch.save(net, f"weights/only_person_{timestamp}.pt")

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"{timestamp}.json"), "w+") as f:
        json.dump(results, f)
