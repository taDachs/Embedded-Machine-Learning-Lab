import time
from faf.utils.yolo import filter_boxes, nms
from torchinfo import summary

import torch
import numpy as np

from typing import List, Tuple
from .utils.loss import iou
import tqdm


def test_net_macs(net) -> int:
    res = summary(net, (1, 3, 320, 320), verbose=0)
    return res.total_mult_adds


def precision_recall(
    ground_truth_boxes: List[torch.Tensor],
    predicted_boxes: List[torch.Tensor],
    iou_threshold: float,
) -> Tuple[float, float, float]:

    """
    expects:
        ground_truth_boxes: List of Torch tensors as input.
                            A GT box is has a shape torch.tensor([x,y,h,w,1.0,class])
        predicted_boxes: List of Torch tensors as input.
                            A GT box is has a shape torch.tensor([x,y,h,w,confidence,class])
    returns:
        Tuple of floats: Number of True Positives, Number of False Positives, Number of False Negatives
    """

    if len(predicted_boxes) == 0:
        return 0, 0, len(ground_truth_boxes)

    if len(ground_truth_boxes) == 0:
        return 0, len(predicted_boxes), 0

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []

    for ipb, pred_box in enumerate(predicted_boxes):
        for igb, gt_box in enumerate(ground_truth_boxes):
            box_iou = iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0))
            if box_iou > iou_threshold:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(float(box_iou))

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        return 0, len(predicted_boxes), len(ground_truth_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]

        if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
            gt_match_idx.append(gt_idx)
            pred_match_idx.append(pr_idx)

    tp = len(gt_match_idx)
    fp = len(predicted_boxes) - len(pred_match_idx)
    fn = len(ground_truth_boxes) - len(gt_match_idx)

    return tp, fp, fn


def precision_recall_levels(
    ground_truth_boxes: torch.Tensor, predicted_boxes: torch.Tensor
) -> Tuple[float, float]:
    """
    expects:
        ground_truth_boxes: Expects a Torch tensor as input
                            Tensor has to be of shape torch.tensor([i,x,y,h,w,1.0,class])
        predicted_boxes: List of Torch tensors as input.
                            A GT box is has a shape torch.tensor([i,x,y,h,w,1.0,class])
    returns:
        Tuple of floats: Precision, Recall
    """

    ground_truth_boxes = [
        ground_truth_boxes[i, :-2]
        for i in range(ground_truth_boxes.shape[0])
        if ground_truth_boxes[i, -1] >= 0
    ]

    predicted_boxes = [predicted_boxes[i] for i in range(predicted_boxes.shape[0])]

    recall = []
    precision = []

    for threshold in np.linspace(0.0, 1.0, 11):
        prediction = list(filter(lambda x: float(x[-2]) > threshold, predicted_boxes))

        tp, fp, fn = precision_recall(ground_truth_boxes, prediction, 0.5)
        try:
            recall.append(tp / (tp + fn))
        except ZeroDivisionError:
            if tp == 0 and fn == 0:
                recall.append(1)
            else:
                recall.append(0)
        try:
            precision.append(tp / (tp + fp))
        except ZeroDivisionError:
            if tp == 0 and fp == 0:
                precision.append(1)
            else:
                precision.append(0)
    return precision, recall


def ap(precision: List[List], recall: List[List]) -> float:
    """
    Calculates the average precision (area under ROC) based on recall and precision values
    expects:
        precision as List of Lists
        recall as List of Lists
    returns:
        float: average_precision
    """
    recall = np.mean(np.array(recall), axis=0)
    precision = np.mean(np.array(precision), axis=0)

    out = []
    for level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recall >= level).flatten()
            prec = max(precision[args])
        except ValueError:
            prec = 0.0
        out.append(prec)
    return np.mean(np.array(out))


def test_net_time(net, testloader, device):

    batch, _ = next(iter(testloader))
    net.to(device)
    batch = batch.to(device)
    t_now = time.time()
    net(batch)
    t_stop = time.time()
    t = t_stop - t_now

    return t


def test_precision(
    net, testloader, device, filter_threshold=0.0, nms_threshold=0.5, num_batches=None
):
    net.to(device)
    precisions = []
    recalls = []
    for i, (batch, targets) in enumerate(tqdm.tqdm(testloader, desc="[EVAL]")):
        if num_batches is not None and i >= num_batches:
            break

        batch = batch.to(device)
        targets = targets.to(device)
        outputs = net(batch)
        outputs = filter_boxes(outputs, filter_threshold)
        outputs = nms(outputs, nms_threshold)

        for output, target in zip(outputs, targets):
            precision, recall = precision_recall_levels(target, output)
            precisions.append(precision)
            recalls.append(recall)
    average_precision = ap(precisions, recalls)

    return average_precision, precisions, recalls