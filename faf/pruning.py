from typing import Dict
import copy
import torch
from torch.utils.data import DataLoader
import logging

from .tinyyolov2 import TinyYoloV2
from .pipeline import Step
from .metrics import test_precision, test_net_macs
from .training import train
from .utils.dataloader import voc_only_person_dataset


class Pruning(Step):
    yaml_tag = "!pruning"

    def __init__(
        self,
        prune_ratio: float = 0.5,
        target_acc: float = 0.3,
        num_eval_batches: int = 5,
        num_train_epochs: int = 3,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prune_ratio = prune_ratio
        self.target_acc = target_acc
        self.num_eval_batches = num_eval_batches
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def run(self, net):
        return iterative_prune(
            net,
            self.prune_ratio,
            self.target_acc,
            self.num_eval_batches,
            self.num_train_epochs,
            self.learning_rate,
            self.batch_size,
            self.data_path,
            self.device,
        )


def iterative_prune(
    net: TinyYoloV2,
    prune_ratio: float,
    target_acc: float,
    num_eval_batches: int,
    num_train_epochs: int,
    learning_rate: float,
    batch_size: int,
    data_path: str,
    device: torch.device,
) -> TinyYoloV2:
    previous_model = net
    test_ds = voc_only_person_dataset(train=False, path=data_path)
    testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    aps = []
    sizes = []
    ap, _, _ = test_precision(net, testloader, device, num_batches=num_eval_batches)
    size = test_net_macs(net)

    logging.info(f"Pruning: starting with AP: {ap} Size: {size}")

    aps.append(ap)
    sizes.append(size)
    while ap > target_acc:
        previous_model = net
        net = prune_net(net, prune_ratio)
        train(net, num_train_epochs, learning_rate, batch_size, data_path, device)
        ap, _, _ = test_precision(net, testloader, device, num_batches=num_eval_batches)
        size = test_net_macs(net)
        aps.append(ap)
        sizes.append(size)
        logging.info(f"Pruning: current AP: {ap} Size: {size}")

    logging.info(f"Pruning: {ap} < {target_acc}")
    return previous_model  # always take the previous model


def prune_net(net: TinyYoloV2, prune_ratio: float):
    sd = net.state_dict()
    sd_pruned = l1_structured_pruning(sd, prune_ratio)
    sd_pruned = densify_state_dict(sd_pruned)
    pruned_net = TinyYoloV2(1, net.use_bias, net.use_batch_norm)
    pruned_net.load_state_dict(sd_pruned)

    return pruned_net


def l1_structured_pruning(state_dict: Dict, prune_ratio: float) -> Dict:
    state_dict = copy.deepcopy(state_dict)
    # 9 layers, last one contains class info so don't prune
    for i in range(3, 9):  # keep first two layers unpruned (same as exercise)
        w_idx = f"conv{i}.weight"

        v = state_dict[w_idx]

        l1_norms = torch.sum(torch.abs(v), dim=[1, 2, 3])
        thresh = torch.quantile(l1_norms, prune_ratio)
        state_dict[w_idx][l1_norms < thresh] = 0

    return state_dict


def densify_state_dict(state_dict: Dict) -> Dict:
    state_dict = copy.deepcopy(state_dict)

    good_indices = None

    for i in range(1, 10):
        w_idx = f"conv{i}.weight"
        b_idx = f"conv{i}.bias"
        w = state_dict[w_idx]

        bn_rm_idx = f"bn{i}.running_mean"
        bn_rv_idx = f"bn{i}.running_var"
        bn_b_idx = f"bn{i}.bias"
        bn_w_idx = f"bn{i}.weight"

        if good_indices is not None:
            w = w[:, good_indices]
        # 9 layers, last one contains class info so don't prune
        good_indices = [i for i in range(w.size()[0]) if torch.any(w[i] != 0)]

        if i != 9 and bn_w_idx in state_dict:  # no batchnorm on last layer
            state_dict[bn_rm_idx] = state_dict[bn_rm_idx][good_indices]
            state_dict[bn_rv_idx] = state_dict[bn_rv_idx][good_indices]
            state_dict[bn_b_idx] = state_dict[bn_b_idx][good_indices]
            state_dict[bn_w_idx] = state_dict[bn_w_idx][good_indices]

        state_dict[w_idx] = w[good_indices]

        # some layers have bias
        if b_idx in state_dict:
            state_dict[b_idx] = state_dict[b_idx][good_indices]

    return state_dict
