from typing import Dict
import os
import copy
import torch
from torchinfo import summary
from torch.utils.data import DataLoader
from datetime import datetime
import json

from faf.utils.dataloader import voc_only_person_dataset
from faf.tinyyolov2 import PrunedTinyYoloV2, TinyYoloV2
from faf.person_only import eval_epoch


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

        if i == 9:  # last layer has bias
            state_dict[b_idx] = state_dict[b_idx][good_indices]

    return state_dict


def net_macs(model_class: torch.nn.Module, state_dict: Dict) -> int:
    net = model_class(1)
    net.load_state_dict(state_dict)
    res = summary(net, (1, 3, 320, 320), verbose=0)
    return res.total_mult_adds


if __name__ == "__main__":
    device = torch.device("cuda:0")
    sd = torch.load("./weights/only_person_20240124-122019.pt")
    net = TinyYoloV2(1)
    net.load_state_dict(sd)
    print(net_macs(TinyYoloV2, sd))

    sd_pruned = l1_structured_pruning(sd, 0.5)
    sd_pruned = densify_state_dict(sd_pruned)
    pruned_net = PrunedTinyYoloV2(1)
    pruned_net.load_state_dict(sd_pruned)
    print(net_macs(PrunedTinyYoloV2, sd_pruned))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("weights", exist_ok=True)
    torch.save(pruned_net.state_dict(), f"weights/only_person_pruned_{timestamp}.pt")

    batch_size = 128
    data_path = "./data/"
    ds_test = voc_only_person_dataset(train=False, path=data_path)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
    epoch_result_unpruned = eval_epoch(net, loader_test, device, single_batch=True)
    epoch_result_pruned = eval_epoch(pruned_net, loader_test, device, single_batch=True)
    print(f"Unpruned Time: {epoch_result_unpruned['time']}")
    print(f"Pruned Time: {epoch_result_pruned['time']}")
