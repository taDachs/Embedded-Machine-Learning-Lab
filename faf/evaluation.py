from torch.utils.data import DataLoader
import torch

from .metrics import test_net_macs, test_precision, test_net_time
from .utils.dataloader import voc_only_person_dataset
import tqdm


def eval_model(
    net,
    data_path,
    batch_size=128,
    iterations=10,
    num_test_batches=None,
    device=torch.device("cpu"),
):
    net.eval()

    ds_test = voc_only_person_dataset(train=False, path=data_path)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    size = test_net_macs(net)
    times = []
    for i in tqdm.tqdm(range(iterations), desc="[EVAL]"):
        time = test_net_time(net, loader_test, device)
        times.append(time)

    average_precision, precision, recall = test_precision(
        net, loader_test, device, num_batches=num_test_batches
    )

    return {
        "times": times,
        "average_precision": average_precision,
        "precision": precision,
        "recall": recall,
        "size": size,
        "batch_size": batch_size,
    }
