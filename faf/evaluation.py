import torch

from .metrics import test_net_macs, test_precision, test_net_time
from faf.data.dataloader import VOCDataLoaderPerson
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

    loader_test = VOCDataLoaderPerson(augment=False, batch_size=batch_size, shuffle=False)

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
