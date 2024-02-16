import torch
import logging

from torch.utils.data import DataLoader
from .utils.dataloader import voc_only_person_dataset
from .utils.loss import YoloLoss
from .pipeline import Step

from .tinyyolov2 import TinyYoloV2
import tqdm


class FineTune(Step):
    yaml_tag = "!finetune"

    def __init__(
        self,
        epochs: int = 15,
        only_last: bool = False,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.epochs = epochs
        self.only_last = only_last
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def run(self, net: TinyYoloV2) -> TinyYoloV2:
        train(
            net,
            num_epochs=self.epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            data_path=self.data_path,
            device=self.device,
            only_last=self.only_last,
        )

        return net


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


def train(
    net, num_epochs, learning_rate, batch_size, data_path, device, only_last=False
):
    if only_last:
        for key, param in net.named_parameters():
            if any(str(x) in key for x in range(1, 9)):
                param.requires_grad = False
    # fine tune model
    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, net.parameters()),
        lr=learning_rate,
    )
    criterion = YoloLoss(anchors=net.anchors)
    ds = voc_only_person_dataset(train=True, path=data_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    loss = []
    for i in range(num_epochs):
        logging.info(f"Epoch {i+1}/{num_epochs}")
        epoch_loss = train_epoch(net, optimizer, criterion, loader, device)
        loss.append(epoch_loss)

    if only_last:
        for key, param in net.named_parameters():
            if any(str(x) in key for x in range(1, 9)):
                param.requires_grad = True

    return loss
