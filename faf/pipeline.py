import yaml
import torch
from .tinyyolov2 import TinyYoloV2

from abc import abstractmethod


class Step(yaml.YAMLObject):
    device = None
    data_path = None
    augment = False

    def __init__(self, name: str = None):
        super().__init__()
        if name is None:
            name = self.__class__.__name__
        if not hasattr(self.__class__, "idx"):
            self.__class__.idx = 0
        self.name = f"{name}{self.__class__.idx}"
        self.__class__.idx += 1

    def set_device(self, device: torch.device):
        self.device = device

    def set_data_path(self, data_path: str):
        self.data_path = data_path

    def set_augment(self, augment: bool):
        self.augment = augment

    @classmethod
    def from_yaml(cls, loader, node):
        values = loader.construct_mapping(node, deep=True)

        return cls(**values)

    def __call__(self, net: TinyYoloV2) -> TinyYoloV2:
        return self.run(net)

    @abstractmethod
    def run(net: TinyYoloV2) -> TinyYoloV2:
        raise NotImplementedError()
