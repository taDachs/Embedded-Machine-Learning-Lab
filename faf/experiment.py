import yaml
from typing import List

# have to be import so they get registered with yaml loader
import faf.training
import faf.person_only
import faf.fuse_conv_bn
import faf.pruning

from .pipeline import Step


class Experiment(yaml.YAMLObject):
    yaml_tag = "!experiment"

    def __init__(self, steps: List[Step], start_weights_path: str, augment: bool):
        super().__init__()
        self.steps = steps
        self.start_weights_path = start_weights_path
        self.augment = augment

    @classmethod
    def from_config(cls, path: str):
        with open(path, "r") as file:
            experiment = yaml.full_load(file)
        return experiment
