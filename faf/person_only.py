from .tinyyolov2 import TinyYoloV2

from .pipeline import Step
from .training import train


class StripClasses(Step):
    yaml_tag = "!strip_classes"

    def __init__(
        self,
        finetune: bool = True,
        finetune_only_last: bool = True,
        finetune_epochs: int = 15,
        finetune_learning_rate: float = 1e-3,
        finetune_batch_size: int = 128,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.finetune = finetune
        self.finetune_only_last = finetune_only_last
        self.finetune_epochs = finetune_epochs
        self.finetune_learning_rate = finetune_learning_rate
        self.finetune_batch_size = finetune_batch_size

    def run(self, net):
        net = strip_classes(net)
        if self.finetune:
            train(
                net,
                self.ds_f,
                self.finetune_epochs,
                self.finetune_learning_rate,
                self.finetune_batch_size,
                self.data_path,
                self.device,
                self.finetune_only_last,
            )
        return net


def strip_classes(net: TinyYoloV2) -> TinyYoloV2:
    new_net = TinyYoloV2(num_classes=1)
    sd = net.state_dict()
    new_net.load_state_dict({k: v for k, v in sd.items() if "9" not in k}, strict=False)

    return new_net
