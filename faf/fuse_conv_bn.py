import torch

from .tinyyolov2 import TinyYoloV2
from .pipeline import Step


class OperatorFusion(Step):
    yaml_tag = "!operator_fusion"

    def run(self, net):
        return operator_fusion(net)


def operator_fusion(net: TinyYoloV2) -> TinyYoloV2:
    sd = net.state_dict()
    fused_sd = {}

    for i in range(1, 9):
        conv_w = sd[f"conv{i}.weight"]
        conv_b = sd.get(f"conv{i}.bias", 0)

        bn_w = sd[f"bn{i}.weight"]
        bn_b = sd[f"bn{i}.bias"]
        bn_rm = sd[f"bn{i}.running_mean"]
        bn_rv = sd[f"bn{i}.running_var"]

        fused_conv, fused_bias = fuse_conv_bn_weights(
            conv_w, conv_b, bn_rm, bn_rv, bn_w, bn_b
        )
        fused_sd[f"conv{i}.weight"] = fused_conv
        fused_sd[f"conv{i}.bias"] = fused_bias

    # Copy over the remaining data
    fused_sd["anchors"] = sd["anchors"]
    fused_sd["conv9.weight"] = sd["conv9.weight"]
    fused_sd["conv9.bias"] = sd["conv9.bias"]

    fused_net = TinyYoloV2(
        num_classes=net.num_classes, use_bias=True, use_batch_norm=False
    )
    fused_net.load_state_dict(fused_sd)
    return fused_net


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_w, bn_b):
    bn_eps = 1e-05

    fused_conv = torch.zeros(conv_w.shape)
    fused_bias = torch.zeros(bn_w.shape)

    rsq = torch.rsqrt(bn_rv + bn_eps)

    fused_conv = conv_w * (bn_w * rsq)[..., None, None, None]
    fused_bias = (conv_b - bn_rm) * rsq * bn_w + bn_b

    # return torch.nn.Parmeter(fused_conv), torch.nn.Parameter(fused_bias)
    return fused_conv, fused_bias
