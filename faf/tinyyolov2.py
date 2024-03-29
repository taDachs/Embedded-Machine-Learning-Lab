import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyYoloV2(nn.Module):
    def __init__(self, num_classes=20, use_bias=False, use_batch_norm=True, use_constant_padding=False, use_leaky_relu=True):
        super().__init__()

        self._anchors = (
            (1.08, 1.19),
            (3.42, 4.41),
            (6.63, 11.38),
            (9.42, 5.11),
            (16.62, 10.52),
        )
        self.use_bias = use_bias
        self.use_batch_norm = use_batch_norm

        self.register_buffer("anchors", torch.tensor(self._anchors))
        self.num_classes = num_classes

        if use_constant_padding:
            self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        else:
            self.pad = nn.ReflectionPad2d((0, 1, 0, 1))

        if use_leaky_relu:
            self.relu = lambda x: F.leaky_relu(x, negative_slope=0.1, inplace=True)
        else:
            self.relu = lambda x: F.relu(x, inplace=True)

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=use_bias)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=use_bias)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=use_bias)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1, bias=use_bias)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1, bias=use_bias)
        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1, bias=use_bias)
        self.conv7 = nn.Conv2d(512, 1024, 3, 1, 1, bias=use_bias)
        self.conv8 = nn.Conv2d(1024, 1024, 3, 1, 1, bias=use_bias)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(128)
            self.bn5 = nn.BatchNorm2d(256)
            self.bn6 = nn.BatchNorm2d(512)
            self.bn7 = nn.BatchNorm2d(1024)
            self.bn8 = nn.BatchNorm2d(1024)

        self.conv9 = nn.Conv2d(1024, len(self._anchors) * (5 + num_classes), 1, 1, 0)

        self._register_load_state_dict_pre_hook(self._sd_hook)

    def _sd_hook(self, state_dict, prefix, *_):
        for key in state_dict:
            if "conv" in key and "weight" in key:
                n = int(key.split("conv")[1].split(".weight")[0])

                if n == 9:
                    continue

                dim_in = state_dict[f"conv{n}.weight"].shape[1]
                dim_out = state_dict[f"conv{n}.weight"].shape[0]

                conv = nn.Conv2d(dim_in, dim_out, 3, 1, padding=1, bias=self.use_bias)

                setattr(self, f"conv{n}", conv)

                if self.use_batch_norm:
                    bn = nn.BatchNorm2d(dim_out)
                    setattr(self, f"bn{n}", bn)

        self.conv9 = nn.Conv2d(
            state_dict["conv8.weight"].shape[1],
            len(self._anchors) * (5 + self.num_classes),
            1,
            1,
            0,
        )

    def forward(self, x, yolo=True):

        x = self.conv1(x)
        x = self.bn1(x) if self.use_batch_norm else x
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x) if self.use_batch_norm else x
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x) if self.use_batch_norm else x
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x) if self.use_batch_norm else x
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x) if self.use_batch_norm else x
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.bn6(x) if self.use_batch_norm else x
        x = self.pad(x)
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = self.relu(x)

        x = self.conv7(x)
        x = self.bn7(x) if self.use_batch_norm else x
        x = self.relu(x)

        x = self.conv8(x)
        x = self.bn8(x) if self.use_batch_norm else x
        x = self.relu(x)

        x = self.conv9(x)
        if yolo:
            nB, _, nH, nW = x.shape

            x = x.view(nB, self.anchors.shape[0], -1, nH, nW).permute(0, 1, 3, 4, 2)

            anchors = self.anchors.to(dtype=x.dtype, device=x.device)
            range_y, range_x, = torch.meshgrid(
                torch.arange(nH, dtype=x.dtype, device=x.device),
                torch.arange(nW, dtype=x.dtype, device=x.device),
            )
            anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]

            x = torch.cat(
                [
                    (x[:, :, :, :, 0:1].sigmoid() + range_x[None, None, :, :, None])
                    / nW,  # x center
                    (x[:, :, :, :, 1:2].sigmoid() + range_y[None, None, :, :, None])
                    / nH,  # y center
                    (x[:, :, :, :, 2:3].exp() * anchor_x[None, :, None, None, None]) / nW,  # Width
                    (x[:, :, :, :, 3:4].exp() * anchor_y[None, :, None, None, None]) / nH,  # Height
                    x[:, :, :, :, 4:5].sigmoid(),  # confidence
                    x[:, :, :, :, 5:].softmax(-1),
                ],
                -1,
            )

        return x

    @classmethod
    def from_saved_state_dict(cls, path: str, **kwargs):
        sd = torch.load(path)
        use_batch_norm = any("bn" in k for k in sd)
        use_bias = any(f"conv{i}.bias" in sd for i in range(1, 9))
        last_dim = sd["conv9.bias"].shape[0]

        num_classes = last_dim // sd["anchors"].shape[0] - 5

        net = TinyYoloV2(num_classes, use_bias, use_batch_norm, **kwargs)
        net.load_state_dict(sd)
        return net
