import torch.nn as nn
import torch
from torchvision.transforms import Resize
from torchvision.ops import box_convert
import numpy as np


class CropModule(nn.Module):
    def __init__(self, size):
        super(CropModule, self).__init__()

        self.resize = Resize(size)

    def forward(self, x):
        img, box = x

        result = []

        for idx, elem in enumerate(box):
            elem = elem.flatten()
            elem = elem * torch.tensor([480, 480, 480, 480])

            elem = (
                box_convert(
                    torch.tensor([[elem[0], elem[1], elem[2], elem[3]]]),
                    in_fmt="cxcywh",
                    out_fmt="xyxy",
                )
                .flatten()
                .int()
            )

            tmp_img = img[idx][:, elem[1] : elem[3], elem[0] : elem[2]]
            tmp_img = self.resize(tmp_img)
            result.append(tmp_img)

        return torch.stack(result).view(len(box), -1)
