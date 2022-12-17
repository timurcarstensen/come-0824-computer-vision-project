# 3rd party imports
import torch
from torchvision.transforms import Resize
from torchvision.ops import box_convert
import pytorch_lightning as pl


class CropModule(pl.LightningModule):
    """
    Crops the images in the batch according to the bounding boxes provided by the DetectionModule and resizes
    to the specified size.
    """

    def __init__(self, size):
        """Initialises the CropModule"""

        super(CropModule, self).__init__()

        self.resize = Resize(size)

    def forward(self, x):
        img, box = x

        result = []

        for idx, elem in enumerate(box):
            elem = elem.flatten()
            elem = elem * torch.tensor([480, 480, 480, 480], device=self.device)

            elem = (
                box_convert(
                    torch.tensor(
                        [[elem[0], elem[1], elem[2], elem[3]]], device=self.device
                    ),
                    in_fmt="cxcywh",
                    out_fmt="xyxy",
                )
                .flatten()
                .int()
            )

            tmp_img = img[idx][:, elem[1] : elem[3], elem[0] : elem[2]]

            # find out in which dimension the image is of dim 0
            if tmp_img.shape[1] == 0:
                elem[1] = elem[1] - 10 if elem[1] - 10 > 0 else 0
                elem[3] = elem[3] - 10 if elem[3] - 10 > 0 else 0

                tmp_img = img[idx][:, elem[1] : elem[3], elem[0] : elem[2]]

            if tmp_img.shape[2] == 0:
                elem[0] = elem[0] + 10 if elem[0] + 10 < 480 else 480
                elem[2] = elem[2] + 10 if elem[2] + 10 < 480 else 480

                tmp_img = img[idx][:, elem[1] : elem[3], elem[0] : elem[2]]

            tmp_img = self.resize(tmp_img)
            result.append(tmp_img)

        return torch.stack(result).view(len(box), -1)
