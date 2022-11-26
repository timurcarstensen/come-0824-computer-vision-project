# Adaption of the roi_pooling module from the original implementation

# standard library imports
from typing import Tuple, Optional

# 3rd party imports
import torch
from torchvision.ops import roi_pool


def roi_pooling_ims(
    t: torch.Tensor,
    rois: torch.Tensor,
    device: torch.device,
    size: Tuple[int, int] = (8, 16),
    spatial_scale: Optional[float] = 1.0,
):
    """
    Wrapper for torchvision.ops.roi_pool
    :param device: device to store the tensor on
    :param t: input tensor of shape (N, C, H, W)
    :param rois: input tensor of shape (N, 5) where N is the number of rois, 5 is (batch_index, x1, y1, x2, y2)
    :param size: output size of the pooling operation
    :param spatial_scale: scale factor for the input coordinates
    :return:
    """
    # checking input arguments for type and shape
    if not (rois.dim() == 2 and len(t) == len(rois) and rois.size(1) == 4):
        raise ValueError("rois should be a 2D tensor of shape (num_rois, 5)")

    if not (isinstance(t, torch.Tensor) and isinstance(rois, torch.Tensor)):
        raise TypeError("t and rois should be torch.Tensor")

    # adapting rois to the format required by torchvision.ops.roi_pool (i.e. (batch_index, x1, y1, x2, y2))
    rois = torch.stack(
        [
            torch.cat(
                (
                    torch.tensor(data=[idx], device=device),
                    elem,
                )
            )
            for idx, elem in enumerate(rois)
        ]
    )

    return roi_pool(input=t, boxes=rois, output_size=size, spatial_scale=spatial_scale)


if __name__ == "__main__":
    pass