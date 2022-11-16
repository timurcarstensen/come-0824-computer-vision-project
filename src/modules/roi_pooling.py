# standard library imports
from typing import List, Tuple, Optional

# 3rd party imports
import torch
from torch.autograd import Variable
from torch.autograd.function import Function
from torch.nn import AdaptiveMaxPool2d


def roi_pooling(
    t: torch.TensorType,
    rois: torch.TensorType,
    size: Tuple[int, int] = (7, 7),
    spatial_scale: Optional[float] = 1.0,
):
    if not rois.dim() == 2 and rois.size(1) == 5:
        raise ValueError("rois should be a 2D tensor of shape (num_rois, 5)")

    if not (isinstance(t, torch.Tensor) and isinstance(rois, torch.Tensor)):
        raise TypeError("t and rois should be torch.Tensor")

    output = []
    rois = rois.data.float()

    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()
    for idx, roi in enumerate(rois):
        im_idx = roi[0]
        im = t.narrow(0, im_idx, 1)[..., roi[2] : (roi[4] + 1), roi[1] : (roi[3] + 1)]
        output.append(AdaptiveMaxPool2d(size)(im))

    return torch.cat(output, 0)


def roi_pooling_ims(
    t: torch.TensorType,
    rois: torch.TensorType,
    size: Tuple[int, int] = (7, 7),
    spatial_scale: Optional[float] = 1.0,
):
    # written for one roi one image
    # size: (w, h)

    if not (rois.dim() == 2 and len(t) == len(rois) and rois.size(1) == 4):
        raise ValueError("rois should be a 2D tensor of shape (num_rois, 5)")

    if not (isinstance(t, torch.Tensor) and isinstance(rois, torch.Tensor)):
        raise TypeError("t and rois should be torch.Tensor")

    output = []
    rois = rois.data.float()

    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()
    for idx, roi in enumerate(rois):
        im = t.narrow(0, idx, 1)[..., roi[1] : (roi[3] + 1), roi[0] : (roi[2] + 1)]
        output.append(AdaptiveMaxPool2d(size)(im))

    return torch.cat(output, 0)


if __name__ == "__main__":
    synthetic_input = Variable(torch.rand(2, 1, 10, 10), requires_grad=True)
    regions_of_interest = Variable(
        torch.LongTensor([[1, 2, 7, 8], [3, 3, 8, 8]]), requires_grad=False
    )

    # AdaptiveMaxPool2d.apply(input, rois, 8, 8)
    out = roi_pooling_ims(synthetic_input, regions_of_interest, size=(8, 8))
    out.backward(out.data.clone().uniform_())

    synthetic_input = Variable(torch.rand(2, 1, 10, 10), requires_grad=True)
    regions_of_interest = Variable(
        torch.LongTensor([[0, 1, 2, 7, 8], [0, 3, 3, 8, 8], [1, 3, 3, 8, 8]]),
        requires_grad=False,
    )
    regions_of_interest = Variable(
        torch.LongTensor([[0, 3, 3, 8, 8]]), requires_grad=False
    )

    # out = adaptive_max_pool(input, (3, 3))
    # out.backward(out.data.clone().uniform_())

    out = roi_pooling(synthetic_input, regions_of_interest, size=(3, 3))
    out.backward(out.data.clone().uniform_())
