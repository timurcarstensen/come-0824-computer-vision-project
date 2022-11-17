# standard library imports
from typing import Optional
import os

# 3rd party imports
import torch.nn as nn
import torch

# local imports (i.e. our own code)
from src.modules.detection import DetectionModule
from src.modules.roi_pooling import roi_pooling_ims


class RecognitionModule(nn.Module):
    def __init__(
        self,
        num_points: Optional[int] = 4,
        pretrained_model_path: Optional[str] = None,
        prov_num: Optional[int] = 38,
        alpha_num: Optional[int] = 25,
        ad_num: Optional[int] = 35,
    ):
        super(RecognitionModule, self).__init__()
        self.load_detection_module(path=pretrained_model_path, num_points=num_points)
        self.classifier1 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, prov_num),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, alpha_num),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, ad_num),
        )
        self.classifier4 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, ad_num),
        )
        self.classifier5 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, ad_num),
        )
        self.classifier6 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, ad_num),
        )
        self.classifier7 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, ad_num),
        )

    def load_detection_module(self, path: str, num_points: int):
        self.detection_module = DetectionModule(num_points)
        if torch.cuda.is_available():
            self.detection_module = torch.nn.DataParallel(
                self.detection_module, device_ids=range(torch.cuda.device_count())
            )
        if path:
            path = f"{os.getenv('MODEL_DIR')}{path}"
            print("Loading detection module from: {}".format(path))
            self.detection_module.load_state_dict(torch.load(f=path))
            print("Detection module loaded successfully.")

    def forward(self, x):
        x0 = self.detection_module.module.features[0](x)
        _x1 = self.detection_module.module.features[1](x0)
        x2 = self.detection_module.module.features[2](_x1)
        _x3 = self.detection_module.module.features[3](x2)
        x4 = self.detection_module.module.features[4](_x3)
        _x5 = self.detection_module.module.features[5](x4)

        x6 = self.detection_module.module.features[6](_x5)
        x7 = self.detection_module.module.features[7](x6)
        x8 = self.detection_module.module.features[8](x7)
        x9 = self.detection_module.module.features[9](x8)
        x9 = x9.view(x9.size(0), -1)
        box_loc = self.detection_module.module.classifier(x9)

        h1, w1 = _x1.data.size()[2], _x1.data.size()[3]
        p1 = torch.tensor(
            data=[[w1, 0, 0, 0], [0, h1, 0, 0], [0, 0, w1, 0], [0, 0, 0, h1]],
            requires_grad=False,
            dtype=torch.float32,
        ).cuda(device=torch.device("cuda") if torch.device.type == "cuda" else None)

        h2, w2 = _x3.data.size()[2], _x3.data.size()[3]

        p2 = torch.tensor(
            data=[[w2, 0, 0, 0], [0, h2, 0, 0], [0, 0, w2, 0], [0, 0, 0, h2]],
            requires_grad=False,
            dtype=torch.float32,
        ).cuda(device=torch.device("cuda") if torch.device.type == "cuda" else None)

        h3, w3 = _x5.data.size()[2], _x5.data.size()[3]
        p3 = torch.tensor(
            data=[[w3, 0, 0, 0], [0, h3, 0, 0], [0, 0, w3, 0], [0, 0, 0, h3]],
            requires_grad=False,
            dtype=torch.float32,
        ).cuda(device=torch.device("cuda") if torch.device.type == "cuda" else None)

        # x, y, w, h --> x1, y1, x2, y2
        if not box_loc.data.size()[1] == 4:
            raise ValueError("box_loc.data.size()[1] != 4")

        postfix = torch.tensor(
            data=[[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            requires_grad=False,
            dtype=torch.float32,
        ).cuda(device=torch.device("cuda") if torch.device.type == "cuda" else None)

        box_new = box_loc.mm(postfix).clamp(min=0, max=1)

        roi1 = roi_pooling_ims(t=_x1, rois=box_new.mm(p1), size=(8, 16))
        roi2 = roi_pooling_ims(t=_x3, rois=box_new.mm(p2), size=(8, 16))
        roi3 = roi_pooling_ims(t=_x5, rois=box_new.mm(p3), size=(8, 16))
        rois = torch.cat((roi1, roi2, roi3), 1)

        _rois = rois.view(rois.size(0), -1)

        y0 = self.classifier1(_rois)
        y1 = self.classifier2(_rois)
        y2 = self.classifier3(_rois)
        y3 = self.classifier4(_rois)
        y4 = self.classifier5(_rois)
        y5 = self.classifier6(_rois)
        y6 = self.classifier7(_rois)
        return box_loc, [y0, y1, y2, y3, y4, y5, y6]
