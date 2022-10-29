import torch
import torch.nn as nn


class ResNetClf(nn.Module):
    def __init__(self, params):
        """
        params: in_channels, num_cls, resnet_name, pretrained=False
        """
        super(ResNetClf, self).__init__()
        self.params = params
        self.model = torch.hub.load("pytorch/vision:v0.10.0", self.params["resnet_name"], self.params["pretrained"])
        self.model.eval()
        self.pre_conv = nn.Conv2d(in_channels=self.params["in_channels"], out_channels=3, kernel_size=3, padding=1)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.params["num_cls"])

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.pre_conv(x)
        pred = self.model(x)

        # (B, num_cls)
        return pred
