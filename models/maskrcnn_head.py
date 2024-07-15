import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskRCNNSegmentationHead(nn.Module):
    def __init__(self, in_channels_list, channel=256, num_classes=19, dropout_rate=0.3):
        super(MaskRCNNSegmentationHead, self).__init__()
        self.fpn_lateral = nn.ModuleList([
            nn.Conv2d(in_channels, channel, kernel_size=1, stride=1, padding=0)
            for in_channels in in_channels_list
        ])
        self.fpn_output = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            )
            for _ in range(len(in_channels_list))
        ])

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.mask_pred = nn.Conv2d(channel, num_classes, kernel_size=1, stride=1)

    def forward(self, features):
        x_list = []
        for i, feature in enumerate(features):
            lateral_conv = self.fpn_lateral[i](feature)
            output_conv = self.fpn_output[i](lateral_conv)
            x_list.append(output_conv)

        x = sum([F.interpolate(x, size=x_list[-1].shape[-2:], mode='bilinear', align_corners=False) for x in x_list])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        mask_pred = self.mask_pred(x)
        return mask_pred


# if __name__ == '__main__':
#     x = torch.randn(2, 3, 224, 224)
#     model = MaskRCNNSegmentationHead([160, 320, 640, 640], 128)
#     x1 = torch.randn(2, 160, 28, 28)
#     x2 = torch.randn(2, 320, 14, 14)
#     x3 = torch.randn(2, 640, 7, 7)
#     x4 = torch.randn(2, 640, 7, 7)
#     y = model([x1, x2, x3, x4])
#     y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
#     print(y.shape)