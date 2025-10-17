# import torch.nn as nn
# import torch.nn.functional as F


# class ArmorCNN(nn.Module):
#     def __init__(self, num_classes=5):
#         super(ArmorCNN, self).__init__()

#         # 针对20×28小尺寸图像的CNN架构
#         # 输入: 3×20×28

#         # 第一个卷积块
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32×20×28
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 32×20×28
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),  # 32×10×14
#             nn.Dropout(0.25),
#         )

#         # 第二个卷积块
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64×10×14
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64×10×14
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),  # 64×5×7
#             nn.Dropout(0.25),
#         )

#         # 第三个卷积块
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128×5×7
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128×5×7
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((3, 4)),  # 128×3×4
#             nn.Dropout(0.25),
#         )

#         # 全连接层
#         self.classifier = nn.Sequential(
#             nn.Linear(128 * 3 * 4, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes),
#         )

#         # 权重初始化
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)

#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)

#         return x


# # 更轻量级的模型（可选）
# class ArmorCNNLite(nn.Module):
#     def __init__(self, num_classes=5):
#         super(ArmorCNNLite, self).__init__()

#         # 输入: 3×20×28
#         self.conv_layers = nn.Sequential(
#             # 第一层
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 16×20×28
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),  # 16×10×14
#             # 第二层
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32×10×14
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),  # 32×5×7
#             # 第三层
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64×5×7
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((2, 3)),  # 64×2×3
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(64 * 2 * 3, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes),
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x


# def create_model(num_classes=5, model_type="standard"):
#     if model_type == "lite":
#         return ArmorCNNLite(num_classes=num_classes)
#     else:
#        return ArmorCNN(num_classes=num_classes)
import torch.nn as nn
import torch.nn.functional as F


class ArmorCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ArmorCNN, self).__init__()

        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32×10×14
            nn.Dropout(0.25),
        )

        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64×10×14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64×10×14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64×5×7
            nn.Dropout(0.25),
        )

        # 第三个卷积块 - 修改这里！
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128×5×7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128×5×7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 替换 AdaptiveAvgPool2d 为 AvgPool2d
            nn.AvgPool2d(kernel_size=2, stride=2),  # 128×2×3
            nn.Dropout(0.25),
        )

        # 全连接层 - 更新输入尺寸
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 3, 256),  # 更新为 128*2*3
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


# 更轻量级的模型（可选）
class ArmorCNNLite(nn.Module):
    def __init__(self, num_classes=5):
        super(ArmorCNNLite, self).__init__()

        # 输入: 3×20×28
        self.conv_layers = nn.Sequential(
            # 第一层
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 16×20×28
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16×10×14
            # 第二层
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32×10×14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32×5×7
            # 第三层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64×5×7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 64×2×3
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 2 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_model(num_classes=5, model_type="standard"):
    if model_type == "lite":
        return ArmorCNNLite(num_classes=num_classes)
    else:
        return ArmorCNN(num_classes=num_classes)
