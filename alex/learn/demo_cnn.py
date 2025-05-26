import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 卷积层1 + BN + ReLU + 池化
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 输入通道3，输出通道16
        self.bn1 = nn.BatchNorm2d(16)  # BN层，通道数与conv1输出一致
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层

        # 卷积层2 + BN + ReLU + 池化
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # 全连接层
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # 假设输入图像大小为32x32，经过两次池化后为8x8
        self.bn3 = nn.BatchNorm1d(128)  # 全连接后的BN
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # 全连接层
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

# 测试网络
if __name__ == "__main__":
    model = SimpleCNN(num_classes=10)
    dummy_input = torch.randn(4, 3, 32, 32)  # 4张3通道32x32的图片
    output = model(dummy_input)
    print("输出形状:", output.shape)  # 应为 [4, 10]