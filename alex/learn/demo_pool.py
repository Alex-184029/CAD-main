import torch
import torch.nn as nn

# 定义一个简单的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 平均池化
        self.fc = nn.Linear(32 * 7 * 7, 10)  # 全连接层

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.fc(x)
        return x

# 创建模型
model = SimpleCNN()

# 输入数据
input_data = torch.randn(1, 1, 28, 28)  # 假设输入是28x28的灰度图像

# 前向传播
output = model(input_data)
print(output.shape)  # 输出形状