import torch
from torch import nn

class alex(nn.Module):
    def __init__(self):
        super(alex, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # [B, 3, 32, 32] -> [B, 64, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # -> [B, 64, 16, 16]

            nn.Conv2d(64, 192, kernel_size=3, padding=1), # -> [B, 192, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # -> [B, 192, 8, 8]

            nn.Conv2d(192, 384, kernel_size=3, padding=1), # -> [B, 384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # -> [B, 256, 8, 8]
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # -> [B, 256, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),         # -> [B, 256, 4, 4]

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.model(x)

# 测试用例
if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 32)
    net = alex()
    y = net(x)
    print("输出形状：", y.shape)
