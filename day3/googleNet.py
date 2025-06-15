import torch
import torch.nn as nn
from torchvision.models import googlenet

class GoogleNet10(nn.Module):
    def __init__(self):
        super(GoogleNet10, self).__init__()
        # 加载预训练模型并修改最后的分类层
        self.model = googlenet(pretrained=False, aux_logits=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224).cuda()
    net = GoogleNet10().cuda()
    y = net(x)
    print(y.shape)
