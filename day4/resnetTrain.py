import os
import time
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 设置 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置数据路径
data_dir = r"D:\Python代码\Python‘s Project\实习\day3\dataset"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# 加载数据集
train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
val_data = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# 加载预训练模型 + 替换分类头
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))  # 修改输出层为你的类别数
model.to(device)

# 损失函数 + 优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 日志与模型保存目录
os.makedirs("model_save", exist_ok=True)
writer = SummaryWriter("logs_train")

# 训练配置
num_epochs = 10
best_acc = 0.0

for epoch in range(num_epochs):
    print(f"----- 第 {epoch+1} 轮训练开始 -----")
    model.train()
    train_loss = 0.0
    loop = tqdm(train_loader, desc=f"[Train Epoch {epoch+1}]")

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    writer.add_scalar("Train/Loss", avg_train_loss, epoch)

    # 验证
    model.eval()
    total_correct = 0
    total_val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            total_val_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = total_correct / len(val_data)

    print(f"验证 Loss: {avg_val_loss:.4f}，准确率: {val_accuracy:.4f}")
    writer.add_scalar("Val/Loss", avg_val_loss, epoch)
    writer.add_scalar("Val/Accuracy", val_accuracy, epoch)

    # 保存最优模型
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(model.state_dict(), "model_save/best_model.pth")
        print("✅ 保存当前最优模型")

    # 每轮都保存
    torch.save(model.state_dict(), f"model_save/resnet_epoch{epoch+1}.pth")

writer.close()
print("🎉 所有训练完成！")
