import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==== 参数设置 ====
MODEL_NAME = "mobilenet"
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 0.001
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 数据预处理 ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 这里用 FakeData 模拟数据集，你可以换成 CIFAR10、ImageFolder 等真实数据
train_data = datasets.FakeData(transform=transform)
val_data = datasets.FakeData(transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# ==== 构建模型 ====
print(f"🚀 正在构建模型：{MODEL_NAME}")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model = model.to(DEVICE)

# ==== 损失函数与优化器 ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==== 训练函数 ====
def train():
    print(f"\n📚 开始训练模型：{MODEL_NAME}，总共 {NUM_EPOCHS} 轮")
    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"📦 Epoch [{epoch+1}/{NUM_EPOCHS}]")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=correct/total)

        acc = correct / total
        print(f"✅ Epoch {epoch+1}: Loss={running_loss:.4f}, Acc={acc:.4f}")

        # 保存最优模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{MODEL_NAME}_best.pth")
            print(f"💾 最优模型已保存：{MODEL_NAME}_best.pth")

# ==== 验证函数 ====
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"\n🎯 验证准确率: {correct / total:.4f}")

# ==== 主程序入口 ====
if __name__ == '__main__':
    train()
    evaluate()
