import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# ==== 可选模型导入（只打开你需要的模型） ====
from torchvision.models import googlenet, GoogLeNet_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ==== 模型选择函数 ====
def get_model(name: str, num_classes: int):
    if name == "googlenet":
        model = googlenet(weights=None, aux_logits=False, init_weights=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "mobilenet":
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    # elif name == "moganet":
    #     model = moganet_s(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model

# ==== 超参数设置 ====
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 0.001
NUM_CLASSES = 10
MODEL_NAME = "googlenet"  # 改成 mobilenet、moganet 即可
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 数据预处理与加载 ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.FakeData(
    size=200, image_size=(3, 224, 224), num_classes=NUM_CLASSES, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = torchvision.datasets.FakeData(
    size=50, image_size=(3, 224, 224), num_classes=NUM_CLASSES, transform=transform
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ==== 模型准备 ====
print(f"\n🚀 正在构建模型：{MODEL_NAME}")
model = get_model(MODEL_NAME, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ==== 训练函数 ====
def train():
    print("\n📚 开始训练...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        loop = tqdm(train_loader, desc=f"📦 Epoch [{epoch+1}/{NUM_EPOCHS}]", leave=False)
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 指标记录
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        print(f"✅ Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Acc={100*correct/total:.2f}%")

# ==== 验证函数 ====
def evaluate():
    print("\n🔍 正在验证模型性能...")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        loop = tqdm(val_loader, desc="🔎 Validating", leave=False)
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"🎯 验证准确率: {100 * correct / total:.2f}%")

# ==== 主程序 ====
if __name__ == '__main__':
    start_time = time.time()
    train()
    evaluate()
    print(f"\n⏱️ 总用时：{time.time() - start_time:.2f} 秒")
