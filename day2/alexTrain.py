import time
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # ✅ 用于添加进度条
from alex import alex  # ✅ 导入你的alex模型

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset_chen",
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root="../dataset_chen",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度：{train_data_size}")
print(f"测试数据集的长度：{test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 创建网络模型
alexnet = alex()
if torch.cuda.is_available():
    alexnet = alexnet.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(alexnet.parameters(), lr=learning_rate)

# 设置训练网络参数
total_train_step = 0
total_test_step = 0
epoch = 50

# TensorBoard 日志记录
writer = SummaryWriter("../logs_alex_train")

# 记录开始时间
start_time = time.time()

for i in range(epoch):
    print(f"----- 第{i + 1}轮训练开始 -----")
    alexnet.train()
    loop = tqdm(train_loader, desc=f"Train Epoch {i + 1}")
    for data in loop:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

        outputs = alexnet(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        loop.set_postfix(loss=loss.item())

        if total_train_step % 500 == 0:
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试部分
    alexnet.eval()
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        loop = tqdm(test_loader, desc=f"Test Epoch {i + 1}")
        for data in loop:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            outputs = alexnet(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            total_accuracy += (outputs.argmax(1) == targets).sum().item()

    print(f"整体测试集上的Loss：{total_test_loss}")
    print(f"整体测试集上的准确率：{total_accuracy / test_data_size:.4f}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    # 保存模型
    torch.save(alexnet, f"model_save/alex_{i}.pth")
    print("模型已保存")

writer.close()

end_time = time.time()
print(f"训练总耗时：{(end_time - start_time):.2f}秒")
