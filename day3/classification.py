import os
import shutil
import random

train_dir = r"dataset/train"
val_dir = r"dataset/val"

val_ratio = 0.3  # 30% 作为验证集

for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        # 创建 val 对应类别目录
        val_class_path = os.path.join(val_dir, class_name)
        os.makedirs(val_class_path, exist_ok=True)

        # 获取并打乱图像列表
        images = os.listdir(class_path)
        random.shuffle(images)

        val_count = int(len(images) * val_ratio)
        val_images = images[:val_count]

        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_class_path, img)
            shutil.move(src, dst)

print("✅ 已成功从 train 中移动部分数据到 val 文件夹，用于验证。")
