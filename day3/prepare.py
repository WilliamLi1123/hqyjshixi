import os


def create_txt_file(root_dir, txt_filename):
    """
    遍历root_dir下的所有类别子文件夹，并生成一个txt文件
    每行格式为：图片完整路径 类别编号（从0开始）
    """
    # 检查路径是否存在
    if not os.path.exists(root_dir):
        print(f"❌ 路径不存在: {root_dir}")
        return

    # 获取所有类别文件夹，按名字排序以保证类别编号一致
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    with open(txt_filename, 'w', encoding='utf-8') as f:
        for class_name in class_names:
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path):
                    line = f"{img_path} {class_to_idx[class_name]}\n"
                    f.write(line)

    print(f"✅ 已成功生成标签文件: {txt_filename}")


# 示例调用
create_txt_file(r'D:\Python代码\Python‘s Project\实习\day3\dataset\train', 'train.txt')
create_txt_file(r'D:\Python代码\Python‘s Project\实习\day3\dataset\val', 'val.txt')
