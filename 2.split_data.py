import os
import random
import shutil

# ==================== 全局配置 ====================
RAW_DATA_DIR = './ore_images'            # 爬虫保存的原始数据目录
SPLIT_DATA_DIR = './ore_images_split'    # 划分后的训练/测试集目录
TRAIN_RATIO = 0.8                        # 训练集比例（8:2）
RANDOM_SEED = 42                         # 固定随机种子，保证每次划分结果一致

# 只统计这些后缀的图片；统一转小写后比较，避免 Windows 下大小写重复匹配
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}


def list_images(folder):
    """列出文件夹中的有效图片文件（不按后缀反复 glob，避免 Windows 重复统计）"""
    images = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in VALID_EXTS:
            images.append(path)
    return images


def reset_split_dir(split_dir):
    """清理旧的划分结果，避免上次残留文件干扰本次统计"""
    if os.path.exists(split_dir):
        print(f'清理旧的划分结果：{split_dir}')
        shutil.rmtree(split_dir)
    for split_name in ('train', 'test'):
        os.makedirs(os.path.join(split_dir, split_name), exist_ok=True)


def split_train_test_dataset(raw_dir=RAW_DATA_DIR,
                             split_dir=SPLIT_DATA_DIR,
                             train_ratio=TRAIN_RATIO):
    """按类别逐个进行 8:2 划分，保证每类都有训练样本和测试样本"""
    if not os.path.isdir(raw_dir):
        print(f'未找到原始数据目录：{raw_dir}')
        print('请先运行：python 1.data_download.py')
        return

    class_names = sorted(
        name for name in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, name))
    )
    if not class_names:
        print(f'目录 {raw_dir} 下没有任何类别文件夹！')
        return

    print('=' * 10 + f' 开始划分训练集/测试集（{int(train_ratio * 10)}:{10 - int(train_ratio * 10)}）' + '=' * 10)
    random.seed(RANDOM_SEED)
    reset_split_dir(split_dir)

    total_train = 0
    total_test = 0
    for class_name in class_names:
        src_dir = os.path.join(raw_dir, class_name)
        images = sorted(list_images(src_dir))   # 先排序，保证划分结果可复现
        if not images:
            print(f'跳过空类别「{class_name}」')
            continue

        random.shuffle(images)
        if len(images) < 2:
            print(f'警告：「{class_name}」只有 {len(images)} 张图片，无法划分测试集，将全部放入训练集')
            train_images, test_images = images, []
        else:
            split_idx = int(len(images) * train_ratio)
            # 至少给训练集留 1 张，至少给测试集留 1 张
            split_idx = max(1, min(split_idx, len(images) - 1))
            train_images = images[:split_idx]
            test_images = images[split_idx:]

        for split_name, part in (('train', train_images), ('test', test_images)):
            dst_dir = os.path.join(split_dir, split_name, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            for img_path in part:
                shutil.copy2(img_path, os.path.join(dst_dir, os.path.basename(img_path)))

        total_train += len(train_images)
        total_test += len(test_images)
        print(f'「{class_name}」：总{len(images)}张 -> 训练集{len(train_images)}张 | 测试集{len(test_images)}张')

    print()
    print('=' * 10 + ' 数据集划分完成 ' + '=' * 10)
    print(f'整体统计：训练集{total_train}张 | 测试集{total_test}张')
    print(f'训练集目录：{os.path.abspath(os.path.join(split_dir, "train"))}')
    print(f'测试集目录：{os.path.abspath(os.path.join(split_dir, "test"))}')
    print(f'类别顺序（DatasetFolder 按文件夹名排序）：{class_names}')


if __name__ == '__main__':
    split_train_test_dataset()
