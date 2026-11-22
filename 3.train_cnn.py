"""教师模型训练：ImageNet 预训练 ResNet18（结构不变，只升级训练/推理策略）

相对旧版的改动（全部不增加新模型）：
  1. 输入 160 -> 224，并改成“保持长宽比 Resize + CenterCrop”，避免把图片压扁；
  2. 训练增强加码：RandomResizedCrop + 随机旋转/翻转 + ColorJitter + RandomErasing；
  3. AdamW 分组学习率（主干小、分类头大）+ 3 epoch warmup + 余弦退火；
  4. 从训练集里再切 15% 做验证集，早停/选模型不再直接看测试集；
  5. 权重 EMA + 验证集 Top-3 checkpoint 平均（模型汤），抑制网图标签噪声带来的过拟合；
  6. 测试时增强（原图 + 水平翻转 + 轻微放大裁剪），多视图概率平均；
  7. 默认在 8.filter_dataset.py 生成的“过滤后测试集”上评估，同时打印完整测试集准确率做对照。
"""
import copy
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')             # 只保存图片，不弹窗口，避免无 GUI 环境报错
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ==================== 全局配置 ====================
SPLIT_DATA_DIR = './ore_images_filtered'   # 8.filter_dataset.py 生成：训练集原样，测试集已人工复核过滤
FULL_TEST_DIR = './ore_images_split/test'   # 原始完整测试集，仅用于打印对照指标
MODEL_DIR = './models'
MODEL_PATH = os.path.join(MODEL_DIR, 'ore_classification_model.pth')
HISTORY_PATH = 'training_history.png'

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
BACKBONE_LR = 2e-4                # 预训练主干用小学习率
HEAD_LR = 1e-3                    # 新初始化的全连接层用大学习率
WEIGHT_DECAY = 0.01               # AdamW 的 L2 正则
LABEL_SMOOTHING = 0.1             # 网图标签有噪声，标签平滑更鲁棒
VAL_RATIO = 0.15                  # 从训练集切 15% 做验证集
EARLY_STOP_PATIENCE = 8
EMA_DECAY = 0.999
USE_WEIGHTED_SAMPLER = False      # 长石样本偏少，可改 True 试试类别均衡
NUM_WORKERS = 4                   # 多进程做数据增强；脚本已用 __main__ 保护
RANDOM_SEED = 42

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# 中文显示设置（Windows 优先使用微软雅黑）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 教师网络：ResNet18 迁移学习 ====================
class OreCNN(nn.Module):
    """教师模型：ImageNet 预训练 ResNet18，只替换最后的全连接层"""

    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# ==================== 数据加载（三层容错） ====================
def is_valid_image(path):
    """第一层校验：先按后缀过滤，再检查文件头是否损坏，不解码像素，速度快"""
    if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS:
        return False
    try:
        with open(path, 'rb') as f:
            Image.open(f).verify()
        return True
    except Exception as e:
        print(f'跳过损坏/无效图片：{path}（{e}）')
        return False


def custom_image_loader(path):
    """自定义加载器：统一转 RGB，兼容灰度图、RGBA 图"""
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        print(f'跳过损坏/无效图片：{path}（{e}）')
        return None


class FilteredDataset(Dataset):
    """过滤 loader 返回 None 的脏样本，并在取样本时应用预处理"""

    def __init__(self, dataset, transform=None, indices=None):
        self.dataset = dataset
        self.transform = transform
        base_indices = list(range(len(dataset))) if indices is None else list(indices)
        self.valid_indices = []
        self.labels = []
        self.paths = []
        for idx in base_indices:
            sample, label = dataset[idx]
            if sample is not None:
                self.valid_indices.append(idx)
                self.labels.append(label)
                self.paths.append(dataset.samples[idx][0])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.valid_indices[index]]
        if image is None:
            # 极少数图片可能偶发读取失败；重试一次，仍失败就用黑图占位，避免整个训练中断
            image, label = self.dataset[self.valid_indices[index]]
        if image is None:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def subset(self, indices, transform=None):
        """按当前数据集的局部下标切子集，避免重复校验图片"""
        return FilteredDataset(
            self.dataset,
            transform=transform if transform is not None else self.transform,
            indices=[self.valid_indices[i] for i in indices],
        )


def build_transforms():
    """训练集做强增强；验证/测试集只做保持长宽比的缩放裁剪 + 标准化"""
    train_transform = transforms.Compose([
        # 直接在原图上随机裁剪缩放，让模型看到不同尺度和构图的矿物纹理
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        # 随机遮挡小块区域，逼迫模型看整体纹理而不是某个水印/标签
        transforms.RandomErasing(p=0.2, value='random'),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(int(round(IMG_SIZE * 8 / 7))),  # 短边放大到 256
        transforms.CenterCrop(IMG_SIZE),                  # 再中心裁剪 224，不压扁图片
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    return train_transform, eval_transform


def build_dataset_dir(root):
    """构建 DatasetFolder，返回 (raw_dataset, class_names)"""
    raw = datasets.DatasetFolder(
        root,
        loader=custom_image_loader,
        is_valid_file=is_valid_image,
    )
    return raw, list(raw.classes)


def stratified_split_indices(labels, val_ratio, seed):
    """按类别分层切分训练/验证集，保证每个类别在验证集中都有样本"""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i, label in enumerate(labels):
        by_class[label].append(i)
    train_idx, val_idx = [], []
    for label, indices in by_class.items():
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_ratio)))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])
    return train_idx, val_idx


def build_dataloaders(data_dir=SPLIT_DATA_DIR):
    """构建训练/验证/测试 DataLoader，并返回类别名列表"""
    train_transform, eval_transform = build_transforms()

    train_raw, train_classes = build_dataset_dir(os.path.join(data_dir, 'train'))
    test_raw, test_classes = build_dataset_dir(os.path.join(data_dir, 'test'))
    if train_classes != test_classes:
        print('训练集与测试集的类别不一致，请重新运行：python 2.split_data.py / 8.filter_dataset.py')
        sys.exit(1)

    class_names = train_classes
    full_train = FilteredDataset(train_raw, transform=train_transform)
    test_dataset = FilteredDataset(test_raw, transform=eval_transform)

    train_idx, val_idx = stratified_split_indices(full_train.labels, VAL_RATIO, RANDOM_SEED)
    train_dataset = full_train.subset(train_idx, transform=train_transform)
    val_dataset = full_train.subset(val_idx, transform=eval_transform)

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print('有效图片数量为 0，请检查数据目录或重新采集数据！')
        sys.exit(1)

    sampler = None
    if USE_WEIGHTED_SAMPLER:
        counts = Counter(train_dataset.labels)
        # 用 1/sqrt(类别样本数) 做温和的类别均衡，避免完全均衡导致整体准确率下降
        weights = [1.0 / (counts[label] ** 0.5) for label in train_dataset.labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=sampler is None,
        sampler=sampler, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0, prefetch_factor=4 if NUM_WORKERS > 0 else None)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0, prefetch_factor=4 if NUM_WORKERS > 0 else None)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0, prefetch_factor=4 if NUM_WORKERS > 0 else None)

    print(f'训练集：{len(train_dataset)} 张 | 验证集：{len(val_dataset)} 张 | 测试集：{len(test_dataset)} 张')
    print(f'类别（共 {len(class_names)} 类）：{class_names}')

    # 检查有没有“零有效样本”的类别（该类别永远不会被训练到）
    train_counts = Counter(train_dataset.labels)
    for class_idx, class_name in enumerate(class_names):
        if train_counts.get(class_idx, 0) == 0:
            print(f'警告：训练集类别「{class_name}」没有有效图片，模型将无法识别该类')

    return train_loader, val_loader, test_loader, class_names


# ==================== EMA / 训练与评估 ====================
class ModelEMA:
    """指数滑动平均：把训练过程中多个 step 的权重平滑起来，通常比最后一轮更稳"""

    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        decay = self.decay
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(model_state[key].detach(), alpha=1 - decay)
            else:
                ema_value.copy_(model_state[key])


def train_one_epoch(model, loader, criterion, optimizer, scaler, ema, device, epoch, epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = device.type == 'cuda'

    progress = tqdm(loader, desc=f'Epoch {epoch}/{epochs} [Train]', ncols=100)
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast(device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)   # 1. 清空上批梯度
        scaler.scale(loss).backward()           # 2. 反向传播求梯度（AMP 缩放）
        scaler.step(optimizer)                  # 3. 更新参数
        scaler.update()
        ema.update(model)

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(loss=f'{total_loss / total:.4f}',
                             acc=f'{correct / total:.4f}')

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total, correct, total


class TTADataset(Dataset):
    """测试时增强：同一张图生成 4 个视图（原图/翻转/放大/放大+翻转）"""

    def __init__(self, dataset, img_size=IMG_SIZE):
        self.dataset = dataset
        resize = int(round(img_size * 8 / 7))
        zoom = int(round(img_size * 9 / 7))
        base = [
            transforms.Resize(resize),          # 单整数：短边缩放，保持长宽比
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
        self.transforms = [
            transforms.Compose(base),
            transforms.Compose([transforms.Resize(resize),
                                transforms.CenterCrop(img_size),
                                transforms.RandomHorizontalFlip(p=1.0),
                                transforms.ToTensor(),
                                transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]),
            transforms.Compose([transforms.Resize(zoom),
                                transforms.CenterCrop(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]),
            transforms.Compose([transforms.Resize(zoom),
                                transforms.CenterCrop(img_size),
                                transforms.RandomHorizontalFlip(p=1.0),
                                transforms.ToTensor(),
                                transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]),
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset.dataset[self.dataset.valid_indices[index]]
        if image is None:
            image, label = self.dataset.dataset[self.dataset.valid_indices[index]]
        if image is None:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        return [tf(image) for tf in self.transforms], label


@torch.no_grad()
def evaluate_tta(model, dataset, device, batch_size=BATCH_SIZE):
    """多视图 TTA 评估，返回 (loss, acc, correct, total, labels, preds)"""
    loader = DataLoader(TTADataset(dataset), batch_size=batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    model.eval()
    correct = 0
    total = 0
    all_labels, all_preds = [], []
    for views, labels in loader:
        probs = None
        for view in views:
            p = F.softmax(model(view.to(device)), dim=1)
            probs = p if probs is None else probs + p
        preds = probs.argmax(dim=1).cpu()
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 0.0, correct / total, correct, total, all_labels, all_preds


def average_state_dicts(states):
    """模型汤：把多个 checkpoint 的同名浮点权重取平均"""
    avg = copy.deepcopy(states[0])
    for key in avg:
        if avg[key].dtype.is_floating_point:
            avg[key] = sum(state[key] for state in states) / len(states)
    return avg


# ==================== 可视化 ====================
def plot_confusion_matrix(labels, preds, class_names, save_path='confusion_matrix.png'):
    """根据给定的真实/预测标签画混淆矩阵"""
    num_classes = len(class_names)
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true_label, pred_label in zip(labels, preds):
        matrix[true_label][pred_label] += 1

    print('混淆矩阵（行=真实，列=预测）：')
    print('真实\\预测'.ljust(10) + ''.join(f'{name:<10}' for name in class_names))
    for i, name in enumerate(class_names):
        row_total = max(1, sum(matrix[i]))
        row = ''.join(f'{value:<10}' for value in matrix[i])
        print(f'{name:<10}{row}该类准确率 {matrix[i][i] / row_total:.2f}')

    try:
        max_value = max(max(row) for row in matrix)
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap='Blues')
        ax.set_xticks(range(num_classes), class_names)
        ax.set_yticks(range(num_classes), class_names)
        ax.set_xlabel('预测类别')
        ax.set_ylabel('真实类别')
        ax.set_title('测试集混淆矩阵（TTA）')
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, matrix[i][j], ha='center', va='center',
                        color='white' if matrix[i][j] > max_value / 2 else 'black')
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'混淆矩阵已保存：{save_path}')
    except Exception as e:
        print(f'混淆矩阵绘图失败：{e}')
    return matrix


def plot_training_history(history, save_path=HISTORY_PATH):
    """保存训练曲线；失败时打印文本表格兜底"""
    try:
        epochs = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        axes[0].plot(epochs, history['train_acc'], 'o-', label='训练准确率')
        axes[0].plot(epochs, history['val_acc_ema'], 's-', label='验证准确率（EMA）')
        axes[0].set_title('准确率曲线')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(epochs, history['train_loss'], 'o-', label='训练损失')
        axes[1].plot(epochs, history['val_loss_ema'], 's-', label='验证损失（EMA）')
        axes[1].set_title('损失曲线')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'训练曲线已保存：{save_path}')
    except Exception as e:
        print(f'训练曲线保存失败：{e}')
        print('Epoch | TrainLoss | TrainAcc | ValLoss | ValAcc')
        for i in range(len(history['train_loss'])):
            print(f'{i + 1:5d} | {history["train_loss"][i]:9.4f} | '
                  f'{history["train_acc"][i]:8.4f} | '
                  f'{history["val_loss_ema"][i]:8.4f} | {history["val_acc_ema"][i]:7.4f}')


def find_sample_image(test_dir, class_names):
    """从测试集中找一张图片用于预测可视化"""
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for name in sorted(os.listdir(class_dir)):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                return os.path.join(class_dir, name), class_name
    return None, None


def save_prediction_example(model, image_path, true_class, class_names, device):
    """预测一张测试图片，并把结果画到 prediction_<类别>.png"""
    try:
        image = Image.open(image_path).convert('RGB')
        # 与 evaluate_tta 一致的 4 视图
        resize = int(round(IMG_SIZE * 8 / 7))
        zoom = int(round(IMG_SIZE * 9 / 7))
        tf_list = [
            transforms.Compose([transforms.Resize(resize), transforms.CenterCrop(IMG_SIZE),
                                transforms.ToTensor(), transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]),
            transforms.Compose([transforms.Resize(resize), transforms.CenterCrop(IMG_SIZE),
                                transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(),
                                transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]),
            transforms.Compose([transforms.Resize(zoom), transforms.CenterCrop(IMG_SIZE),
                                transforms.ToTensor(), transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]),
            transforms.Compose([transforms.Resize(zoom), transforms.CenterCrop(IMG_SIZE),
                                transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(),
                                transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]),
        ]
        model.eval()
        with torch.no_grad():
            probs = None
            for tf in tf_list:
                tensor = tf(image).unsqueeze(0).to(device)
                p = F.softmax(model(tensor), dim=1)
                probs = p if probs is None else probs + p
            probs = probs / len(tf_list)
            confidence, pred_idx = torch.max(probs, 1)

        pred_class = class_names[pred_idx.item()]
        confidence = confidence.item() * 100

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.imshow(image)
        ax.set_title(f'真实类别：{true_class}\n'
                     f'预测：{pred_class}（TTA 置信度 {confidence:.2f}%）')
        ax.axis('off')

        save_path = f'prediction_{pred_class}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'预测示例已保存：{save_path}（真实：{true_class}，预测：{pred_class}）')
    except Exception as e:
        print(f'预测示例生成失败：{e}')


# ==================== 主流程 ====================
def main():
    train_dir = os.path.join(SPLIT_DATA_DIR, 'train')
    test_dir = os.path.join(SPLIT_DATA_DIR, 'test')

    if not os.path.isdir(SPLIT_DATA_DIR):
        print(f'未找到过滤后的数据集目录：{SPLIT_DATA_DIR}')
        print('请先运行：python 8.filter_dataset.py')
        return
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        print('训练集或测试集目录不存在，请重新运行：python 8.filter_dataset.py')
        return

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader, class_names = build_dataloaders()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用计算设备：{device}')

    model = OreCNN(num_classes=len(class_names), pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型：ResNet18（ImageNet 预训练）| 参数量：{total_params:,} | 输入：{IMG_SIZE}x{IMG_SIZE}')

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    head_params = list(model.backbone.fc.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': BACKBONE_LR},
        {'params': head_params, 'lr': HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    warmup_epochs = 3

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, EPOCHS - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress)) * 0.99 + 0.01

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    ema = ModelEMA(model, decay=EMA_DECAY)

    history = {'train_loss': [], 'train_acc': [],
               'val_loss_raw': [], 'val_acc_raw': [],
               'val_loss_ema': [], 'val_acc_ema': []}

    print('开始训练模型...')
    start_time = time.time()

    best_raw = (0.0, -1, None)      # (acc, epoch, state)
    best_ema = (0.0, -1, None)
    top_ema_states = []             # 验证集表现最好的 3 个 EMA 权重
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, ema, device, epoch, EPOCHS)
        val_loss_raw, val_acc_raw, _, _ = evaluate(model, val_loader, criterion, device)
        val_loss_ema, val_acc_ema, _, _ = evaluate(ema.ema, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss_raw'].append(val_loss_raw)
        history['val_acc_raw'].append(val_acc_raw)
        history['val_loss_ema'].append(val_loss_ema)
        history['val_acc_ema'].append(val_acc_ema)

        improved = False
        if val_acc_raw > best_raw[0]:
            best_raw = (val_acc_raw, epoch, copy.deepcopy(model.state_dict()))
            improved = True
        if val_acc_ema > best_ema[0]:
            best_ema = (val_acc_ema, epoch, copy.deepcopy(ema.ema.state_dict()))
            top_ema_states.append((val_acc_ema, epoch, copy.deepcopy(ema.ema.state_dict())))
            top_ema_states.sort(key=lambda item: -item[0])
            top_ema_states = top_ema_states[:3]
            improved = True
        no_improve = 0 if improved else no_improve + 1

        print(f'Epoch {epoch} 总结：')
        print(f'训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}')
        print(f'验证损失: {val_loss_ema:.4f} | 验证准确率(EMA): {val_acc_ema:.4f}'
              f'{"  <- 新最优" if improved else ""}')
        print(f'学习率: {scheduler.get_last_lr()[0]:.6f}')

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f'连续 {EARLY_STOP_PATIENCE} 个 epoch 没有提升，提前停止训练')
            break

    # 候选模型：验证集最好的原始权重、最好的 EMA、Top-3 EMA 平均（模型汤）
    candidates = {}
    if best_raw[2] is not None:
        raw_model = copy.deepcopy(model)
        raw_model.load_state_dict(best_raw[2])
        candidates['best_raw'] = raw_model
    if best_ema[2] is not None:
        ema_model = copy.deepcopy(ema.ema)
        ema_model.load_state_dict(best_ema[2])
        candidates['best_ema'] = ema_model
    if len(top_ema_states) >= 2:
        soup_model = copy.deepcopy(ema.ema)
        soup_model.load_state_dict(average_state_dicts([state for _, _, state in top_ema_states]))
        candidates['soup'] = soup_model
    if not candidates:                 # 兜底：验证集一直没提升时用最后一轮权重
        candidates['last'] = model

    # 用验证集选最终模型，测试集只在最后看一次
    val_scores = {}
    for name, candidate in candidates.items():
        _, acc, _, _ = evaluate(candidate, val_loader, criterion, device)
        val_scores[name] = acc
        print(f'候选模型 {name}：验证准确率 {acc:.4f}')
    final_name = max(val_scores, key=val_scores.get) if val_scores else 'best_ema'
    final_model = candidates[final_name]
    best_epoch = {'best_raw': best_raw[1], 'best_ema': best_ema[1]}.get(
        final_name, top_ema_states[0][1] if top_ema_states else -1)

    # 过滤后测试集（主指标）+ 完整原始测试集（对照）
    final_loss, final_acc, final_correct, final_total, labels, preds = evaluate_tta(
        final_model, test_loader.dataset, device)
    elapsed = (time.time() - start_time) / 60
    print(f'训练完成，总耗时：{elapsed:.1f} 分钟')
    print(f'最终模型：{final_name} | 最佳 Epoch：{best_epoch}')
    print(f'过滤后测试集准确率（TTA）：{final_acc:.4f} ({final_correct}/{final_total})')

    full_acc = None
    if os.path.isdir(FULL_TEST_DIR):
        full_raw, _ = build_dataset_dir(FULL_TEST_DIR)
        full_dataset = FilteredDataset(full_raw, transform=build_transforms()[1])
        _, full_acc, fc, ft, _, _ = evaluate_tta(final_model, full_dataset, device)
        print(f'完整原始测试集准确率（TTA，仅供参考，含已剔除的脏样本）：'
              f'{full_acc:.4f} ({fc}/{ft})')

    # 保存模型（类别名必须一起保存，推理时靠它把索引翻译成中文类别）
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        backup_path = os.path.join(MODEL_DIR, 'ore_classification_model_backup.pth')
        if not os.path.exists(backup_path):
            shutil.copy2(MODEL_PATH, backup_path)
            print(f'已备份旧模型：{backup_path}')
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_name': 'resnet18',
        'class_names': class_names,
        'final_accuracy': final_acc,
        'full_test_accuracy': full_acc,
        'best_epoch': best_epoch,
        'img_size': IMG_SIZE,
        'epochs': len(history['train_loss']),
        'history': history,
        'eval_tta': True,
        'test_filtered': True,
    }, MODEL_PATH)
    print(f'模型已保存为：{MODEL_PATH}')

    plot_training_history(history, HISTORY_PATH)
    plot_confusion_matrix(labels, preds, class_names)

    # 用测试集里的一张图片演示预测，并保存可视化结果
    sample_path, true_class = find_sample_image(test_dir, class_names)
    if sample_path:
        save_prediction_example(final_model, sample_path, true_class, class_names, device)


if __name__ == '__main__':
    main()
