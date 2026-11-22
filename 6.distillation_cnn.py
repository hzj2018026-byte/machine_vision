import os
import sys
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ==================== 全局配置 ====================
SPLIT_DATA_DIR = './ore_images_filtered'   # 8.filter_dataset.py 生成：测试集已过滤
MODEL_DIR = './models'
TEACHER_MODEL_PATH = os.path.join(MODEL_DIR, 'ore_classification_model.pth')
STUDENT_MODEL_PATH = os.path.join(MODEL_DIR, 'distilled_ore_model.pth')

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
NUM_WORKERS = 0
RANDOM_SEED = 42

# 蒸馏超参数
TEMPERATURE = 2.0      # 温度 T：越大分布越平滑，经验范围 2~20
ALPHA = 0.7            # 软损失权重，硬损失权重为 1-alpha

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


# ==================== 教师网络：ResNet18 迁移学习 ====================
class OreCNN(nn.Module):
    """教师模型：与 3.train_cnn.py 完全一致的 ImageNet 预训练 ResNet18"""

    def __init__(self, num_classes=5, pretrained=False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# ==================== 学生网络：手写 4 层 CNN ====================
class StudentCNN(nn.Module):
    """蒸馏学生模型：结构小巧，自适应池化保证不同输入尺寸都能用"""

    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ==================== 数据加载 ====================
def is_valid_image(path):
    # 新版 torchvision 不允许同时传 extensions 和 is_valid_file，后缀过滤在这里完成
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
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        print(f'跳过损坏/无效图片：{path}（{e}）')
        return None


class FilteredDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.valid_indices = []
        for idx in range(len(dataset)):
            sample, _ = dataset[idx]
            if sample is not None:
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.valid_indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(int(round(IMG_SIZE * 8 / 7))),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    return train_transform, test_transform


def build_dataloaders(train_dir, test_dir):
    train_transform, test_transform = build_transforms()

    train_raw = datasets.DatasetFolder(
        train_dir,
        loader=custom_image_loader,
        is_valid_file=is_valid_image,
    )
    test_raw = datasets.DatasetFolder(
        test_dir,
        loader=custom_image_loader,
        is_valid_file=is_valid_image,
    )

    if train_raw.classes != test_raw.classes:
        print('训练集与测试集的类别不一致，请重新运行：python 2.split_data.py')
        sys.exit(1)

    class_names = list(train_raw.classes)
    train_dataset = FilteredDataset(train_raw, transform=train_transform)
    test_dataset = FilteredDataset(test_raw, transform=test_transform)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print('有效图片数量为 0，请检查数据目录或重新采集数据！')
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False)

    print(f'训练集：{len(train_dataset)} 张 | 测试集：{len(test_dataset)} 张')
    print(f'类别（共 {len(class_names)} 类）：{class_names}')

    train_counts = Counter(train_raw.targets[i] for i in train_dataset.valid_indices)
    for class_idx, class_name in enumerate(class_names):
        if train_counts.get(class_idx, 0) == 0:
            print(f'警告：训练集类别「{class_name}」没有有效图片')

    return train_loader, test_loader, class_names


# ==================== 教师模型加载 ====================
def load_teacher_model(path, device):
    if not os.path.exists(path):
        print(f'未找到教师模型：{path}')
        print('请先运行：python 3.train_cnn.py')
        return None, None

    checkpoint = torch.load(path, map_location=device)
    class_names = checkpoint.get('class_names')
    if not class_names:
        print('教师模型 checkpoint 中缺少 class_names，无法继续蒸馏！')
        return None, None

    model = OreCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 冻结教师参数
    for param in model.parameters():
        param.requires_grad = False

    return model, list(class_names)


# ==================== 蒸馏损失 ====================
class DistillationLoss(nn.Module):
    """损失 = alpha * T^2 * KL(学生软标签 || 教师软标签) + (1-alpha) * 交叉熵"""

    def __init__(self, temperature=10.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        # 乘 T^2 补偿温度带来的梯度尺度变化
        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        hard_loss = self.ce_loss(student_logits, labels)

        total_loss = self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss
        return total_loss, soft_loss, hard_loss


# ==================== 评估 ====================
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total if total else 0.0


# ==================== 主流程 ====================
def main():
    train_dir = os.path.join(SPLIT_DATA_DIR, 'train')
    test_dir = os.path.join(SPLIT_DATA_DIR, 'test')

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        print(f'未找到划分后的数据集：{SPLIT_DATA_DIR}')
        print('请先运行：python 2.split_data.py，再运行 python 8.filter_dataset.py')
        return

    torch.manual_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用计算设备：{device}')

    teacher_model, teacher_class_names = load_teacher_model(TEACHER_MODEL_PATH, device)
    if teacher_model is None:
        return

    train_loader, test_loader, class_names = build_dataloaders(train_dir, test_dir)

    # 教师与数据集类别顺序必须一致，否则软标签索引会完全错位
    if teacher_class_names != class_names:
        print('教师模型与数据集的类别顺序不一致！')
        print(f'教师：{teacher_class_names}')
        print(f'数据集：{class_names}')
        print('请重新训练教师模型，或检查数据集类别文件夹。')
        return

    teacher_acc = evaluate(teacher_model, test_loader, device)
    print(f'教师模型在测试集上的准确率：{teacher_acc:.4f} ({teacher_acc * 100:.2f}%)')

    # 教师是 ResNet18，学生是手写的小 CNN：既演示蒸馏，也演示"大模型带小模型"
    student_model = StudentCNN(num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    distill_criterion = DistillationLoss(temperature=TEMPERATURE, alpha=ALPHA)

    os.makedirs(MODEL_DIR, exist_ok=True)

    best_test_acc = -1.0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'soft_loss': [], 'hard_loss': [],
               'test_acc': []}

    print('开始蒸馏训练...')
    for epoch in range(1, EPOCHS + 1):
        student_model.train()
        total_loss = 0.0
        total_soft = 0.0
        total_hard = 0.0
        correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS} [Distill]', ncols=110)
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher_model(images)   # 教师只做推理

            student_logits = student_model(images)

            loss, soft_loss, hard_loss = distill_criterion(
                student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_soft += soft_loss.item() * batch_size
            total_hard += hard_loss.item() * batch_size

            _, predicted = torch.max(student_logits, 1)
            correct += (predicted == labels).sum().item()
            total += batch_size

            progress.set_postfix(loss=f'{total_loss / total:.4f}',
                                 acc=f'{correct / total:.4f}')

        scheduler.step()

        train_loss = total_loss / total
        train_acc = correct / total
        test_acc = evaluate(student_model, test_loader, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['soft_loss'].append(total_soft / total)
        history['hard_loss'].append(total_hard / total)
        history['test_acc'].append(test_acc)

        print(f'Epoch {epoch} 总结：')
        print(f'平均训练损失：{train_loss:.4f}（软损失 {total_soft / total:.4f} | 硬损失 {total_hard / total:.4f}）')
        print(f'学生模型测试准确率：{test_acc:.4f} ({test_acc * 100:.2f}%)')
        print(f'教师模型基准准确率：{teacher_acc:.4f} ({teacher_acc * 100:.2f}%)')

        # 只保存测试准确率最高的学生模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'model_name': 'studentcnn',
                'best_test_acc': best_test_acc,
                'class_names': class_names,
                'img_size': IMG_SIZE,
                'temperature': TEMPERATURE,
                'alpha': ALPHA,
                'teacher_acc': teacher_acc,
                'history': history,
            }, STUDENT_MODEL_PATH)
            print(f'最优模型已保存（测试准确率：{best_test_acc:.4f}）')

    print()
    print('=' * 12 + ' 蒸馏训练完成 ' + '=' * 12)
    print(f'最优 Epoch：{best_epoch}')
    print(f'学生模型最优测试准确率：{best_test_acc:.4f} ({best_test_acc * 100:.2f}%)')
    print(f'教师模型测试准确率：{teacher_acc:.4f} ({teacher_acc * 100:.2f}%)')
    print(f'蒸馏模型保存路径：{os.path.abspath(STUDENT_MODEL_PATH)}')


if __name__ == '__main__':
    main()
