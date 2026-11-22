"""矿石分类模型加载与推理服务。

兼容两种 checkpoint：
- 旧版 160×160：Resize((160,160)) 直接压缩；
- 新版 224×224：短边 Resize + CenterCrop，并可选择水平翻转 TTA。
"""
import threading
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from ..config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_CLASS_NAMES,
    MODEL_PATH,
    USE_TTA,
)

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


class OreCNN(nn.Module):
    """与 3.train_cnn.py 完全一致的 ResNet18 教师模型。"""

    def __init__(self, num_classes: int = 5, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class ModelService:
    """模型只加载一次；推理加锁，避免多请求同时进入模型。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model: OreCNN | None = None
        self.class_names: list[str] = []
        self.img_size = 160
        self.use_tta = USE_TTA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.error: str | None = None
        self.transform = None
        self.load()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def _build_transform(self, img_size: int, center_crop: bool):
        if center_crop:
            return transforms.Compose([
                transforms.Resize(int(round(img_size * 8 / 7))),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ])
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ])

    def load(self) -> None:
        path = Path(MODEL_PATH)
        if not path.exists():
            self.error = f"模型文件不存在：{path}"
            return
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            class_names = checkpoint.get("class_names") or DEFAULT_CLASS_NAMES
            img_size = int(checkpoint.get("img_size", 160))
            # 新版训练脚本会写入 eval_tta=True；旧版用直接 Resize 压缩
            center_crop = bool(checkpoint.get("eval_tta")) or img_size >= 224

            model = OreCNN(num_classes=len(class_names))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()

            self.model = model
            self.class_names = list(class_names)
            self.img_size = img_size
            self.transform = self._build_transform(img_size, center_crop)
            # 旧版 160 模型实测水平翻转 TTA 反而略降准确率，默认关闭；
            # 新版 224 模型（checkpoint 带 eval_tta=True）再启用。
            self.use_tta = USE_TTA and center_crop
            self.error = None
        except Exception as exc:  # noqa: BLE001
            self.error = f"模型加载失败：{exc}"

    @torch.no_grad()
    def predict(self, image: Image.Image, topk: int = 3) -> dict:
        if self.model is None or self.transform is None:
            raise RuntimeError(self.error or "模型未加载")

        image = image.convert("RGB")
        views = [image]
        if self.use_tta:
            views.append(image.transpose(Image.FLIP_LEFT_RIGHT))

        probs = None
        with self._lock:
            for view in views:
                tensor = self.transform(view).unsqueeze(0).to(self.device)
                logits = self.model(tensor)
                batch_probs = torch.softmax(logits, dim=1)
                probs = batch_probs if probs is None else probs + batch_probs
            probs = probs / len(views)

        values, indices = torch.topk(probs[0], k=min(topk, len(self.class_names)))
        top = [
            {"label": self.class_names[i], "confidence": round(float(v), 4)}
            for v, i in zip(values.tolist(), indices.tolist())
        ]
        label = top[0]["label"]
        confidence = top[0]["confidence"]
        return {
            "label": label,
            "confidence": confidence,
            "uncertain": confidence < CONFIDENCE_THRESHOLD,
            "top3": top,
        }


model_service = ModelService()
