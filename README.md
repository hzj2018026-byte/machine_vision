# 深度学习综合实践

矿石图像分类系统（PyTorch）：数据采集 → 数据划分 → CNN 训练 → 图形界面推理，以及知识蒸馏与集成预测。

- **分类类别（5 类）**：石英、方解石、黄铁矿、赤铁矿、长石
- **教师模型**：ImageNet 预训练 ResNet18 微调（迁移学习，输入 224×224）
- **学生模型**：手写 4 层 CNN，通过知识蒸馏向教师模型学习
- **原始数据目录**：`ore_images/{类别名}/`
- **划分后目录**：`ore_images_split/train|test/{类别名}/`
- **过滤后目录**：`ore_images_filtered/train|test/{类别名}/`（测试集已人工复核剔除脏样本）
- **模型目录**：`models/`

## 运行顺序

```powershell
# 1. 采集图片（百度图片，每类多个关键词；自动去重，可重复运行续爬）
python 1.data_download.py

# 2. 按 8:2 划分训练集/测试集
python 2.split_data.py

# 8. 过滤测试集中的广告/截图/拼图/珠宝/明显错标等脏样本
#    生成 ore_images_filtered/ 和 filter_manifest.csv
python 8.filter_dataset.py

# 3. 训练教师模型（ResNet18 224px + 强增强 + EMA + 模型汤 + TTA）
#    输出 models/ore_classification_model.pth
python 3.train_cnn.py

# 4. 图形界面分类（本地图片）
python 4.classifier_ui.py

# 5. 图形界面分类（摄像头）
python 5.classifier_camera_ui.py

# 6. 知识蒸馏训练学生模型（手写 CNN，输出 models/distilled_ore_model.pth）
python 6.distillation_cnn.py

# 7. 教师/学生/集成预测界面
python 7.distillation_predictor.py
```

> 说明：`8.filter_dataset.py` 会保留原始 `ore_images_split`，只把复核通过的测试图片硬链接/复制到
> `ore_images_filtered`。过滤后的准确率与完整测试集准确率不能直接比较，报告时应同时给出两个数字
> （`3.train_cnn.py` 会打印“过滤后测试集”和“完整原始测试集”两行指标）。
> 过滤清单及原因见 `ore_images_filtered/filter_manifest.csv`。

## Web 版：识矿乐园（ore_recognition_web）

在训练脚本之外，项目新增了一个前后端分离的矿石识别科普 Web 项目：

- 后端：FastAPI + SQLAlchemy + MySQL（也可用 SQLite 演示）
- 前端：Vue 3 + Vite
- 模型：直接读取 `models/ore_classification_model.pth`，自动适配 160/224 输入尺寸
- 功能：拍照识别、Top-3、知识卡、图鉴、识别记录、纠错、登录注册、管理端

快速启动：

```powershell
# 建库并配置 backend/.env 后
cd ore_recognition_web\backend
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000

# 另开终端
cd ore_recognition_web\frontend
npm install
npm run dev
```

访问 <http://localhost:5173>，管理员 `admin / 123456`。详细说明见
[识矿乐园项目开发说明.md](识矿乐园项目开发说明.md) 和
[ore_recognition_web/README.md](ore_recognition_web/README.md)。

单类别调试爬虫：

```powershell
python single_data_download.py            # 默认石英，10 页
python single_data_download.py 黄铁矿 5    # 指定类别和页数
```

## 默认账号

```
admin / 123456
user  / 123456
```

详细原理、参数说明与排错可参考 `图像分类项目开发说明文档.md`（该文档以水果分类为例，本项目已替换为矿石类别）。
