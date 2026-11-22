import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import torch
import torch.nn as nn
from PIL import Image, ImageTk
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

# ==================== 全局配置 ====================
SPLIT_DATA_DIR = './ore_images_filtered'   # 8.filter_dataset.py 生成：测试集已过滤
TEACHER_MODEL_PATH = os.path.join('models', 'ore_classification_model.pth')
STUDENT_MODEL_PATH = os.path.join('models', 'distilled_ore_model.pth')
USER_INFO_PATH = 'user_info.txt'

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 420

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

DEFAULT_CLASS_NAMES = ['方解石', '石英', '赤铁矿', '长石', '黄铁矿']


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


# ==================== 数据加载（用于集成准确率评估） ====================
def is_valid_image(path):
    # 新版 torchvision 不允许同时传 extensions 和 is_valid_file，后缀过滤在这里完成
    if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS:
        return False
    try:
        with open(path, 'rb') as f:
            Image.open(f).verify()
        return True
    except Exception:
        return False


def custom_image_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except Exception:
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


def build_transform():
    """与 3.train_cnn.py / 6.distillation_cnn.py 的测试预处理保持一致"""
    return transforms.Compose([
        transforms.Resize(int(round(IMG_SIZE * 8 / 7))),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def build_test_loader():
    """构建测试集 DataLoader（评估集成准确率时在后台线程调用）"""
    test_dir = os.path.join(SPLIT_DATA_DIR, 'test')
    if not os.path.isdir(test_dir):
        raise RuntimeError(f'未找到测试集目录：{test_dir}\n请先运行 2.split_data.py，再运行 8.filter_dataset.py')

    test_raw = datasets.DatasetFolder(
        test_dir,
        loader=custom_image_loader,
        is_valid_file=is_valid_image,
    )
    test_dataset = FilteredDataset(test_raw, transform=build_transform())
    if len(test_dataset) == 0:
        raise RuntimeError('测试集中没有有效图片！')

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False)
    return test_loader, list(test_raw.classes)


# ==================== 通用工具 ====================
def center_window(root, width, height):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    root.geometry(f'{width}x{height}+{x}+{y}')


def load_user_info(path=USER_INFO_PATH):
    default_users = {'admin': '123456', 'user': '123456'}
    if not os.path.exists(path):
        return default_users

    users = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' in line:
                    username, password = line.split(':', 1)
                elif '/' in line:
                    username, password = line.split('/', 1)
                else:
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    username, password = parts
                users[username.strip().lower()] = password.strip()
    except Exception as e:
        print(f'读取账号文件失败：{e}，使用默认账号')
        return default_users

    return users or default_users


def build_model_from_checkpoint(checkpoint, num_classes):
    """根据 checkpoint 中的 model_name / 权重键名判断网络结构，兼容两种模型"""
    state_dict = checkpoint.get('model_state_dict', {})
    model_name = checkpoint.get('model_name', '')
    if model_name == 'resnet18' or 'backbone.conv1.weight' in state_dict:
        return OreCNN(num_classes=num_classes)
    return StudentCNN(num_classes=num_classes)


def load_single_model(path, device):
    """加载单个模型，返回 (model, class_names, num_classes)；失败返回 (None, [], 0)"""
    if not os.path.exists(path):
        print(f'未找到模型文件：{path}')
        return None, [], 0

    try:
        checkpoint = torch.load(path, map_location=device)
        class_names = checkpoint.get('class_names', [])
        if not class_names:
            print(f'模型 {path} 缺少 class_names 字段')
            return None, [], 0

        model = build_model_from_checkpoint(checkpoint, len(class_names))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, list(class_names), len(class_names)
    except Exception as e:
        print(f'模型加载失败 {path}：{e}')
        return None, [], 0


def open_camera(index=0, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, warmup_reads=5):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for _ in range(warmup_reads):
            ret, _ = cap.read()
            if ret:
                return cap
            time.sleep(0.1)
        cap.release()
    return None


# ==================== 集成预测主界面 ====================
class OreEnsembleApp:
    def __init__(self, root, teacher_path, student_path):
        self.root = root
        self.root.title('矿石图像分类系统（集成预测版）')
        center_window(self.root, 860, 840)
        self.root.resizable(False, False)
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = build_transform()

        # ---------- 加载双模型并校验一致性 ----------
        self.teacher_model, teacher_classes, teacher_num = load_single_model(
            teacher_path, self.device)
        self.student_model, student_classes, student_num = load_single_model(
            student_path, self.device)

        self.class_names = []
        self.ready = False
        self.load_error = ''

        if self.teacher_model is None or self.student_model is None:
            self.load_error = (
                '模型加载失败，请确认以下文件存在且可读取：\n'
                f'教师模型：{teacher_path}\n'
                f'学生模型：{student_path}\n'
                '请先运行 3.train_cnn.py 和 6.distillation_cnn.py。'
            )
        elif teacher_num != student_num:
            self.load_error = (f'类别数不一致！教师：{teacher_num} | 学生：{student_num}')
        elif teacher_classes != student_classes:
            self.load_error = (
                '类别顺序不一致，集成结果将完全错误！\n'
                f'教师：{teacher_classes}\n'
                f'学生：{student_classes}'
            )
        else:
            self.ready = True
            self.class_names = teacher_classes

        # ---------- 运行状态 ----------
        self.current_image = None
        self.current_image_path = None
        self.current_frame = None
        self.cap = None
        self.is_camera_open = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.camera_thread = None

        self.eval_running = False
        self.eval_queue = None
        self.eval_max_batches = 1

        self.build_ui()

        if not self.ready:
            self.root.after(200, lambda: messagebox.showerror('模型加载错误', self.load_error))

    # ---------- 界面 ----------
    def build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main, text='矿石图像分类系统（集成预测版）',
            font=('Microsoft YaHei', 17, 'bold')
        ).pack(pady=(0, 4))

        if self.ready:
            status_text = f'教师 + 学生模型已加载 | 类别：{"/".join(self.class_names)}'
            status_color = '#2e7d32'
        else:
            status_text = '模型未就绪：请先训练教师模型和学生模型'
            status_color = '#c62828'

        ttk.Label(
            main, text=status_text,
            font=('Microsoft YaHei', 9), foreground=status_color
        ).pack(pady=(0, 6))

        mode_frame = ttk.LabelFrame(main, text='预测模式', padding=8)
        mode_frame.pack(fill=tk.X, pady=(0, 8))
        self.mode_var = tk.StringVar(value='集成模型')
        for mode in ('教师模型', '学生模型', '集成模型'):
            ttk.Radiobutton(
                mode_frame, text=mode, value=mode, variable=self.mode_var
            ).pack(side=tk.LEFT, padx=18)

        self.display_frame = tk.Frame(
            main, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, bg='#1e1e1e')
        self.display_frame.pack(pady=5)
        self.display_frame.pack_propagate(False)

        self.display_label = tk.Label(
            self.display_frame,
            text='请选择本地图片，或打开摄像头',
            bg='#1e1e1e', fg='#bbbbbb',
            font=('Microsoft YaHei', 13)
        )
        self.display_label.pack(expand=True, fill=tk.BOTH)

        button_frame = ttk.Frame(main)
        button_frame.pack(pady=10)

        self.select_button = ttk.Button(
            button_frame, text='选择本地图片', command=self.select_image, width=13)
        self.select_button.grid(row=0, column=0, padx=5)

        self.open_button = ttk.Button(
            button_frame, text='打开摄像头', command=self.start_camera, width=12)
        self.open_button.grid(row=0, column=1, padx=5)

        self.close_button = ttk.Button(
            button_frame, text='关闭摄像头', command=self.close_camera,
            width=12, state=tk.DISABLED)
        self.close_button.grid(row=0, column=2, padx=5)

        self.predict_button = ttk.Button(
            button_frame, text='执行预测', command=self.execute_prediction, width=12)
        self.predict_button.grid(row=0, column=3, padx=5)

        self.eval_button = ttk.Button(
            button_frame, text='评估集成准确率', command=self.eval_ensemble_accuracy,
            width=15, state=tk.NORMAL if self.ready else tk.DISABLED)
        self.eval_button.grid(row=0, column=4, padx=5)

        self.progress_bar = ttk.Progressbar(main, mode='determinate', maximum=1, value=0)
        self.progress_bar.pack(fill=tk.X, padx=30, pady=(4, 0))

        self.status_label = ttk.Label(
            main, text='摄像头未打开', font=('Microsoft YaHei', 9))
        self.status_label.pack(pady=(4, 0))

        self.result_label = tk.Label(
            main,
            text='预测结果将显示在这里',
            font=('Microsoft YaHei', 12),
            justify=tk.LEFT,
            anchor='w',
            wraplength=800,
            fg='#333333'
        )
        self.result_label.pack(fill=tk.X, padx=30, pady=8)

    # ---------- 摄像头 ----------
    def start_camera(self):
        if self.is_camera_open:
            return

        cap = open_camera()
        if cap is None:
            messagebox.showerror(
                '摄像头错误',
                '无法打开摄像头。\n'
                '请检查摄像头是否被占用、Windows 相机权限是否开启。'
            )
            return

        self.cap = cap
        self.is_camera_open = True
        self.current_frame = None
        self.current_image = None
        self.current_image_path = None
        self.frame_queue = queue.Queue(maxsize=1)

        self.camera_thread = threading.Thread(
            target=self.camera_read_loop, daemon=True)
        self.camera_thread.start()

        self.open_button.config(state=tk.DISABLED)
        self.close_button.config(state=tk.NORMAL)
        self.status_label.config(text='摄像头已打开')
        self.result_label.config(text='摄像头已打开，点击「执行预测」开始识别')
        self.update_camera_display()

    def close_camera(self, reset_display=True):
        self.is_camera_open = False
        cap, self.cap = self.cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        self.current_frame = None

        self.open_button.config(state=tk.NORMAL)
        self.close_button.config(state=tk.DISABLED)
        self.status_label.config(text='摄像头未打开')

        if reset_display:
            self.display_label.config(image='', text='摄像头未打开')
            self.display_label.photo = None

    def camera_read_loop(self):
        while self.is_camera_open:
            cap = self.cap
            if cap is None:
                break

            ret, frame = cap.read()
            if not ret:
                if self.is_camera_open:
                    try:
                        self.root.after(0, self.handle_camera_failure)
                    except tk.TclError:
                        pass
                break

            self.current_frame = frame
            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            try:
                self.frame_queue.put_nowait(display_frame)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait(display_frame)
                except queue.Full:
                    pass

    def handle_camera_failure(self):
        if not self.is_camera_open:
            return
        self.close_camera()
        messagebox.showwarning(
            '提示', '摄像头读取失败，已自动关闭。\n请检查设备是否被其他程序占用。')

    def update_camera_display(self):
        if not self.is_camera_open:
            return

        try:
            frame = self.frame_queue.get_nowait()
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
            self.display_label.config(image=photo, text='')
            self.display_label.photo = photo
        except queue.Empty:
            pass

        self.root.after(30, self.update_camera_display)

    # ---------- 本地图片 ----------
    def select_image(self):
        if self.is_camera_open:
            self.close_camera()

        file_path = filedialog.askopenfilename(
            title='选择矿石图片',
            filetypes=[
                ('图片文件', '*.jpg *.jpeg *.png *.bmp *.webp *.gif'),
                ('所有文件', '*.*'),
            ]
        )
        if not file_path:
            return

        try:
            image = Image.open(file_path).convert('RGB')
        except Exception as e:
            messagebox.showerror('打开失败', f'无法打开该图片：\n{e}')
            return

        self.current_image = image
        self.current_image_path = file_path

        display_image = image.copy()
        display_image.thumbnail((DISPLAY_WIDTH - 20, DISPLAY_HEIGHT - 20))
        photo = ImageTk.PhotoImage(display_image)
        self.display_label.config(image=photo, text='')
        self.display_label.photo = photo

        self.status_label.config(text=f'已选择本地图片：{os.path.basename(file_path)}')
        self.result_label.config(text='点击「执行预测」开始识别')

    # ---------- 预测 ----------
    def get_current_image(self):
        if self.is_camera_open and self.current_frame is not None:
            frame = self.current_frame.copy()
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return image, '摄像头实时画面'
        if self.current_image is not None:
            name = os.path.basename(self.current_image_path or '')
            return self.current_image, f'本地图片（{name}）'
        return None, ''

    def _format_prediction(self, name, logits):
        probs = torch.softmax(logits, dim=1)[0]
        confidence, pred_idx = torch.max(probs, 0)
        return (f'{name}：{self.class_names[pred_idx.item()]}'
                f'（置信度 {confidence.item() * 100:.2f}%）')

    def execute_prediction(self):
        if not self.ready:
            messagebox.showwarning('提示', '模型未就绪，无法预测。\n' + self.load_error)
            return

        image, source = self.get_current_image()
        if image is None:
            messagebox.showwarning('提示', '请先选择本地图片或打开摄像头！')
            return

        mode = self.mode_var.get()

        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if mode == '教师模型':
                    teacher_logits = self.teacher_model(img_tensor)
                    text = ('模式：教师模型\n'
                            + self._format_prediction('教师模型', teacher_logits))
                elif mode == '学生模型':
                    student_logits = self.student_model(img_tensor)
                    text = ('模式：学生模型\n'
                            + self._format_prediction('学生模型', student_logits))
                else:
                    teacher_logits = self.teacher_model(img_tensor)
                    student_logits = self.student_model(img_tensor)
                    # 先平均 logits，再 softmax（保留 logit 的尺度信息）
                    ensemble_logits = (teacher_logits + student_logits) / 2.0
                    text = ('模式：集成模型（教师 + 学生 logits 平均）\n'
                            + self._format_prediction('教师模型', teacher_logits) + '\n'
                            + self._format_prediction('学生模型', student_logits) + '\n'
                            + self._format_prediction('集成模型', ensemble_logits))

            self.result_label.config(text=f'图片来源：{source}\n{text}')
        except Exception as e:
            messagebox.showerror('预测失败', str(e))

    # ---------- 集成准确率评估 ----------
    def eval_ensemble_accuracy(self):
        if not self.ready:
            messagebox.showwarning('提示', '模型未就绪，无法评估。')
            return
        if self.eval_running:
            return
        if not os.path.isdir(os.path.join(SPLIT_DATA_DIR, 'test')):
            messagebox.showwarning('提示', '未找到测试集目录，请先运行 2.split_data.py 和 8.filter_dataset.py')
            return

        self.eval_running = True
        self.eval_queue = queue.Queue()
        self.eval_max_batches = 1
        self.eval_button.config(state=tk.DISABLED)
        self.progress_bar.config(maximum=1, value=0)
        self.status_label.config(text='正在评估集成准确率...')

        threading.Thread(target=self._eval_worker, daemon=True).start()
        self.root.after(100, self._poll_eval_queue)

    def _eval_worker(self):
        """后台线程：只计算 + 往队列里放结果，不碰任何 Tk 控件"""
        try:
            test_loader, test_classes = build_test_loader()
            if test_classes != self.class_names:
                raise ValueError(
                    f'测试集类别与模型类别不一致！\n测试集：{test_classes}\n模型：{self.class_names}')

            total_batches = len(test_loader)
            teacher_correct = 0
            student_correct = 0
            ensemble_correct = 0
            total = 0

            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_loader, start=1):
                    images, labels = images.to(self.device), labels.to(self.device)

                    teacher_logits = self.teacher_model(images)
                    student_logits = self.student_model(images)
                    ensemble_logits = (teacher_logits + student_logits) / 2.0

                    _, teacher_pred = torch.max(teacher_logits, 1)
                    _, student_pred = torch.max(student_logits, 1)
                    _, ensemble_pred = torch.max(ensemble_logits, 1)

                    teacher_correct += (teacher_pred == labels).sum().item()
                    student_correct += (student_pred == labels).sum().item()
                    ensemble_correct += (ensemble_pred == labels).sum().item()
                    total += labels.size(0)

                    self.eval_queue.put(('progress', batch_idx, total_batches))

            self.eval_queue.put((
                'result',
                teacher_correct / total,
                student_correct / total,
                ensemble_correct / total,
                total,
            ))
        except Exception as e:
            self.eval_queue.put(('error', str(e)))

    def _poll_eval_queue(self):
        finished = False

        try:
            while True:
                message = self.eval_queue.get_nowait()
                kind = message[0]

                if kind == 'progress':
                    _, done, total_batches = message
                    self.eval_max_batches = total_batches
                    self.progress_bar.config(
                        maximum=max(total_batches, 1), value=done)
                    self.status_label.config(
                        text=f'正在评估集成准确率...（{done}/{total_batches} 批）')
                elif kind == 'result':
                    finished = True
                    _, teacher_acc, student_acc, ensemble_acc, sample_count = message
                    self.progress_bar.config(value=self.eval_max_batches)
                    self.status_label.config(text='集成准确率评估完成')
                    messagebox.showinfo(
                        '集成准确率评估',
                        f'测试样本数：{sample_count}\n'
                        f'教师模型准确率：{teacher_acc * 100:.2f}%\n'
                        f'学生模型准确率：{student_acc * 100:.2f}%\n'
                        f'集成模型准确率：{ensemble_acc * 100:.2f}%'
                    )
                elif kind == 'error':
                    finished = True
                    self.status_label.config(text='集成准确率评估失败')
                    messagebox.showerror('评估失败', message[1])
        except queue.Empty:
            pass

        if finished:
            self.eval_running = False
            self.eval_button.config(state=tk.NORMAL)
        elif self.eval_running:
            self.root.after(100, self._poll_eval_queue)

    # ---------- 退出 ----------
    def on_close(self):
        self.close_camera(reset_display=False)
        self.root.destroy()


# ==================== 登录窗口 ====================
def login_window():
    users = load_user_info()

    login_root = tk.Tk()
    login_root.title('矿石图像分类系统 - 登录')
    center_window(login_root, 380, 270)
    login_root.resizable(False, False)

    frame = ttk.Frame(login_root, padding=20)
    frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(
        frame, text='矿石图像分类系统',
        font=('Microsoft YaHei', 16, 'bold')
    ).pack(pady=(0, 15))

    form = ttk.Frame(frame)
    form.pack()

    ttk.Label(form, text='用户名：', font=('Microsoft YaHei', 11)) \
        .grid(row=0, column=0, sticky='e', pady=6)
    username_entry = ttk.Entry(form, width=22, font=('Microsoft YaHei', 11))
    username_entry.grid(row=0, column=1, pady=6)

    ttk.Label(form, text='密　码：', font=('Microsoft YaHei', 11)) \
        .grid(row=1, column=0, sticky='e', pady=6)
    password_entry = ttk.Entry(form, width=22, show='*', font=('Microsoft YaHei', 11))
    password_entry.grid(row=1, column=1, pady=6)

    def do_login(event=None):
        username = username_entry.get().strip().lower()
        password = password_entry.get().strip()
        if username in users and users[username] == password:
            login_root.destroy()
            main_root = tk.Tk()
            OreEnsembleApp(main_root, TEACHER_MODEL_PATH, STUDENT_MODEL_PATH)
            main_root.mainloop()
        else:
            messagebox.showerror('登录失败', '用户名或密码错误，请重试！', parent=login_root)
            password_entry.delete(0, tk.END)
            password_entry.focus_set()

    ttk.Button(frame, text='登 录', command=do_login, width=14).pack(pady=16)

    username_entry.focus_set()
    login_root.bind('<Return>', do_login)
    login_root.mainloop()


if __name__ == '__main__':
    login_window()
