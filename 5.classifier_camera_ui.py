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
from torchvision import models, transforms

# ==================== 全局配置 ====================
MODEL_PATH = os.path.join('models', 'ore_classification_model.pth')
USER_INFO_PATH = 'user_info.txt'

IMG_SIZE = 224               # 兜底值；实际以 checkpoint 里的 img_size 为准
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

DISPLAY_WIDTH = 780
DISPLAY_HEIGHT = 400

DEFAULT_CLASS_NAMES = ['方解石', '石英', '赤铁矿', '长石', '黄铁矿']


# ==================== 教师网络：ResNet18（必须与训练脚本完全一致） ====================
class OreCNN(nn.Module):
    """教师模型：ImageNet 预训练 ResNet18，只替换最后的全连接层

    推理时不需要联网下载预训练权重，网络结构一致即可加载训练好的 checkpoint。
    """

    def __init__(self, num_classes=5, pretrained=False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


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


def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(MODEL_PATH):
        messagebox.showwarning(
            '提示',
            f'未找到模型文件：{MODEL_PATH}\n'
            '请先运行 3.train_cnn.py 训练模型。\n'
            '当前界面仍可运行，但无法预测。'
        )
        return None, device, DEFAULT_CLASS_NAMES, IMG_SIZE

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        class_names = checkpoint.get('class_names', DEFAULT_CLASS_NAMES)
        img_size = int(checkpoint.get('img_size', IMG_SIZE))
        model = OreCNN(num_classes=len(class_names))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f'模型加载成功，类别：{class_names}，输入尺寸：{img_size}')
        return model, device, class_names, img_size
    except Exception as e:
        messagebox.showwarning(
            '提示',
            f'模型加载失败：{e}\n当前界面仍可运行，但无法预测。'
        )
        return None, device, DEFAULT_CLASS_NAMES, IMG_SIZE


def build_transform(img_size=IMG_SIZE):
    """与训练脚本的测试预处理保持一致：短边放大后中心裁剪，不压扁图片"""
    return transforms.Compose([
        transforms.Resize(int(round(img_size * 8 / 7))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def open_camera(index=0, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, warmup_reads=5):
    """DSHOW 优先打开摄像头，并预热丢掉无效首帧"""
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


# ==================== 主界面 ====================
class CameraClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title('矿石图像分类系统（摄像头版）')
        center_window(self.root, 840, 780)
        self.root.resizable(False, False)
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        self.model, self.device, self.class_names, self.img_size = load_model()
        self.transform = build_transform(self.img_size)

        self.current_image = None        # 本地图片（PIL，用于预测）
        self.current_image_path = None
        self.current_frame = None        # 摄像头原始帧（BGR numpy）
        self.cap = None
        self.is_camera_open = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.camera_thread = None

        self.build_ui()

    def build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main, text='矿石图像分类系统（摄像头版）',
            font=('Microsoft YaHei', 17, 'bold')
        ).pack(pady=(0, 4))

        status_text = (f'模型已加载 | 类别：{"/".join(self.class_names)}'
                       if self.model is not None
                       else '模型未加载：请先运行 3.train_cnn.py 训练模型')
        ttk.Label(
            main, text=status_text,
            font=('Microsoft YaHei', 9),
            foreground='#2e7d32' if self.model is not None else '#c62828'
        ).pack(pady=(0, 8))

        self.display_frame = tk.Frame(
            main, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, bg='#1e1e1e')
        self.display_frame.pack(pady=5)
        self.display_frame.pack_propagate(False)

        self.display_label = tk.Label(
            self.display_frame,
            text='摄像头未打开\n可打开摄像头，或选择本地图片',
            bg='#1e1e1e', fg='#bbbbbb',
            font=('Microsoft YaHei', 13)
        )
        self.display_label.pack(expand=True, fill=tk.BOTH)

        button_frame = ttk.Frame(main)
        button_frame.pack(pady=12)

        self.open_button = ttk.Button(
            button_frame, text='打开摄像头', command=self.start_camera, width=13)
        self.open_button.grid(row=0, column=0, padx=6)

        self.close_button = ttk.Button(
            button_frame, text='关闭摄像头', command=self.close_camera,
            width=13, state=tk.DISABLED)
        self.close_button.grid(row=0, column=1, padx=6)

        self.select_button = ttk.Button(
            button_frame, text='选择本地图片', command=self.select_image, width=13)
        self.select_button.grid(row=0, column=2, padx=6)

        self.predict_button = ttk.Button(
            button_frame, text='拍照预测', command=self.predict, width=13)
        self.predict_button.grid(row=0, column=3, padx=6)

        self.status_label = ttk.Label(
            main, text='摄像头未打开', font=('Microsoft YaHei', 9))
        self.status_label.pack(pady=(6, 0))

        self.result_label = tk.Label(
            main,
            text='识别结果将显示在这里',
            font=('Microsoft YaHei', 13),
            justify=tk.CENTER,
            wraplength=780,
            fg='#333333'
        )
        self.result_label.pack(pady=10)

    # ---------- 摄像头 ----------
    def start_camera(self):
        if self.is_camera_open:
            return

        cap = open_camera()
        if cap is None:
            messagebox.showerror(
                '摄像头错误',
                '无法打开摄像头。\n'
                '请检查：\n'
                '1. 摄像头是否被其他程序占用\n'
                '2. Windows 相机隐私权限是否开启\n'
                '3. 设备索引是否正确（可尝试 cv2.VideoCapture(1)）'
            )
            return

        self.cap = cap
        self.is_camera_open = True
        self.current_frame = None
        self.current_image = None
        self.current_image_path = None
        self.frame_queue = queue.Queue(maxsize=1)   # 清空旧帧

        self.camera_thread = threading.Thread(
            target=self.camera_read_loop, daemon=True)
        self.camera_thread.start()

        self.open_button.config(state=tk.DISABLED)
        self.close_button.config(state=tk.NORMAL)
        self.status_label.config(text='摄像头已打开')
        self.result_label.config(text='摄像头已打开，点击「拍照预测」开始识别')
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
        """子线程只负责读帧并放入队列，不操作任何 Tk 控件"""
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
                    self.frame_queue.get_nowait()      # 丢旧帧，保证实时性
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
        """主线程定时取帧刷新界面（Tkinter 不是线程安全的）"""
        if not self.is_camera_open:
            return

        try:
            frame = self.frame_queue.get_nowait()
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # BGR -> RGB
            photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
            self.display_label.config(image=photo, text='')
            self.display_label.photo = photo
        except queue.Empty:
            pass

        self.root.after(30, self.update_camera_display)   # 约 33 FPS

    # ---------- 本地图片与预测 ----------
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
        self.result_label.config(text='点击「拍照预测」开始识别')

    def predict(self):
        if self.model is None:
            messagebox.showwarning('提示', '模型未加载，无法预测。\n请先运行 3.train_cnn.py')
            return

        if self.is_camera_open and self.current_frame is not None:
            frame = self.current_frame.copy()
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            source = '摄像头实时画面'
        elif self.current_image is not None:
            image = self.current_image
            source = f'本地图片（{os.path.basename(self.current_image_path or "")}）'
        else:
            messagebox.showwarning('提示', '请先打开摄像头或选择本地图片！')
            return

        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                prob = torch.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(prob, 1)

            pred_class = self.class_names[pred_idx.item()]
            confidence = confidence.item() * 100

            self.result_label.config(
                text=f'来源：{source}\n'
                     f'预测结果：{pred_class}\n'
                     f'置信度：{confidence:.2f}%'
            )
        except Exception as e:
            messagebox.showerror('预测失败', str(e))

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
            CameraClassifierUI(main_root)
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
