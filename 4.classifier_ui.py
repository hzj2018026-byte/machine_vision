import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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

# 模型缺失时的兜底类别（DatasetFolder 按文件夹名排序的结果）
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
    """读取账号文件，兼容 `用户名:密码`、`用户名 / 密码`、`用户名 密码` 三种写法"""
    default_users = {'admin': '123456', 'user': '123456'}
    if not os.path.exists(path):
        print(f'未找到账号文件 {path}，使用默认账号 admin/123456、user/123456')
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
    """加载模型；失败时返回 None，界面仍可正常运行（只是不能预测）"""
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


# ==================== 主界面 ====================
class ClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title('矿石图像分类系统')
        center_window(self.root, 720, 780)
        self.root.resizable(False, False)

        self.model, self.device, self.class_names, self.img_size = load_model()
        self.transform = build_transform(self.img_size)
        self.current_image_path = None

        self.build_ui()

    def build_ui(self):
        main = ttk.Frame(self.root, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main, text='矿石图像分类系统',
            font=('Microsoft YaHei', 18, 'bold')
        ).pack(pady=(0, 5))

        status_text = (f'模型已加载 | 类别：{"/".join(self.class_names)}'
                       if self.model is not None
                       else '模型未加载：请先运行 3.train_cnn.py 训练模型')
        ttk.Label(
            main, text=status_text,
            font=('Microsoft YaHei', 9),
            foreground='#2e7d32' if self.model is not None else '#c62828'
        ).pack(pady=(0, 10))

        # 固定大小的图片显示区，避免图片撑大窗口
        self.image_frame = tk.Frame(main, width=520, height=300, bg='#1e1e1e')
        self.image_frame.pack(pady=5)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(
            self.image_frame,
            text='请选择一张矿石图片',
            bg='#1e1e1e', fg='#bbbbbb',
            font=('Microsoft YaHei', 13)
        )
        self.image_label.pack(expand=True, fill=tk.BOTH)

        button_frame = ttk.Frame(main)
        button_frame.pack(pady=15)

        self.select_button = ttk.Button(
            button_frame, text='选择矿石图片', command=self.select_image, width=16)
        self.select_button.grid(row=0, column=0, padx=10)

        self.predict_button = ttk.Button(
            button_frame, text='开始预测', command=self.predict,
            width=16, state=tk.DISABLED)
        self.predict_button.grid(row=0, column=1, padx=10)

        self.result_label = tk.Label(
            main,
            text='预测结果将显示在这里',
            font=('Microsoft YaHei', 13),
            justify=tk.CENTER,
            wraplength=560,
            fg='#333333'
        )
        self.result_label.pack(pady=10)

    def select_image(self):
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

        image.thumbnail((500, 280))
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo, text='')
        self.image_label.photo = photo   # 必须保留引用，否则会被垃圾回收成空白

        self.current_image_path = file_path
        self.predict_button.config(state=tk.NORMAL)
        self.result_label.config(text='已选择图片，请点击「开始预测」')

    def predict(self):
        if self.model is None:
            messagebox.showwarning('提示', '模型未加载，无法预测。\n请先运行 3.train_cnn.py')
            return
        if not self.current_image_path:
            messagebox.showwarning('提示', '请先选择一张矿石图片！')
            return

        try:
            image = Image.open(self.current_image_path).convert('RGB')
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                prob = torch.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(prob, 1)

            pred_class = self.class_names[pred_idx.item()]
            confidence = confidence.item() * 100

            self.result_label.config(
                text=f'预测结果：{pred_class}\n置信度：{confidence:.2f}%'
            )
        except Exception as e:
            messagebox.showerror('预测失败', str(e))


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
        username = username_entry.get().strip().lower()   # 用户名忽略大小写
        password = password_entry.get().strip()           # 密码区分大小写
        if username in users and users[username] == password:
            login_root.destroy()
            main_root = tk.Tk()
            ClassifierUI(main_root)
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
