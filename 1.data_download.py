import hashlib
import io
import json
import os
import random
import time
from urllib.parse import urlparse, quote

import requests
from PIL import Image


# ========== 数据采集：百度图片爬虫（矿石图像） ==========
class BaiduImageSpider:
    """按关键词爬取矿石图片，每个类别单独保存到一个文件夹

    相比旧版的三点增强：
    1. 同一类别支持多个搜索词（中英文混合），扩大样本来源
    2. 全局 md5 去重：已有图片、其他类别已下载的重复图都会被跳过
    3. 文件名从现有最大编号续写，重复运行不会覆盖旧数据
    """

    def __init__(self, word, save_root='./ore_images', seen_hashes=None):
        # 百度图片异步 JSON 接口（新版参数：ala / nojc / newReq 缺一不可）
        self.url = ('https://image.baidu.com/search/acjson'
                    '?tn=resultjson_com&word={}&pn={}&rn=30'
                    '&ala=0&nojc=0&newReq=1')
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
            'Accept': 'text/plain, */*; q=0.01',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Referer': 'https://image.baidu.com/search/index?tn=baiduimage&word=',
            'X-Requested-With': 'XMLHttpRequest'
        }
        self.word = word          # 矿石名称，同时也是该类的标签名和保存目录名
        self.queries = [word]     # 搜索词列表，main 中会替换成多种关键词
        self.seen_urls = set()    # 本次运行内的 URL 去重
        self.seen_hashes = seen_hashes if seen_hashes is not None else set()  # 全局内容去重
        self.i = 0                # 序号，初始化时续写已有最大编号
        self.added = 0            # 本次新增数量
        self.directory = os.path.join(save_root, word)  # 保存目录
        os.makedirs(self.directory, exist_ok=True)
        self.i = self._max_existing_index()

    def _max_existing_index(self):
        """扫描已有文件名（{类别}_{编号}.ext），返回最大编号，避免覆盖"""
        max_idx = 0
        for name in os.listdir(self.directory):
            stem = os.path.splitext(name)[0]
            if '_' not in stem:
                continue
            prefix, num = stem.rsplit('_', 1)
            if prefix == self.word and num.isdigit():
                max_idx = max(max_idx, int(num))
        return max_idx

    @staticmethod
    def fix_image_url(img_url):
        """修复图片URL格式"""
        if not img_url:
            return None
        parsed_url = urlparse(img_url)
        if not parsed_url.scheme:
            fixed_url = f'https://{img_url}'
            parsed_fixed = urlparse(fixed_url)
            if parsed_fixed.scheme and parsed_fixed.netloc:
                return fixed_url
            return None
        return img_url.replace('http://', 'https://')

    @staticmethod
    def get_extension(content_type):
        """根据响应头中的 Content-Type 决定保存的扩展名"""
        content_type = (content_type or '').lower()
        if 'png' in content_type:
            return '.png'
        if 'webp' in content_type:
            return '.webp'
        if 'gif' in content_type:
            return '.gif'
        if 'bmp' in content_type:
            return '.bmp'
        return '.jpg'

    def get_image_urls(self, query, page):
        """获取指定搜索词、指定页的图片URL列表"""
        try:
            url = self.url.format(quote(query), page * 30)
            response = requests.get(url=url, headers=self.headers, timeout=10)
            data = json.loads(response.text)

            # 触发反爬时返回 {"antiFlag": ..., "message": "Forbid spider access"}
            if 'antiFlag' in data:
                print(f"[{self.word}] 「{query}」第{page + 1}页被反爬拦截：{data.get('message')}")
                return []

            payload = data.get('data') or {}
            # 新版接口：data 是字典，图片列表在 data['images']
            # 旧版接口：data 直接是图片列表
            if isinstance(payload, dict):
                items = payload.get('images') or []
            else:
                items = payload

            image_urls = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                # 新版字段名为小写 thumburl / objurl，旧版为 hoverURL / objURL
                img_url = (item.get('thumburl') or item.get('objurl')
                           or item.get('hoverURL') or item.get('objURL'))
                fixed_url = self.fix_image_url(img_url)
                if fixed_url and fixed_url not in self.seen_urls:
                    self.seen_urls.add(fixed_url)
                    image_urls.append(fixed_url)
            return image_urls
        except Exception as e:
            print(f"[{self.word}] 获取URL失败：{e}")
            return []

    def download_image(self, img_url):
        """下载并校验单张图片，成功返回 True"""
        try:
            response = requests.get(
                img_url,
                headers={'User-Agent': self.headers['User-Agent']},
                timeout=10
            )
            if response.status_code != 200:
                return False
            content_type = response.headers.get('Content-Type', '')
            if 'image' not in content_type.lower():
                return False
            content = response.content
            if len(content) < 4096:  # 过滤无效小文件
                return False

            # 内容哈希去重：跨类别、跨运行都生效
            digest = hashlib.md5(content).hexdigest()
            if digest in self.seen_hashes:
                return False

            # 用 PIL 校验能正常打开，并过滤过小的图标类图片
            try:
                image = Image.open(io.BytesIO(content))
                width, height = image.size
                image.verify()
            except Exception:
                return False
            if width < 120 or height < 120:
                return False

            self.seen_hashes.add(digest)
            self.i += 1
            self.added += 1
            ext = self.get_extension(content_type)
            filename = os.path.join(self.directory, f'{self.word}_{self.i:03d}{ext}')
            with open(filename, 'wb') as f:
                f.write(content)
            return True
        except Exception:
            return False

    def run(self, pages_per_query=6):
        """按多个搜索词依次爬取，每个词最多翻 pages_per_query 页（每页约 30 张）"""
        print(f"开始爬取「{self.word}」：搜索词 {self.queries}，每个词最多 {pages_per_query} 页")
        for query in self.queries:
            for page in range(pages_per_query):
                image_urls = self.get_image_urls(query, page)
                if not image_urls:
                    break
                for img_url in image_urls:
                    self.download_image(img_url)
                    time.sleep(random.uniform(0.15, 0.3))  # 控制下载速度，避免被限流
                time.sleep(random.uniform(0.8, 1.5))       # 翻页间隔
            print(f"  「{query}」完成，本次累计新增 {self.added} 张")
        total = len(os.listdir(self.directory))
        print(f"「{self.word}」爬取完成：本次新增 {self.added} 张，目录现有 {total} 个文件")


def collect_existing_hashes(root):
    """收集已有所有图片的 md5，用于跨类别、跨运行去重"""
    seen = set()
    if not os.path.isdir(root):
        return seen
    for cls in os.listdir(root):
        class_dir = os.path.join(root, cls)
        if not os.path.isdir(class_dir):
            continue
        for name in os.listdir(class_dir):
            path = os.path.join(class_dir, name)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, 'rb') as f:
                    seen.add(hashlib.md5(f.read()).hexdigest())
            except OSError:
                pass
    return seen


# ========== 主函数（按类别依次爬取） ==========
if __name__ == '__main__':
    # 选定的 5 类矿石：类别名 = 保存目录名 = 训练标签
    # 每个类别用多个搜索词（中英文混合），扩大数据量并增加样本多样性
    ORE_QUERIES = {
        '石英': ['石英矿石', '石英晶体 原石', '白石英 矿物 标本', 'quartz crystal mineral'],
        '方解石': ['方解石矿石', '方解石晶体 原石', '冰洲石 矿物 标本', 'calcite mineral'],
        '黄铁矿': ['黄铁矿矿石', '黄铁矿晶体 原石', '愚人金 矿物 标本', 'pyrite mineral'],
        '赤铁矿': ['赤铁矿矿石', '赤铁矿原石 标本', '镜铁矿 矿物', 'hematite mineral'],
        '长石': ['长石矿石', '钾长石 原石 标本', '钠长石 矿物', 'feldspar mineral'],
    }
    SAVE_ROOT = './ore_images'
    PAGES_PER_QUERY = 6  # 每个搜索词最多 6 页，约 180 条结果

    seen_hashes = collect_existing_hashes(SAVE_ROOT)
    print(f"已有图片指纹 {len(seen_hashes)} 个（作为全局去重基准）\n")

    for ore, queries in ORE_QUERIES.items():
        spider = BaiduImageSpider(ore, save_root=SAVE_ROOT, seen_hashes=seen_hashes)
        spider.queries = queries
        spider.run(pages_per_query=PAGES_PER_QUERY)
        print()

    # 爬取结束后统计每个类别的图片数量
    print("各类别图片数量统计：")
    for ore in ORE_QUERIES:
        folder = os.path.join(SAVE_ROOT, ore)
        count = len(os.listdir(folder)) if os.path.isdir(folder) else 0
        print(f"  {ore}: {count} 张")
    print(f"\n所有矿石图片爬取完成！保存在 {os.path.abspath(SAVE_ROOT)} 目录下")
