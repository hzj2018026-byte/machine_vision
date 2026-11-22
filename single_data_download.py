"""单类别矿石爬虫：只爬一个关键词，用于调试或快速补充某一类数据。

用法：
    python single_data_download.py              # 默认爬「石英」，最多 10 页
    python single_data_download.py 黄铁矿 5      # 指定关键词和页数
"""

import json
import os
import random
import sys
import time
from urllib.parse import urlparse, quote

import requests


class BaiduImageSpider:
    def __init__(self, word, save_root='./ore_images'):
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
        self.word = word
        self.i = 0
        self.seen_urls = set()
        self.directory = os.path.join(save_root, word)
        os.makedirs(self.directory, exist_ok=True)

    @staticmethod
    def fix_image_url(img_url):
        if not img_url:
            return None
        parsed = urlparse(img_url)
        if not parsed.scheme:
            fixed = f'https://{img_url}'
            parsed_fixed = urlparse(fixed)
            if parsed_fixed.scheme and parsed_fixed.netloc:
                return fixed
            return None
        return img_url.replace('http://', 'https://')

    @staticmethod
    def get_extension(content_type):
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

    def get_image_urls(self, page):
        try:
            url = self.url.format(quote(self.word), page * 30)
            response = requests.get(url=url, headers=self.headers, timeout=10)
            data = json.loads(response.text)

            if 'antiFlag' in data:
                print(f"[{self.word}] 第{page + 1}页被反爬拦截：{data.get('message')}")
                return []

            payload = data.get('data') or {}
            if isinstance(payload, dict):
                items = payload.get('images') or []
            else:
                items = payload

            image_urls = []
            for item in items:
                if not isinstance(item, dict):
                    continue
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
            if len(content) < 1024:
                return False

            self.i += 1
            ext = self.get_extension(content_type)
            filename = os.path.join(self.directory, f'{self.word}_{self.i:03d}{ext}')
            with open(filename, 'wb') as f:
                f.write(content)
            return True
        except Exception:
            return False

    def run(self, max_pages=10):
        print(f'开始爬取「{self.word}」图片（最多 {max_pages} 页）...')
        for page in range(max_pages):
            image_urls = self.get_image_urls(page)
            if not image_urls:
                break
            for img_url in image_urls:
                self.download_image(img_url)
                time.sleep(random.uniform(0.2, 0.4))
            time.sleep(random.uniform(1.0, 2.0))
        print(f'「{self.word}」爬取完成，共下载 {self.i} 张，保存于 {os.path.abspath(self.directory)}')


if __name__ == '__main__':
    word = sys.argv[1] if len(sys.argv) > 1 else '石英'
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    spider = BaiduImageSpider(word)
    spider.run(max_pages=max_pages)
