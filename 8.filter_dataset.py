"""按人工复核清单过滤测试集中的低质量/错标/无关图片，生成可直接训练评估的新数据集目录。

过滤标准（只保留满足全部条件的图片）：
  1. 是清晰的矿石/矿物标本照片，而不是广告、截图、文档、书页、拼图、厂房、风景、人物；
  2. 画面主体不是珠宝首饰、手串、粉末、砂料等加工/商品图；
  3. 画面主体与文件夹标签一致，不是明显其他矿物（如方解石目录里的黄铁矿）；
  4. 是单张照片，不是多格拼图、紫外灯多图、带大量文字标注的网店图。

输出：
  ore_images_filtered/train/  （原样保留，供训练）
  ore_images_filtered/test/   （仅保留复核通过的图片）
  ore_images_filtered/filter_manifest.csv  （被剔除文件的清单和原因）

注意：过滤后的测试集指标不能和过滤前的指标直接比较，报告里应同时给出两个数字。
"""
import csv
import os
import shutil

SPLIT_DATA_DIR = './ore_images_split'
FILTERED_DIR = './ore_images_filtered'
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}

# 依据 _review/sheet_*.jpg 的人工复核结果（编号 = 按类别、文件名排序后的全局序号）
REMOVE = {
    '方解石': [25, 34, 35, 36, 38, 40, 42, 43, 50, 53, 55, 56, 58, 60, 61, 63, 66, 69, 73, 76,
              91, 92, 93, 95, 103, 106, 107, 110, 114, 115, 118],
    '石英': [135, 137, 138, 139, 140, 141, 142, 143, 144, 146, 147, 148, 149, 150, 151, 152,
             153, 154, 155, 156, 158, 159, 163, 164, 170, 171, 173, 174, 176, 179, 181, 182,
             183, 184, 185, 186, 191, 197, 204, 206, 208, 212, 213, 214, 223, 225, 226, 233,
             239, 240, 244, 249, 250, 253, 254, 255, 258, 259, 260, 261, 262, 266, 267, 270,
             271, 272, 274, 282, 283, 284, 285, 288, 289, 291, 293, 296, 297],
    '赤铁矿': [312, 317, 319, 320, 321, 322, 323, 325, 326, 328, 329, 330, 340, 341, 342, 343,
               346, 350, 351, 357, 359, 361, 362, 364, 366, 367, 371, 373, 374, 375, 379, 383,
               387, 388, 389, 390, 394, 396, 398, 399, 400, 401, 402, 406, 407, 408, 412, 414,
               415, 416, 417, 418, 419, 420, 422, 424, 425, 426, 427, 428, 429, 430, 431, 432,
               433, 434, 436, 437],
    '长石': [439, 440, 446, 447, 448, 449, 450, 451, 452, 455, 456, 457, 458, 459, 460, 461,
             466, 467, 468, 471, 472, 473, 474, 477, 479, 480, 481, 482, 483, 484, 485, 486,
             487, 488, 491, 492, 499, 500, 501, 505, 506, 507, 509, 510],
    '黄铁矿': [523, 524, 526, 535, 539, 540, 541, 543, 560, 576, 580, 591, 597, 598, 599, 602,
               604, 605, 608, 611, 614, 615, 617, 620, 628, 630, 631, 635, 640, 641, 644],
}

REASON = '人工复核剔除：无关/广告/截图/拼图/非矿石物体/明显错标或非单一标本'

# 复核清单是按当前 8:2 划分结果生成的；如果重新爬取/重新划分导致数量变化，
# 编号会整体错位，必须先重新人工复核，不能直接沿用。
EXPECTED_TEST_COUNTS = {'方解石': 133, '石英': 167, '赤铁矿': 138, '长石': 74, '黄铁矿': 136}


def list_images(folder):
    return sorted(
        name for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name))
        and os.path.splitext(name)[1].lower() in VALID_EXTS
    )


def link_or_copy(src, dst):
    """优先硬链接节省空间；失败则复制"""
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    if os.path.isdir(FILTERED_DIR):
        print(f'清理旧的过滤结果：{FILTERED_DIR}')
        shutil.rmtree(FILTERED_DIR)

    classes = sorted(
        name for name in os.listdir(os.path.join(SPLIT_DATA_DIR, 'train'))
        if os.path.isdir(os.path.join(SPLIT_DATA_DIR, 'train', name))
    )

    # 数量校验：防止数据集被重新划分后，旧编号错误地删掉别的图片
    for cls in classes:
        actual = len(list_images(os.path.join(SPLIT_DATA_DIR, 'test', cls)))
        expected = EXPECTED_TEST_COUNTS.get(cls)
        if expected is not None and actual != expected:
            print(f'测试集「{cls}」数量为 {actual}，与复核时的 {expected} 不一致。')
            print('说明数据集已变化，旧的人工复核编号不能再用，请重新复核后再过滤。')
            return

    manifest = []

    for split in ('train', 'test'):
        for cls in classes:
            src_dir = os.path.join(SPLIT_DATA_DIR, split, cls)
            dst_dir = os.path.join(FILTERED_DIR, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            files = list_images(src_dir)

            # 复核编号是按“类别、文件名排序”的全局序号；这里按同样顺序重建
            global_offset = sum(
                len(list_images(os.path.join(SPLIT_DATA_DIR, 'test', c)))
                for c in classes[:classes.index(cls)]
            )
            removed = set()
            if split == 'test':
                removed = set(REMOVE.get(cls, []))
                # REMOVE 中存的是全局编号，换算成本类内序号
                removed = {i - global_offset for i in removed}
            kept = 0
            for local_idx, name in enumerate(files):
                src = os.path.join(src_dir, name)
                if local_idx in removed:
                    manifest.append([split, cls, name, REASON])
                    continue
                link_or_copy(src, os.path.join(dst_dir, name))
                kept += 1
            total = len(files)
            print(f'{split}/{cls}: 保留 {kept}/{total} | 剔除 {total - kept}')

    os.makedirs(FILTERED_DIR, exist_ok=True)
    with open(os.path.join(FILTERED_DIR, 'filter_manifest.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['split', 'class', 'filename', 'reason'])
        w.writerows(manifest)

    print(f'\n过滤后数据集：{os.path.abspath(FILTERED_DIR)}')
    print(f'剔除清单：{os.path.abspath(os.path.join(FILTERED_DIR, "filter_manifest.csv"))}（共 {len(manifest)} 张）')


if __name__ == '__main__':
    main()
