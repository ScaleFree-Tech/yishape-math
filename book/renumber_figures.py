#!/usr/bin/env python3
"""精准图号重写脚本：遍历所有markdown，找出所有图片引用，
按教材规范（图号=章.节.图序号）重写caption"""
import re, os

BASE = '/home/reremouse/work/yishape-math/book/'

# 收集所有文件中的图片引用
image_refs = {}  # md_file → [(line_num, figure_filename, caption_line_num), ...]

for root, dirs, files in os.walk(BASE):
    for fn in files:
        if not fn.endswith('.md'):
            continue
        fpath = os.path.join(root, fn)
        rel = os.path.relpath(fpath, BASE)
        with open(fpath, encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            m = re.match(r'(!\[[^\]]*\]\([^)]+\))', line)
            if m:
                md_link = m.group(1)
                fname_m = re.search(r'figures/(fig_[^)]+)', md_link)
                if fname_m:
                    fname = fname_m.group(1)
                    image_refs.setdefault(rel, []).append((i, fname, md_link))

# 统计
total = sum(len(v) for v in image_refs.values())
print(f"Found {total} image refs across {len(image_refs)} files")

# 打印每个文件的图片
for f, refs in sorted(image_refs.items()):
    print(f"\n{f}:")
    for line_num, fname, link in refs:
        print(f"  L{line_num+1}: {fname}")
