#!/usr/bin/env python3
"""修正图编号：从文件路径提取章节号，从文件名提取图序号"""
import re, os, shutil

base = "/home/reremouse/work/yishape-math/book/"

# 修复1: 4.2.md 中错误放置的 4.1 图 → 重命名为 4.2 并更新引用
# fig_4_1_1_six_distributions.png → fig_4_2_1_six_distributions.png
# fig_4_1_3_qq_plot.png → fig_4_2_3_qq_plot.png
renames = {
    "fig_4_1_1_six_distributions.png": "fig_4_2_1_six_distributions.png",
    "fig_4_1_3_qq_plot.png": "fig_4_2_3_qq_plot.png",
    "fig_4_1_2_clt_detail.png": "fig_4_2_2_clt_detail.png",
    # 5.1 中 5.1.1 → 5.1.1 正确（filename 5_1_1 = section 5.1，对）
    # 5.2 中 5.2.1 → 5.2.1 正确
}

for old_name, new_name in renames.items():
    old_path = base + f"figures/{old_name}"
    new_path = base + f"figures/{new_name}"
    if os.path.exists(old_path) and not os.path.exists(new_path):
        os.rename(old_path, new_path)
        print(f"RENAME: {old_name} → {new_name}")
    elif os.path.exists(new_path):
        print(f"ALREADY EXISTS: {new_name}")
    else:
        print(f"NOT FOUND: {old_path}")

# 修复2: 更新markdown中的引用
fixes = {
    ("4.2.md", "fig_4_1_1_six_distributions.png"): ("fig_4_2_1_six_distributions.png", "图4.2.1", "六大分布对比"),
    ("4.2.md", "fig_4_1_3_qq_plot.png"): ("fig_4_2_3_qq_plot.png", "图4.2.3", "Q-Q图"),
    ("4.2.md", "fig_4_1_2_clt_detail.png"): ("fig_4_2_2_clt_detail.png", "图4.2.2", "CLT详细演示"),
}

for (md_file, old_fig), (new_fig, new_num, new_title) in fixes.items():
    path = base + f"Chapter4_Statistics/{md_file}"
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 替换 ![old_title](old_fig) 为 ![new_title](new_fig)
    # 同时替换 *图X.X.X old_title...* 为 *图X.X.X new_title...*
    content = content.replace(f"![六大分布对比](../figures/{old_fig})",
                               f"![{new_title}](../figures/{new_fig})")
    content = content.replace(f"![{new_title}](../figures/{new_fig})\n*图4.4.2 六大分布对比",
                               f"![{new_title}](../figures/{new_fig})\n*{new_num} {new_title}。")
    content = content.replace(f"![Q-Q图](../figures/{old_fig})",
                               f"![{new_title}](../figures/{new_fig})")
    content = content.replace(f"![{new_title}](../figures/{new_fig})\n*图4.4.2 Q-Q图",
                               f"![{new_title}](../figures/{new_fig})\n*{new_num} {new_title}。")
    content = content.replace(f"![CLT详细演示](../figures/{old_fig})",
                               f"![{new_title}](../figures/{new_fig})")
    content = content.replace(f"![{new_title}](../figures/{new_fig})\n*图4.4.1 CLT详细演示",
                               f"![{new_title}](../figures/{new_fig})\n*{new_num} {new_title}。")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"FIXED refs in {md_file}: {old_fig} → {new_fig}")

# 修复3: 删除错误章节的重复图（4.1.md中错误插入了来自4.2的图）
# 检查 4.1.md 是否有来自 4.2 的图的引用
path_41 = base + "Chapter4_Statistics/4.1.md"
with open(path_41, 'r', encoding='utf-8') as f:
    c41 = f.read()

# 查找所有 ![...](fig_4_2_*.png) 引用
bad_refs = re.findall(r'!\[[^\]]*\]\(\.\.\/figures\/fig_4_2_[^\)]+\)', c41)
print(f"\n4.1.md has bad 4.2 refs: {bad_refs}")

# 这些需要从4.1.md中移除（它们属于4.2.md）
# 只移除新插入的（包含 *图4.4.1 或 *图4.4.2 的部分）
# 找所有 *图4.4.1 或 *图4.4.2 的段落并删除
c41_lines = c41.split('\n')
new_c41 = []
skip_next = False
for i, line in enumerate(c41_lines):
    if re.search(r'\*\s*图4\.4\.[12]\s', line):
        # 跳过这一行（标题行）
        # 同时检查上一行是否是图片行
        if new_c41 and new_c41[-1].startswith('!['):
            new_c41.pop()  # 移除图片行
            continue
        continue
    new_c41.append(line)

with open(path_41, 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_c41))
print(f"Cleaned 4.1.md - removed {len(c41_lines) - len(new_c41)} bad lines")

# 修复4: 清理 4.2.md 中错误章节号的插图说明
# 把所有 *图4.4.2 * → *图4.2.1 * 或 *图4.2.2 * 等
path_42 = base + "Chapter4_Statistics/4.2.md"
with open(path_42, 'r', encoding='utf-8') as f:
    c42 = f.read()

# 修正 CLT detail (fig_4_2_2 → 图4.2.2)
c42 = c42.replace('*图4.4.1 CLT详细演示。', '*图4.2.2 CLT详细演示。')
# 修正 Q-Q plot (fig_4_2_3 → 图4.2.3)
c42 = c42.replace('*图4.4.2 Q-Q图。', '*图4.2.3 Q-Q图。')
# 修正 distributions (fig_4_2_1 → 图4.2.1)
c42 = c42.replace('*图4.4.2 六大分布对比。正态分布', '*图4.2.1 六大分布对比。正态分布')

with open(path_42, 'w', encoding='utf-8') as f:
    f.write(c42)
print("Fixed 4.2.md chapter numbers")

print("\nAll fixes applied!")
