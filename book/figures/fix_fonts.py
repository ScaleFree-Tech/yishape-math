#!/usr/bin/env python3
"""Fix all figure scripts to use FontProperties for CJK text"""
import os, re

SCRIPT_HEADER = '''# -*- coding: utf-8 -*-
# 中文字体配置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

# 查找可用的中文支持字体
_noto_cjk = [f for f in fm.fontManager.ttflist if 'Noto Sans CJK JP' in f.name]
_uming = [f for f in fm.fontManager.ttflist if 'UMing' in f.name]

_cjk_fp = None
if _noto_cjk:
    # NotoSansCJK TTC 文件在 fname 中包含具体文件名
    for f in _noto_cjk:
        if 'Regular' in f.fname or 'Medium' in f.fname:
            _cjk_fp = FontProperties(fname=f.fname)
            break
    if _cjk_fp is None:
        _cjk_fp = FontProperties(fname=_noto_cjk[0].fname)
elif _uming:
    _cjk_fp = FontProperties(fname=_uming[0].fname)

def cjk(size=10, bold=False):
    """返回中文字体 FontProperties"""
    if _cjk_fp is None:
        return {'fontsize': size}
    fp = FontProperties(fname=_cjk_fp.fname)
    fp.set_size(size)
    if bold:
        fp.set_weight('bold')
    return fp

# 小清新风格设置
plt.style.use('seaborn-v0_8-whitegrid')

'''

def fix_script(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'FontProperties from matplotlib.font_manager import FontProperties' in content:
        return  # Already fixed
    
    lines = content.split('\n')
    new_lines = []
    header_done = False
    skip_old_header = False
    
    for line in lines:
        # Skip old font config lines
        if "plt.rcParams['font.sans-serif']" in line:
            continue
        if '中文字体配置' in line and not header_done:
            continue  # Skip old header marker
        if skip_old_header and line.strip() and not line.startswith('#') and not line.startswith('import'):
            skip_old_header = False
        if 'matplotlib.use' in line and "use('Agg')" in line:
            skip_old_header = True
            continue
        
        if not header_done:
            new_lines.append(SCRIPT_HEADER)
            header_done = True
            skip_old_header = False
        
        new_lines.append(line)
    
    # Remove the shebang and old first line since we have our own header
    while new_lines and (new_lines[0].startswith('#!') or new_lines[0].startswith('# -*- coding')):
        new_lines.pop(0)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    print(f'Fixed: {os.path.basename(path)}')

scripts_dir = '/home/reremouse/work/yishape-math/book/figures'
for fname in sorted(os.listdir(scripts_dir)):
    if fname.endswith('.py') and fname not in ['fix_fonts.py', 'fix_fonts_and_run.py']:
        fix_script(os.path.join(scripts_dir, fname))

print('All scripts fixed!')
