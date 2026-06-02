#!/usr/bin/env python3
"""Add Chinese font config and fix CLT layout, then run all scripts"""
import subprocess, sys

FONT_CONFIG = '''# 中文字体配置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

'''

scripts = [
    'fig_pca_svd.py',
    'fig_ml_concepts.py',
    'fig_timeseries_optimization.py',
    'fig_classification_metrics.py',
    'fig_ensemble_neural.py',
]

for script in scripts:
    path = f'/home/reremouse/work/yishape-math/book/figures/{script}'
    with open(path, 'r') as f:
        content = f.read()
    
    if '中文字体配置' not in content:
        # Remove old matplotlib.use lines and add font config at top
        lines = content.split('\n')
        new_lines = []
        skip_until_imports = True
        for line in lines:
            if skip_until_imports and not line.startswith('#') and line.strip():
                if line.startswith('import'):
                    new_lines.append(FONT_CONFIG)
                    skip_until_imports = False
            new_lines.append(line)
        content = '\n'.join(new_lines)
        
        with open(path, 'w') as f:
            f.write(content)
        print(f'Fixed fonts: {script}')
    else:
        print(f'Already has font config: {script}')

print('Done fixing fonts')
