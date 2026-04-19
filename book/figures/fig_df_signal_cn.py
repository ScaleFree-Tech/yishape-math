#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图2.x DataFrame操作 + 图7.2/7.3 音频处理 + 图7.4 音乐挖掘
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def savefig(fig, name):
    fig.savefig(f'/home/reremouse/work/yishape-math/book/figures/{name}',
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {name}'); plt.close(fig)

# ========== 图2.1 DataFrame 结构可视化 ==========
fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(-0.5, 9); ax.set_ylim(-1, 8); ax.axis('off')
# 列标题行
col_labels = ['id', 'name', 'age', 'city', 'revenue', 'is_active', 'score']
col_widths = [0.8, 1.2, 0.8, 1.2, 1.0, 1.0, 0.8]
col_x = [0.5]
for w in col_widths[:-1]:
    col_x.append(col_x[-1] + w + 0.05)
# 表头
for cx, cl, cw in zip(col_x, col_labels, col_widths):
    ax.add_patch(plt.Rectangle((cx, 6.5), cw, 0.7, facecolor='#3498DB', edgecolor='white', lw=1.5))
    ax.text(cx+cw/2, 6.85, cl, ha='center', va='center', fontsize=10,
            fontweight='bold', color='white')
# 数据行
data = [
    [1, 'Alice', 28, '北京', 85000, True, 4.2],
    [2, 'Bob', 35, '上海', 92000, True, 3.8],
    [3, 'Carol', 24, '深圳', 71000, False, 4.5],
    [4, 'David', 42, '北京', 110000, True, 4.9],
]
row_colors = ['#EBF5FB', '#FDFEFE']
for ri, row in enumerate(data):
    y_base = 5.8 - ri * 1.3
    for cx, val, cw in zip(col_x, row, col_widths):
        color = '#E8F8F5' if ri%2==1 else '#FDFEFE'
        ax.add_patch(plt.Rectangle((cx, y_base-0.5), cw, 0.65,
                                   facecolor=row_colors[ri%2], edgecolor='#BDC3C7', lw=0.5))
        fc = '#2C3E50'
        ax.text(cx+cw/2, y_base-0.17, str(val), ha='center', va='center',
                fontsize=9, color=fc)
    ax.text(col_x[0]-0.15, y_base-0.17, f'行{ri}', fontsize=8, color='#95A5A6', va='center')
# 行列标注
ax.annotate('列（Column）\n= 特征 = 变量', xy=(4.5, 7.5), xytext=(7, 7.5),
            fontsize=10, color='#3498DB', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#3498DB', lw=2))
ax.annotate('行（Row）\n= 样本 = 观测', xy=(-0.2, 3.5), xytext=(-1.5, 3.5),
            fontsize=10, color='#E74C3C', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))
ax.set_title('DataFrame 数据结构\n'
             '行为样本（观测值），列为特征（变量）\n'
             '类型自动检测：数值列、字符串列、布尔列、缺失值标记',
             fontsize=13, fontweight='bold')
# 索引标注
ax.add_patch(plt.Rectangle((-0.7, 5.3), 0.5, 0.65, facecolor='#F39C12', edgecolor='white', lw=1))
ax.text(-0.45, 5.62, '索引\n(Index)', ha='center', va='center', fontsize=8,
        fontweight='bold', color='white')
for ri in range(4):
    ax.add_patch(plt.Rectangle((-0.7, 5.8-ri*1.3-0.5), 0.5, 0.65, facecolor='#F39C12', edgecolor='white', lw=0.5))
    ax.text(-0.45, 5.8-ri*1.3-0.17, str(ri), ha='center', va='center', fontsize=8, color='white')
savefig(fig, 'fig_2_1_1_dataframe_structure.png')

# ========== 图2.2 缺失值与数据清洗 ==========
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
np.random.seed(0)
# 原始有缺失的数据
data_orig = np.random.randn(50, 3) * 0.5 + np.array([5, 50, 100])
mask = np.random.rand(50, 3) < 0.15
data_missing = data_orig.copy()
data_missing[mask] = np.nan
ax = axes[0]
im = ax.imshow(np.where(mask, np.nan, data_orig), aspect='auto', cmap='Blues')
ax.set_title('缺失值热力图（红色=缺失）\n'
             '白色区域为缺失数据位置', fontsize=11, fontweight='bold')
ax.set_xlabel('特征'); ax.set_ylabel('样本')
# 插值前后
ax = axes[1]
valid_idx = ~np.isnan(data_missing[:,0])
valid_x = np.where(valid_idx)[0]
valid_y = data_missing[valid_idx, 0]
interp_y = np.interp(np.arange(len(data_missing)), valid_x, valid_y)
ax.plot(range(len(data_missing)), data_missing[:,0], 'o', ms=4, alpha=0.4,
        color='#E74C3C', label='原始数据（含缺失）')
ax.plot(range(len(data_missing)), interp_y, '-', lw=2, color='#27AE60', label='线性插值')
ax.set_title('缺失值处理：线性插值\n'
             '用相邻有效值的直线填充缺失区间', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
# 重复值检测
ax = axes[2]
dup_data = np.array([[1,2,3],[1,2,3],[4,5,6],[1,2,3],[7,8,9]])
for i, (row, color) in enumerate(zip(dup_data, ['#3498DB']*5)):
    for j, val in enumerate(row):
        ax.add_patch(plt.Rectangle((j*1.1, 4-i*0.9), 0.9, 0.7,
                                   facecolor=color, alpha=0.8 if i!=1 and i!=3 else 0.3,
                                   edgecolor='black', lw=0.5))
        ax.text(j*1.1+0.45, 4-i*0.9+0.35, str(val), ha='center', va='center', fontsize=10)
ax.text(1.65, 0.2, '↑ 重复行（第2行和第4行完全相同）', ha='center', fontsize=10,
        color='#E74C3C', fontweight='bold')
ax.set_xlim(-0.2, 4); ax.set_ylim(-0.5, 5)
ax.axis('off')
ax.set_title('重复值检测\n'
             '完全相同的行需要去重（dropDuplicates）', fontsize=11, fontweight='bold')
savefig(fig, 'fig_2_2_2_missing_outlier.png')

# ========== 图7.3.1 音频特征：波形 + 频谱 + MFCC ==========
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
np.random.seed(42)
sample_rate = 8000
duration = 1.5  # 秒
time = np.linspace(0, duration, int(sample_rate*duration))
# 合成语音-like信号（基频+谐波+噪声）
f0 = 200  # 基频
fundamental = np.sin(2*np.pi*f0*time)
harmonics = sum(0.3/k * np.sin(2*np.pi*f0*k*time) for k in range(2,6))
noise = np.random.randn(len(time)) * 0.1
audio = fundamental + harmonics + noise
# 波形
ax = axes[0]
ax.plot(time, audio, color='#3498DB', lw=0.8, alpha=0.8)
ax.set_title('音频波形（Waveform）：时域信号\n'
             '横轴=时间，纵轴=振幅', fontsize=12, fontweight='bold')
ax.set_ylabel('振幅', fontsize=11); ax.set_xlabel('')
# 频谱
ax = axes[1]
from scipy.fft import rfft, rfftfreq
fft_vals = np.abs(rfft(audio))
freqs = rfftfreq(len(audio), 1/sample_rate)
ax.plot(freqs[:100], fft_vals[:100], color='#E74C3C', lw=1.5)
ax.axvline(f0, color='#27AE60', lw=2, ls='--', label=f'基频 f0={f0}Hz')
for k in range(2, 6):
    ax.axvline(f0*k, color='#27AE60', lw=1, ls=':', alpha=0.6)
ax.set_title('频谱图（Spectrum）：频域表示\n'
             '峰值对应基频及其谐波（共振峰）', fontsize=12, fontweight='bold')
ax.set_ylabel('幅值', fontsize=11); ax.legend(fontsize=10); ax.set_xlabel('')
# MFCC示意
ax = axes[2]
n_mfcc = 13
mfcc_vals = np.sin(np.linspace(0, 2*np.pi, n_mfcc)) * np.exp(-np.linspace(0, 1, n_mfcc))
mfcc_vals += np.random.randn(n_mfcc)*0.1
ax.bar(range(n_mfcc), mfcc_vals, color='#9B59B6', alpha=0.8)
ax.set_title('MFCC（梅尔频率倒谱系数）：语音识别核心特征\n'
             '模拟人耳对频率的感知特性，保留语音关键信息',
             fontsize=12, fontweight='bold')
ax.set_xlabel('MFCC 系数编号', fontsize=11); ax.set_ylabel('系数值', fontsize=11)
plt.tight_layout()
savefig(fig, 'fig_7_3_1_audio_features.png')

# ========== 图7.4.1 音乐挖掘：节拍检测 + 和弦识别 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
np.random.seed(0)
# 振幅包络（ onset检测）
sample_rate2 = 22050
t_audio = np.linspace(0, 8, sample_rate2*8)
beat_times = [0.5, 1.0, 1.5, 2.0, 2.8, 3.3, 3.8, 4.2, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
energy = np.zeros(len(t_audio))
for bt in beat_times:
    idx = int(bt * sample_rate2)
    if idx < len(energy):
        energy[idx:idx+2000] = 1.0
energy_smooth = np.convolve(energy, np.ones(500)/500, mode='same')
ax = axes[0,0]
ax.plot(t_audio, energy_smooth, color='#3498DB', lw=1.5)
for bt in beat_times:
    ax.axvline(bt, color='#E74C3C', lw=1.5, ls='--', alpha=0.7)
ax.scatter(beat_times, [energy_smooth[int(b*sample_rate2)] for b in beat_times],
           s=80, color='#E74C3C', zorder=10, label='检测到的节拍')
ax.set_title('节拍检测（Beat Detection）\n'
             '通过振幅包络峰值检测音乐节拍位置', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.set_xlabel('时间 (s)'); ax.set_ylabel('能量')
# 频谱图
ax = axes[0,1]
# 简化的频谱图
freq_bins = 50; time_bins = 100
spec = np.random.rand(time_bins, freq_bins)
# 添加谐波结构
for f in range(0, freq_bins, 8):
    spec[:, f] += 2
    if f+1 < freq_bins: spec[:, f+1] += 1
    if f-1 >= 0: spec[:, f-1] += 1
spec = np.abs(np.fft.fft2(spec)[:, :freq_bins//2])
spec = spec / spec.max()
im = ax.imshow(spec.T, aspect='auto', origin='lower', cmap='magma',
               extent=[0, 8, 0, freq_bins//2])
ax.set_title('频谱图（Spectrogram）：时频二维表示\n'
             '颜色越亮=该时刻该频率能量越高', fontsize=11, fontweight='bold')
ax.set_xlabel('时间 (s)'); ax.set_ylabel('频率 (Hz)')
plt.colorbar(im, ax=ax, shrink=0.8)
# 和弦识别
ax = axes[1,0]
chord_names = ['C', 'Am', 'F', 'G', 'C', 'Am', 'F', 'G', 'D', 'Em', 'F', 'G']
chord_pitches = [
    [1,0,0,0,1,0,0,1,0,0,0,0],  # C: C E G
    [1,0,0,0,1,0,0,0,1,0,0,0],  # Am: A C E
    [0,0,0,1,0,0,0,1,0,0,1,0],  # F: F A C
    [0,0,0,1,0,0,0,1,0,0,0,1],  # G: G B D
    [1,0,0,0,1,0,0,1,0,0,0,0],
    [1,0,0,0,1,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,1,0,0,1,0],
    [0,0,0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,0,0,0,1,0,0,1,0],  # D: D F# A
    [1,0,0,0,0,0,0,0,1,0,0,0],  # Em: E G B
    [0,0,0,1,0,0,0,1,0,0,1,0],
    [0,0,0,1,0,0,0,1,0,0,0,1],
]
pitch_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
chroma_data = np.array(chord_pitches).T
im = ax.imshow(chroma_data, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(chord_names)))
ax.set_xticklabels(chord_names, fontsize=10)
ax.set_yticks(range(12))
ax.set_yticklabels(pitch_names, fontsize=9)
ax.set_title('和弦识别：色度图（Chromagram）\n'
             '每列表示一个和弦的音高组成', fontsize=11, fontweight='bold')
# 音乐推荐
ax = axes[1,1]
songs = ['歌曲A', '歌曲B', '歌曲C', '歌曲D', '歌曲E']
features = np.array([
    [0.8, 0.3, 0.6, 0.9, 0.2],  # A
    [0.6, 0.7, 0.4, 0.3, 0.8],  # B
    [0.3, 0.2, 0.9, 0.7, 0.1],  # C
    [0.9, 0.5, 0.3, 0.2, 0.7],  # D
    [0.4, 0.8, 0.5, 0.6, 0.3],  # E
])
im = ax.imshow(features, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(5)); ax.set_xticklabels(['能量', '情感', '节奏', '人声', '舞蹈'], fontsize=10)
ax.set_yticks(range(5)); ax.set_yticklabels(songs, fontsize=10)
ax.set_title('音乐特征矩阵 → 协同过滤推荐\n'
             '计算用户-物品相似度，推荐相似歌曲', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
savefig(fig, 'fig_7_4_1_music_mining.png')

print("DataFrame and audio/music figures done!")
