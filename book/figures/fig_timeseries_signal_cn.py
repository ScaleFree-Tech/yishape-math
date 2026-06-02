#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图7.1/7.2 时间序列与信号处理
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def savefig(fig, name):
    fig.savefig(f'/home/reremouse/work/yishape-math/book/figures/{name}',
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {name}'); plt.close(fig)

n = 300; dt = 0.01; t = np.arange(n) * dt
fs = 1.0 / dt  # 100 Hz 采样率

# ========== 图7.1.1 傅里叶变换频率分解 ==========
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
# 原始信号 = 低频 + 高频 + 噪声
f1, f2, f3 = 2.0, 10.0, 30.0  # Hz
sig1 = 2*np.sin(2*np.pi*f1*t)
sig2 = 0.7*np.sin(2*np.pi*f2*t)
sig3 = 0.3*np.sin(2*np.pi*f3*t)
noise = np.random.randn(n)*0.2
sig = sig1 + sig2 + sig3 + noise
axes[0].plot(t, sig, color='#3498DB', lw=1.2, alpha=0.8)
axes[0].set_title('原始信号 s(t) = sin(2π·2t) + 0.7·sin(2π·10t) + 0.3·sin(2π·30t) + 噪声',
                  fontsize=12, fontweight='bold')
axes[0].set_ylabel('幅值', fontsize=11)
# 频谱
freqs = np.fft.rfftfreq(n, d=dt)
fft_vals = np.abs(np.fft.rfft(sig))
axes[1].stem(freqs[:80], fft_vals[:80], linefmt='#E74C3C', markerfmt='ro', basefmt='gray')
axes[1].set_title('傅里叶变换频谱 |S(f)|（峰值对应各频率成分）',
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('频率 f (Hz)', fontsize=11)
axes[1].set_ylabel('幅值', fontsize=11)
axes[1].set_xlim(0, 40)
# 滤波后重建 - 低通截止 15Hz
sos = scipy_signal.butter(4, 15, btype='low', fs=fs, output='sos')
filtered = scipy_signal.sosfilt(sos, sig)
axes[2].plot(t, filtered, color='#27AE60', lw=1.5)
axes[2].set_title('低通滤波后（截止频率 15Hz，保留低频，滤除高频噪声）',
                  fontsize=12, fontweight='bold')
axes[2].set_xlabel('时间 t (s)', fontsize=11)
axes[2].set_ylabel('幅值', fontsize=11)
plt.tight_layout()
savefig(fig, 'fig_7_1_1_fourier_decompose.png')

# ========== 图7.1.2 滑动平均与指数平滑 ==========
fig, axes = plt.subplots(3, 1, figsize=(14, 9))
np.random.seed(0)
trend = np.linspace(0, 5, n)
seasonal = 1.5 * np.sin(t/3)
noise = np.random.randn(n)*0.3
y = trend + seasonal + noise
axes[0].plot(t, y, color='#3498DB', lw=1.2, alpha=0.7, label='原始序列')
axes[0].plot(t, trend + seasonal, color='#E74C3C', lw=2, ls='--', label='真实趋势+季节')
axes[0].set_title('原始时间序列（含趋势、季节、噪声）', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].set_ylabel('y', fontsize=11)
# 滑动平均
window = 15
ma = np.convolve(y, np.ones(window)/window, mode='same')
axes[1].plot(t, y, color='#3498DB', lw=1, alpha=0.5, label='原始')
axes[1].plot(t, ma, color='#E74C3C', lw=2.5, label=f'滑动平均 (窗口={window})')
axes[1].set_title('滑动平均（MA）：平滑序列，暴露趋势', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].set_ylabel('y', fontsize=11)
# 指数平滑
def exp_smoothing(y, alpha):
    result = np.zeros(len(y))
    result[0] = y[0]
    for i in range(1, len(y)):
        result[i] = alpha*y[i] + (1-alpha)*result[i-1]
    return result
axes[2].plot(t, y, color='#3498DB', lw=1, alpha=0.5, label='原始')
for alpha, color in [(0.3, '#27AE60'), (0.7, '#F39C12')]:
    es = exp_smoothing(y, alpha)
    axes[2].plot(t, es, color=color, lw=2, label=f'指数平滑 α={alpha}')
axes[2].set_title('指数平滑（EWMA）：近期数据权重更大', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].set_xlabel('时间 t', fontsize=11)
axes[2].set_ylabel('y', fontsize=11)
plt.tight_layout()
savefig(fig, 'fig_7_1_2_smoothing_methods.png')

# ========== 图7.1.3 差分与平稳性 ==========
fig, axes = plt.subplots(3, 1, figsize=(14, 9))
np.random.seed(0)
# 非平稳：带趋势
t_seq = np.arange(200)
y_nonstationary = 0.05*t_seq + np.random.randn(200)*0.5 + 2*np.sin(t_seq/10)
axes[0].plot(t_seq, y_nonstationary, color='#3498DB', lw=2)
axes[0].set_title('非平稳序列（有趋势）：均值随时间变化',
                  fontsize=12, fontweight='bold')
axes[0].set_ylabel('y', fontsize=11)
# 一阶差分
y_diff1 = np.diff(y_nonstationary)
axes[1].plot(t_seq[1:], y_diff1, color='#E74C3C', lw=2)
axes[1].set_title('一阶差分 Δyt = yt - yt-1（去除趋势）',
                  fontsize=12, fontweight='bold')
axes[1].set_ylabel('Δyt', fontsize=11)
# 二阶差分（季节差分）
y_seasonal = 3*np.sin(t_seq/20) + np.random.randn(200)*0.5
y_diff2 = np.diff(y_seasonal, n=2)
axes[2].plot(t_seq[2:], y_diff2, color='#27AE60', lw=2)
axes[2].set_title('二阶差分（季节性差分）：去除周期性波动',
                  fontsize=12, fontweight='bold')
axes[2].set_xlabel('时间 t', fontsize=11)
axes[2].set_ylabel('Δ2yt', fontsize=11)
plt.tight_layout()
savefig(fig, 'fig_7_1_3_differencing.png')

# ========== 图7.2.1 滤波器类型 ==========
fig, axes = plt.subplots(4, 1, figsize=(14, 11))
fs = 1000; nyq = fs/2
# 低通
sos_lp = scipy_signal.butter(6, 300, btype='low', fs=fs, output='sos')
w, h = scipy_signal.sosfreqz(sos_lp, worN=2000, fs=fs)
axes[0].plot(w, 20*np.log10(np.abs(h)), color='#3498DB', lw=2)
axes[0].axvline(300, color='#E74C3C', ls='--', lw=1.5, label='截止频率 300Hz')
axes[0].set_title('低通滤波器：保留 <300Hz，衰减 >300Hz', fontsize=12, fontweight='bold')
axes[0].set_ylabel('幅值 (dB)', fontsize=11)
axes[0].legend(fontsize=10)
# 带通
sos_bp = scipy_signal.butter(6, [100, 400], btype='bandpass', fs=fs, output='sos')
w, h = scipy_signal.sosfreqz(sos_bp, worN=2000, fs=fs)
axes[1].plot(w, 20*np.log10(np.abs(h)), color='#27AE60', lw=2)
axes[1].set_title('带通滤波器：只保留 100~400Hz（提取特定频段信号）',
                  fontsize=12, fontweight='bold')
axes[1].set_ylabel('幅值 (dB)', fontsize=11)
# 高通
sos_hp = scipy_signal.butter(6, 100, btype='high', fs=fs, output='sos')
w, h = scipy_signal.sosfreqz(sos_hp, worN=2000, fs=fs)
axes[2].plot(w, 20*np.log10(np.abs(h)), color='#F39C12', lw=2)
axes[2].axvline(100, color='#E74C3C', ls='--', lw=1.5, label='截止频率 100Hz')
axes[2].set_title('高通滤波器：去除低频漂移（如直流分量、基线漂移）',
                  fontsize=12, fontweight='bold')
axes[2].set_ylabel('幅值 (dB)', fontsize=11)
axes[2].legend(fontsize=10)
# 时域对比
np.random.seed(0)
time = np.arange(500)/fs*1000  # ms
raw_sig = np.sin(2*np.pi*50*time/1000) + 0.5*np.sin(2*np.pi*500*time/1000) + np.random.randn(500)*0.1
filt_sig = scipy_signal.sosfilt(sos_lp, raw_sig)
axes[3].plot(time, raw_sig, alpha=0.5, color='#3498DB', lw=1, label='原始信号（含高频噪声）')
axes[3].plot(time, filt_sig, color='#E74C3C', lw=1.5, label='低通滤波后')
axes[3].set_title('时域信号对比：低频正弦波 + 高频噪声 → 滤波后平滑',
                  fontsize=12, fontweight='bold')
axes[3].set_xlabel('时间 (ms)', fontsize=11)
axes[3].set_ylabel('幅值', fontsize=11)
axes[3].legend(fontsize=10)
plt.tight_layout()
savefig(fig, 'fig_7_2_1_filter_types.png')

print("时间序列与信号处理 figures done!")
