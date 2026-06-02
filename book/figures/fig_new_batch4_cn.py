#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新增图批次4：Ch7时间序列/信号/音频"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def savefig(fig, name):
    fig.savefig(f'/home/reremouse/work/yishape-math/book/figures/{name}', dpi=150, bbox_inches='tight')
    print(f"Saved: {name}")

# ============================================================
# 图7.1.4 ARIMA结构示意
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('图7.1.4 ARIMA模型结构 | ARIMA Model Structure', fontsize=14, fontweight='bold')

np.random.seed(42)
n = 200
t = np.arange(n)
# AR(1) process: y_t = 0.7*y_{t-1} + noise
y_ar = np.zeros(n)
y_ar[0] = np.random.randn()
for i in range(1, n):
    y_ar[i] = 0.7 * y_ar[i-1] + np.random.randn()

# MA(2) process: y_t = noise + 0.3*noise_{t-1} + 0.1*noise_{t-2}
noise = np.random.randn(n)
y_ma = np.zeros(n)
for i in range(n):
    y_ma[i] = noise[i] + 0.3*noise[i-1] if i>0 else noise[i]

# ARMA(1,1)
y_arma = np.zeros(n)
for i in range(1, n):
    y_arma[i] = 0.5*y_arma[i-1] + noise[i] + 0.2*noise[i-1]

# ARIMA(1,1,1) - differenced ARMA
y_arima = np.diff(y_arma)

ax = axes[0,0]
ax.plot(t, y_ar, color='#3498DB', lw=1.2, alpha=0.8)
ax.set_title('AR(1)：自回归过程（当前值依赖前一步）\ny_t = 0.7·y_{t-1} + ε_t', fontsize=11)
ax.set_xlabel('时间 t', fontsize=10)
ax.set_ylabel('y_t', fontsize=10)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.95, 'p=1（只看前1步）', transform=ax.transAxes, fontsize=9,
        va='top', bbox=dict(boxstyle='round', facecolor='#E3F2FD'))

ax2 = axes[0,1]
ax2.plot(t, y_ma, color='#E74C3C', lw=1.2, alpha=0.8)
ax2.set_title('MA(2)：移动平均过程（噪声的加权滑动）\ny_t = ε_t + 0.3·ε_{t-1}', fontsize=11)
ax2.set_xlabel('时间 t', fontsize=10)
ax2.set_ylabel('y_t', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.text(0.02, 0.95, 'q=2（滑动窗口宽度2）', transform=ax2.transAxes, fontsize=9,
         va='top', bbox=dict(boxstyle='round', facecolor='#FFEBEE'))

ax3 = axes[1,0]
ax3.plot(t, y_arma, color='#27AE60', lw=1.2, alpha=0.8)
ax3.set_title('ARMA(1,1)：混合过程（AR+MA）\ny_t = 0.5·y_{t-1} + ε_t + 0.2·ε_{t-1}', fontsize=11)
ax3.set_xlabel('时间 t', fontsize=10)
ax3.set_ylabel('y_t', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.text(0.02, 0.95, 'p=1, q=1', transform=ax3.transAxes, fontsize=9,
         va='top', bbox=dict(boxstyle='round', facecolor='#E8F5E9'))

# ARIMA: show differencing effect
ax4 = axes[1,1]
ax4.plot(t[:-1], y_arima, color='#9B59B6', lw=1.2, alpha=0.8)
ax4.set_title('ARIMA(1,1,1)：差分后的ARMA\n先做一阶差分（去掉趋势）→ ARMA平稳过程', fontsize=11)
ax4.set_xlabel('时间 t', fontsize=10)
ax4.set_ylabel('Δy_t', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.text(0.02, 0.95, 'd=1（一阶差分）', transform=ax4.transAxes, fontsize=9,
         va='top', bbox=dict(boxstyle='round', facecolor='#F3E5F5'))

plt.tight_layout(rect=[0, 0, 1, 0.96])
savefig(fig, 'fig_7_1_4_arima_structure.png')

# ============================================================
# 图7.2.2 语谱图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('图7.2.2 语谱图 | Spectrogram', fontsize=14, fontweight='bold')

fs = 8000  # 8kHz采样率
t_vowel = np.linspace(0, 1, fs)
# 模拟元音 "a" - 基频f0=200Hz, 共振峰F1=800Hz, F2=1200Hz
f0 = 200
F1, F2, F3 = 800, 1200, 2400
vowel_a = (np.sin(2*np.pi*f0*t_vowel) * 0.5 +
           np.sin(2*np.pi*F1*t_vowel) * 0.3 +
           np.sin(2*np.pi*F2*t_vowel) * 0.2 +
           np.sin(2*np.pi*F3*t_vowel) * 0.1 +
           np.random.randn(fs)*0.05)
# 模拟静音段
silence = np.zeros(fs//4)

signal_combined = np.concatenate([silence, vowel_a, silence])

ax = axes[0,0]
time_axis = np.linspace(0, len(signal_combined)/fs, len(signal_combined))
ax.plot(time_axis, signal_combined, color='#3498DB', lw=0.8, alpha=0.8)
ax.set_title('时域波形：元音 "a" + 静音', fontsize=12)
ax.set_xlabel('时间 (s)', fontsize=11)
ax.set_ylabel('幅值', fontsize=11)
ax.set_xlim(0, 1.5)

# 语谱图（窄带）
f, t_spec, Sxx = scipy_signal.spectrogram(signal_combined, fs=fs, nperseg=512, noverlap=400)
ax2 = axes[0,1]
im = ax2.pcolormesh(t_spec, f[:80], 10*np.log10(Sxx[:80,:]+1e-10), cmap='magma', shading='gouraud')
ax2.set_title('语谱图（窄带：频率分辨率高）', fontsize=12)
ax2.set_xlabel('时间 (s)', fontsize=11)
ax2.set_ylabel('频率 (Hz)', fontsize=11)
plt.colorbar(im, ax=ax2, shrink=0.8, label='功率/dB')
ax2.axhline(f0, color='cyan', lw=1, ls='--', label=f'基频 f0≈{f0}Hz')
ax2.axhline(F1, color='yellow', lw=1, ls=':', label=f'F1≈{F1}Hz')
ax2.axhline(F2, color='orange', lw=1, ls=':', label=f'F2≈{F2}Hz')
ax2.legend(fontsize=8, loc='upper right')
ax2.set_ylim(0, 4000)

# 宽带语谱图
ax3 = axes[1,0]
f_wide, t_wide, Sxx_wide = scipy_signal.spectrogram(signal_combined, fs=fs, nperseg=128, noverlap=96)
im3 = ax3.pcolormesh(t_wide, f_wide[:80], 10*np.log10(Sxx_wide[:80,:]+1e-10), cmap='magma', shading='gouraud')
ax3.set_title('语谱图（宽带：时间分辨率高）', fontsize=12)
ax3.set_xlabel('时间 (s)', fontsize=11)
ax3.set_ylabel('频率 (Hz)', fontsize=11)
plt.colorbar(im3, ax=ax3, shrink=0.8, label='功率/dB')
ax3.set_ylim(0, 4000)

# FFT频谱（单帧）
ax4 = axes[1,1]
frame_start = fs // 2
frame = signal_combined[frame_start:frame_start+1024]
fft_frame = np.abs(np.fft.rfft(frame * np.hanning(1024)))
freqs_frame = np.fft.rfftfreq(1024, d=1/fs)
ax4.plot(freqs_frame[:200], 20*np.log10(fft_frame[:200]+1e-10), color='#2C3E50', lw=1.5)
for f_val, label, col in [(f0, 'f0(基频)', 'cyan'), (F1, 'F1(第1共振峰)', 'yellow'),
                            (F2, 'F2(第2共振峰)', 'orange'), (F3, 'F3', 'red')]:
    ax4.axvline(f_val, color=col, lw=1.5, ls='--', alpha=0.8, label=f'{label}={f_val}Hz')
ax4.set_title('单帧FFT频谱（1秒处）', fontsize=12)
ax4.set_xlabel('频率 (Hz)', fontsize=11)
ax4.set_ylabel('功率/dB', fontsize=11)
ax4.legend(fontsize=8, loc='upper right')
ax4.set_xlim(0, 3500)
ax4.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
savefig(fig, 'fig_7_2_2_spectrogram.png')

# ============================================================
# 图7.2.3 小波变换
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('图7.2.3 小波变换 | Wavelet Transform', fontsize=14, fontweight='bold')

np.random.seed(0)
t_wavelet = np.linspace(0, 4, 2000)
# 信号：低频 → 高频 跳变
sig_wavelet = np.concatenate([
    np.sin(2*np.pi*5*t_wavelet[:500]),
    np.sin(2*np.pi*20*t_wavelet[500:1000]),
    np.sin(2*np.pi*5*t_wavelet[1000:1500]),
    np.sin(2*np.pi*40*t_wavelet[1500:])
]) + np.random.randn(2000)*0.2

ax = axes[0,0]
ax.plot(t_wavelet, sig_wavelet, color='#3498DB', lw=1, alpha=0.8)
ax.set_title('原始信号（频率随时间变化）', fontsize=12)
ax.set_xlabel('时间 (s)', fontsize=11)
ax.set_ylabel('幅值', fontsize=11)
ax.grid(True, alpha=0.3)

# 短时傅里叶变换 (STFT)
ax2 = axes[0,1]
f_stft, t_stft, Zxx = scipy_signal.stft(sig_wavelet, fs=1/(t_wavelet[1]-t_wavelet[0]),
                                          nperseg=128, noverlap=96)
im2 = ax2.pcolormesh(t_stft, f_stft[:50], np.abs(Zxx[:50,:]), cmap='viridis', shading='gouraud')
ax2.set_title('STFT（固定窗：频率分辨率固定）', fontsize=12)
ax2.set_xlabel('时间 (s)', fontsize=11)
ax2.set_ylabel('频率 (Hz)', fontsize=11)
plt.colorbar(im2, ax=ax2, shrink=0.8)

# 连续小波变换 (CWT) using Morlet wavelet
ax3 = axes[1,0]
# Use scipy's cwt with moral wavelet
widths = np.logspace(np.log10(3), np.log10(200), 100)
cwtmatr = scipy_signal.cwt(sig_wavelet, scipy_signal.ricker, widths)
im3 = ax3.pcolormesh(t_wavelet, widths, np.abs(cwtmatr), cmap='viridis', shading='gouraud')
ax3.set_yscale('log')
ax3.set_title('CWT小波变换（多尺度：高频→细尺度，低频→粗尺度）', fontsize=12)
ax3.set_xlabel('时间 (s)', fontsize=11)
ax3.set_ylabel('尺度（与频率成反比）', fontsize=11)
plt.colorbar(im3, ax=ax3, shrink=0.8)

# 小波基底示意
ax4 = axes[1,1]
t_mother = np.linspace(-2, 2, 500)
# Morlet小波
mother_wavelet = np.exp(-t_mother**2/2) * np.cos(5*t_mother)
# 不同尺度的小波
for scale, alpha in [(1, 1.0), (2, 0.7), (4, 0.4)]:
    ax4.plot(t_mother*scale, mother_wavelet*np.sqrt(scale), lw=2, alpha=alpha,
             label=f'尺度={scale}')
ax4.set_title('Morlet母小波（不同尺度拉伸）', fontsize=12)
ax4.set_xlabel('时间', fontsize=11)
ax4.set_ylabel('幅值', fontsize=11)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.text(0.5, -1.5, '小波变换：用不同尺度的小波去「匹配」信号的不同频率成分\nSTFT用固定窗，小波用自适应窗 → 对非平稳信号更灵活',
         fontsize=9, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.96])
savefig(fig, 'fig_7_2_3_wavelet.png')

# ============================================================
# 图7.3.2 MFCC滤波器组
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('图7.3.2 MFCC滤波器组 | Mel Filterbank', fontsize=14, fontweight='bold')

fs_audio = 16000  # 16kHz采样
n_mels = 40
n_fft = 1024

# Mel刻度频率
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

f_max = fs_audio // 2
mel_max = hz_to_mel(f_max)
mel_points = np.linspace(0, mel_max, n_mels + 2)
hz_points = mel_to_hz(mel_points)
hz_points = hz_points[1:-1]  # 只保留n_mels个点

# 构建Mel滤波器组
def mel_filterbank(n_mels, n_fft, fs):
    f_max = fs // 2
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(0, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / fs).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        left = bin_points[i-1]
        center = bin_points[i]
        right = bin_points[i+1]
        for k in range(left, center):
            fb[i-1, k] = (k - left) / (center - left)
        for k in range(center, right):
            fb[i-1, k] = (right - k) / (right - center)
    return fb

fb = mel_filterbank(n_mels, n_fft, fs_audio)

ax = axes[0,0]
for i in range(n_mels):
    ax.fill_between(np.arange(len(fb[i])),
                     0, fb[i], alpha=0.3, color='orange')
ax.plot(np.arange(len(fb[0])), fb.T, 'orange', lw=0.8, alpha=0.5)
ax.set_xlim(0, len(fb[0])//4)
ax.set_title('Mel滤波器组（40个三角形滤波器）', fontsize=12)
ax.set_xlabel('频率bin（对应Hz）', fontsize=11)
ax.set_ylabel('滤波器幅值', fontsize=11)
ax.text(0.02, 0.95, f'Mel刻度：人耳对低频分辨更细，对高频分辨更粗\n非线性变换模拟人耳感知特性',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.9))

# 频率响应对比：线性 vs Mel
ax2 = axes[0,1]
freqs_lin = np.fft.rfftfreq(n_fft, d=1/fs_audio)
linear_filter = np.ones(len(freqs_lin)//8)
ax2.plot(freqs_lin[:len(freqs_lin)//8] * 1000, linear_filter, 'b-', lw=2, label='线性频带划分')
ax2_twin = ax2.twinx()
fb_one = fb[10]  # 第10个Mel滤波器
ax2_twin.plot(freqs_lin[:len(freqs_lin)//4] * 1000, fb_one[:len(freqs_lin)//4], 'orange', lw=2, label='第10个Mel滤波器')
ax2.set_title('线性频带 vs Mel频带（对比）', fontsize=12)
ax2.set_xlabel('频率 (Hz)', fontsize=11)
ax2.set_ylabel('线性滤波器幅值', fontsize=11, color='blue')
ax2_twin.set_ylabel('Mel滤波器幅值', fontsize=11, color='orange')
ax2.legend(fontsize=9, loc='upper left')
ax2_twin.legend(fontsize=9, loc='upper right')
ax2.set_xlim(0, 4000)

# MFCC流程图
ax3 = axes[1,0]
ax3.axis('off')
ax3.set_title('MFCC计算流程', fontsize=12)
steps = [
    ('1. 预加重', '高频提升\n(1 - 0.97z⁻¹)', '#E3F2FD', '#1976D2'),
    ('2. 分帧', '每帧25ms\n跳帧10ms', '#E8F5E9', '#388E3C'),
    ('3. 加窗FFT', '加汉宁窗\n做FFT', '#FFF9C4', '#F57F17'),
    ('4. Mel滤波器组', '40个三角形\n滤波器', '#FCE4EC', '#C2185B'),
    ('5. 对数运算', 'log(滤波器能量)', '#E0F7FA', '#00796B'),
    ('6. DCT变换', '倒谱分析\n提取13维', '#F3E5F5', '#7B1FA2'),
]
for i, (title, desc, bg, edge) in enumerate(steps):
    x_pos = 0.15 + (i % 3) * 0.35
    y_pos = 0.7 - (i // 3) * 0.45
    box = plt.matplotlib.patches.FancyBboxPatch((x_pos-0.12, y_pos-0.15), 0.24, 0.3,
                                                  boxstyle='round,pad=0.03',
                                                  facecolor=bg, edgecolor=edge, lw=2)
    ax3.add_patch(box)
    ax3.text(x_pos, y_pos+0.08, title, ha='center', va='center', fontsize=9,
             fontweight='bold', color=edge)
    ax3.text(x_pos, y_pos-0.05, desc, ha='center', va='center', fontsize=7.5,
             color='#333333')
    if i < len(steps) - 1:
        next_x = x_pos + 0.35 if i % 3 < 2 else x_pos
        next_y = y_pos if i % 3 < 2 else y_pos - 0.45
        ax3.annotate('', xy=(next_x, next_y),
                    xytext=(x_pos + 0.12 if i % 3 < 2 else x_pos,
                           next_y + 0.15 if i % 3 == 2 else y_pos),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

# MFCC系数可视化
ax4 = axes[1,1]
# 模拟MFCC系数（13维，随时间变化）
t_mfcc = np.arange(50)
mfcc_features = np.random.randn(50, 13) * 0.5
mfcc_features[:, 0] = np.arange(50) * 0.1 + np.random.randn(50) * 0.2  # energy trend
im4 = ax4.imshow(mfcc_features.T, aspect='auto', cmap='coolwarm', origin='lower')
ax4.set_title('MFCC系数矩阵（13维 × 50帧）', fontsize=12)
ax4.set_xlabel('帧序号', fontsize=11)
ax4.set_ylabel('MFCC系数维度', fontsize=11)
ax4.set_yticks([0, 3, 7, 12])
ax4.set_yticklabels(['C₁(能量)', 'C₄', 'C₈', 'C₁₃'])
plt.colorbar(im4, ax=ax4, shrink=0.8, label='系数值')
ax4.text(0.5, -0.15, 'C₁包含大部分能量信息（最大值）\n高阶系数C₈~C₁₃描述频谱包络细节',
         transform=ax4.transAxes, fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.96])
savefig(fig, 'fig_7_3_2_mfcc_bank.png')

print("批次4完成!")
