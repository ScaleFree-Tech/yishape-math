# 第 7 章总结 / Chapter 7 Summary

## 这一章解决了什么问题

**和时间打交道的数据，和静态数据是不同的物种。**

静态数据（横截面数据）：每个样本独立，从分布里抽样。
时间序列数据：相邻时刻不独立——过去影响未来。

学完这章，你应该能：

- 判断一个业务指标是否「平稳」（趋势和季节性有多强）
- 用 ARIMA 做短期销售或流量预测，并给出预测区间
- 用 FFT 分析一段混合信号的频率成分（高频噪音 vs 低频信号）
- 理解为什么微信语音转文字之前要先提取 MFCC 特征

---

## 核心概念回顾

| 任务 | 工具 | 回答什么问题 | YiShape-Math API |
|------|------|-------------|-----------------|
| 时序分解 | 趋势 + 季节 + 残差 | 这个指标的波动来源是什么？ | `TSA.decompose.additive(series, period)` |
| 短期预测 | ARIMA / Holt-Winters | 下个月/下周的销售大概是多少？ | `TSA.forecast.arima(series, p, d, q)` |
| 时序滤波 | 移动平均 / 指数平滑 | 如何去除噪声？ | `TSA.filter.movingAverage(series, window)` |
| 协整分析 | Engle-Granger | 两个序列是否长期均衡？ | `TSA.cointegrate.engleGranger(y1, y2)` |
| 频域分析 | FFT | 这段信号里有哪些频率成分？ | `Signals.xform.fft(complexArray)` |
| 滤波 | 巴特沃斯 / 卡尔曼 | 如何去除特定频率噪声？ | `Signals.filt.butterworthLowPass(...)` |
| 时频分析 | STFT / 小波 | 频率如何随时间变化？ | `Signals.analyze.shortTimeFourierTransform(...)` |
| 音频特征 | MFCC（手动实现） | 这段语音说的是什么内容？ | 基于 `Signals.xform.fft` + Mel 滤波器组 |
| 音乐分析 | BPM + 调性（手动实现） | 这首歌和哪首歌最像？ | 基于 `Signals.analyze` + 色度分析 |

> ⚠️ **注意**：YiShape-Math 当前版本**未提供独立的 Audio/Music 模块**。音频和音乐特征需要基于 `Signals` 模块手动实现。

---

## 本章 API 速查

```java
// 时间序列分析（TSA 门面）
var analyzer = TSA.analyzer(series, "sales");  // 创建时序分析器
analyzer.quickAnalyze();                         // 快速诊断

// 时序分解
var decomp = TSA.decompose.additive(series, 12);  // 加法模型（周期=12）
var decomp = TSA.decompose.multiplicative(series, 12);  // 乘法模型

// 时序预测
var forecast = TSA.forecast.arima(series, 1, 1, 1);  // ARIMA(1,1,1)
var forecast = TSA.forecast.exponentialSmoothing(series, "triple");  // 三重指数平滑

// 时序滤波
var filtered = TSA.filter.movingAverage(series, 7);  // 7 点移动平均

// 协整分析
var result = TSA.cointegrate.engleGranger(y1, y2);

// 信号变换（Signals.xform）
var fft = Signals.xform.fft(complexArray);  // FFT
var ifft = Signals.xform.ifft(complexArray);  // IFFT
var mag = Signals.xform.magnitudeSpectrum(complexArray);  // 幅度谱
var dwt = Signals.xform.discreteWaveletTransform(signal, "haar", 3);  // 小波

// 信号滤波（Signals.filt）
var lowpass = Signals.filt.butterworthLowPass(signal, 100, 1000, 4);  // 低通
var highpass = Signals.filt.butterworthHighPass(signal, 50, 1000, 4);  // 高通
var bandpass = Signals.filt.bandPass(signal, 100, 500, 1000, 4);  // 带通
var smoothed = Signals.filt.movingAverage(signal, 5);  // 移动平均
var denoised = Signals.filt.kalmanFilter(signal, 0.01, 0.1);  // 卡尔曼

// 信号生成（Signals.gen）
var sine = Signals.gen.sineWave(1000, 50, 1000, 1.0, 0);  // 正弦波
var chirp = Signals.gen.chirpSignal(1000, 10, 500, 1000);  // 线性调频

// 信号分析（Signals.analyze）
var psd = Signals.analyze.powerSpectralDensity(signal, 256, 0.5, 1000);  // PSD
var stft = Signals.analyze.shortTimeFourierTransform(signal, 256, 128, 1000);  // STFT
var acf = Signals.analyze.autocorrelation(signal);  // 自相关
var snr = Signals.analyze.signalToNoiseRatio(signal, noise);  // 信噪比
```

---

## 与其他章节的联系

- **第 4 章统计学**：平稳性检验是单位根检验；预测区间的宽度来自统计分布假设；白噪声 = 独立同分布的随机变量序列
- **第 1 章线性代数**：FFT 是酉矩阵变换；卷积运算可以用矩阵乘法实现；时频分析依赖线性代数框架
- **第 5 章机器学习**：MFCC 特征 + RNN/Transformer = 语音识别；音乐特征相似度 + K-Means = 歌单聚类
- **第 6 章最优化**：ARIMA 参数估计是最优化问题（最大似然）；滤波器设计也是约束优化

---

## 常见误区

1. **对非平稳序列直接跑均值预测**：趋势明显的数据，用均值预测会把趋势当噪音——先差分（$d$ 次）使序列平稳
2. **把时序图和频谱图混用**：时域图看趋势，频域图看周期——用错工具会错过关键信息
3. **忽略季节性就上线预测模型**：双十一、春节有强季节性，不建模会周期性预测失败
4. **MFCC 维度选择随意**：13 维 MFCC 是语音识别的经验最优（考虑了人耳感知的 Mel 频率刻度），不是随便选的

---

*第 7 章的核心只有一句话：**和时间做朋友的关键，是理解「趋势是暂时的，还是持久的；波动是随机的，还是有周期的」。***
