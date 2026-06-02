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
| 时序分解 | 趋势 + 季节 + 残差 | 这个指标的波动来源是什么？ | `Series.decompose()` |
| 短期预测 | ARIMA / Holt-Winters | 下个月/下周的销售大概是多少？ | `Series.forecastArima()` |
| 频域分析 | FFT | 这段信号里有哪些频率成分？ | `Signals.fft()` |
| 音频特征 | MFCC | 这段语音说的是什么内容？ | `extractor.extractAudioFeatures()` |
| 音乐相似度 | BPM + 调性 + MFCC | 这首歌和哪首歌最像？ | `Musics.createBasicMusicAnalyzer()` |

---

## 本章 API 速查

```java
// 时间序列
Series.decompose(series, "additive");   // 趋势+季节+残差
Series.decompose(series, "multiplicative");
Series.forecastArima(series, p, d, q);  // ARIMA 预测

// 信号处理
Signals.fft(double[] signal);           // 返回复数数组
Signals.ifft(Complex[] freqDomain);     // 逆变换
Signals.filter(signal, FilterKernel kernel); // 滤波器

// 音频特征
IAudioFeatureExtractor extractor = new RereAudioFeatureExtractor();
AudioFeatureResult afr = extractor.extractAudioFeatures(audio);
var mfcc = afr.getMfcc();          // 13维MFCC系数
var spectral = afr.getSpectralCentroid(); // 频谱质心

// 音乐分析
MusicAnalyzer analyzer = Musics.createBasicMusicAnalyzer();
MusicFeatures mf = analyzer.analyze("song.wav");
double bpm = mf.getBPM();
String key = mf.getKey();               // "C minor", "G major" 等
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
