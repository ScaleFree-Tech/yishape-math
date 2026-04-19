# 第 7 章：时间序列与信号处理——和时间做朋友


> **💡 工厂模式约定**：本章大量使用 `Linalg.vector()`、`Decomps.svd()`、`Opts.lbfgs()` 等工厂类入口。
> 不需要记住具体实现类名——所有数学对象通过统一的工厂类创建，代码简洁且风格一致。


## 开场：老板要你预测下个月的销售

季度末，老板把你叫到办公室：

> 「下个月销售能到多少？」

你翻出过去 36 个月的销售数据——一条起起伏伏的曲线。有增长趋势，有明显的年末高峰（双十一），也有些异常波动。

**这不是一道静态的统计题，这是一道和时间有关的题。**

你不能只拿平均值糊弄过去——因为相邻月份之间是有关系的（10 月高了，11 月往往也高）。这就是时间序列分析的核心挑战：**数据点不独立，过去影响未来。**

但时间序列不只是销售预测：

- 手机陀螺仪信号里藏着用户活动和手势——**信号处理**
- 微信语音消息转成文字——**音频特征提取（MFCC）**
- Spotify 推荐「这首歌像那首歌」——**音乐信息检索（MIR）**

它们都处理的是「随时间变化的信号」，只是采样率和分析目的不同。

---

## 学完这章你能做什么

- 用 **ARIMA** 预测业务指标的短期走势，并给出预测区间（而不是点估计）
- 用 **FFT** 把混杂在一起的频率成分分开——比如从一段录音里分离人声和背景噪音
- 理解 **MFCC** 的管线——为什么微信语音转文字之前要先做频域分析
- 用 **音乐特征相似度** 做歌单推荐——不用人工标签，让音频自己说话

---

## 本章知识地图

```
时间序列与信号
  ├── 7.1 时间序列
  │     分解（趋势/季节/残差）、ARIMA/Holt-Winters、预测区间
  ├── 7.2 信号处理
  │     FFT、频域分析、滤波器——嘈杂咖啡馆里听懂朋友的话
  ├── 7.3 音频处理
  │     波形→MFCC→文字的管线、语音识别前端
  └── 7.4 音乐挖掘
        节拍、调性、音色相似度、Spotify推荐系统
```

---

## 时间序列 vs 信号处理 vs 音频：什么时候用哪个？

| 问题类型 | 采样率 | 典型方法 | 典型应用 |
|---------|--------|---------|---------|
| 业务指标（销售、流量） | 日/周/月 | ARIMA、指数平滑 | 需求预测、库存管理 |
| 传感器信号 | kHz~MHz | FFT、滤波器 | 振动分析、异常检测 |
| 语音/音频 | 8~44 kHz | MFCC、滤波器组 | 语音识别、哼唱搜索 |
| 音乐 | 44 kHz+ | 节拍跟踪、调性分析 | 歌单推荐、风格分类 |

四者的数学工具（傅里叶变换、时频分析、滤波）是相通的——本章帮你建立这个共同基础。

---

## YiShape-Math 时序与信号模块

```java
// 时间序列分解与预测
Series.decompose(series, "additive");      // 趋势+季节+残差
Series.forecastArima(series, p, d, q);    // ARIMA预测

// 信号处理
Signals.fft(double[] signal);            // 快速傅里叶变换
Signals.filter(double[] signal, FilterKernel kernel);  // 滤波器

// 音频特征提取
IAudioFeatureExtractor extractor = new RereAudioFeatureExtractor();
AudioFeatureResult afr = extractor.extractAudioFeatures(audio);
var mfcc = afr.getMfcc();           // 13维MFCC

// 音乐分析
MusicAnalyzer analyzer = Musics.createBasicMusicAnalyzer();
MusicFeatures mf = analyzer.analyze("song.wav");
```

> **API 注意**：`AudioIO.readAudio()` 读取音频文件（不是 `load()`）；`IAudioFeatureExtractor.extractAudioFeatures()` 提取特征（不是 `extract()`）；`Musics.createBasicMusicAnalyzer()` 是音乐分析的入口工厂类。

---
[← 第6章：返回上一章](introduction.md) ｜ [下一章：数据分析实战 →](7.1. Time series.md)
