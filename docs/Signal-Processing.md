# 信号处理模块文档 / Signal Processing Module Documentation

## 概述 / Overview

YiShape-Math 信号处理模块提供了一套完整的数字信号处理功能，包括信号生成、滤波、分析、变换等核心操作。该模块采用工厂模式和策略模式设计，具有良好的扩展性和灵活性。

The YiShape-Math signal processing module provides a complete set of digital signal processing functions, including signal generation, filtering, analysis, transformation and other core operations. The module uses factory and strategy patterns for design, providing good extensibility and flexibility.

## 快速上手 / Quick Start

```java
// 生成信号 / Generate signals
// sineWave(length, frequency, samplingRate, amplitude, phase)
IVector<Double> sine = Signals.sineWave(1000, 50.0, 1000.0, 2.0, 0.0);  // 1000采样点，50Hz，1000Hz采样率
IVector<Double> noise = Signals.whiteNoise(1000, 0.1);  // 1000点白噪声，功率0.1

// 信号处理 / Process signals
ButterworthFilter filter = new ButterworthFilter(4, 200.0, 1000.0);  // 4阶，截止200Hz，采样率1000Hz
IVector<Double> filtered = filter.process(sine);

// 频谱分析 / Spectrum analysis (FFT)
ISignalTransform<Double, Double> fft = SignalProcessorFactory.getInstance().createTransform("fft");
IVector<Double> spectrum = fft.process(filtered);
```

## 核心类 / Core Classes

### Signals 类 / Signals Class

信号处理入口工厂类，提供统一的信号处理接口，封装了信号生成、滤波、分析、变换等核心功能。

Signal processing entry factory class providing a unified signal processing interface that encapsulates core functions such as signal generation, filtering, analysis, and transformation.

### ISignalProcessor 接口 / ISignalProcessor Interface

信号处理器基础接口，定义了一维和二维信号处理的标准接口。

Base interface for signal processors, defining standard interfaces for one-dimensional and two-dimensional signal processing.

### ISignalFilter 接口 / ISignalFilter Interface

信号滤波器接口，定义滤波器的标准接口和框架。

Signal filter interface, defining standard interfaces and framework for filters.

### ISignalAnalyzer 接口 / ISignalAnalyzer Interface

信号分析器接口，提供频谱分析、功率谱密度、自相关、互相关等功能。

Signal analyzer interface providing spectral analysis, power spectral density, autocorrelation, cross-correlation and other functions.

### ISignalGenerator 接口 / ISignalGenerator Interface

信号生成器接口，支持各种信号类型的生成。

Signal generator interface supporting generation of various signal types.

### ISignalTransform 接口 / ISignalTransform Interface

信号变换接口，定义信号变换的标准接口。

Signal transform interface, defining standard interfaces for signal transformation.

### SignalProcessorFactory 类 / SignalProcessorFactory Class

信号处理器工厂类，使用单例模式管理所有信号处理器的创建。

Signal processor factory class using singleton pattern to manage creation of all signal processors.

### TimeSeriesFiltering 类 / TimeSeriesFiltering Class

时间序列滤波类，提供时间序列数据的滤波功能。

Time series filtering class providing filtering functionality for time series data.

### SignalUtilities 类 / SignalUtilities Class

信号工具类，提供各种信号处理工具函数，包括窗函数、重采样、信号拼接、信号检测等。

Signal utilities class providing various signal processing utility functions including window functions, resampling, signal concatenation, signal detection, etc.

### WaveletAnalysis 类 / WaveletAnalysis Class

小波分析类，提供各种小波变换功能，包括连续小波变换(CWT)、离散小波变换(DWT)、小波包变换等。

Wavelet analysis class providing various wavelet transform functionality including Continuous Wavelet Transform (CWT), Discrete Wavelet Transform (DWT), Wavelet Packet Transform, etc.

### RereFFT 类 / RereFFT Class

快速傅里叶变换类，提供FFT和IFFT功能，以及相关的频谱分析功能。

Fast Fourier Transform class providing FFT and IFFT functionality, as well as related spectral analysis functions.

### Complex 类 / Complex Class

复数类，提供复数运算功能，支持各种复数数学操作。

Complex number class providing complex number operations, supporting various complex mathematical operations.

### RereDCT 类 / RereDCT Class

离散余弦变换类，提供各种类型的DCT变换，用于信号压缩和频域分析。

Discrete Cosine Transform class providing various types of DCT transforms for signal compression and frequency domain analysis.

### RereHilbert 类 / RereHilbert Class

希尔伯特变换类，提供希尔伯特变换和解析信号分析功能。

Hilbert Transform class providing Hilbert transform and analytic signal analysis functionality.

### SignalPlots 类 / SignalPlots Class

信号可视化类，提供信号数据的可视化功能。

Signal visualization class providing signal data visualization functionality.

## 子包结构 / Subpackage Structure

### 1. analysis 包 - 信号分析接口
**功能**: 定义信号分析的标准接口和框架。

**核心类**:
- `ISignalAnalyzer<T>` - 信号分析器接口
  - 支持多种分析类型：频谱、功率谱、自相关、互相关、相干性、包络、瞬时频率、时频分析等
  - 提供分析参数配置和结果封装
  - 支持批量分析和比较分析

### 2. core 包 - 核心框架
**功能**: 提供信号处理的核心框架和基础类。

**核心类**:
- `ISignalProcessor<T>` - 信号处理器基础接口
  - 定义一维和二维信号处理接口
  - 提供参数验证和处理器管理功能
- `AbstractSignalProcessor<T>` - 抽象信号处理器基类
  - 实现模板方法模式，定义标准处理流程
  - 提供预处理、核心处理、后处理的框架
  - 包含输入输出验证和错误处理
- `SignalProcessingException` - 信号处理异常类

### 3. factory 包 - 工厂模式
**功能**: 提供信号处理器的创建和管理。

**核心类**:
- `SignalProcessorFactory` - 信号处理器工厂类
  - 使用单例模式管理处理器创建
  - 支持插件式注册新的处理器实现
  - 提供按分类查询处理器功能
  - 支持变换器、滤波器、生成器、分析器的创建

### 4. filter 包 - 滤波器实现
**功能**: 提供各种高级滤波器的具体实现。

**核心类**:
- `ISignalFilter<T>` - 信号滤波器接口
  - 定义滤波器类型和实现类型枚举
  - 提供滤波器系数和频率响应计算
  - 支持FIR和IIR滤波器
- `ChebyshevFilter` - 切比雪夫滤波器实现
  - 支持I型和II型切比雪夫滤波器
  - 提供低通、高通、带通、带阻滤波
  - 包含滤波器系数计算和频率响应分析
- `EllipticFilter` - 椭圆滤波器实现

### 5. generation 包 - 信号生成接口
**功能**: 定义信号生成的标准接口。

**核心类**:
- `ISignalGenerator<T>` - 信号生成器接口
  - 支持多种信号类型：正弦、余弦、方波、三角波、锯齿波、脉冲、调频、噪声等
  - 提供信号参数配置和验证
  - 支持多分量信号生成和噪声添加

### 6. transform 包 - 信号变换实现
**功能**: 提供各种信号变换的具体实现。

**核心类**:
- `ISignalTransform<T, R>` - 信号变换接口
  - 定义正向和逆向变换接口
  - 支持一维和二维变换
  - 提供变换核大小和长度支持检查
- `ChirpZTransform` - Chirp-Z变换实现
  - 高效计算Z变换在任意复平面路径上的值
  - 特别适用于高分辨率频谱分析
  - 支持频率范围和对数频率扫描
- `WalshHadamardTransform` - 沃尔什-哈达玛变换
- `ZTransform` - Z变换实现

## 主要功能 / Main Features

### 1. 信号生成 / Signal Generation

#### 1.1 基本波形生成 / Basic Waveform Generation

```java

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

// 生成正弦波信号 / Generate sine wave signal
IVector<Double>sineWave=Signals.sineWave(
        1000,      // 信号长度 / Signal length
        50.0,      // 频率 (Hz) / Frequency (Hz)
        1000.0,    // 采样率 (Hz) / Sampling rate (Hz)
        1.0,       // 幅度 / Amplitude
        0.0        // 相位 (弧度) / Phase (radians)
        );

// 生成余弦波信号 / Generate cosine wave signal
        IVector<Double>cosineWave=Signals.cosineWave(
        1000,50.0,1000.0,1.0,0.0
        );

// 生成方波信号 / Generate square wave signal
        IVector<Double>squareWave=Signals.squareWave(
        1000,      // 信号长度 / Signal length
        50.0,      // 频率 (Hz) / Frequency (Hz)
        1000.0,    // 采样率 (Hz) / Sampling rate (Hz)
        1.0,       // 幅度 / Amplitude
        0.5        // 占空比 (0-1) / Duty cycle (0-1)
        );

// 生成三角波信号 / Generate triangular wave signal
        IVector<Double>triangularWave=Signals.triangularWave(
        1000,50.0,1000.0,1.0,0.5  // 0.5为对称三角波 / 0.5 for symmetric triangular wave
        );

// 生成锯齿波信号 / Generate sawtooth wave signal
        IVector<Double>sawtoothWave=Signals.sawtoothWave(
        1000,50.0,1000.0,1.0
        );

// 使用工厂模式创建信号生成器 / Using factory pattern to create signal generators
import com.yishape.lab.math.signal.factory.SignalProcessorFactory;
import com.yishape.lab.math.signal.generation.ISignalGenerator;
import com.yishape.lab.math.signal.generation.ISignalGenerator.SignalType;
import com.yishape.lab.math.signal.generation.ISignalGenerator.SignalParameters;

// 方法1: 使用工厂创建不同类型的生成器 / Method 1: Using factory to create different types of generators
try{
SignalProcessorFactory factory = SignalProcessorFactory.getInstance();

// 注册不同类型的信号生成器 / Register different types of signal generators
    factory.

registerProcessor("sine_generator",SineWaveGenerator .class,
                  SignalProcessorFactory.ProcessorCategory.GENERATOR, 
        "Sine Wave Generator","1.0.0");
    
    factory.

registerProcessor("square_generator",SquareWaveGenerator .class,
                  SignalProcessorFactory.ProcessorCategory.GENERATOR, 
        "Square Wave Generator","1.0.0");
    
    factory.

registerProcessor("noise_generator",NoiseGenerator .class,
                  SignalProcessorFactory.ProcessorCategory.GENERATOR, 
        "Noise Generator","1.0.0");

// 创建正弦波生成器 / Create sine wave generator
ISignalGenerator<Double> sineGenerator = factory.createGenerator("sine_generator");

// 创建方波生成器 / Create square wave generator
ISignalGenerator<Double> squareGenerator = factory.createGenerator("square_generator");

// 创建噪声生成器 / Create noise generator
ISignalGenerator<Double> noiseGenerator = factory.createGenerator("noise_generator");

// 使用参数化生成 / Using parameterized generation
SignalParameters params = new SignalParameters()
        .amplitude(1.0)
        .frequency(50.0)
        .phase(0.0)
        .samplingRate(1000.0)
        .dutyCycle(0.5);

// 生成不同类型的信号 / Generate different types of signals
IVector<Double> sineSignal = sineGenerator.generate(SignalType.SINE, 1000, params);
IVector<Double> squareSignal = squareGenerator.generate(SignalType.SQUARE, 1000, params);
IVector<Double> noiseSignal = noiseGenerator.generate(SignalType.WHITE_NOISE, 1000, params);
    
    System.out.

println("工厂创建生成器成功");
    System.out.

println("正弦波信号长度: "+sineSignal.length());
        System.out.

println("方波信号长度: "+squareSignal.length());
        System.out.

println("噪声信号长度: "+noiseSignal.length());

        }catch(
Exception e){
        System.err.

println("工厂创建生成器失败: "+e.getMessage());
        }

// 方法2: 使用生成器链模式 / Method 2: Using generator chain pattern
class SignalGeneratorChain {
    private final List<ISignalGenerator<Double>> generators;

    public SignalGeneratorChain() {
        this.generators = new ArrayList<>();
    }

    public SignalGeneratorChain addGenerator(ISignalGenerator<Double> generator) {
        generators.add(generator);
        return this;
    }

    public IVector<Double> generateComposite(SignalType[] types, int length, SignalParameters[] params) {
        if (types.length != generators.size() || types.length != params.length) {
            throw new IllegalArgumentException("类型、生成器和参数数量必须匹配");
        }

        IVector<Double> result = Linalg.zeros(length);
        for (int i = 0; i < types.length; i++) {
            IVector<Double> component = generators.get(i).generate(types[i], length, params[i]);
            result = result.add(component);
        }
        return result;
    }
}

// 使用生成器链 / Using generator chain
try{
SignalGeneratorChain chain = new SignalGeneratorChain()
        .addGenerator(factory.createGenerator("sine_generator"))
        .addGenerator(factory.createGenerator("square_generator"))
        .addGenerator(factory.createGenerator("noise_generator"));

SignalType[] types = {SignalType.SINE, SignalType.SQUARE, SignalType.WHITE_NOISE};
SignalParameters[] paramArray = {
        new SignalParameters().frequency(10.0).amplitude(1.0).samplingRate(1000.0),
        new SignalParameters().frequency(50.0).amplitude(0.5).samplingRate(1000.0),
        new SignalParameters().noiseVariance(0.1).samplingRate(1000.0)
};

IVector<Double> compositeSignal = chain.generateComposite(types, 1000, paramArray);
    System.out.

println("生成器链创建复合信号成功，长度: "+compositeSignal.length());

        }catch(
Exception e){
        System.err.

println("生成器链创建失败: "+e.getMessage());
        }
```

#### 1.2 噪声信号生成 / Noise Signal Generation

```java
// 生成白噪声信号 / Generate white noise signal
IVector<Double> whiteNoise = Signals.whiteNoise(
    1000,      // 信号长度 / Signal length
    0.1        // 噪声功率 / Noise power
);

// 生成粉红噪声信号 / Generate pink noise signal
IVector<Double> pinkNoise = Signals.pinkNoise(
    1000,      // 信号长度 / Signal length
    0.1        // 噪声功率 / Noise power
);
```

#### 1.3 复合信号生成 / Composite Signal Generation

```java
// 生成复合信号（多个正弦波叠加）/ Generate composite signal (multiple sine waves)
ISignalGenerator.SignalType[] signalTypes = {
    ISignalGenerator.SignalType.SINE,
    ISignalGenerator.SignalType.SINE,
    ISignalGenerator.SignalType.SINE
};

ISignalGenerator.SignalParameters[] parameters = {
    new ISignalGenerator.SignalParameters().frequency(10.0).amplitude(1.0).samplingRate(1000.0),
    new ISignalGenerator.SignalParameters().frequency(50.0).amplitude(0.5).samplingRate(1000.0),
    new ISignalGenerator.SignalParameters().frequency(100.0).amplitude(0.3).samplingRate(1000.0)
};

IVector<Double> compositeSignal = Signals.compositeSignal(signalTypes, 1000, parameters);

// 生成带噪声的信号 / Generate signal with noise
IVector<Double> noisySignal = Signals.addNoise(
    compositeSignal, 
    ISignalGenerator.SignalType.WHITE_NOISE, 
    new ISignalGenerator.SignalParameters().noiseVariance(0.1).samplingRate(1000.0)
);
```

#### 1.4 特殊信号生成 / Special Signal Generation

```java
// 生成阶跃信号 / Generate step signal
IVector<Double> stepSignal = Signals.stepSignal(1000, 1.0, 0.5, 1000.0);

// 生成单位阶跃信号 / Generate unit step signal
IVector<Double> unitStep = Signals.unitStep(1000, 0.5, 1000.0);

// 生成脉冲信号 / Generate pulse signal
IVector<Double> pulseSignal = Signals.pulseSignal(1000, 1.0, 10, 10.0, 1000.0);

// 生成单位脉冲信号 / Generate unit impulse signal
IVector<Double> unitImpulse = Signals.unitImpulse(1000, 500);

// 生成线性调频信号 / Generate linear chirp signal
IVector<Double> chirpSignal = Signals.chirpSignal(1000, 10.0, 100.0, 1000.0, 1.0);

// 生成狄拉克δ函数 / Generate Dirac delta function
IVector<Double> diracDelta = Signals.diracDelta(1000, 100, 1.0);
```

### 2. 信号滤波 / Signal Filtering

#### 算法原理 / Algorithm Principles

滤波的本质是**对信号每个时刻的值进行加权平均**，权重由滤波器类型决定。

**时域视角**：
- 移动平均：窗口内 N 个点取算术平均，等权重
- 高斯滤波：窗口内按高斯分布加权（中间点权重大，两边小）
- 中值滤波：取窗口内排序后的中值，**对脉冲噪声（尖刺）鲁棒**

**频域视角**：
滤波器让某些频率通过（通带），阻止另一些（阻带）：
- **低通**：保留低频，去高频（平滑、去噪）
- **高通**：保留高频，去低频（去基线漂移、边缘检测）
- **带通**：只保留某一频段（提取特定信号成分）
- **带阻**：去除某一频段（去除工频干扰 50/60Hz）

**滤波器阶数的含义**：阶数越高，通带与阻带之间的过渡越陡峭（越接近理想滤波器），但计算量越大，数值稳定性可能下降。Butterworth 滤波器在通带内最平坦（无纹波）。

**选型建议** / Filter Selection Guide:

| 需求 | 推荐滤波器 |
|------|----------|
| 一般去噪、平滑 | 移动平均或高斯滤波 |
| 去除脉冲噪声 | 中值滤波 |
| 保留边缘同时去噪 | 高斯滤波（优于移动平均）|
| 提取特定频段 | Butterworth 带通/带阻 |
| 去除基线漂移 | Butterworth 高通 |
| 跟踪时变信号最优估计 | 卡尔曼滤波 |

#### 2.1 基础滤波方法 / Basic Filtering Methods

```java
import com.yishape.lab.math.signal.Signals;

// 移动平均滤波 / Moving average filtering
IVector<Double> filtered = Signals.movingAverage(signal, 5);

// 中值滤波 / Median filtering
IVector<Double> medianFiltered = Signals.medianFilter(signal, 5);

// 高斯滤波 / Gaussian filtering
IVector<Double> gaussianFiltered = Signals.gaussianFilter(signal, 1.0);
```

#### 选型指南 / Filter Selection Guide

| 需求 | 推荐滤波器 | 关键参数 |
|------|----------|---------|
| 一般去噪、平滑 | 移动平均 | 窗口大小（越大越平滑）|
| 去除脉冲噪声（尖刺）| 中值滤波 | 窗口大小 |
| 保留边缘同时去噪 | 高斯滤波 | σ 值（越大越平滑）|
| 提取低频（去高频）| Butterworth 低通 | `cutoffFreq`, `order` |
| 提取高频（去低频/基线漂移）| Butterworth 高通 | `cutoffFreq`, `order` |
| 只保留某频段 | Butterworth 带通 | `lowFreq`, `highFreq`, `order` |
| 去除工频干扰 50/60Hz | Butterworth 带阻 | `lowFreq`, `highFreq`, `order` |
| 跟踪动态系统最优估计 | 卡尔曼滤波 | `processNoise`, `measurementNoise` |

**参数快速参考**：
- 截止频率 `cutoffFreq`：通常取信号最高频率的 1.2~2 倍
- 采样率 `samplingRate`：必须 > 2× 信号最高频率（Nyquist）
- 阶数 `order`：越高越陡峭，但计算量越大、稳定性越差（通常 2~6）

#### 2.2 频域滤波方法 / Frequency Domain Filtering Methods

```java
// 巴特沃斯低通滤波 / Butterworth low-pass filtering
IVector<Double> lowPassFiltered = Signals.butterworthLowPass(
    signal, 50.0, 1000.0, 1  // 信号, 截止频率, 采样率, 阶数
);

// 巴特沃斯高通滤波 / Butterworth high-pass filtering
IVector<Double> highPassFiltered = Signals.butterworthHighPass(
    signal, 10.0, 1000.0, 1
);

// 带通滤波 / Band-pass filtering
IVector<Double> bandPassFiltered = Signals.bandPass(
    signal, 10.0, 50.0, 1000.0, 1  // 信号, 低截止频率, 高截止频率, 采样率, 阶数
);

// 带阻滤波 / Band-stop filtering
IVector<Double> bandStopFiltered = Signals.bandStop(
    signal, 45.0, 55.0, 1000.0, 1  // 信号, 低截止频率, 高截止频率, 采样率, 阶数
);
```

#### 2.3 滤波器功能 / Filter Functionality

``java
// 卡尔曼滤波 / Kalman filtering
IVector<Double> kalmanFiltered = Signals.kalmanFilter(
    signal, 1.0, 1.0  // 信号, 过程噪声方差, 测量噪声方差
);

// 维纳滤波 / Wiener filtering
IVector<Double> wienerFiltered = Signals.wienerFilter(
    signal, 1.0, 0.1, 10  // 信号, 信号功率, 噪声功率, 滤波器长度
);

// 使用高级滤波器接口 / Using advanced filter interface
import com.yishape.lab.math.signal.filter.ISignalFilter;
import com.yishape.lab.math.signal.filter.ChebyshevFilter;
import com.yishape.lab.math.signal.filter.ISignalFilter.FilterType;
import com.yishape.lab.math.signal.filter.ISignalFilter.FilterImplementation;

// 创建切比雪夫滤波器 / Create Chebyshev filter
try {
    ChebyshevFilter chebyshevFilter = new ChebyshevFilter(
        ChebyshevFilter.ChebyshevType.TYPE_I,  // I型切比雪夫滤波器
        4,                                      // 滤波器阶数
        50.0,                                   // 截止频率
        1000.0,                                 // 采样率
        0.5                                     // 波纹参数 (dB)
    );
    
    // 应用滤波器 / Apply filter
    IVector<Double> chebyshevFiltered = chebyshevFilter.filter(signal);
    
    // 获取滤波器信息 / Get filter information
    System.out.println("滤波器类型: " + chebyshevFilter.getFilterType());
    System.out.println("实现类型: " + chebyshevFilter.getImplementationType());
    System.out.println("滤波器阶数: " + chebyshevFilter.getOrder());
    System.out.println("截止频率: " + chebyshevFilter.getCutoffFrequencies()[0]);
    
    // 计算频率响应 / Calculate frequency response
    double[] frequencies = {10.0, 25.0, 50.0, 75.0, 100.0};
    ISignalFilter.FrequencyResponse response = chebyshevFilter.getFrequencyResponse(frequencies);
    
    System.out.println("频率响应:");
    for (int i = 0; i < frequencies.length; i++) {
        System.out.printf("频率: %.1f Hz, 幅度: %.3f, 相位: %.3f rad%n", 
            frequencies[i], response.getMagnitude()[i], response.getPhase()[i]);
    }
    
} catch (Exception e) {
    System.err.println("滤波器创建失败: " + e.getMessage());
}

// 使用滤波器工厂创建滤波器 / Using filter factory to create filters
import com.yishape.lab.math.signal.factory.SignalProcessorFactory;

try {
    SignalProcessorFactory factory = SignalProcessorFactory.getInstance();
    
    // 注册切比雪夫滤波器 / Register Chebyshev filter
    factory.registerProcessor("chebyshev", ChebyshevFilter.class, 
        SignalProcessorFactory.ProcessorCategory.FILTER, 
        "Chebyshev Filter", "1.0.0");
    
    // 创建滤波器实例 / Create filter instance
    ISignalFilter<Double> filter = factory.createFilter("chebyshev");
    
    // 使用滤波器 / Use filter
    IVector<Double> filteredSignal = filter.filter(signal);
    
} catch (Exception e) {
    System.err.println("工厂创建滤波器失败: " + e.getMessage());
}
```

### 3. 信号分析 / Signal Analysis

#### 3.1 频谱分析 / Spectral Analysis

```java
import com.yishape.lab.math.signal.Signals;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;

// 计算功率谱密度 (PSD) / Calculate Power Spectral Density (PSD)
Tuple2<IVector<Double>, IVector<Double>> psdResult = Signals.powerSpectralDensity(
    signal, 256, 0.5, 1000.0  // 信号, 窗大小, 重叠比例, 采样率
);
IVector<Double> frequencies = psdResult._1;
IVector<Double> psd = psdResult._2;

// 计算信号频谱 / Calculate signal spectrum
Tuple3<IVector<Double>, IVector<Double>, IVector<Double>> spectrumResult = 
    Signals.spectrum(signal, 1000.0);
IVector<Double> freq = spectrumResult._1;
IVector<Double> magnitude = spectrumResult._2;
IVector<Double> phase = spectrumResult._3;

// 计算短时傅里叶变换 (STFT) / Calculate Short-Time Fourier Transform (STFT)
// Note: STFT not currently implemented in Signals class
// IMatrix<Double> stft = Signals.shortTimeFourierTransform(
//     signal, 256, 64, 1000.0  // 信号, 窗大小, 跳跃大小, 采样率
// );
```

#### 3.2 相关分析 / Correlation Analysis

```java
// 计算自相关函数 / Calculate autocorrelation function
IVector<Double> autocorr = Signals.autocorrelation(signal);

// 计算互相关函数 / Calculate cross-correlation function
IVector<Double> crossCorr = Signals.crossCorrelation(signal1, signal2);

// 计算带最大滞后的互相关函数 / Calculate cross-correlation with maximum lag
IVector<Double> crossCorrLimited = Signals.crossCorrelation(signal1, signal2, 100);
```

#### 3.3 信噪比分析 / Signal-to-Noise Ratio Analysis

```java
// 计算信噪比 (SNR) / Calculate Signal-to-Noise Ratio (SNR)

double snr=Signals.signalToNoiseRatio(originalSignal,noiseSignal);

// 计算峰值信噪比 (PSNR) / Calculate Peak Signal-to-Noise Ratio (PSNR)
        double psnr=Signals.peakSignalToNoiseRatio(originalSignal,reconstructedSignal);

// 使用信号分析器接口的高级分析 / Advanced analysis using signal analyzer interface
import com.yishape.lab.math.signal.analysis.ISignalAnalyzer;
import com.yishape.lab.math.signal.analysis.ISignalAnalyzer.AnalysisType;
import com.yishape.lab.math.signal.analysis.ISignalAnalyzer.AnalysisParameters;
import com.yishape.lab.math.signal.analysis.ISignalAnalyzer.AnalysisResult;

// 创建自定义信号分析器 / Create custom signal analyzer
ISignalAnalyzer<Double> analyzer = new ISignalAnalyzer<Double>() {
    @Override
    public <R> AnalysisResult<R> analyze(IVector<Double> signal, AnalysisType analysisType, AnalysisParameters parameters) {
        switch (analysisType) {
            case SPECTRUM:
                Tuple3<IVector<Double>, IVector<Double>, IVector<Double>> spectrumResult =
                        Signals.spectrum(signal, parameters.getSamplingRate());
                return new AnalysisResult<>(analysisType, (R) spectrumResult,
                        new String[]{"frequencies", "magnitude", "phase"},
                        "Signal spectrum analysis", parameters.getConfidenceLevel());

            case POWER_SPECTRUM:
                Tuple2<IVector<Double>, IVector<Double>> psdResult =
                        Signals.powerSpectralDensity(signal, parameters.getWindowSize(),
                                parameters.getOverlap(), parameters.getSamplingRate());
                return new AnalysisResult<>(analysisType, (R) psdResult,
                        new String[]{"frequencies", "psd"},
                        "Power spectral density analysis", parameters.getConfidenceLevel());

            case AUTOCORRELATION:
                IVector<Double> autocorr = Signals.autocorrelation(signal);
                return new AnalysisResult<>(analysisType, (R) autocorr,
                        new String[]{"autocorrelation"},
                        "Autocorrelation analysis", parameters.getConfidenceLevel());

            case SNR:
                double snr = Signals.signalToNoiseRatio(signal, signal); // 简化示例
                return new AnalysisResult<>(analysisType, (R) Double.valueOf(snr),
                        new String[]{"snr"},
                        "Signal-to-noise ratio analysis", parameters.getConfidenceLevel());

            default:
                throw new UnsupportedOperationException("Unsupported analysis type: " + analysisType);
        }
    }

    @Override
    public AnalysisResult<?>[] batchAnalyze(IVector<Double> signal, AnalysisType[] analysisTypes, AnalysisParameters parameters) {
        AnalysisResult<?>[] results = new AnalysisResult[analysisTypes.length];
        for (int i = 0; i < analysisTypes.length; i++) {
            results[i] = analyze(signal, analysisTypes[i], parameters);
        }
        return results;
    }

    @Override
    public <R> AnalysisResult<R> compareAnalyze(IVector<Double> signal1, IVector<Double> signal2, AnalysisType analysisType, AnalysisParameters parameters) {
        // 比较分析实现 / Comparison analysis implementation
        double snr1 = Signals.signalToNoiseRatio(signal1, signal1);
        double snr2 = Signals.signalToNoiseRatio(signal2, signal2);
        double snrDiff = snr1 - snr2;

        return new AnalysisResult<>(analysisType, (R) Double.valueOf(snrDiff),
                new String[]{"snr_difference"},
                "SNR comparison analysis", parameters.getConfidenceLevel());
    }

    @Override
    public AnalysisType[] getSupportedAnalysisTypes() {
        return new AnalysisType[]{AnalysisType.SPECTRUM, AnalysisType.POWER_SPECTRUM,
                AnalysisType.AUTOCORRELATION, AnalysisType.SNR};
    }

    @Override
    public boolean validateParameters(AnalysisType analysisType, AnalysisParameters parameters) {
        return parameters.getSamplingRate() > 0 && parameters.getWindowSize() > 0;
    }

    @Override
    public AnalysisParameters getRecommendedParameters(IVector<Double> signal, AnalysisType analysisType) {
        AnalysisParameters params = new AnalysisParameters();
        params.samplingRate(1000.0)
                .windowSize(Math.min(256, signal.length() / 4))
                .overlap(0.5)
                .confidenceLevel(0.95);
        return params;
    }
};

// 使用分析器进行信号分析 / Using analyzer for signal analysis
AnalysisParameters analysisParams = new AnalysisParameters()
        .samplingRate(1000.0)
        .windowSize(256)
        .overlap(0.5)
        .confidenceLevel(0.95);

// 单个分析 / Single analysis
AnalysisResult<Tuple3<IVector<Double>, IVector<Double>, IVector<Double>>> spectrumResult =
        analyzer.analyze(signal, AnalysisType.SPECTRUM, analysisParams);

// 批量分析 / Batch analysis
AnalysisType[] analysisTypes = {AnalysisType.SPECTRUM, AnalysisType.POWER_SPECTRUM, AnalysisType.SNR};
AnalysisResult<?>[] batchResults = analyzer.batchAnalyze(signal, analysisTypes, analysisParams);

// 比较分析 / Comparison analysis
AnalysisResult<Double> comparisonResult = analyzer.compareAnalyze(signal1, signal2, AnalysisType.SNR, analysisParams);
```

### 4. 信号工具 / Signal Utilities

#### 4.1 窗函数 / Window Functions

```java
import com.yishape.lab.math.signal.SignalUtilities;

// 生成各种窗函数 / Generate various window functions
IVector<Double> hanningWindow = SignalUtilities.window(256, SignalUtilities.WindowType.HANNING);
IVector<Double> hammingWindow = SignalUtilities.window(256, SignalUtilities.WindowType.HAMMING);
IVector<Double> blackmanWindow = SignalUtilities.window(256, SignalUtilities.WindowType.BLACKMAN);
IVector<Double> kaiserWindow = SignalUtilities.window(256, SignalUtilities.WindowType.KAISER, 8.6);
IVector<Double> gaussianWindow = SignalUtilities.window(256, SignalUtilities.WindowType.GAUSSIAN, 0.4);
```

#### 4.2 信号操作 / Signal Operations

```java

// 信号重采样 / Signal resampling
IVector<Double> resampled = SignalUtilities.resample(signal, 2000);  // 重采样到2000个点

// 信号拼接 / Signal concatenation
IVector<Double> concatenated = SignalUtilities.concatenate(signal1, signal2, signal3);

// 信号分割 / Signal segmentation
IVector<Double>[] segments = SignalUtilities.segment(signal, 256, 128);  // 256点段，128点重叠
```

#### 4.3 信号检测 / Signal Detection

```java
// 峰值检测 / Peak detection
int[] peaks = SignalUtilities.detectPeaks(signal, 0.5, 10);  // 阈值0.5，最小距离10

// 过零检测 / Zero crossing detection
int[] zeroCrossings = SignalUtilities.detectZeroCrossings(signal);

// 突变检测 / Change point detection
int[] changePoints = SignalUtilities.detectChangePoints(signal, 50, 0.1);  // 窗口50，阈值0.1
```

#### 4.4 信号预处理 / Signal Preprocessing

```java
// 信号归一化 / Signal normalization
IVector<Double> normalized = SignalUtilities.normalize(signal, -1.0, 1.0);

// 信号去趋势 / Signal detrending
IVector<Double> detrended = SignalUtilities.detrend(signal);

// 信号平滑 / Signal smoothing
IVector<Double> smoothed = SignalUtilities.smooth(signal, 5);
```

### 5. 小波分析 / Wavelet Analysis

#### 5.1 离散小波变换 (DWT) / Discrete Wavelet Transform (DWT)

```java
import com.yishape.lab.math.signal.WaveletAnalysis;

// 执行离散小波变换 / Perform discrete wavelet transform
WaveletAnalysis.WaveletCoefficients coeffs = WaveletAnalysis.discreteWaveletTransform(
    signal, WaveletAnalysis.WaveletType.DAUBECHIES, 4, 4.0
);

// 获取近似系数和细节系数 / Get approximation and detail coefficients
IVector<Double> approximation = coeffs.approximation;
IVector<Double>[] details = coeffs.details;

// 小波逆变换 / Inverse wavelet transform
IVector<Double> reconstructed = WaveletAnalysis.inverseDiscreteWaveletTransform(
    coeffs, WaveletAnalysis.WaveletType.DAUBECHIES, 4.0
);
```

#### 5.2 连续小波变换 (CWT) / Continuous Wavelet Transform (CWT)

```java
// 生成尺度数组 / Generate scale array
IVector<Double> scales = Linalg.linspace(1.0, 50.0, 20);

// 执行连续小波变换 / Perform continuous wavelet transform
IMatrix<Double> cwt = WaveletAnalysis.continuousWaveletTransform(
    signal, WaveletAnalysis.WaveletType.MORLET, scales, 5.0  // 信号, 小波类型, 尺度, 参数
);
```

#### 5.3 小波包变换 / Wavelet Packet Transform

```java
// 执行小波包变换 / Perform wavelet packet transform
WaveletAnalysis.WaveletPacketTree packetTree = WaveletAnalysis.waveletPacketTransform(
    signal, WaveletAnalysis.WaveletType.DAUBECHIES, 4, 4.0
);
```

#### 5.4 小波去噪和压缩 / Wavelet Denoising and Compression

```java
// 小波去噪 / Wavelet denoising
IVector<Double> denoised = WaveletAnalysis.waveletDenoising(
    noisySignal, WaveletAnalysis.WaveletType.DAUBECHIES, 4, 0.1, 4.0
);

// 小波压缩 / Wavelet compression
IVector<Double> compressed = WaveletAnalysis.waveletCompression(
    signal, WaveletAnalysis.WaveletType.DAUBECHIES, 4, 0.5, 4.0  // 压缩比0.5
);
```

#### 5.5 小波特征分析 / Wavelet Feature Analysis

```java
// 小波能量分析 / Wavelet energy analysis

IVector<Double>energy=WaveletAnalysis.waveletEnergyAnalysis(coeffs);

// 小波特征提取 / Wavelet feature extraction
        IVector<Double>features=WaveletAnalysis.waveletFeatureExtraction(coeffs);

// 使用信号变换接口进行小波分析 / Using signal transform interface for wavelet analysis
import com.yishape.lab.math.signal.transform.ISignalTransform;
import com.yishape.lab.math.signal.transform.ChirpZTransform;
import com.yishape.lab.math.signal.core.ISignalProcessor;
import com.yishape.lab.math.signal.core.AbstractSignalProcessor;
import com.yishape.lab.math.signal.core.SignalProcessingException;

// 创建自定义小波变换器 / Create custom wavelet transformer
ISignalTransform<Double, WaveletAnalysis.WaveletCoefficients> waveletTransformer =
        new ISignalTransform<Double, WaveletAnalysis.WaveletCoefficients>() {

            @Override
            public WaveletAnalysis.WaveletCoefficients forward(IVector<Double> signal) throws SignalProcessingException {
                return WaveletAnalysis.discreteWaveletTransform(
                        signal, WaveletAnalysis.WaveletType.DAUBECHIES, 4, 4.0);
            }

            @Override
            public IVector<Double> inverse(WaveletAnalysis.WaveletCoefficients transformed) throws SignalProcessingException {
                return WaveletAnalysis.inverseDiscreteWaveletTransform(
                        transformed, WaveletAnalysis.WaveletType.DAUBECHIES, 4.0);
            }

            @Override
            public boolean validateInput(IVector<Double> input) {
                return input != null && input.length() > 0;
            }

            @Override
            public String getName() {
                return "Custom Wavelet Transformer";
            }

            @Override
            public ISignalProcessor<Double> clone() {
                return this; // 简化实现
            }
        };

// 使用变换器 / Using transformer
try{
WaveletAnalysis.WaveletCoefficients coeffs = waveletTransformer.forward(signal);
IVector<Double> reconstructed = waveletTransformer.inverse(coeffs);
    
    System.out.

println("小波变换完成，分解层数: "+coeffs.levels);
    System.out.

println("近似系数长度: "+coeffs.approximation.length());
        System.out.

println("细节系数数量: "+coeffs.details.length);
    
}catch(
SignalProcessingException e){
        System.err.

println("小波变换失败: "+e.getMessage());
        }

// 使用Chirp-Z变换进行高分辨率频谱分析 / Using Chirp-Z transform for high-resolution spectral analysis
        try{
// 创建Chirp-Z变换器 / Create Chirp-Z transformer
ChirpZTransform czt = ChirpZTransform.forFrequencyRange(0.01, 0.5, 1000);

// 执行变换 / Perform transform
Complex[] cztResult = czt.forward(signal);

// 计算幅度谱 / Calculate magnitude spectrum
IVector<Double> magnitude = Linalg.zeros(cztResult.length);
    for(
int i = 0;
i<cztResult.length;i++){
        magnitude.

set(i, cztResult[i].magnitude());
        }

        System.out.

println("Chirp-Z变换完成，输出点数: "+cztResult.length);
    
}catch(
SignalProcessingException e){
        System.err.

println("Chirp-Z变换失败: "+e.getMessage());
        }

// 使用核心框架进行信号处理 / Using core framework for signal processing
AbstractSignalProcessor<Double> customProcessor = new AbstractSignalProcessor<Double>("Custom Processor", "1.0.0") {
    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        // 预处理：归一化 / Preprocessing: normalization
        IVector<Double> normalized = SignalUtilities.normalize(input, -1.0, 1.0);

        // 核心处理：小波去噪 / Core processing: wavelet denoising
        IVector<Double> denoised = WaveletAnalysis.waveletDenoising(
                normalized, WaveletAnalysis.WaveletType.DAUBECHIES, 4, 0.1, 4.0);

        // 后处理：平滑 / Postprocessing: smoothing
        return SignalUtilities.smooth(denoised, 3);
    }

    @Override
    protected boolean validateSpecificInput(IVector<Double> input) {
        return input.length() >= 16; // 最小长度要求
    }

    @Override
    public AbstractSignalProcessor<Double> clone() {
        return new AbstractSignalProcessor<Double>("Custom Processor", "1.0.0") {
            @Override
            protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
                return customProcessor.doProcess(input);
            }

            @Override
            protected boolean validateSpecificInput(IVector<Double> input) {
                return customProcessor.validateSpecificInput(input);
            }
        };
    }
};

// 使用自定义处理器 / Using custom processor
try{
IVector<Double> processed = customProcessor.process(signal);
    System.out.

println("自定义处理器处理完成，输出长度: "+processed.length());
        }catch(
SignalProcessingException e){
        System.err.

println("自定义处理器处理失败: "+e.getMessage());
        }
```

### 6. 快速傅里叶变换 (FFT) / Fast Fourier Transform (FFT)

#### 算法原理 / Algorithm Principles

FFT（快速傅里叶变换）是离散傅里叶变换（DFT）的加速算法，将 O(n²) 的复杂度降到 O(n log n)。

**DFT 的核心思想**：任何周期信号都可以表示为一系列正弦和余弦波的叠加。DFT 把时域信号转换为频域，得到每个频率成分的幅度和相位。

**为什么 FFT 输出不对称？**

FFT 输出的是复数频谱，通常包含正频率和负频率两部分（共轭对称）。对于实数输入信号，正负频率的幅度互为镜像，相位互为相反数。直观上：一个实数正弦波 `sin(2πft)` 只对应频谱上频率 f 和 -f 两个冲激，而不是一个。这是数学上正交基变换的必然结果，**不是 bug**。

实际应用中只取前半部分（N/2 个频率点）即可：
- 第 0 点：直流分量（DC，偏置）
- 第 1 ~ N/2-1 点：正频率部分
- 第 N/2 点：Nyquist 频率（采样率的 1/2）

**幅度谱的物理含义**：第 k 点对应的频率 = `k × Fs / N`，其中 Fs 是采样率，N 是信号长度。

**使用建议** / Usage Tips:

| 场景 | 建议 |
|------|------|
| 只想看正频率幅度 | 取前半段 `spectrum[0:length/2]` |
| 信号有直流偏移（均值不为0）| 先去均值：`signal = signal - signal.mean()` |
| 频谱分辨率不够 | 增加信号长度（补零不能提高真实分辨率，只能插值）|
| 分析非平稳信号 | 用短时傅里叶变换（STFT）或小波分析 |

#### 6.1 基本FFT操作 / Basic FFT Operations

```java
import com.yishape.lab.math.signal.RereFFT;
import com.yishape.lab.math.signal.Complex;

// 将实数信号转换为复数数组 / Convert real signal to complex array
Complex[] complexSignal = new Complex[signal.length()];
for (int i = 0; i < signal.length(); i++) {
    complexSignal[i] = new Complex(signal.get(i), 0);
}

// 执行FFT / Perform FFT
Complex[] fftResult = RereFFT.fft(complexSignal);

// 执行逆FFT / Perform IFFT
Complex[] ifftResult = RereFFT.ifft(fftResult);
```

#### 6.2 频谱分析 / Spectral Analysis

```java
// 计算幅度谱 / Calculate magnitude spectrum
double[] magnitude = RereFFT.magnitudeSpectrum(fftResult);

// 计算相位谱 / Calculate phase spectrum
double[] phase = RereFFT.phaseSpectrum(fftResult);

// 计算功率谱 / Calculate power spectrum
double[] power = RereFFT.powerSpectrum(fftResult);
```

### 7. 复数运算 / Complex Number Operations

#### 7.1 基本复数操作 / Basic Complex Operations

```java
import com.yishape.lab.math.signal.Complex;

// 创建复数 / Create complex numbers
Complex c1 = new Complex(3.0, 4.0);  // 3 + 4i
Complex c2 = new Complex(1.0, 2.0);  // 1 + 2i

// 基本运算 / Basic operations
Complex sum = c1.add(c2);           // 加法 / Addition
Complex diff = c1.subtract(c2);     // 减法 / Subtraction
Complex product = c1.multiply(c2);  // 乘法 / Multiplication
Complex quotient = c1.divide(c2);   // 除法 / Division

// 标量运算 / Scalar operations
Complex scaled = c1.scale(2.0);     // 标量乘法 / Scalar multiplication

// 复数属性 / Complex properties
double magnitude = c1.magnitude();  // 幅度 / Magnitude
double phase = c1.phase();          // 相位 / Phase
Complex conjugate = c1.conjugate(); // 共轭 / Conjugate
```

#### 7.2 高级复数操作 / Advanced Complex Operations

```java
// 复数的幂运算 / Power operations
Complex square = c1.square();       // 平方 / Square
Complex power = c1.power(3);        // 3次幂 / 3rd power
Complex sqrt = c1.sqrt();           // 平方根 / Square root

// 复数的三角函数 / Trigonometric functions
Complex sin = c1.sin();             // 正弦 / Sine
Complex cos = c1.cos();             // 余弦 / Cosine

// 复数的指数和对数 / Exponential and logarithm
Complex exp = c1.exp();             // 指数 / Exponential
Complex log = c1.log();             // 对数 / Logarithm

// 从极坐标创建复数 / Create complex from polar coordinates
Complex fromPolar = Complex.fromPolar(5.0, Math.PI/4);  // 幅度5，相位π/4
Complex unit = Complex.unit(Math.PI/3);                 // 单位复数，相位π/3
```

## 使用示例 / Usage Examples

### 完整信号处理流程 / Complete Signal Processing Workflow

``java
public class CompleteSignalProcessingExample {
    public static void main(String[] args) {
        // 1. 生成测试信号 / Generate test signal
        // Note: Using Signals class static methods instead of deleted SignalGeneration class
        IVector<Double> signal1 = Signals.sineWave(1000, 10.0, 1000.0, 1.0, 0.0);
        IVector<Double> signal2 = Signals.sineWave(1000, 50.0, 1000.0, 0.5, 0.0);
        IVector<Double> signal3 = Signals.sineWave(1000, 100.0, 1000.0, 0.3, 0.0);
        IVector<Double> signal = signal1.add(signal2).add(signal3);
        
        // 添加噪声 / Add noise
        IVector<Double> noise = Signals.whiteNoise(1000, 0.1);
        IVector<Double> noisySignal = signal.add(noise);
        
        System.out.println("=== 完整信号处理流程 / Complete Signal Processing Workflow ===");
        
        // 2. 信号预处理 / Signal preprocessing
        System.out.println("\n--- 步骤1: 信号预处理 / Step 1: Signal Preprocessing ---");
        IVector<Double> normalized = SignalUtilities.normalize(noisySignal, -1.0, 1.0);
        IVector<Double> detrended = SignalUtilities.detrend(normalized);
        System.out.println("预处理完成，信号长度: " + detrended.length());
        
        // 3. 信号滤波 / Signal filtering
        System.out.println("\n--- 步骤2: 信号滤波 / Step 2: Signal Filtering ---");
        IVector<Double> filtered = Signals.butterworthLowPass(detrended, 80.0, 1000.0, 2);
        System.out.println("滤波完成");
        
        // 4. 频谱分析 / Spectral analysis
        System.out.println("\n--- 步骤3: 频谱分析 / Step 3: Spectral Analysis ---");
        Tuple2<IVector<Double>, IVector<Double>> psdResult = Signals.powerSpectralDensity(
            filtered, 256, 0.5, 1000.0
        );
        System.out.println("功率谱密度计算完成");
        
        // 5. 小波分析 / Wavelet analysis
        System.out.println("\n--- 步骤4: 小波分析 / Step 4: Wavelet Analysis ---");
        WaveletAnalysis.WaveletCoefficients coeffs = WaveletAnalysis.discreteWaveletTransform(
            filtered, WaveletAnalysis.WaveletType.DAUBECHIES, 4, 4.0
        );
        System.out.println("小波分解完成，层数: " + coeffs.levels);
        
        // 6. 特征提取 / Feature extraction
        System.out.println("\n--- 步骤5: 特征提取 / Step 5: Feature Extraction ---");
        IVector<Double> features = WaveletAnalysis.waveletFeatureExtraction(coeffs);
        System.out.println("特征提取完成，特征数量: " + features.length());
        
        // 7. 信号重建 / Signal reconstruction
        System.out.println("\n--- 步骤6: 信号重建 / Step 6: Signal Reconstruction ---");
        IVector<Double> reconstructed = WaveletAnalysis.inverseDiscreteWaveletTransform(
            coeffs, WaveletAnalysis.WaveletType.DAUBECHIES, 4.0
        );
        System.out.println("信号重建完成");
        
        // 8. 性能评估 / Performance evaluation
        System.out.println("\n--- 步骤7: 性能评估 / Step 7: Performance Evaluation ---");
        double snr = Signals.signalToNoiseRatio(signal, signal.sub(reconstructed));
        double psnr = Signals.peakSignalToNoiseRatio(signal, reconstructed);
        System.out.println("信噪比: " + snr + " dB");
        System.out.println("峰值信噪比: " + psnr + " dB");
    }
}
```

### 实时信号处理示例 / Real-time Signal Processing Example

``java
public class RealTimeSignalProcessingExample {
    public static void main(String[] args) {
        // 模拟实时信号处理 / Simulate real-time signal processing
        int bufferSize = 256;
        int numBuffers = 10;
        
        System.out.println("=== 实时信号处理示例 / Real-time Signal Processing Example ===");
        
        for (int buffer = 0; buffer < numBuffers; buffer++) {
            // 生成当前缓冲区的信号 / Generate signal for current buffer
            // Note: Using Signals class static methods instead of deleted SignalGeneration class
            IVector<Double> currentSignal = Signals.sineWave(
                bufferSize, 
                50.0 + buffer * 5.0,  // 频率递增 / Increasing frequency
                1000.0, 
                1.0, 
                0.0
            );
            
            // 添加噪声 / Add noise
            IVector<Double> noise = Signals.whiteNoise(bufferSize, 0.05);
            IVector<Double> noisySignal = currentSignal.add(noise);
            
            // 实时滤波 / Real-time filtering
            IVector<Double> filtered = Signals.movingAverage(noisySignal, 5);
            
            // 实时频谱分析 / Real-time spectral analysis
            Tuple2<IVector<Double>, IVector<Double>> psdResult = Signals.powerSpectralDensity(
                filtered, 128, 0.5, 1000.0
            );
            
            // 峰值检测 / Peak detection
            int[] peaks = SignalUtilities.detectPeaks(filtered, 0.3, 10);
            
            System.out.printf("缓冲区 %d: 检测到 %d 个峰值 / Buffer %d: %d peaks detected%n", 
                            buffer + 1, peaks.length);
        }
    }
}
```

## 性能特性 / Performance Features

### 计算效率 / Computational Efficiency

- 高效的FFT实现 / Efficient FFT implementation
- 优化的滤波算法 / Optimized filtering algorithms
- 向量化操作支持 / Vectorized operation support
- 内存友好的数据结构 / Memory-friendly data structures

### 数值稳定性 / Numerical Stability

- 稳定的数值算法 / Stable numerical algorithms
- 边界情况处理 / Boundary case handling
- 精度控制 / Precision control
- 异常处理 / Exception handling

### 扩展性 / Extensibility

- 模块化设计 / Modular design
- 接口抽象 / Interface abstraction
- 插件式架构 / Plugin architecture
- 自定义滤波器支持 / Custom filter support

## 注意事项 / Notes

1. **采样率选择** / **Sampling Rate Selection**: 确保采样率满足奈奎斯特定理 / Ensure sampling rate satisfies Nyquist theorem
2. **窗函数选择** / **Window Function Selection**: 根据应用选择合适的窗函数 / Choose appropriate window function based on application
3. **滤波器设计** / **Filter Design**: 注意滤波器的稳定性和线性相位特性 / Pay attention to filter stability and linear phase characteristics
4. **小波选择** / **Wavelet Selection**: 根据信号特性选择合适的小波基 / Choose appropriate wavelet basis based on signal characteristics
5. **内存管理** / **Memory Management**: 对于大规模信号，注意内存使用 / For large-scale signals, pay attention to memory usage

## 与Python库功能对照表 / Python Library Functionality Comparison Table

| 功能类别 / Function Category | yishape-math | Python库 / Python Libraries | 说明 / Description |
|---------|-------------|-------------------|------|
| **信号生成 / Signal Generation** | | | |
| 正弦波 / Sine wave | `Signals.sineWave()` | `numpy.sin()`, `scipy.signal.chirp` | 正弦波生成 / Sine wave generation |
| 方波 / Square wave | `Signals.squareWave()` | `scipy.signal.square()` | 方波生成 / Square wave generation |
| 三角波 / Triangular wave | `Signals.triangularWave()` | `scipy.signal.sawtooth()` | 三角波生成 / Triangular wave generation |
| 白噪声 / White noise | `Signals.whiteNoise()` | `numpy.random.normal()` | 白噪声生成 / White noise generation |
| 粉红噪声 / Pink noise | `Signals.pinkNoise()` | `pinknoise` | 粉红噪声生成 / Pink noise generation |
| **信号滤波 / Signal Filtering** | | | |
| 移动平均 / Moving average | `Signals.movingAverage()` | `scipy.signal.savgol_filter` | 移动平均滤波 / Moving average filtering |
| 中值滤波 / Median filter | `Signals.medianFilter()` | `scipy.signal.medfilt()` | 中值滤波 / Median filtering |
| 高斯滤波 / Gaussian filter | `Signals.gaussianFilter()` | `scipy.ndimage.gaussian_filter1d()` | 高斯滤波 / Gaussian filtering |
| 巴特沃斯滤波 / Butterworth filter | `Signals.butterworthLowPass()` | `scipy.signal.butter()` | 巴特沃斯滤波 / Butterworth filtering |
| 卡尔曼滤波 / Kalman filter | `Signals.kalmanFilter()` | `filterpy.kalman` | 卡尔曼滤波 / Kalman filtering |
| 维纳滤波 / Wiener filter | `Signals.wienerFilter()` | `scipy.signal.wiener()` | 维纳滤波 / Wiener filtering |
| 带通滤波 / Bandpass filter | `Signals.bandPass()` | `scipy.signal.butter()` | 带通滤波 / Bandpass filtering |
| 带阻滤波 / Band-stop filter | `Signals.bandStop()` | `scipy.signal.butter()` | 带阻滤波 / Band-stop filtering |
| **信号分析 / Signal Analysis** | | | |
| 功率谱密度 / PSD | `Signals.powerSpectralDensity()` | `scipy.signal.welch()` | 功率谱密度计算 / Power spectral density calculation |
| 自相关 / Autocorrelation | `Signals.autocorrelation()` | `numpy.correlate()` | 自相关函数 / Autocorrelation function |
| 互相关 / Cross-correlation | `Signals.crossCorrelation()` | `numpy.correlate()` | 互相关函数 / Cross-correlation function |
| 短时傅里叶变换 / STFT | `Signals.shortTimeFourierTransform()` | `scipy.signal.stft()` | 短时傅里叶变换 / Short-time Fourier transform |
| **小波分析 / Wavelet Analysis** | | | |
| 离散小波变换 / DWT | `WaveletAnalysis.discreteWaveletTransform()` | `pywt.dwt()` | 离散小波变换 / Discrete wavelet transform |
| 连续小波变换 / CWT | `WaveletAnalysis.continuousWaveletTransform()` | `pywt.cwt()` | 连续小波变换 / Continuous wavelet transform |
| 小波去噪 / Wavelet denoising | `WaveletAnalysis.waveletDenoising()` | `pywt.threshold()` | 小波去噪 / Wavelet denoising |
| 小波包变换 / Wavelet packet | `WaveletAnalysis.waveletPacketTransform()` | `pywt.WaveletPacket()` | 小波包变换 / Wavelet packet transform |
| **FFT变换 / FFT Transform** | | | |
| 快速傅里叶变换 / FFT | `RereFFT.fft()` | `numpy.fft.fft()` | 快速傅里叶变换 / Fast Fourier transform |
| 逆FFT / IFFT | `RereFFT.ifft()` | `numpy.fft.ifft()` | 逆快速傅里叶变换 / Inverse FFT |
| 幅度谱 / Magnitude spectrum | `RereFFT.magnitudeSpectrum()` | `numpy.abs()` | 幅度谱计算 / Magnitude spectrum calculation |
| 相位谱 / Phase spectrum | `RereFFT.phaseSpectrum()` | `numpy.angle()` | 相位谱计算 / Phase spectrum calculation |
| **信号工具 / Signal Utilities** | | | |
| 窗函数 / Window functions | `SignalUtilities.window()` | `scipy.signal.windows` | 窗函数生成 / Window function generation |
| 信号重采样 / Resampling | `SignalUtilities.resample()` | `scipy.signal.resample()` | 信号重采样 / Signal resampling |
| 峰值检测 / Peak detection | `SignalUtilities.detectPeaks()` | `scipy.signal.find_peaks()` | 峰值检测 / Peak detection |
| 过零检测 / Zero crossing | `SignalUtilities.detectZeroCrossings()` | `scipy.signal.zero_crossings()` | 过零检测 / Zero crossing detection |

## 最佳实践建议 / Best Practices Recommendations

### 信号预处理 / Signal Preprocessing

1. **去趋势处理** / **Detrending**: 去除信号中的线性趋势 / Remove linear trends from signals
2. **归一化** / **Normalization**: 将信号归一化到合适的范围 / Normalize signals to appropriate range
3. **异常值处理** / **Outlier Handling**: 检测并处理异常值 / Detect and handle outliers
4. **窗函数选择** / **Window Function Selection**: 根据应用选择合适的窗函数 / Choose appropriate window function based on application

### 滤波器设计 / Filter Design

1. **截止频率选择** / **Cutoff Frequency Selection**: 根据信号特性选择合适的截止频率 / Choose appropriate cutoff frequency based on signal characteristics
2. **滤波器阶数** / **Filter Order**: 平衡滤波效果和计算复杂度 / Balance filtering effect and computational complexity
3. **稳定性检查** / **Stability Check**: 确保滤波器的稳定性 / Ensure filter stability
4. **相位特性** / **Phase Characteristics**: 考虑滤波器的相位特性 / Consider filter phase characteristics

### 频谱分析 / Spectral Analysis

1. **窗函数选择** / **Window Function Selection**: 选择合适的窗函数减少频谱泄漏 / Choose appropriate window function to reduce spectral leakage
2. **重叠处理** / **Overlap Processing**: 使用重叠窗口提高频谱估计精度 / Use overlapping windows to improve spectral estimation accuracy
3. **零填充** / **Zero Padding**: 使用零填充提高频率分辨率 / Use zero padding to improve frequency resolution
4. **平均化** / **Averaging**: 使用平均化减少噪声影响 / Use averaging to reduce noise effects

### 小波分析 / Wavelet Analysis

1. **小波选择** / **Wavelet Selection**: 根据信号特性选择合适的小波基 / Choose appropriate wavelet basis based on signal characteristics
2. **分解层数** / **Decomposition Levels**: 选择合适的分解层数 / Choose appropriate number of decomposition levels
3. **阈值选择** / **Threshold Selection**: 选择合适的阈值进行去噪 / Choose appropriate threshold for denoising
4. **边界处理** / **Boundary Handling**: 注意边界效应的影响 / Pay attention to boundary effects

---

**信号处理** - 让数字信号处理更简单、更高效！

**Signal Processing** - Making digital signal processing simpler and more efficient!

## 常见问题 / FAQ

### Q1: FFT 结果不对称，正负频率都有？

**原因**：FFT 对实数输入的输出是共轭对称的，正负频率幅度相同，这是数学性质，不是错误。

**解决**：如果只需要正频率，取前半段即可：
```java
int halfN = fftResult.length / 2;
for (int i = 0; i < halfN; i++) {
    System.out.println("频率 " + i + ": " + fftResult[i].magnitude());
}
```

### Q2: 滤波后信号幅值变小了？

**原因**：某些滤波器（如 Butterworth）在通带内有小于 1 的增益，导致整体幅值衰减。

**解决**：滤波后进行幅值归一化：
```java
IVector<Double> filtered = filter.process(signal);
double maxAmp = filtered.abs().max();
IVector<Double> normalized = filtered.multiplyScalar(1.0 / maxAmp);
```

### Q3: 小波去噪后信号看起来还是噪？

**排查顺序**：
1. 阈值是否过大 → 尝试更小的软阈值：`WaveletAnalysis.waveletDenoising(signal, type, level, 0.1)`
2. 小波类型是否合适 → `DAUBECHIES` 适合突变信号，`MORLET` 适合振荡信号
3. 分解层数是否合适 → 层数过多会丢失细节，一般 3~5 层足够

### Q4: 卡尔曼滤波和维纳滤波该用哪个？

| 场景 | 推荐 |
|------|------|
| 跟踪动态系统（雷达、导航）| 卡尔曼滤波（状态空间模型）|
| 已知噪声统计特性的平稳信号 | 维纳滤波 |
| 实时跟踪、时变系统 | 卡尔曼滤波 |
| 非平稳噪声 | 卡尔曼滤波（自适应）|

### Q5: 采样率设多少合适？

**Nyquist 采样定理**：采样率 Fs 必须 > 2 × 信号最高频率成分。

- 若信号最高频率为 1000Hz，采样率至少 2000Hz，建议 4000Hz 以上
- 采样率过低会导致频谱混叠（aliasing），信号失真无法恢复
- 实际中通常取信号最高频率的 3~5 倍

### Q6: 移动平均和中值滤波哪个好？

| 噪声类型 | 推荐 |
|---------|------|
| 高斯噪声（连续随机扰动）| 移动平均或高斯滤波 |
| 脉冲噪声（偶尔的尖刺）| 中值滤波 |
| 混合噪声 | 先中值滤波去脉冲，再移动平均去高斯噪声 |