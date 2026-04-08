# 信号处理示例文档 / Signal Processing Examples Documentation

## 概述 / Overview

本文档提供了YiShape-Math信号处理模块的完整使用示例，从基础到高级，帮助用户快速掌握各种信号处理功能的使用方法。

This document provides complete usage examples of the YiShape-Math signal processing module, from basic to advanced, helping users quickly master the use of various signal processing functions.

## 目录 / Table of Contents

1. [基础信号处理示例 / Basic Signal Processing Examples](#1-基础信号处理示例--basic-signal-processing-examples)
2. [高级信号处理示例 / Advanced Signal Processing Examples](#2-高级信号处理示例--advanced-signal-processing-examples)
3. [实时信号处理示例 / Real-time Signal Processing Examples](#3-实时信号处理示例--real-time-signal-processing-examples)
4. [滤波器功能示例 / Filter Functionality Examples](#4-滤波器功能示例--filter-functionality-examples)

## 第一部分：基础信号处理示例 (Level 1 - Basic Signal Processing Examples) / Part 1: Basic Signal Processing Examples (Level 1 - Basic Signal Processing Examples)

### 1.1 基本信号生成 / Basic Signal Generation

``java
import com.yishape.lab.math.signal.Signals;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.Tuple2;

public class BasicSignalGenerationExample {
    public static void main(String[] args) {
        System.out.println("=== 基本信号生成 / Basic Signal Generation ===");
        
        // 生成各种基本波形 / Generate various basic waveforms
        IVector<Double> sineWave = Signals.sineWave(1000, 50.0, 1000.0, 1.0, 0.0);
        IVector<Double> cosineWave = Signals.cosineWave(1000, 50.0, 1000.0, 1.0, 0.0);
        IVector<Double> squareWave = Signals.squareWave(1000, 50.0, 1000.0, 1.0, 0.5);
        IVector<Double> triangularWave = Signals.triangularWave(1000, 50.0, 1000.0, 1.0, 0.5);
        IVector<Double> sawtoothWave = Signals.sawtoothWave(1000, 50.0, 1000.0, 1.0);
        
        // 生成噪声信号 / Generate noise signals
        IVector<Double> whiteNoise = Signals.whiteNoise(1000, 0.1);
        IVector<Double> pinkNoise = Signals.pinkNoise(1000, 0.1);
        
        // 输出信号统计信息 / Output signal statistics
        printSignalStatistics(sineWave, "正弦波 / Sine Wave");
        printSignalStatistics(cosineWave, "余弦波 / Cosine Wave");
        printSignalStatistics(whiteNoise, "白噪声 / White Noise");
        
        System.out.println("基本信号生成完成 / Basic signal generation completed");
    }
    
    private static void printSignalStatistics(IVector<Double> signal, String signalName) {
        System.out.println(signalName + " 统计信息 / " + signalName + " Statistics:");
        System.out.println("  长度: " + signal.length());
        System.out.println("  均值: " + String.format("%.3f", signal.mean()));
        System.out.println("  标准差: " + String.format("%.3f", signal.std()));
        System.out.println("  最大值: " + String.format("%.3f", signal.max()));
        System.out.println("  最小值: " + String.format("%.3f", signal.min()));
        System.out.println("  方差: " + String.format("%.3f", signal.var()));
    }
}
```

### 1.2 复合信号生成 / Composite Signal Generation

``java
public class CompositeSignalGenerationExample {
    public static void main(String[] args) {
        System.out.println("=== 复合信号生成 / Composite Signal Generation ===");
        
        // 使用新添加的复合信号生成方法 / Using newly added composite signal generation methods
        // 生成阶跃信号 / Generate step signal
        IVector<Double> stepSignal = Signals.stepSignal(1000, 1.0, 0.5, 1000.0);
        System.out.println("阶跃信号生成完成，长度: " + stepSignal.length());
        
        // 生成单位阶跃信号 / Generate unit step signal
        IVector<Double> unitStep = Signals.unitStep(1000, 0.5, 1000.0);
        System.out.println("单位阶跃信号生成完成，长度: " + unitStep.length());
        
        // 生成脉冲信号 / Generate pulse signal
        IVector<Double> pulseSignal = Signals.pulseSignal(1000, 1.0, 10, 10.0, 1000.0);
        System.out.println("脉冲信号生成完成，长度: " + pulseSignal.length());
        
        // 生成单位脉冲信号 / Generate unit impulse signal
        IVector<Double> unitImpulse = Signals.unitImpulse(1000, 500);
        System.out.println("单位脉冲信号生成完成，长度: " + unitImpulse.length());
        
        // 生成线性调频信号 / Generate linear chirp signal
        IVector<Double> chirpSignal = Signals.chirpSignal(1000, 10.0, 100.0, 1000.0, 1.0);
        System.out.println("线性调频信号生成完成，长度: " + chirpSignal.length());
        
        // 生成狄拉克δ函数 / Generate Dirac delta function
        IVector<Double> diracDelta = Signals.diracDelta(1000, 100, 1.0);
        System.out.println("狄拉克δ函数生成完成，长度: " + diracDelta.length());
        
        // 使用复合信号生成方法 / Using composite signal generation method
        com.yishape.lab.math.signal.generation.ISignalGenerator.SignalType[] signalTypes = {
            com.yishape.lab.math.signal.generation.ISignalGenerator.SignalType.SINE,
            com.yishape.lab.math.signal.generation.ISignalGenerator.SignalType.COSINE
        };
        
        com.yishape.lab.math.signal.generation.ISignalGenerator.SignalParameters[] parameters = {
            new com.yishape.lab.math.signal.generation.ISignalGenerator.SignalParameters().frequency(10.0).amplitude(1.0).samplingRate(1000.0),
            new com.yishape.lab.math.signal.generation.ISignalGenerator.SignalParameters().frequency(20.0).amplitude(0.5).samplingRate(1000.0)
        };
        
        IVector<Double> compositeSignal = Signals.compositeSignal(signalTypes, 1000, parameters);
        System.out.println("复合信号生成完成，长度: " + compositeSignal.length());
        
        // 添加噪声 / Add noise
        IVector<Double> noisySignal = Signals.addNoise(
            compositeSignal, 
            com.yishape.lab.math.signal.generation.ISignalGenerator.SignalType.WHITE_NOISE, 
            new com.yishape.lab.math.signal.generation.ISignalGenerator.SignalParameters().noiseVariance(0.1).samplingRate(1000.0)
        );
        System.out.println("带噪声信号生成完成，长度: " + noisySignal.length());
        
        System.out.println("所有复合信号生成完成 / All composite signals generation completed");
    }
}
```

### 1.3 基础滤波示例 / Basic Filtering Examples

``java
public class BasicFilteringExample {
    public static void main(String[] args) {
        System.out.println("=== 基础滤波示例 / Basic Filtering Examples ===");
        
        // 生成测试信号 / Generate test signal
        IVector<Double> cleanSignal = Signals.sineWave(1000, 50.0, 1000.0, 1.0, 0.0);
        IVector<Double> noise = Signals.whiteNoise(1000, 0.3);
        IVector<Double> noisySignal = cleanSignal.add(noise);
        
        System.out.println("测试信号生成完成 / Test signal generation completed");
        
        // 应用各种基础滤波器 / Apply various basic filters
        IVector<Double> movingAvgFiltered = Signals.movingAverage(noisySignal, 5);
        System.out.println("移动平均滤波完成 / Moving average filtering completed");
        
        IVector<Double> medianFiltered = Signals.medianFilter(noisySignal, 5);
        System.out.println("中值滤波完成 / Median filtering completed");
        
        IVector<Double> gaussianFiltered = Signals.gaussianFilter(noisySignal, 1.0);
        System.out.println("高斯滤波完成 / Gaussian filtering completed");
        
        // 应用频域滤波器 / Apply frequency domain filters
        IVector<Double> lowPassFiltered = Signals.butterworthLowPass(noisySignal, 100.0, 1000.0, 2);
        System.out.println("巴特沃斯低通滤波完成 / Butterworth low-pass filtering completed");
        
        IVector<Double> highPassFiltered = Signals.butterworthHighPass(noisySignal, 10.0, 1000.0, 2);
        System.out.println("巴特沃斯高通滤波完成 / Butterworth high-pass filtering completed");
        
        IVector<Double> bandPassFiltered = Signals.bandPass(noisySignal, 20.0, 80.0, 1000.0, 2);
        System.out.println("带通滤波完成 / Band-pass filtering completed");
        
        // 计算信噪比改善 / Calculate SNR improvement
        double originalSNR = Signals.signalToNoiseRatio(cleanSignal, noise);
        double filteredSNR = Signals.signalToNoiseRatio(cleanSignal, cleanSignal.sub(movingAvgFiltered));
        
        System.out.println("原始信噪比: " + String.format("%.2f", originalSNR) + " dB");
        System.out.println("滤波后信噪比: " + String.format("%.2f", filteredSNR) + " dB");
        System.out.println("信噪比改善: " + String.format("%.2f", filteredSNR - originalSNR) + " dB");
    }
}
```

## 第二部分：高级信号处理示例 (Level 2 - Advanced Signal Processing Examples) / Part 2: Advanced Signal Processing Examples (Level 2 - Advanced Signal Processing Examples)

### 2.1 完整信号处理流程 / Complete Signal Processing Workflow

```java
import com.yishape.lab.math.signal.Signals;
import com.yishape.lab.math.signal.SignalUtilities;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.Tuple2;

public class CompleteSignalProcessingExample {
    public static void main(String[] args) {
        System.out.println("=== 完整信号处理流程 / Complete Signal Processing Workflow ===");
        
        // 1. 生成测试信号 / Generate test signal
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
        com.yishape.lab.math.signal.WaveletAnalysis.WaveletCoefficients coeffs = com.yishape.lab.math.signal.WaveletAnalysis.discreteWaveletTransform(
            filtered, com.yishape.lab.math.signal.WaveletAnalysis.WaveletType.DAUBECHIES, 4, 4.0
        );
        System.out.println("小波分解完成，层数: " + coeffs.levels);
        
        // 6. 特征提取 / Feature extraction
        System.out.println("\n--- 步骤5: 特征提取 / Step 5: Feature Extraction ---");
        IVector<Double> features = com.yishape.lab.math.signal.WaveletAnalysis.waveletFeatureExtraction(coeffs);
        System.out.println("特征提取完成，特征数量: " + features.length());
        
        // 7. 信号重建 / Signal reconstruction
        System.out.println("\n--- 步骤6: 信号重建 / Step 6: Signal Reconstruction ---");
        IVector<Double> reconstructed = com.yishape.lab.math.signal.WaveletAnalysis.inverseDiscreteWaveletTransform(
            coeffs, com.yishape.lab.math.signal.WaveletAnalysis.WaveletType.DAUBECHIES, 4.0
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

### 2.2 高级小波分析 / Advanced Wavelet Analysis

``java
public class AdvancedWaveletAnalysisExample {
    public static void main(String[] args) {
        System.out.println("=== 高级小波分析 / Advanced Wavelet Analysis ===");
        
        // 生成复杂测试信号 / Generate complex test signal
        IVector<Double> signal = generateComplexTestSignal();
        
        // 添加噪声 / Add noise
        IVector<Double> noise = Signals.whiteNoise(signal.length(), 0.15);
        IVector<Double> noisySignal = signal.add(noise);
        
        System.out.println("测试信号生成完成，长度: " + signal.length());
        
        // 比较不同小波类型 / Compare different wavelet types
        com.yishape.lab.math.signal.WaveletAnalysis.WaveletType[] waveletTypes = {
            com.yishape.lab.math.signal.WaveletAnalysis.WaveletType.HAAR,
            com.yishape.lab.math.signal.WaveletAnalysis.WaveletType.DAUBECHIES,
            com.yishape.lab.math.signal.WaveletAnalysis.WaveletType.MORLET
        };
        
        for (com.yishape.lab.math.signal.WaveletAnalysis.WaveletType waveletType : waveletTypes) {
            System.out.println("\n--- 使用 " + waveletType + " 小波 / Using " + waveletType + " Wavelet ---");
            
            // 小波分解 / Wavelet decomposition
            com.yishape.lab.math.signal.WaveletAnalysis.WaveletCoefficients coeffs = com.yishape.lab.math.signal.Signals.discreteWaveletTransform(
                noisySignal, waveletType, 5, getWaveletParameter(waveletType)
            );
            
            // 小波去噪 / Wavelet denoising
            IVector<Double> denoised = com.yishape.lab.math.signal.WaveletAnalysis.waveletDenoising(
                noisySignal, waveletType, 5, 0.1, getWaveletParameter(waveletType)
            );
            
            // 小波压缩 / Wavelet compression
            IVector<Double> compressed = com.yishape.lab.math.signal.WaveletAnalysis.waveletCompression(
                noisySignal, waveletType, 5, 0.3, getWaveletParameter(waveletType)
            );
            
            // 小波包变换 / Wavelet packet transform
            com.yishape.lab.math.signal.WaveletAnalysis.WaveletPacketTree packetTree = com.yishape.lab.math.signal.WaveletAnalysis.waveletPacketTransform(
                noisySignal, waveletType, 4, getWaveletParameter(waveletType)
            );
            
            // 分析结果 / Analyze results
            analyzeWaveletResults(signal, noisySignal, denoised, compressed, coeffs, waveletType);
        }
        
        // 连续小波变换分析 / Continuous wavelet transform analysis
        System.out.println("\n--- 连续小波变换分析 / Continuous Wavelet Transform Analysis ---");
        performContinuousWaveletAnalysis(noisySignal);
    }
    
    private static IVector<Double> generateComplexTestSignal() {
        // 生成包含瞬态和调频的复杂信号
        // Generate complex signal with transients and frequency modulation
        IVector<Double> baseSignal1 = Signals.sineWave(2048, 20.0, 2048.0, 1.0, 0.0);
        IVector<Double> baseSignal2 = Signals.sineWave(2048, 60.0, 2048.0, 0.8, 0.0);
        IVector<Double> baseSignal3 = Signals.sineWave(2048, 120.0, 2048.0, 0.6, 0.0);
        IVector<Double> baseSignal = baseSignal1.add(baseSignal2).add(baseSignal3);
        
        // 添加调频信号 / Add frequency modulated signal
        IVector<Double> fmSignal = Signals.chirpSignal(2048, 10.0, 50.0, 2048.0, 0.5);
        
        // 添加瞬态信号 / Add transient signals
        IVector<Double> transient1 = Signals.diracDelta(2048, 512, 2.0);
        IVector<Double> transient2 = Signals.diracDelta(2048, 1024, 1.5);
        IVector<Double> transient3 = Signals.diracDelta(2048, 1536, 1.8);
        
        return baseSignal.add(fmSignal).add(transient1).add(transient2).add(transient3);
    }
    
    private static double getWaveletParameter(com.yishape.lab.math.signal.WaveletAnalysis.WaveletType waveletType) {
        switch (waveletType) {
            case DAUBECHIES:
                return 4.0;  // db4
            case MORLET:
                return 5.0;  // 中心频率 / Center frequency
            case GAUSSIAN:
                return 2.0;  // 高斯阶数 / Gaussian order
            default:
                return 0.0;
        }
    }
    
    private static void analyzeWaveletResults(IVector<Double> original, IVector<Double> noisy,
                                            IVector<Double> denoised, IVector<Double> compressed,
                                            com.yishape.lab.math.signal.WaveletAnalysis.WaveletCoefficients coeffs,
                                            com.yishape.lab.math.signal.WaveletAnalysis.WaveletType waveletType) {
        // 计算信噪比 / Calculate signal-to-noise ratio
        double originalSNR = Signals.signalToNoiseRatio(original, original.sub(noisy));
        double denoisedSNR = Signals.signalToNoiseRatio(original, original.sub(denoised));
        double compressedSNR = Signals.signalToNoiseRatio(original, original.sub(compressed));
        
        // 计算峰值信噪比 / Calculate peak signal-to-noise ratio
        double originalPSNR = Signals.peakSignalToNoiseRatio(original, noisy);
        double denoisedPSNR = Signals.peakSignalToNoiseRatio(original, denoised);
        double compressedPSNR = Signals.peakSignalToNoiseRatio(original, compressed);
        
        // 小波能量分析 / Wavelet energy analysis
        IVector<Double> energy = com.yishape.lab.math.signal.WaveletAnalysis.waveletEnergyAnalysis(coeffs);
        
        // 小波特征提取 / Wavelet feature extraction
        IVector<Double> features = com.yishape.lab.math.signal.WaveletAnalysis.waveletFeatureExtraction(coeffs);
        
        System.out.println("小波类型: " + waveletType);
        System.out.println("原始信噪比: " + String.format("%.2f", originalSNR) + " dB");
        System.out.println("去噪后信噪比: " + String.format("%.2f", denoisedSNR) + " dB");
        System.out.println("压缩后信噪比: " + String.format("%.2f", compressedSNR) + " dB");
        System.out.println("原始峰值信噪比: " + String.format("%.2f", originalPSNR) + " dB");
        System.out.println("去噪后峰值信噪比: " + String.format("%.2f", denoisedPSNR) + " dB");
        System.out.println("压缩后峰值信噪比: " + String.format("%.2f", compressedPSNR) + " dB");
        System.out.println("小波能量分布: " + energy);
        System.out.println("小波特征数量: " + features.length());
    }
    
    private static void performContinuousWaveletAnalysis(IVector<Double> signal) {
        // 生成尺度数组 / Generate scale array
        IVector<Double> scales = Linalg.linspace(1.0, 100.0, 50);
        
        // 执行连续小波变换 / Perform continuous wavelet transform
        com.yishape.lab.math.linalg.IMatrix<Double> cwt = com.yishape.lab.math.signal.WaveletAnalysis.continuousWaveletTransform(
            signal, com.yishape.lab.math.signal.WaveletAnalysis.WaveletType.MORLET, scales, 5.0
        );
        
        System.out.println("连续小波变换完成");
        System.out.println("CWT矩阵大小: " + cwt.getRowNum() + " x " + cwt.getColNum());
        System.out.println("尺度范围: " + scales.get(0) + " - " + scales.get(scales.length()-1));
        
        // 分析CWT结果 / Analyze CWT results
        double maxCWT = cwt.max();
        double minCWT = cwt.min();
        System.out.println("CWT最大值: " + String.format("%.3f", maxCWT));
        System.out.println("CWT最小值: " + String.format("%.3f", minCWT));
    }
}
```

## 第三部分：实时信号处理示例 (Level 3 - Real-time Signal Processing Examples) / Part 3: Real-time Signal Processing Examples (Level 3 - Real-time Signal Processing Examples)

### 3.1 实时信号处理 / Real-time Signal Processing

```java
import com.yishape.lab.math.signal.Signals;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.Tuple2;

public class RealTimeSignalProcessingExample {
    public static void main(String[] args) {
        System.out.println("=== 实时信号处理 / Real-time Signal Processing ===");
        
        // 模拟实时信号处理 / Simulate real-time signal processing
        int bufferSize = 512;
        int numBuffers = 20;
        double samplingRate = 1000.0;
        
        // 初始化滤波器状态 / Initialize filter state
        IVector<Double> previousBuffer = Linalg.zeros(bufferSize);
        
        for (int buffer = 0; buffer < numBuffers; buffer++) {
            System.out.println("\n--- 处理缓冲区 " + (buffer + 1) + " / Processing Buffer " + (buffer + 1) + " ---");
            
            // 生成当前缓冲区的信号 / Generate signal for current buffer
            IVector<Double> currentSignal = generateRealTimeSignal(buffer, bufferSize, samplingRate);
            
            // 添加噪声 / Add noise
            IVector<Double> noise = Signals.whiteNoise(bufferSize, 0.05);
            IVector<Double> noisySignal = currentSignal.add(noise);
            
            // 实时滤波 / Real-time filtering
            IVector<Double> filtered = performRealTimeFiltering(noisySignal, previousBuffer);
            
            // 实时频谱分析 / Real-time spectral analysis
            RealTimeSpectralResult spectralResult = performRealTimeSpectralAnalysis(filtered, samplingRate);
            
            // 实时特征提取 / Real-time feature extraction
            IVector<Double> features = performRealTimeFeatureExtraction(filtered, spectralResult);
            
            // 实时检测 / Real-time detection
            RealTimeDetectionResult detectionResult = performRealTimeDetection(filtered);
            
            // 输出结果 / Output results
            System.out.println("信号长度: " + filtered.length());
            System.out.println("主要频率: " + String.format("%.1f", spectralResult.dominantFreq) + " Hz");
            System.out.println("信号功率: " + String.format("%.3f", spectralResult.power));
            System.out.println("检测到峰值: " + detectionResult.peakCount + " 个");
            System.out.println("特征数量: " + features.length());
            
            // 更新状态 / Update state
            previousBuffer = filtered.copy();
        }
    }
    
    private static IVector<Double> generateRealTimeSignal(int bufferIndex, int bufferSize, double samplingRate) {
        // 生成随时间变化的信号 / Generate time-varying signal
        double baseFreq = 50.0 + bufferIndex * 2.0;  // 频率递增 / Increasing frequency
        double amplitude = 1.0 + 0.1 * Math.sin(bufferIndex * 0.1);  // 幅度调制 / Amplitude modulation
        
        return Signals.sineWave(
            bufferSize,
            baseFreq,
            samplingRate,
            amplitude,
            0.0
        );
    }
    
    private static IVector<Double> performRealTimeFiltering(IVector<Double> signal, IVector<Double> previousBuffer) {
        // 简单的移动平均滤波 / Simple moving average filtering
        return Signals.movingAverage(signal, 5);
    }
    
    private static RealTimeSpectralResult performRealTimeSpectralAnalysis(IVector<Double> signal, double samplingRate) {
        // 计算功率谱密度 / Calculate power spectral density
        Tuple2<IVector<Double>, IVector<Double>> psdResult = Signals.powerSpectralDensity(
            signal, 128, 0.5, samplingRate
        );
        
        IVector<Double> frequencies = psdResult._1;
        IVector<Double> psd = psdResult._2;
        
        // 找到主要频率 / Find dominant frequency
        int maxIndex = 0;
        double maxPower = psd.get(0);
        for (int i = 1; i < psd.length(); i++) {
            if (psd.get(i) > maxPower) {
                maxPower = psd.get(i);
                maxIndex = i;
            }
        }
        
        double dominantFreq = frequencies.get(maxIndex);
        double power = psd.sum();
        
        return new RealTimeSpectralResult(dominantFreq, power, psd);
    }
    
    private static IVector<Double> performRealTimeFeatureExtraction(IVector<Double> signal, 
                                                                   RealTimeSpectralResult spectralResult) {
        java.util.List<Double> features = new java.util.ArrayList<>();
        
        // 时域特征 / Time domain features
        features.add(signal.mean());
        features.add(signal.std());
        features.add(signal.max());
        features.add(signal.min());
        
        // 频域特征 / Frequency domain features
        features.add(spectralResult.dominantFreq);
        features.add(spectralResult.power);
        
        // 转换为向量 / Convert to vector
        double[] featureArray = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            featureArray[i] = features.get(i);
        }
        
        return Linalg.vector(featureArray);
    }
    
    private static RealTimeDetectionResult performRealTimeDetection(IVector<Double> signal) {
        // 峰值检测 / Peak detection
        int[] peaks = SignalUtilities.detectPeaks(signal, 0.3, 20);
        
        // 过零检测 / Zero crossing detection
        int[] zeroCrossings = SignalUtilities.detectZeroCrossings(signal);
        
        return new RealTimeDetectionResult(peaks.length, zeroCrossings.length);
    }
    
    // 结果类 / Result classes
    private static class RealTimeSpectralResult {
        double dominantFreq;
        double power;
        IVector<Double> psd;
        
        RealTimeSpectralResult(double dominantFreq, double power, IVector<Double> psd) {
            this.dominantFreq = dominantFreq;
            this.power = power;
            this.psd = psd;
        }
    }
    
    private static class RealTimeDetectionResult {
        int peakCount;
        int zeroCrossingCount;
        
        RealTimeDetectionResult(int peakCount, int zeroCrossingCount) {
            this.peakCount = peakCount;
            this.zeroCrossingCount = zeroCrossingCount;
        }
    }
}
```

## 第四部分：滤波器功能示例 (Level 4 - Filter Functionality Examples) / Part 4: Filter Functionality Examples (Level 4 - Filter Functionality Examples)

### 4.1 卡尔曼滤波 / Kalman Filtering

``java
public class KalmanFilteringExample {
    public static void main(String[] args) {
        System.out.println("=== 卡尔曼滤波 / Kalman Filtering ===");
        
        // 生成带噪声的测试信号 / Generate noisy test signal
        IVector<Double> cleanSignal = Signals.sineWave(1000, 10.0, 1000.0, 1.0, 0.0);
        IVector<Double> noise = Signals.whiteNoise(1000, 0.1);
        IVector<Double> noisySignal = cleanSignal.add(noise);
        
        // 应用卡尔曼滤波 / Apply Kalman filtering
        IVector<Double> kalmanFiltered = Signals.kalmanFilter(noisySignal, 0.1, 0.1);
        
        // 计算性能指标 / Calculate performance metrics
        double originalSNR = Signals.signalToNoiseRatio(cleanSignal, noise);
        double filteredSNR = Signals.signalToNoiseRatio(cleanSignal, cleanSignal.sub(kalmanFiltered));
        
        System.out.println("原始信噪比: " + String.format("%.2f", originalSNR) + " dB");
        System.out.println("卡尔曼滤波后信噪比: " + String.format("%.2f", filteredSNR) + " dB");
        System.out.println("信噪比改善: " + String.format("%.2f", filteredSNR - originalSNR) + " dB");
        
        // 通过工厂模式使用卡尔曼滤波器 / Using Kalman filter through factory pattern
        try {
            com.yishape.lab.math.signal.filter.ISignalFilter<Double> kalmanFilter = Signals.createFilter("kalman");
            IVector<Double> factoryFiltered = kalmanFilter.process(noisySignal);
            System.out.println("工厂模式卡尔曼滤波完成");
        } catch (Exception e) {
            System.err.println("工厂模式卡尔曼滤波失败: " + e.getMessage());
        }
    }
}
```

### 4.2 维纳滤波 / Wiener Filtering

``java
public class WienerFilteringExample {
    public static void main(String[] args) {
        System.out.println("=== 维纳滤波 / Wiener Filtering ===");
        
        // 生成带噪声的测试信号 / Generate noisy test signal
        IVector<Double> cleanSignal = Signals.sineWave(1000, 10.0, 1000.0, 1.0, 0.0);
        IVector<Double> noise = Signals.whiteNoise(1000, 0.1);
        IVector<Double> noisySignal = cleanSignal.add(noise);
        
        // 应用维纳滤波 / Apply Wiener filtering
        IVector<Double> wienerFiltered = Signals.wienerFilter(noisySignal, 1.0, 0.1, 10);
        
        // 计算性能指标 / Calculate performance metrics
        double originalSNR = Signals.signalToNoiseRatio(cleanSignal, noise);
        double filteredSNR = Signals.signalToNoiseRatio(cleanSignal, cleanSignal.sub(wienerFiltered));
        
        System.out.println("原始信噪比: " + String.format("%.2f", originalSNR) + " dB");
        System.out.println("维纳滤波后信噪比: " + String.format("%.2f", filteredSNR) + " dB");
        System.out.println("信噪比改善: " + String.format("%.2f", filteredSNR - originalSNR) + " dB");
        
        // 通过工厂模式使用维纳滤波器 / Using Wiener filter through factory pattern
        try {
            com.yishape.lab.math.signal.filter.ISignalFilter<Double> wienerFilter = Signals.createFilter("wiener");
            IVector<Double> factoryFiltered = wienerFilter.process(noisySignal);
            System.out.println("工厂模式维纳滤波完成");
        } catch (Exception e) {
            System.err.println("工厂模式维纳滤波失败: " + e.getMessage());
        }
    }
}
```

### 4.3 带阻滤波 / Band-stop Filtering

``java
public class BandStopFilteringExample {
    public static void main(String[] args) {
        System.out.println("=== 带阻滤波 / Band-stop Filtering ===");
        
        // 生成包含干扰频率的测试信号 / Generate test signal with interference frequencies
        IVector<Double> signal1 = Signals.sineWave(2000, 10.0, 2000.0, 1.0, 0.0);
        IVector<Double> signal2 = Signals.sineWave(2000, 50.0, 2000.0, 0.8, 0.0); // 干扰频率 / Interference frequency
        IVector<Double> signal3 = Signals.sineWave(2000, 100.0, 2000.0, 0.6, 0.0);
        IVector<Double> signal4 = Signals.sineWave(2000, 150.0, 2000.0, 0.4, 0.0); // 干扰频率 / Interference frequency
        IVector<Double> cleanSignal = signal1.add(signal2).add(signal3).add(signal4);
        
        // 添加噪声 / Add noise
        IVector<Double> noise = Signals.whiteNoise(2000, 0.05);
        IVector<Double> noisySignal = cleanSignal.add(noise);
        
        // 应用带阻滤波去除干扰频率 / Apply band-stop filtering to remove interference frequencies
        IVector<Double> bandStopFiltered1 = Signals.bandStop(noisySignal, 45.0, 55.0, 2000.0, 2); // 去除50Hz干扰 / Remove 50Hz interference
        IVector<Double> bandStopFiltered2 = Signals.bandStop(bandStopFiltered1, 145.0, 155.0, 2000.0, 2); // 去除150Hz干扰 / Remove 150Hz interference
        
        // 计算性能指标 / Calculate performance metrics
        double originalSNR = Signals.signalToNoiseRatio(cleanSignal, noise);
        double filteredSNR = Signals.signalToNoiseRatio(cleanSignal, cleanSignal.sub(bandStopFiltered2));
        
        System.out.println("原始信噪比: " + String.format("%.2f", originalSNR) + " dB");
        System.out.println("带阻滤波后信噪比: " + String.format("%.2f", filteredSNR) + " dB");
        System.out.println("信噪比改善: " + String.format("%.2f", filteredSNR - originalSNR) + " dB");
        
        // 通过工厂模式使用带阻滤波器 / Using band-stop filter through factory pattern
        try {
            com.yishape.lab.math.signal.filter.ISignalFilter<Double> bandStopFilter = Signals.createFilter("bandstop");
            IVector<Double> factoryFiltered = bandStopFilter.process(noisySignal);
            System.out.println("工厂模式带阻滤波完成");
        } catch (Exception e) {
            System.err.println("工厂模式带阻滤波失败: " + e.getMessage());
        }
    }
}
```

## 辅助方法 / Helper Methods

``java
// 生成时间轴的辅助方法 / Helper method for generating time axis
private static IVector<Double> generateTimeAxis(int length, double samplingRate) {
    return Linalg.range(length).multiplyScalar(1.0 / samplingRate);
}

// 计算信号统计信息的辅助方法 / Helper method for calculating signal statistics
private static void printSignalStatistics(IVector<Double> signal, String signalName) {
    System.out.println(signalName + " 统计信息 / " + signalName + " Statistics:");
    System.out.println("  长度: " + signal.length());
    System.out.println("  均值: " + String.format("%.3f", signal.mean()));
    System.out.println("  标准差: " + String.format("%.3f", signal.std()));
    System.out.println("  最大值: " + String.format("%.3f", signal.max()));
    System.out.println("  最小值: " + String.format("%.3f", signal.min()));
    System.out.println("  方差: " + String.format("%.3f", signal.var()));
}

// 带通滤波辅助方法 / Helper method for bandpass filtering
private static IVector<Double> applyBandpassFilter(IVector<Double> signal, double lowFreq, double highFreq, 
                                                 double samplingRate, int order) {
    return Signals.bandPass(signal, lowFreq, highFreq, samplingRate, order);
}
```

---

**信号处理示例** - 从基础到高级的完整学习路径！

**Signal Processing Examples** - Complete learning path from basic to advanced!