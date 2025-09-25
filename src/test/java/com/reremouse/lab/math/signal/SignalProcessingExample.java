package com.reremouse.lab.math.signal;

import com.reremouse.lab.math.signal.core.SignalUtilities;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.core.Complex;
import com.reremouse.lab.math.signal.core.RereDCT;
import com.reremouse.lab.math.signal.core.RereFFT;
import com.reremouse.lab.math.signal.core.RereHilbert;
import com.reremouse.lab.util.Tuple2;

/**
 * 信号处理示例类 / Signal Processing Example Class
 * <p>
 * 展示如何使用math.signal包中的各种信号处理功能。
 * Demonstrates how to use various signal processing functions in the math.signal package.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class SignalProcessingExample {

    public static void main(String[] args) {
        System.out.println("=== 信号处理功能示例 / Signal Processing Example ===");
        
        // 生成测试信号 / Generate test signal
        int length = 1000;
        double samplingRate = 1000.0; // 1kHz
        
        // 生成复合信号：正弦波 + 噪声 / Generate composite signal: sine wave + noise
        IVector<Double> signal = Signals.sineWave(length, 10.0, samplingRate, 1.0, 0.0)
                .add(Signals.sineWave(length, 50.0, samplingRate, 0.5, 0.0))
                .add(Signals.whiteNoise(length, 0.1));
        
        System.out.println("原始信号统计:");
        System.out.println("  长度: " + signal.length());
        System.out.println("  均值: " + signal.mean());
        System.out.println("  标准差: " + signal.std());
        System.out.println("  峰值: " + signal.max());
        
        // 信号滤波 / Signal filtering
        System.out.println("\n=== 信号滤波 / Signal Filtering ===");
        
        // 移动平均滤波 / Moving average filter
        IVector<Double> filteredMA = Signals.movingAverage(signal, 20);
        System.out.println("移动平均滤波后:");
        System.out.println("  均值: " + filteredMA.mean());
        System.out.println("  标准差: " + filteredMA.std());
        
        // 高斯滤波 / Gaussian filter
        IVector<Double> filteredGaussian = Signals.gaussianFilter(signal, 2.0, 0);
        System.out.println("高斯滤波后:");
        System.out.println("  均值: " + filteredGaussian.mean());
        System.out.println("  标准差: " + filteredGaussian.std());
        
        // 巴特沃斯低通滤波 / Butterworth low-pass filter
        IVector<Double> filteredLP = Signals.butterworthLowPass(signal, 50.0, 1000.0, 1);
        System.out.println("巴特沃斯低通滤波后:");
        System.out.println("  均值: " + filteredLP.mean());
        System.out.println("  标准差: " + filteredLP.std());
        
        // 信号分析 / Signal analysis
        System.out.println("\n=== 信号分析 / Signal Analysis ===");
        
        // 计算功率谱密度 / Calculate power spectral density
        Tuple2<IVector<Double>, IVector<Double>> psd = 
                Signals.powerSpectralDensity(signal, 256, 0.5, samplingRate);
        System.out.println("功率谱密度:");
        System.out.println("  频率范围: " + psd._1.get(0) + " - " + psd._1.get(psd._1.length()-1) + " Hz");
        // 找到最大功率的索引 / Find index of maximum power
        int maxIndex = 0;
        double maxPower = psd._2.get(0);
        for (int i = 1; i < psd._2.length(); i++) {
            if (psd._2.get(i) > maxPower) {
                maxPower = psd._2.get(i);
                maxIndex = i;
            }
        }
        System.out.println("  最大功率: " + maxPower + " at " + 
                psd._1.get(maxIndex) + " Hz");
        
        // 计算自相关 / Calculate autocorrelation
        IVector<Double> autocorr = Signals.autocorrelation(signal);
        System.out.println("自相关函数:");
        System.out.println("  最大值: " + autocorr.max());
        System.out.println("  零延迟值: " + autocorr.get(0));
        
        // 计算瞬时幅度和相位 / Calculate instantaneous amplitude and phase
        IVector<Double> envelope = RereHilbert.instantaneousAmplitude(signal);
        IVector<Double> phase = RereHilbert.instantaneousPhase(signal);
        System.out.println("瞬时特性:");
        System.out.println("  包络均值: " + envelope.mean());
        System.out.println("  相位范围: " + phase.min() + " - " + phase.max() + " rad");
        
        // 信号变换 / Signal transforms
        System.out.println("\n=== 信号变换 / Signal Transforms ===");
        
        // FFT变换 / FFT transform
        Complex[] fftResult = RereFFT.fft(convertToComplex(signal.slice(0, 256)));
        double[] magnitude = RereFFT.magnitudeSpectrum(fftResult);
        System.out.println("FFT变换:");
        System.out.println("  最大幅度: " + Linalg.vector(magnitude).max());
        
        // DCT变换 / DCT transform
        IVector<Double> dctResult = RereDCT.dct2(signal.slice(0, 256));
        System.out.println("DCT变换:");
        System.out.println("  前10个系数: ");
        for (int i = 0; i < Math.min(10, dctResult.length()); i++) {
            System.out.printf("    DCT[%d] = %.3f\n", i, dctResult.get(i));
        }
        
        // 信号工具 / Signal utilities
        System.out.println("\n=== 信号工具 / Signal Utilities ===");
        
        // 窗函数 / Window functions
        IVector<Double> hanningWindow = SignalUtilities.window(64, SignalUtilities.WindowType.HANNING);
        IVector<Double> hammingWindow = SignalUtilities.window(64, SignalUtilities.WindowType.HAMMING);
        System.out.println("窗函数:");
        System.out.println("  汉宁窗最大值: " + hanningWindow.max());
        System.out.println("  汉明窗最大值: " + hammingWindow.max());
        
        // 信号检测 / Signal detection
        int[] peaks = SignalUtilities.detectPeaks(signal, 0.5, 50);
        int[] zeroCrossings = SignalUtilities.detectZeroCrossings(signal);
        System.out.println("信号检测:");
        System.out.println("  检测到峰值数: " + peaks.length);
        System.out.println("  检测到过零点数: " + zeroCrossings.length);
        
        // 信号压缩 / Signal compression
        IVector<Double> compressed = RereDCT.compress(signal, 0.1);
        double compressionError = RereDCT.compressionError(signal, compressed);
        System.out.println("信号压缩:");
        System.out.println("  压缩比: 10%");
        System.out.println("  压缩误差: " + compressionError);
        
        System.out.println("\n=== 示例完成 / Example Complete ===");
    }
    
    /**
     * 将向量转换为复数数组 / Convert vector to complex array
     */
    private static Complex[] convertToComplex(IVector<Double> signal) {
        Complex[] complex = new Complex[signal.length()];
        for (int i = 0; i < signal.length(); i++) {
            complex[i] = new Complex(signal.get(i), 0);
        }
        return complex;
    }
}
