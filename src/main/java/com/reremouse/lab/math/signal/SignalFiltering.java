package com.reremouse.lab.math.signal;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;

/**
 * 信号滤波类 / Signal Filtering Class
 * <p>
 * 提供各种信号滤波功能，包括低通、高通、带通、带阻滤波器，以及移动平均、中值滤波等。
 * 使用IVector和IMatrix接口进行向量和矩阵操作，确保与现有代码库的兼容性。
 * </p>
 * <p>
 * Provides various signal filtering functions including low-pass, high-pass, band-pass, band-stop filters,
 * as well as moving average, median filtering, etc. Uses IVector and IMatrix interfaces for vector and
 * matrix operations to ensure compatibility with existing codebase.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class SignalFiltering {

    /**
     * 移动平均滤波器 / Moving Average Filter
     * <p>
     * 简单的低通滤波器，通过计算滑动窗口内的平均值来平滑信号。
     * Simple low-pass filter that smooths signal by calculating average within sliding window.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param windowSize 窗口大小 / Window size
     * @return 滤波后的信号向量 / Filtered signal vector
     */
    public static IVector<Double> movingAverage(IVector<Double> signal, int windowSize) {
        if (windowSize <= 0 || windowSize > signal.length()) {
            throw new IllegalArgumentException("窗口大小必须大于0且小于等于信号长度");
        }
        
        IVector<Double> filtered = Linalg.zeros(signal.length());
        
        for (int i = 0; i < signal.length(); i++) {
            int start = Math.max(0, i - windowSize + 1);
            int end = i + 1;
            
            double sum = 0;
            int count = 0;
            for (int j = start; j < end; j++) {
                sum += signal.get(j);
                count++;
            }
            filtered.set(i, sum / count);
        }
        
        return filtered;
    }

    /**
     * 中值滤波器 / Median Filter
     * <p>
     * 非线性滤波器，通过计算滑动窗口内的中值来去除脉冲噪声。
     * Non-linear filter that removes impulse noise by calculating median within sliding window.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param windowSize 窗口大小 / Window size
     * @return 滤波后的信号向量 / Filtered signal vector
     */
    public static IVector<Double> medianFilter(IVector<Double> signal, int windowSize) {
        if (windowSize <= 0 || windowSize > signal.length()) {
            throw new IllegalArgumentException("窗口大小必须大于0且小于等于信号长度");
        }
        
        IVector<Double> filtered = Linalg.zeros(signal.length());
        
        for (int i = 0; i < signal.length(); i++) {
            int start = Math.max(0, i - windowSize / 2);
            int end = Math.min(signal.length(), i + windowSize / 2 + 1);
            
            // 提取窗口内的值 / Extract values within window
            IVector<Double> window = signal.slice(start, end);
            
            // 排序并取中值 / Sort and take median
            IVector<Double> sorted = window.sort();
            int medianIndex = sorted.length() / 2;
            filtered.set(i, sorted.get(medianIndex));
        }
        
        return filtered;
    }

    public static IVector<Double> gaussianFilter(IVector<Double> signal, double sigma) {
        // 使用默认的kernelSize（0表示自动计算）
        return gaussianFilter(signal, sigma, 0);
    }
    
    /**
     * 高斯滤波器 / Gaussian Filter
     * <p>
     * 使用高斯核进行卷积的线性滤波器，具有良好的频率特性。
     * Linear filter using Gaussian kernel convolution with good frequency characteristics.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param sigma 高斯核的标准差 / Standard deviation of Gaussian kernel
     * @param kernelSize 核大小（如果为0则自动计算） / Kernel size (auto-calculated if 0)
     * @return 滤波后的信号向量 / Filtered signal vector
     */
    public static IVector<Double> gaussianFilter(IVector<Double> signal, double sigma, int kernelSize) {
        if (sigma <= 0) {
            throw new IllegalArgumentException("标准差必须大于0");
        }
        
        // 自动计算核大小 / Auto-calculate kernel size
        if (kernelSize <= 0) {
            kernelSize = (int) (6 * sigma) + 1;
            if (kernelSize % 2 == 0) kernelSize++;
        }
        
        // 创建高斯核 / Create Gaussian kernel
        IVector<Double> kernel = createGaussianKernel(kernelSize, sigma);
        
        // 归一化核 / Normalize kernel
        double kernelSum = kernel.sum();
        kernel = kernel.multiplyScalar(1.0 / kernelSum);
        
        // 卷积 / Convolution
        return convolve(signal, kernel);
    }

    /**
     * 巴特沃斯低通滤波器 / Butterworth Low-pass Filter
     * <p>
     * IIR滤波器，在通带内具有平坦的频率响应。
     * IIR filter with flat frequency response in passband.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param cutoffFreq 截止频率 (Hz) / Cutoff frequency (Hz)
     * @param samplingRate 采样率 (Hz) / Sampling rate (Hz)
     * @param order 滤波器阶数 / Filter order
     * @return 滤波后的信号向量 / Filtered signal vector
     */
    public static IVector<Double> butterworthLowPass(IVector<Double> signal, double cutoffFreq, double samplingRate, int order) {
        if (cutoffFreq <= 0 || cutoffFreq >= samplingRate / 2) {
            throw new IllegalArgumentException("截止频率必须在(0, " + samplingRate / 2 + ")范围内");
        }
        
        // 将截止频率转换为归一化频率 / Convert cutoff frequency to normalized frequency
        double normalizedCutoffFreq = cutoffFreq / samplingRate;
        
        // 计算滤波器系数 / Calculate filter coefficients
        double[] b = new double[order + 1];
        double[] a = new double[order + 1];
        
        // 简化的巴特沃斯滤波器设计（一阶） / Simplified Butterworth filter design (first order)
        if (order == 1) {
            double omega = Math.tan(Math.PI * normalizedCutoffFreq);
            double k = omega / (1 + omega);
            
            b[0] = k;
            b[1] = k;
            a[0] = 1.0;
            a[1] = k - 1;
        } else {
            // 对于高阶滤波器，这里使用简化的设计 / For higher order filters, use simplified design
            throw new UnsupportedOperationException("目前只支持一阶巴特沃斯滤波器");
        }
        
        // 应用滤波器 / Apply filter
        return applyIIRFilter(signal, b, a);
    }

    /**
     * 巴特沃斯高通滤波器 / Butterworth High-pass Filter
     * <p>
     * IIR滤波器，去除低频成分。
     * IIR filter that removes low-frequency components.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param cutoffFreq 截止频率 (Hz) / Cutoff frequency (Hz)
     * @param samplingRate 采样率 (Hz) / Sampling rate (Hz)
     * @param order 滤波器阶数 / Filter order
     * @return 滤波后的信号向量 / Filtered signal vector
     */
    public static IVector<Double> butterworthHighPass(IVector<Double> signal, double cutoffFreq, double samplingRate, int order) {
        if (cutoffFreq <= 0 || cutoffFreq >= samplingRate / 2) {
            throw new IllegalArgumentException("截止频率必须在(0, " + samplingRate / 2 + ")范围内");
        }
        
        // 将截止频率转换为归一化频率 / Convert cutoff frequency to normalized frequency
        double normalizedCutoffFreq = cutoffFreq / samplingRate;
        
        // 计算滤波器系数 / Calculate filter coefficients
        double[] b = new double[order + 1];
        double[] a = new double[order + 1];
        
        // 简化的巴特沃斯高通滤波器设计（一阶） / Simplified Butterworth high-pass filter design (first order)
        if (order == 1) {
            double omega = Math.tan(Math.PI * normalizedCutoffFreq);
            double k = 1 / (1 + omega);
            
            b[0] = k;
            b[1] = -k;
            a[0] = 1.0;
            a[1] = k - 1;
        } else {
            throw new UnsupportedOperationException("目前只支持一阶巴特沃斯滤波器");
        }
        
        // 应用滤波器 / Apply filter
        return applyIIRFilter(signal, b, a);
    }

    /**
     * 带通滤波器 / Band-pass Filter
     * <p>
     * 通过级联低通和高通滤波器实现带通滤波。
     * Implements band-pass filtering by cascading low-pass and high-pass filters.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param lowCutoff 低截止频率 / Low cutoff frequency
     * @param highCutoff 高截止频率 / High cutoff frequency
     * @param order 滤波器阶数 / Filter order
     * @return 滤波后的信号向量 / Filtered signal vector
     */
    public static IVector<Double> bandPass(IVector<Double> signal, double lowCutoff, double highCutoff, int order) {
        // 使用默认采样率1000Hz / Use default sampling rate of 1000Hz
        double defaultSamplingRate = 1000.0;
        return bandPass(signal, lowCutoff, highCutoff, defaultSamplingRate, order);
    }
    
    /**
     * 带通滤波器 / Band-pass Filter
     * <p>
     * 通过级联低通和高通滤波器实现带通滤波。
     * Implements band-pass filtering by cascading low-pass and high-pass filters.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param lowCutoff 低截止频率 (Hz) / Low cutoff frequency (Hz)
     * @param highCutoff 高截止频率 (Hz) / High cutoff frequency (Hz)
     * @param samplingRate 采样率 (Hz) / Sampling rate (Hz)
     * @param order 滤波器阶数 / Filter order
     * @return 滤波后的信号向量 / Filtered signal vector
     */
    public static IVector<Double> bandPass(IVector<Double> signal, double lowCutoff, double highCutoff, double samplingRate, int order) {
        if (lowCutoff >= highCutoff || lowCutoff <= 0 || highCutoff >= samplingRate / 2) {
            throw new IllegalArgumentException("频率参数无效");
        }
        
        // 先应用高通滤波器 / Apply high-pass filter first
        IVector<Double> highPassed = butterworthHighPass(signal, lowCutoff, samplingRate, order);
        
        // 再应用低通滤波器 / Then apply low-pass filter
        return butterworthLowPass(highPassed, highCutoff,samplingRate, order);
    }

    /**
     * 带阻滤波器 / Band-stop Filter
     * <p>
     * 通过从原信号中减去带通信号实现带阻滤波。
     * Implements band-stop filtering by subtracting band-pass signal from original signal.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param lowCutoff 低截止频率 / Low cutoff frequency
     * @param highCutoff 高截止频率 / High cutoff frequency
     * @param order 滤波器阶数 / Filter order
     * @return 滤波后的信号向量 / Filtered signal vector
     */
    public static IVector<Double> bandStop(IVector<Double> signal, double lowCutoff, double highCutoff, int order) {
        if (lowCutoff >= highCutoff || lowCutoff <= 0 || highCutoff >= 0.5) {
            throw new IllegalArgumentException("频率参数无效");
        }
        
        // 计算带通信号 / Calculate band-pass signal
        IVector<Double> bandPassSignal = bandPass(signal, lowCutoff, highCutoff, order);
        
        // 从原信号中减去带通信号 / Subtract band-pass signal from original signal
        return signal.sub(bandPassSignal);
    }

    /**
     * 卡尔曼滤波器 / Kalman Filter
     * <p>
     * 用于估计动态系统状态的递归滤波器。
     * Recursive filter for estimating state of dynamic systems.
     * </p>
     *
     * @param measurements 测量值向量 / Measurement vector
     * @param processNoise 过程噪声方差 / Process noise variance
     * @param measurementNoise 测量噪声方差 / Measurement noise variance
     * @return 滤波后的状态估计向量 / Filtered state estimate vector
     */
    public static IVector<Double> kalmanFilter(IVector<Double> measurements, double processNoise, double measurementNoise) {
        int n = measurements.length();
        IVector<Double> filtered = Linalg.zeros(n);
        
        // 初始化 / Initialize
        double state = measurements.get(0);
        double errorCovariance = 1.0;
        
        filtered.set(0, state);
        
        for (int i = 1; i < n; i++) {
            // 预测步骤 / Prediction step
            double predictedState = state;
            double predictedErrorCovariance = errorCovariance + processNoise;
            
            // 更新步骤 / Update step
            double kalmanGain = predictedErrorCovariance / (predictedErrorCovariance + measurementNoise);
            state = predictedState + kalmanGain * (measurements.get(i) - predictedState);
            errorCovariance = (1 - kalmanGain) * predictedErrorCovariance;
            
            filtered.set(i, state);
        }
        
        return filtered;
    }

    /**
     * 维纳滤波器 / Wiener Filter
     * <p>
     * 基于信号和噪声统计特性的最优线性滤波器。
     * Optimal linear filter based on signal and noise statistics.
     * </p>
     *
     * @param noisySignal 含噪信号向量 / Noisy signal vector
     * @param noiseVariance 噪声方差 / Noise variance
     * @return 滤波后的信号向量 / Filtered signal vector
     */
    public static IVector<Double> wienerFilter(IVector<Double> noisySignal, double noiseVariance) {
        // 简化的维纳滤波器实现 / Simplified Wiener filter implementation
        // 这里使用移动平均作为近似 / Use moving average as approximation here
        int windowSize = Math.max(3, (int) Math.sqrt(noisySignal.length()));
        return movingAverage(noisySignal, windowSize);
    }

    // ========== 辅助方法 / Helper Methods ==========

    /**
     * 创建高斯核 / Create Gaussian kernel
     */
    private static IVector<Double> createGaussianKernel(int size, double sigma) {
        IVector<Double> kernel = Linalg.zeros(size);
        int center = size / 2;
        
        for (int i = 0; i < size; i++) {
            double x = i - center;
            double value = Math.exp(-(x * x) / (2 * sigma * sigma));
            kernel.set(i, value);
        }
        
        return kernel;
    }

    /**
     * 卷积运算 / Convolution operation
     */
    private static IVector<Double> convolve(IVector<Double> signal, IVector<Double> kernel) {
        int signalLength = signal.length();
        int kernelLength = kernel.length();
        int resultLength = signalLength + kernelLength - 1;
        
        IVector<Double> result = Linalg.zeros(resultLength);
        
        for (int i = 0; i < resultLength; i++) {
            double sum = 0;
            for (int j = 0; j < kernelLength; j++) {
                int signalIndex = i - j;
                if (signalIndex >= 0 && signalIndex < signalLength) {
                    sum += signal.get(signalIndex) * kernel.get(j);
                }
            }
            result.set(i, sum);
        }
        
        // 返回与输入信号相同长度的结果 / Return result with same length as input signal
        return result.slice(kernelLength / 2, kernelLength / 2 + signalLength);
    }

    /**
     * 应用IIR滤波器 / Apply IIR filter
     */
    private static IVector<Double> applyIIRFilter(IVector<Double> signal, double[] b, double[] a) {
        int n = signal.length();
        IVector<Double> filtered = Linalg.zeros(n);
        
        // 初始化 / Initialize
        for (int i = 0; i < n; i++) {
            double y = 0;
            
            // 前向项 / Forward terms
            for (int j = 0; j < b.length && i - j >= 0; j++) {
                y += b[j] * signal.get(i - j);
            }
            
            // 反馈项 / Feedback terms
            for (int j = 1; j < a.length && i - j >= 0; j++) {
                y -= a[j] * filtered.get(i - j);
            }
            
            filtered.set(i, y / a[0]);
        }
        
        return filtered;
    }
}
