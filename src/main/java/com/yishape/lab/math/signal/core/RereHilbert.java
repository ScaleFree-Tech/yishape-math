package com.yishape.lab.math.signal.core;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * 希尔伯特变换类 / Hilbert Transform Class
 * <p>
 * 提供希尔伯特变换功能，用于计算信号的解析信号、瞬时幅度、瞬时相位等。
 * 使用IVector接口进行向量操作，确保与现有代码库的兼容性。
 * </p>
 * <p>
 * Provides Hilbert transform functionality for calculating analytic signal, instantaneous amplitude,
 * instantaneous phase, etc. Uses IVector interface for vector operations to ensure compatibility with existing codebase.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RereHilbert {

    /**
     * 希尔伯特变换 / Hilbert Transform
     * <p>
     * 计算信号的希尔伯特变换，使用FFT方法实现。
     * Calculate Hilbert transform of signal using FFT method.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @return 希尔伯特变换结果 / Hilbert transform result
     */
    public static IVector<Double> hilbertTransform(IVector<Double> signal) {
        int n = signal.length();
        
        // 零填充到2的幂 / Zero-pad to power of 2
        int fftSize = nextPowerOfTwo(2 * n - 1);
        Complex[] paddedSignal = new Complex[fftSize];
        
        for (int i = 0; i < n; i++) {
            paddedSignal[i] = new Complex(signal.get(i), 0);
        }
        for (int i = n; i < fftSize; i++) {
            paddedSignal[i] = new Complex(0, 0);
        }
        
        // 计算FFT / Calculate FFT
        Complex[] fftResult = RereFFT.fft(paddedSignal);
        
        // 应用希尔伯特滤波器 / Apply Hilbert filter
        for (int i = 0; i < fftSize; i++) {
            if (i == 0 || i == fftSize / 2) {
                fftResult[i] = new Complex(0, 0);
            } else if (i < fftSize / 2) {
                fftResult[i] = fftResult[i].scale(2);
            } else {
                fftResult[i] = new Complex(0, 0);
            }
        }
        
        // 计算逆FFT / Calculate IFFT
        Complex[] hilbertComplex = RereFFT.ifft(fftResult);
        
        // 提取结果 / Extract result
        IVector<Double> hilbert = Linalg.zeros(n);
        for (int i = 0; i < n; i++) {
            hilbert.set(i, hilbertComplex[i].imag);
        }
        
        return hilbert;
    }

    /**
     * 计算解析信号 / Calculate Analytic Signal
     * <p>
     * 计算信号的解析信号，即原信号加上其希尔伯特变换的虚部。
     * Calculate analytic signal, which is original signal plus imaginary part of its Hilbert transform.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @return 解析信号的实部和虚部 / Real and imaginary parts of analytic signal
     */
    public static Complex[] analyticSignal(IVector<Double> signal) {
        IVector<Double> hilbert = hilbertTransform(signal);
        
        Complex[] analytic = new Complex[signal.length()];
        for (int i = 0; i < signal.length(); i++) {
            analytic[i] = new Complex(signal.get(i), hilbert.get(i));
        }
        
        return analytic;
    }

    /**
     * 计算瞬时幅度 / Calculate Instantaneous Amplitude
     * <p>
     * 计算解析信号的幅度，即包络线。
     * Calculate amplitude of analytic signal, i.e., envelope.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @return 瞬时幅度向量 / Instantaneous amplitude vector
     */
    public static IVector<Double> instantaneousAmplitude(IVector<Double> signal) {
        Complex[] analytic = analyticSignal(signal);
        
        IVector<Double> amplitude = Linalg.zeros(signal.length());
        for (int i = 0; i < signal.length(); i++) {
            amplitude.set(i, analytic[i].magnitude());
        }
        
        return amplitude;
    }

    /**
     * 计算瞬时相位 / Calculate Instantaneous Phase
     * <p>
     * 计算解析信号的相位。
     * Calculate phase of analytic signal.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @return 瞬时相位向量 / Instantaneous phase vector
     */
    public static IVector<Double> instantaneousPhase(IVector<Double> signal) {
        Complex[] analytic = analyticSignal(signal);
        
        IVector<Double> phase = Linalg.zeros(signal.length());
        for (int i = 0; i < signal.length(); i++) {
            phase.set(i, Math.atan2(analytic[i].imag, analytic[i].real));
        }
        
        return phase;
    }

    /**
     * 计算瞬时频率 / Calculate Instantaneous Frequency
     * <p>
     * 通过相位差分计算瞬时频率。
     * Calculate instantaneous frequency by phase differentiation.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param samplingRate 采样率 / Sampling rate
     * @return 瞬时频率向量 / Instantaneous frequency vector
     */
    public static IVector<Double> instantaneousFrequency(IVector<Double> signal, double samplingRate) {
        IVector<Double> phase = instantaneousPhase(signal);
        
        // 计算相位差分 / Calculate phase difference
        IVector<Double> phaseDiff = Linalg.zeros(signal.length());
        for (int i = 1; i < signal.length(); i++) {
            double diff = phase.get(i) - phase.get(i - 1);
            
            // 相位解包 / Phase unwrapping
            while (diff > Math.PI) diff -= 2 * Math.PI;
            while (diff < -Math.PI) diff += 2 * Math.PI;
            
            phaseDiff.set(i, diff);
        }
        
        // 计算瞬时频率 / Calculate instantaneous frequency
        return phaseDiff.multiplyScalar(samplingRate / (2 * Math.PI));
    }

    /**
     * 计算信号的包络线 / Calculate Signal Envelope
     * <p>
     * 使用希尔伯特变换计算信号的包络线。
     * Calculate signal envelope using Hilbert transform.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @return 包络线向量 / Envelope vector
     */
    public static IVector<Double> envelope(IVector<Double> signal) {
        return instantaneousAmplitude(signal);
    }

    /**
     * 计算信号的瞬时功率 / Calculate Instantaneous Power
     * <p>
     * 计算解析信号的功率。
     * Calculate power of analytic signal.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @return 瞬时功率向量 / Instantaneous power vector
     */
    public static IVector<Double> instantaneousPower(IVector<Double> signal) {
        IVector<Double> amplitude = instantaneousAmplitude(signal);
        return amplitude.multiply(amplitude);
    }

    /**
     * 计算信号的瞬时能量 / Calculate Instantaneous Energy
     * <p>
     * 计算解析信号的能量。
     * Calculate energy of analytic signal.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @return 瞬时能量向量 / Instantaneous energy vector
     */
    public static IVector<Double> instantaneousEnergy(IVector<Double> signal) {
        return instantaneousPower(signal);
    }

    /**
     * 计算信号的瞬时熵 / Calculate Instantaneous Entropy
     * <p>
     * 基于瞬时功率计算信号的瞬时熵。
     * Calculate instantaneous entropy based on instantaneous power.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param windowSize 窗口大小 / Window size
     * @return 瞬时熵向量 / Instantaneous entropy vector
     */
    public static IVector<Double> instantaneousEntropy(IVector<Double> signal, int windowSize) {
        IVector<Double> power = instantaneousPower(signal);
        IVector<Double> entropy = Linalg.zeros(signal.length());
        
        for (int i = 0; i < signal.length(); i++) {
            int start = Math.max(0, i - windowSize / 2);
            int end = Math.min(signal.length(), i + windowSize / 2 + 1);
            
            // 计算窗口内的功率分布 / Calculate power distribution within window
            IVector<Double> windowPower = power.slice(start, end);
            double totalPower = windowPower.sum();
            
            if (totalPower > 0) {
                // 归一化功率 / Normalize power
                IVector<Double> normalizedPower = windowPower.multiplyScalar(1.0 / totalPower);
                
                // 计算熵 / Calculate entropy
                double entropyValue = 0;
                for (int j = 0; j < normalizedPower.length(); j++) {
                    double p = normalizedPower.get(j);
                    if (p > 0) {
                        entropyValue -= p * Math.log(p);
                    }
                }
                entropy.set(i, (Double) entropyValue);
            } else {
                entropy.set(i, (Double) 0.0);
            }
        }
        
        return entropy;
    }

    /**
     * 计算信号的瞬时带宽 / Calculate Instantaneous Bandwidth
     * <p>
     * 基于瞬时幅度和频率计算瞬时带宽。
     * Calculate instantaneous bandwidth based on instantaneous amplitude and frequency.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param samplingRate 采样率 / Sampling rate
     * @return 瞬时带宽向量 / Instantaneous bandwidth vector
     */
    public static IVector<Double> instantaneousBandwidth(IVector<Double> signal, double samplingRate) {
        IVector<Double> amplitude = instantaneousAmplitude(signal);
        
        // 计算幅度导数的绝对值 / Calculate absolute value of amplitude derivative
        IVector<Double> amplitudeDerivative = Linalg.zeros(signal.length());
        for (int i = 1; i < signal.length(); i++) {
            double deriv = (amplitude.get(i) - amplitude.get(i - 1)) * samplingRate;
            amplitudeDerivative.set(i, (Double) Math.abs(deriv));
        }
        
        // 计算瞬时带宽 / Calculate instantaneous bandwidth
        IVector<Double> bandwidth = Linalg.zeros(signal.length());
        for (int i = 0; i < signal.length(); i++) {
            if (amplitude.get(i) > 0) {
                bandwidth.set(i, (Double) (amplitudeDerivative.get(i) / amplitude.get(i)));
            } else {
                bandwidth.set(i, (Double) 0.0);
            }
        }
        
        return bandwidth;
    }

    // ========== 辅助方法 / Helper Methods ==========

    /**
     * 计算大于等于n的最小2的幂 / Calculate smallest power of 2 >= n
     */
    private static int nextPowerOfTwo(int n) {
        if (n <= 0) return 1;
        if ((n & (n - 1)) == 0) return n;
        
        int power = 1;
        while (power < n) {
            power <<= 1;
        }
        return power;
    }
}
