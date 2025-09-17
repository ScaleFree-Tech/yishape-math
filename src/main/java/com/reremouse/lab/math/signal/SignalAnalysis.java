package com.reremouse.lab.math.signal;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;

/**
 * 信号分析类 / Signal Analysis Class
 * <p>
 * 提供各种信号分析功能，包括频谱分析、功率谱密度、自相关、互相关等。
 * 使用IVector和IMatrix接口进行向量和矩阵操作，确保与现有代码库的兼容性。
 * </p>
 * <p>
 * Provides various signal analysis functions including spectral analysis, power spectral density,
 * autocorrelation, cross-correlation, etc. Uses IVector and IMatrix interfaces for vector and matrix
 * operations to ensure compatibility with existing codebase.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class SignalAnalysis {

    /**
     * 计算信号的功率谱密度 (PSD) / Calculate Power Spectral Density (PSD)
     * <p>
     * 使用Welch方法计算功率谱密度，这是一种改进的周期图方法。
     * Uses Welch's method to calculate power spectral density, which is an improved periodogram method.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param windowSize 窗函数大小 / Window size
     * @param overlap 重叠比例 (0-1) / Overlap ratio (0-1)
     * @param samplingRate 采样率 / Sampling rate
     * @return 包含频率和功率谱密度的元组 / Tuple containing frequencies and power spectral density
     */
    public static Tuple2<IVector<Double>, IVector<Double>> powerSpectralDensity(
            IVector<Double> signal, int windowSize, double overlap, double samplingRate) {
        
        int hopSize = (int) (windowSize * (1 - overlap));
        int numWindows = (signal.length() - windowSize) / hopSize + 1;
        
        // 创建汉宁窗 / Create Hanning window
        IVector<Double> window = hanningWindow(windowSize);
        
        // 计算每个窗口的功率谱 / Calculate power spectrum for each window
        IVector<Double> psdSum = Linalg.zeros(windowSize / 2 + 1);
        
        for (int i = 0; i < numWindows; i++) {
            int start = i * hopSize;
            int end = Math.min(start + windowSize, signal.length());
            
            // 提取窗口信号 / Extract window signal
            IVector<Double> windowSignal = signal.slice(start, end);
            
            // 应用窗函数 / Apply window function
            IVector<Double> windowedSignal = windowSignal.multiply(window);
            
            // 计算FFT / Calculate FFT
            Complex[] fftResult = RereFFT.fft(convertToComplex(windowedSignal));
            
            // 计算功率谱 / Calculate power spectrum
            IVector<Double> powerSpectrum = Linalg.zeros(fftResult.length / 2 + 1);
            for (int j = 0; j < powerSpectrum.length(); j++) {
                double magnitude = fftResult[j].magnitude();
                powerSpectrum.set(j, magnitude * magnitude);
            }
            
            // 累加功率谱 / Accumulate power spectrum
            psdSum = psdSum.add(powerSpectrum);
        }
        
        // 平均化 / Average
        IVector<Double> psd = psdSum.multiplyScalar(1.0 / numWindows);
        
        // 归一化 / Normalize
        double windowPower = window.multiply(window).sum();
        psd = psd.multiplyScalar(2.0 / (samplingRate * windowPower));
        
        // 生成频率轴 / Generate frequency axis
        IVector<Double> frequencies = Linalg.range(psd.length())
                .multiplyScalar(samplingRate / windowSize);
        
        return new Tuple2<>(frequencies, psd);
    }

    /**
     * 计算信号的自相关函数 / Calculate Autocorrelation Function
     * <p>
     * 使用FFT方法计算自相关函数，比直接计算更高效。
     * Uses FFT method to calculate autocorrelation function, more efficient than direct calculation.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @return 自相关函数向量 / Autocorrelation function vector
     */
    public static IVector<Double> autocorrelation(IVector<Double> signal) {
        int n = signal.length();
        int fftSize = nextPowerOfTwo(2 * n - 1);
        
        // 零填充到FFT大小 / Zero-pad to FFT size
        IVector<Double> paddedSignal = Linalg.zeros(fftSize);
        for (int i = 0; i < n; i++) {
            paddedSignal.set(i, signal.get(i));
        }
        
        // 计算FFT / Calculate FFT
        Complex[] fftResult = RereFFT.fft(convertToComplex(paddedSignal));
        
        // 计算功率谱 / Calculate power spectrum
        Complex[] powerSpectrum = new Complex[fftSize];
        for (int i = 0; i < fftSize; i++) {
            double magnitude = fftResult[i].magnitude();
            powerSpectrum[i] = new Complex(magnitude * magnitude, 0);
        }
        
        // 计算逆FFT得到自相关 / Calculate IFFT to get autocorrelation
        Complex[] autocorrComplex = RereFFT.ifft(powerSpectrum);
        
        // 转换为实数向量 / Convert to real vector
        IVector<Double> autocorr = Linalg.zeros(n);
        for (int i = 0; i < n; i++) {
            autocorr.set(i, autocorrComplex[i].real);
        }
        
        // 归一化 / Normalize
        double maxValue = autocorr.get(0);
        if (maxValue != 0) {
            autocorr = autocorr.multiplyScalar(1.0 / maxValue);
        }
        
        return autocorr;
    }

    /**
     * 计算两个信号的互相关函数 / Calculate Cross-correlation Function
     * <p>
     * 使用FFT方法计算互相关函数，自动确定最大滞后为两个信号长度中的较小值。
     * Uses FFT method to calculate cross-correlation function with automatic maximum lag determination.
     * </p>
     *
     * @param signal1 第一个信号向量 / First signal vector
     * @param signal2 第二个信号向量 / Second signal vector
     * @return 互相关函数向量 / Cross-correlation function vector
     */
    public static IVector<Double> crossCorrelation(IVector<Double> signal1, IVector<Double> signal2) {
        // 自动确定最大滞后为两个信号长度中的较小值 / Automatically determine maximum lag as smaller of two signal lengths
        int maxLag = Math.min(signal1.length(), signal2.length()) - 1;
        
        // 调用带maxLag参数的方法 / Call the method with maxLag parameter
        return crossCorrelation(signal1, signal2, maxLag);
    }
    
    /**
     * 计算两个信号的互相关函数 / Calculate Cross-correlation Function
     * <p>
     * 使用FFT方法计算互相关函数，并限制在指定的最大滞后范围内。
     * Uses FFT method to calculate cross-correlation function with limited maximum lag.
     * </p>
     *
     * @param signal1 第一个信号向量 / First signal vector
     * @param signal2 第二个信号向量 / Second signal vector
     * @param maxLag  最大滞后 / Maximum lag
     * @return 互相关函数向量，长度为2*maxLag+1 / Cross-correlation function vector with length 2*maxLag+1
     */
    public static IVector<Double> crossCorrelation(IVector<Double> signal1, IVector<Double> signal2, int maxLag) {
        int n1 = signal1.length();
        int n2 = signal2.length();
        int maxLength = Math.max(n1, n2);
        int fftSize = nextPowerOfTwo(2 * maxLength - 1);
        
        // 零填充到FFT大小 / Zero-pad to FFT size
        IVector<Double> paddedSignal1 = Linalg.zeros(fftSize);
        IVector<Double> paddedSignal2 = Linalg.zeros(fftSize);
        
        for (int i = 0; i < n1; i++) {
            paddedSignal1.set(i, signal1.get(i));
        }
        for (int i = 0; i < n2; i++) {
            paddedSignal2.set(i, signal2.get(i));
        }
        
        // 计算FFT / Calculate FFT
        Complex[] fft1 = RereFFT.fft(convertToComplex(paddedSignal1));
        Complex[] fft2 = RereFFT.fft(convertToComplex(paddedSignal2));
        
        // 计算互功率谱 / Calculate cross power spectrum
        Complex[] crossPowerSpectrum = new Complex[fftSize];
        for (int i = 0; i < fftSize; i++) {
            crossPowerSpectrum[i] = fft1[i].multiply(fft2[i].conjugate());
        }
        
        // 计算逆FFT得到互相关 / Calculate IFFT to get cross-correlation
        Complex[] crossCorrComplex = RereFFT.ifft(crossPowerSpectrum);
        
        // 提取指定滞后范围内的互相关值 / Extract cross-correlation values within specified lag range
        int resultLength = 2 * maxLag + 1;
        IVector<Double> crossCorr = Linalg.zeros(resultLength);
        
        // 计算中心位置（零滞后位置）/ Calculate center position (zero lag position)
        int center = n1 - 1;
        
        for (int i = 0; i < resultLength; i++) {
            int lag = i - maxLag;
            int index = center + lag;
            
            // 确保索引在有效范围内 / Ensure index is within valid range
            if (index >= 0 && index < crossCorrComplex.length) {
                crossCorr.set(i, crossCorrComplex[index].real);
            } else {
                crossCorr.set(i, 0.0);
            }
        }
        
        return crossCorr;
    }

    /**
     * 计算信号的频谱 / Calculate Signal Spectrum
     * <p>
     * 计算信号的幅度谱和相位谱。
     * Calculate magnitude spectrum and phase spectrum of the signal.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param samplingRate 采样率 / Sampling rate
     * @return 包含频率、幅度谱和相位谱的元组 / Tuple containing frequencies, magnitude spectrum, and phase spectrum
     */
    public static Tuple3<IVector<Double>, IVector<Double>, IVector<Double>> spectrum(
            IVector<Double> signal, double samplingRate) {
        
        // 计算FFT / Calculate FFT
        Complex[] fftResult = RereFFT.fft(convertToComplex(signal));
        
        int n = fftResult.length;
        int halfN = n / 2;
        
        // 计算幅度谱和相位谱 / Calculate magnitude and phase spectrum
        IVector<Double> magnitudeSpectrum = Linalg.zeros(halfN);
        IVector<Double> phaseSpectrum = Linalg.zeros(halfN);
        
        for (int i = 0; i < halfN; i++) {
            magnitudeSpectrum.set(i, fftResult[i].magnitude());
            phaseSpectrum.set(i, Math.atan2(fftResult[i].imag, fftResult[i].real));
        }
        
        // 生成频率轴 / Generate frequency axis
        IVector<Double> frequencies = Linalg.range(halfN)
                .multiplyScalar(samplingRate / n);
        
        return new Tuple3<>(frequencies, magnitudeSpectrum, phaseSpectrum);
    }

    /**
     * 计算信号的短时傅里叶变换 (STFT) / Calculate Short-Time Fourier Transform (STFT)
     * <p>
     * 使用滑动窗口计算信号的时频表示。
     * Calculate time-frequency representation of signal using sliding window.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param windowSize 窗函数大小 / Window size
     * @param hopSize 跳跃大小 / Hop size
     * @param samplingRate 采样率 / Sampling rate
     * @return STFT矩阵，行为频率，列为时间 / STFT matrix with rows as frequencies and columns as time
     */
    public static IMatrix<Double> shortTimeFourierTransform(
            IVector<Double> signal, int windowSize, int hopSize, double samplingRate) {
        
        int numFrames = (signal.length() - windowSize) / hopSize + 1;
        int numFreqs = windowSize / 2 + 1;
        
        // 创建汉宁窗 / Create Hanning window
        IVector<Double> window = hanningWindow(windowSize);
        
        // 初始化STFT矩阵 / Initialize STFT matrix
        IMatrix<Double> stft = Linalg.zeros(numFreqs, numFrames);
        
        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopSize;
            int end = Math.min(start + windowSize, signal.length());
            
            // 提取窗口信号 / Extract window signal
            IVector<Double> windowSignal = signal.slice(start, end);
            
            // 应用窗函数 / Apply window function
            IVector<Double> windowedSignal = windowSignal.multiply(window);
            
            // 计算FFT / Calculate FFT
            Complex[] fftResult = RereFFT.fft(convertToComplex(windowedSignal));
            
            // 存储幅度谱 / Store magnitude spectrum
            for (int freq = 0; freq < numFreqs; freq++) {
                stft.set(freq, frame, fftResult[freq].magnitude());
            }
        }
        
        return stft;
    }

    /**
     * 计算信号的信噪比 (SNR) / Calculate Signal-to-Noise Ratio (SNR)
     * <p>
     * 计算信号功率与噪声功率的比值，以分贝为单位。
     * Calculate the ratio of signal power to noise power in decibels.
     * </p>
     *
     * @param signal 原始信号向量 / Original signal vector
     * @param noise 噪声信号向量 / Noise signal vector
     * @return 信噪比 (dB) / Signal-to-noise ratio (dB)
     */
    public static double signalToNoiseRatio(IVector<Double> signal, IVector<Double> noise) {
        double signalPower = signal.multiply(signal).sum();
        double noisePower = noise.multiply(noise).sum();
        
        if (noisePower == 0) {
            return Double.POSITIVE_INFINITY;
        }
        
        return 10 * Math.log10(signalPower / noisePower);
    }

    /**
     * 计算信号的峰值信噪比 (PSNR) / Calculate Peak Signal-to-Noise Ratio (PSNR)
     * <p>
     * 计算信号的最大可能功率与噪声功率的比值。
     * Calculate the ratio of maximum possible signal power to noise power.
     * </p>
     *
     * @param original 原始信号向量 / Original signal vector
     * @param reconstructed 重建信号向量 / Reconstructed signal vector
     * @return 峰值信噪比 (dB) / Peak signal-to-noise ratio (dB)
     */
    public static double peakSignalToNoiseRatio(IVector<Double> original, IVector<Double> reconstructed) {
        double maxSignal = original.max();
        double mse = original.sub(reconstructed)
                .multiply(original.sub(reconstructed))
                .mean();
        
        if (mse == 0) {
            return Double.POSITIVE_INFINITY;
        }
        
        return 10 * Math.log10((maxSignal * maxSignal) / mse);
    }

    // ========== 辅助方法 / Helper Methods ==========

    /**
     * 创建汉宁窗 / Create Hanning window
     */
    private static IVector<Double> hanningWindow(int size) {
        IVector<Double> window = Linalg.zeros(size);
        for (int i = 0; i < size; i++) {
            double value = 0.5 * (1 - Math.cos(2 * Math.PI * i / (size - 1)));
            window.set(i, value);
        }
        return window;
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
