package com.yishape.lab.math.signal.core;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Objects;

/**
 * 快速傅里叶变换工具类 / Fast Fourier Transform Utility Class
 * <p>
 * 提供离散傅里叶变换（DFT）的快速算法实现，支持复数信号和实数信号的
 * 正向和逆向变换，以及幅度谱、相位谱、功率谱的计算。
 * </p>
 * <p>
 * Provides efficient implementations of Fast Fourier Transform (FFT) algorithms,
 * supporting forward and inverse transforms for both complex and real signals,
 * as well as magnitude, phase, and power spectrum calculations.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RereFFT {

    private static final Logger log = LoggerFactory.getLogger(RereFFT.class);

    /**
     * 快速傅里叶变换（递归实现）/ Fast Fourier Transform (Recursive Implementation)
     * <p>
     * 对输入的复数数组进行离散傅里叶变换，数组长度必须是2的幂。
     * Performs discrete Fourier transform on input complex array with length that must be a power of 2.
     * </p>
     *
     * @param x 输入的复数数组 / Input complex array
     * @return 变换后的复数数组 / Transformed complex array
     */
    public static Complex[] fft(Complex[] x) {
        int n = x.length;
        
        // 基本情况
        if (n == 1) {
            return new Complex[] { x[0] };
        }
        
        // 检查n是否为2的幂
        if ((n & (n - 1)) != 0) {
            throw new IllegalArgumentException("数组长度必须是2的幂");
        }
        
        // 分别计算偶数和奇数索引的FFT
        Complex[] even = new Complex[n/2];
        Complex[] odd = new Complex[n/2];
        
        for (int k = 0; k < n/2; k++) {
            even[k] = x[2*k];
            odd[k] = x[2*k + 1];
        }
        
        Complex[] evenFFT = fft(even);
        Complex[] oddFFT = fft(odd);
        
        // 合并结果
        Complex[] y = new Complex[n];
        for (int k = 0; k < n/2; k++) {
            double angle = -2 * Math.PI * k / n;
            Complex wk = new Complex(Math.cos(angle), Math.sin(angle));
            Complex term = wk.multiply(oddFFT[k]);
            
            y[k] = evenFFT[k].add(term);
            y[k + n/2] = evenFFT[k].subtract(term);
        }
        
        return y;
    }
    
    // 快速傅里叶逆变换 / Inverse Fast Fourier Transform
    /**
     * 快速傅里叶逆变换 / Inverse Fast Fourier Transform
     * <p>
     * 对输入的复数频谱进行逆变换，恢复时域信号。通过取共轭、FFT、再取共轭并缩放实现。
     * Performs inverse FFT by conjugating, computing FFT, then conjugating and scaling.
     * </p>
     *
     * @param x 输入的复数频谱 / Input complex frequency spectrum
     * @return 逆变换后的时域复数信号 / Inverse transformed time-domain complex signal
     */
    public static Complex[] ifft(Complex[] x) {
        int n = x.length;
        Complex[] conjugated = new Complex[n];
        
        // 取共轭
        for (int i = 0; i < n; i++) {
            conjugated[i] = x[i].conjugate();
        }
        
        // 计算FFT
        Complex[] temp = fft(conjugated);
        
        // 再次取共轭并缩放
        Complex[] result = new Complex[n];
        for (int i = 0; i < n; i++) {
            result[i] = temp[i].conjugate().scale(1.0 / n);
        }
        
        return result;
    }

    /**
     * 实数输入的离散傅里叶变换（半谱）：内部补零至 2 的幂后做复 FFT，返回前 {@code N/2+1} 个频点。
     *
     * @param x 实数序列（任意长度）
     * @return 长度 {@code nextPowerOfTwo(x.length)/2 + 1} 的复频谱
     */
    public static Complex[] rfft(double[] x) {
        if (x.length == 0) {
            throw new IllegalArgumentException("空序列 / empty sequence");
        }
        int n = nextPowerOfTwo(x.length);
        Complex[] cx = new Complex[n];
        for (int i = 0; i < x.length; i++) {
            cx[i] = new Complex(x[i], 0);
        }
        for (int i = x.length; i < n; i++) {
            cx[i] = new Complex(0, 0);
        }
        Complex[] full = fft(cx);
        return Arrays.copyOf(full, n / 2 + 1);
    }

    /**
     * {@link #rfft(double[])} 的逆变换：由半谱还原指定长度的实数序列。
     *
     * @param y 长度须为 {@code nextPowerOfTwo(n)/2 + 1}
     * @param n 输出实数长度
     */
    public static double[] irfft(Complex[] y, int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("n > 0");
        }
        Objects.requireNonNull(y, "y");
        int nfft = nextPowerOfTwo(n);
        if (y.length != nfft / 2 + 1) {
            throw new IllegalArgumentException(
                    "频谱长度须为 nextPowerOfTwo(n)/2+1 / spectrum length must be nextPowerOfTwo(n)/2+1");
        }
        Complex[] full = new Complex[nfft];
        full[0] = y[0];
        for (int k = 1; k < nfft / 2; k++) {
            full[k] = y[k];
            full[nfft - k] = y[k].conjugate();
        }
        full[nfft / 2] = y[nfft / 2];
        Complex[] time = ifft(full);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = time[i].real;
        }
        return out;
    }

    /**
     * 不小于 n 的最小 2 的幂（用于 FFT 补零）。
     */
    public static int nextPowerOfTwoLength(int n) {
        return nextPowerOfTwo(n);
    }

    /**
     * 计算信号的幅度谱 / Calculate Magnitude Spectrum
     * <p>
     * 计算FFT结果的幅度谱。
     * Calculate magnitude spectrum from FFT result.
     * </p>
     *
     * @param fftResult FFT结果 / FFT result
     * @return 幅度谱向量 / Magnitude spectrum vector
     */
    public static double[] magnitudeSpectrum(Complex[] fftResult) {
        double[] magnitude = new double[fftResult.length];
        for (int i = 0; i < fftResult.length; i++) {
            magnitude[i] = fftResult[i].magnitude();
        }
        return magnitude;
    }

    /**
     * 计算信号的相位谱 / Calculate Phase Spectrum
     * <p>
     * 计算FFT结果的相位谱。
     * Calculate phase spectrum from FFT result.
     * </p>
     *
     * @param fftResult FFT结果 / FFT result
     * @return 相位谱向量 / Phase spectrum vector
     */
    public static double[] phaseSpectrum(Complex[] fftResult) {
        double[] phase = new double[fftResult.length];
        for (int i = 0; i < fftResult.length; i++) {
            phase[i] = Math.atan2(fftResult[i].imag, fftResult[i].real);
        }
        return phase;
    }

    /**
     * 计算信号的功率谱 / Calculate Power Spectrum
     * <p>
     * 计算FFT结果的功率谱。
     * Calculate power spectrum from FFT result.
     * </p>
     *
     * @param fftResult FFT结果 / FFT result
     * @return 功率谱向量 / Power spectrum vector
     */
    public static double[] powerSpectrum(Complex[] fftResult) {
        double[] power = new double[fftResult.length];
        for (int i = 0; i < fftResult.length; i++) {
            double magnitude = fftResult[i].magnitude();
            power[i] = magnitude * magnitude;
        }
        return power;
    }

    /**
     * 零填充到2的幂 / Zero-pad to Power of 2
     * <p>
     * 将信号零填充到最近的2的幂长度，以便进行FFT。
     * Zero-pad signal to nearest power of 2 length for FFT.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @return 零填充后的信号 / Zero-padded signal
     */
    public static Complex[] zeroPadToPowerOfTwo(Complex[] signal) {
        int n = signal.length;
        int paddedLength = nextPowerOfTwo(n);
        
        if (paddedLength == n) {
            return signal.clone();
        }
        
        Complex[] padded = new Complex[paddedLength];
        for (int i = 0; i < n; i++) {
            padded[i] = signal[i];
        }
        for (int i = n; i < paddedLength; i++) {
            padded[i] = new Complex(0, 0);
        }
        
        return padded;
    }

    /**
     * 计算大于等于n的最小2的幂 / Calculate smallest power of 2 >= n
     *
     * @param n 输入的正整数 / Input positive integer
     * @return 大于等于n的最小2的幂 / Smallest power of 2 >= n
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
    
    // 测试方法 / Test method
    /**
     * 主测试方法 / Main test method
     * <p>
     * 测试FFT和IFFT的正确性，创建测试信号并进行变换。
     * Tests FFT and IFFT correctness by creating test signals and performing transforms.
     * </p>
     *
     * @param args 命令行参数 / Command line arguments
     */
    public static void main(String[] args) {
        // 创建测试信号：2个正弦波的叠加
        int N = 8; // 必须是2的幂
        double[] signal = new double[N];
        
        // 生成测试信号：sin(2πt) + 0.5*sin(4πt)
        for (int i = 0; i < N; i++) {
            double t = (double) i / N;
            signal[i] = Math.sin(2 * Math.PI * t) + 0.5 * Math.sin(4 * Math.PI * t);
        }
        
        // 转换为复数数组（虚部为0）
        Complex[] x = new Complex[N];
        for (int i = 0; i < N; i++) {
            x[i] = new Complex(signal[i], 0);
        }
        
        log.debug("原始信号:");
        for (int i = 0; i < N; i++) {
            log.debug(String.format("x[%d] = %.3f\n", i, signal[i]));
        }
        
        // 执行FFT
        Complex[] spectrum = fft(x);
        
        log.debug("\n频域表示 (FFT结果):");
        for (int i = 0; i < N; i++) {
            log.debug(String.format("X[%d] = %s, 幅度: %.3f\n",
                i, spectrum[i], spectrum[i].magnitude()));
        }
        
        // 执行逆FFT
        Complex[] reconstructed = ifft(spectrum);
        
        log.debug("\n重建的信号 (IFFT结果):");
        for (int i = 0; i < N; i++) {
            log.debug(String.format("x[%d] = %.3f (原始: %.3f)\n",
                i, reconstructed[i].real, signal[i]));
        }
    }
}