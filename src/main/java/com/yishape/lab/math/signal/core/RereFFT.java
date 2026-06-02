package com.yishape.lab.math.signal.core;

import com.yishape.lab.util.YishapeLogger;

import java.lang.ref.SoftReference;
import java.util.Arrays;
import java.util.Objects;

/**
 * 快速傅里叶变换工具类 / Fast Fourier Transform Utility Class
 * <p>
 * 提供迭代 Cooley-Tukey FFT，支持任意长度输入（自动补零至 2 的幂），
 * 通过 ThreadLocal 缓存工作缓冲区和预计算旋转因子，消除递归分配开销。
 * </p>
 * <p>
 * Provides iterative Cooley-Tukey FFT with automatic zero-padding for arbitrary lengths,
 * ThreadLocal work-buffer caching, and precomputed twiddle factors.
 * </p>
 *
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class RereFFT {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereFFT.class);

    /** 预计算旋转因子的最大 FFT 长度 (2^maxLogN) */
    private static final int MAX_LOG_N = 20; // up to 2^20 = 1,048,576
    private static final int DEFAULT_MIN_POW2 = 6; // min precompute size = 2^6 = 64

    /** 线程本地 Complex 缓冲区和旋转因子缓存 */
    private static final ThreadLocal<SoftReference<Complex[]>> complexBufRef = new ThreadLocal<>();
    private static final ThreadLocal<SoftReference<Complex[][]>> twiddleCacheRef = new ThreadLocal<>();

    // ========== 公共 API / Public API ==========

    /**
     * 快速傅里叶变换 / Fast Fourier Transform
     * <p>
     * 迭代 Cooley-Tukey 实现。若输入长度非 2 的幂，自动补零至下一 2 的幂。
     * Iterative Cooley-Tukey implementation. Auto-pads to next power of two.
     * </p>
     *
     * @param x 输入复数数组（任意长度）/ Input complex array (any length)
     * @return 变换后的复数数组 / Transformed complex array
     */
    public static Complex[] fft(Complex[] x) {
        int n = x.length;
        if (n == 0) return new Complex[0];
        int fftSize = nextPowerOfTwo(n);

        Complex[] work = getWorkBuffer(fftSize);
        // 复制并补零
        for (int i = 0; i < n; i++) {
            work[i] = x[i];
        }
        for (int i = n; i < fftSize; i++) {
            work[i] = Complex.ZERO;
        }

        iterativeFFT(work, fftSize, false);

        // 截取或返回完整结果
        if (n == fftSize) {
            Complex[] result = new Complex[n];
            System.arraycopy(work, 0, result, 0, n);
            return result;
        }
        return Arrays.copyOf(work, fftSize);
    }

    /**
     * 快速傅里叶逆变换 / Inverse Fast Fourier Transform
     */
    public static Complex[] ifft(Complex[] x) {
        int n = x.length;
        if (n == 0) return new Complex[0];
        int fftSize = nextPowerOfTwo(n);

        Complex[] work = getWorkBuffer(fftSize);
        // 共轭输入
        for (int i = 0; i < n; i++) {
            work[i] = x[i].conjugate();
        }
        for (int i = n; i < fftSize; i++) {
            work[i] = Complex.ZERO;
        }

        iterativeFFT(work, fftSize, false);

        // 共轭并缩放
        Complex[] result = new Complex[n];
        double scale = 1.0 / fftSize;
        for (int i = 0; i < n; i++) {
            result[i] = work[i].conjugate().scale(scale);
        }
        return result;
    }

    /**
     * 实数输入的离散傅里叶变换（半谱）：内部补零至 2 的幂后做复 FFT，返回前 N/2+1 个频点。
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
            cx[i] = Complex.ZERO;
        }
        Complex[] full = fft(cx);
        return Arrays.copyOf(full, n / 2 + 1);
    }

    /**
     * {@link #rfft(double[])} 的逆变换：由半谱还原指定长度的实数序列。
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
     */
    public static double[] powerSpectrum(Complex[] fftResult) {
        double[] power = new double[fftResult.length];
        for (int i = 0; i < fftResult.length; i++) {
            double m = fftResult[i].magnitude();
            power[i] = m * m;
        }
        return power;
    }

    /**
     * 零填充到2的幂 / Zero-pad to Power of 2
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
            padded[i] = Complex.ZERO;
        }
        return padded;
    }

    // ========== 核心迭代 FFT / Core Iterative FFT ==========

    /**
     * 迭代原位 Cooley-Tukey FFT。
     * Iterative in-place Cooley-Tukey FFT.
     *
     * @param a    复数数组（长度必须为 2 的幂）
     * @param n    FFT 长度（2 的幂）
     * @param inverse true 为正变换，保留符号处理给调用方
     */
    private static void iterativeFFT(Complex[] a, int n, boolean inverse) {
        // 位翻转置换
        int bits = Integer.numberOfTrailingZeros(n);
        for (int i = 0; i < n; i++) {
            int j = Integer.reverse(i) >>> (32 - bits);
            if (i < j) {
                Complex tmp = a[i];
                a[i] = a[j];
                a[j] = tmp;
            }
        }

        // 蝴蝶运算 — 使用预计算旋转因子
        Complex[][] twiddles = getTwiddleCache(n);
        for (int len = 2, level = 0; len <= n; len <<= 1, level++) {
            int halfLen = len >> 1;
            Complex[] w = twiddles[level];
            for (int i = 0; i < n; i += len) {
                for (int j = 0; j < halfLen; j++) {
                    Complex u = a[i + j];
                    // t = w * a[i + j + halfLen]
                    Complex v = a[i + j + halfLen];
                    double tRe = w[j].real * v.real - w[j].imag * v.imag;
                    double tIm = w[j].real * v.imag + w[j].imag * v.real;
                    a[i + j].real = u.real + tRe;
                    a[i + j].imag = u.imag + tIm;
                    a[i + j + halfLen].real = u.real - tRe;
                    a[i + j + halfLen].imag = u.imag - tIm;
                }
            }
        }
    }

    // ========== ThreadLocal 缓存 / ThreadLocal Caching ==========

    /**
     * 获取或分配线程本地工作缓冲区。
     */
    private static Complex[] getWorkBuffer(int minSize) {
        SoftReference<Complex[]> ref = complexBufRef.get();
        Complex[] buf = (ref != null) ? ref.get() : null;
        if (buf == null || buf.length < minSize) {
            buf = new Complex[minSize];
            complexBufRef.set(new SoftReference<>(buf));
        }
        return buf;
    }

    /**
     * 获取或预计算各蝶形层的旋转因子。
     * 第 level 层 (长度 len = 2^(level+1)): exp(-2πi·k/len) for k=0..halfLen-1
     */
    private static Complex[][] getTwiddleCache(int maxN) {
        SoftReference<Complex[][]> ref = twiddleCacheRef.get();
        Complex[][] cache = (ref != null) ? ref.get() : null;
        int maxLogN = 32 - Integer.numberOfLeadingZeros(maxN);
        if (cache == null || cache.length < maxLogN) {
            cache = new Complex[maxLogN][];
        }
        // 按需填充尚未计算的层
        for (int logLen = 1; logLen <= maxLogN; logLen++) {
            int len = 1 << logLen;
            int halfLen = len >> 1;
            if (cache[logLen - 1] == null || cache[logLen - 1].length < halfLen) {
                cache[logLen - 1] = new Complex[halfLen];
                double angle = -2.0 * Math.PI / len;
                for (int k = 0; k < halfLen; k++) {
                    cache[logLen - 1][k] = new Complex(Math.cos(angle * k), Math.sin(angle * k));
                }
            }
        }
        twiddleCacheRef.set(new SoftReference<>(cache));
        return cache;
    }

    // ========== 工具方法 / Utility Methods ==========

    private static int nextPowerOfTwo(int n) {
        if (n <= 1) return 1;
        if ((n & (n - 1)) == 0) return n;
        return Integer.highestOneBit(n) << 1;
    }

    // ========== 测试 / Test ==========

    public static void main(String[] args) {
        int N = 8;
        double[] signal = new double[N];
        for (int i = 0; i < N; i++) {
            double t = (double) i / N;
            signal[i] = Math.sin(2 * Math.PI * t) + 0.5 * Math.sin(4 * Math.PI * t);
        }
        Complex[] x = new Complex[N];
        for (int i = 0; i < N; i++) {
            x[i] = new Complex(signal[i], 0);
        }
        log.debug("原始信号:");
        for (int i = 0; i < N; i++) {
            log.debug(String.format("x[%d] = %.3f", i, signal[i]));
        }
        Complex[] spectrum = fft(x);
        log.debug("\n频域表示 (FFT结果):");
        for (int i = 0; i < spectrum.length; i++) {
            log.debug(String.format("X[%d] = %s, 幅度: %.3f",
                    i, spectrum[i], spectrum[i].magnitude()));
        }
        Complex[] reconstructed = ifft(spectrum);
        log.debug("\n重建的信号 (IFFT结果):");
        for (int i = 0; i < N; i++) {
            log.debug(String.format("x[%d] = %.3f (原始: %.3f)",
                    i, reconstructed[i].real, signal[i]));
        }
    }
}
