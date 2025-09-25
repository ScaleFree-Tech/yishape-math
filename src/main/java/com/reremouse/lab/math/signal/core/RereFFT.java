package com.reremouse.lab.math.signal.core;

/**
 *
 * @author lteb2
 */
public class RereFFT {
    
    // 快速傅里叶变换
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
    
    // 快速傅里叶逆变换
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
    
    // 测试方法
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
        
        System.out.println("原始信号:");
        for (int i = 0; i < N; i++) {
            System.out.printf("x[%d] = %.3f\n", i, signal[i]);
        }
        
        // 执行FFT
        Complex[] spectrum = fft(x);
        
        System.out.println("\n频域表示 (FFT结果):");
        for (int i = 0; i < N; i++) {
            System.out.printf("X[%d] = %s, 幅度: %.3f\n", 
                i, spectrum[i], spectrum[i].magnitude());
        }
        
        // 执行逆FFT
        Complex[] reconstructed = ifft(spectrum);
        
        System.out.println("\n重建的信号 (IFFT结果):");
        for (int i = 0; i < N; i++) {
            System.out.printf("x[%d] = %.3f (原始: %.3f)\n", 
                i, reconstructed[i].real, signal[i]);
        }
    }
}