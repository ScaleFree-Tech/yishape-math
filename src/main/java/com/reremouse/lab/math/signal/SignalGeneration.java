package com.reremouse.lab.math.signal;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import java.util.Random;

/**
 * 信号生成类 / Signal Generation Class
 * <p>
 * 提供各种信号生成功能，包括正弦波、方波、三角波、噪声信号等。
 * 使用IVector接口进行向量操作，确保与现有代码库的兼容性。
 * </p>
 * <p>
 * Provides various signal generation functions including sine waves, square waves, triangular waves,
 * noise signals, etc. Uses IVector interface for vector operations to ensure compatibility with existing codebase.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class SignalGeneration {

    private static final Random random = new Random();

    /**
     * 生成正弦波信号 / Generate Sine Wave Signal
     * <p>
     * 生成指定频率、幅度和相位的正弦波信号。
     * Generate sine wave signal with specified frequency, amplitude and phase.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param frequency 频率 (Hz) / Frequency (Hz)
     * @param samplingRate 采样率 (Hz) / Sampling rate (Hz)
     * @param amplitude 幅度 / Amplitude
     * @param phase 相位 (弧度) / Phase (radians)
     * @return 正弦波信号向量 / Sine wave signal vector
     */
    public static IVector<Double> sineWave(int length, double frequency, double samplingRate, 
                                         double amplitude, double phase) {
        if (frequency <= 0 || samplingRate <= 0 || amplitude <= 0) {
            throw new IllegalArgumentException("参数必须为正数");
        }
        
        IVector<Double> time = Linalg.range(length).multiplyScalar(1.0 / samplingRate);
        IVector<Double> signal = time.multiplyScalar(2 * Math.PI * frequency)
                .addScalar(phase)
                .sin()
                .multiplyScalar(amplitude);
        
        return signal;
    }

    /**
     * 生成余弦波信号 / Generate Cosine Wave Signal
     * <p>
     * 生成指定频率、幅度和相位的余弦波信号。
     * Generate cosine wave signal with specified frequency, amplitude and phase.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param frequency 频率 (Hz) / Frequency (Hz)
     * @param samplingRate 采样率 (Hz) / Sampling rate (Hz)
     * @param amplitude 幅度 / Amplitude
     * @param phase 相位 (弧度) / Phase (radians)
     * @return 余弦波信号向量 / Cosine wave signal vector
     */
    public static IVector<Double> cosineWave(int length, double frequency, double samplingRate, 
                                           double amplitude, double phase) {
        if (frequency <= 0 || samplingRate <= 0 || amplitude <= 0) {
            throw new IllegalArgumentException("参数必须为正数");
        }
        
        IVector<Double> time = Linalg.range(length).multiplyScalar(1.0 / samplingRate);
        IVector<Double> signal = time.multiplyScalar(2 * Math.PI * frequency)
                .addScalar(phase)
                .cos()
                .multiplyScalar(amplitude);
        
        return signal;
    }

    /**
     * 生成方波信号 / Generate Square Wave Signal
     * <p>
     * 生成指定频率、幅度和占空比的方波信号。
     * Generate square wave signal with specified frequency, amplitude and duty cycle.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param frequency 频率 (Hz) / Frequency (Hz)
     * @param samplingRate 采样率 (Hz) / Sampling rate (Hz)
     * @param amplitude 幅度 / Amplitude
     * @param dutyCycle 占空比 (0-1) / Duty cycle (0-1)
     * @return 方波信号向量 / Square wave signal vector
     */
    public static IVector<Double> squareWave(int length, double frequency, double samplingRate, 
                                           double amplitude, double dutyCycle) {
        if (frequency <= 0 || samplingRate <= 0 || amplitude <= 0 || dutyCycle < 0 || dutyCycle > 1) {
            throw new IllegalArgumentException("参数无效");
        }
        
        IVector<Double> time = Linalg.range(length).multiplyScalar(1.0 / samplingRate);
        IVector<Double> signal = Linalg.zeros(length);
        
        double period = 1.0 / frequency;
        
        for (int i = 0; i < length; i++) {
            double t = time.get(i);
            double phase = (t % period) / period;
            signal.set(i, phase < dutyCycle ? amplitude : -amplitude);
        }
        
        return signal;
    }

    /**
     * 生成三角波信号 / Generate Triangular Wave Signal
     * <p>
     * 生成指定频率、幅度和对称性的三角波信号。
     * Generate triangular wave signal with specified frequency, amplitude and symmetry.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param frequency 频率 (Hz) / Frequency (Hz)
     * @param samplingRate 采样率 (Hz) / Sampling rate (Hz)
     * @param amplitude 幅度 / Amplitude
     * @param symmetry 对称性 (0-1, 0.5为对称) / Symmetry (0-1, 0.5 for symmetric)
     * @return 三角波信号向量 / Triangular wave signal vector
     */
    public static IVector<Double> triangularWave(int length, double frequency, double samplingRate, 
                                               double amplitude, double symmetry) {
        if (frequency <= 0 || samplingRate <= 0 || amplitude <= 0 || symmetry < 0 || symmetry > 1) {
            throw new IllegalArgumentException("参数无效");
        }
        
        IVector<Double> time = Linalg.range(length).multiplyScalar(1.0 / samplingRate);
        IVector<Double> signal = Linalg.zeros(length);
        
        double period = 1.0 / frequency;
        
        for (int i = 0; i < length; i++) {
            double t = time.get(i);
            double phase = (t % period) / period;
            
            double value;
            if (phase < symmetry) {
                value = 2 * amplitude * phase / symmetry - amplitude;
            } else {
                value = 2 * amplitude * (1 - phase) / (1 - symmetry) - amplitude;
            }
            signal.set(i, value);
        }
        
        return signal;
    }

    /**
     * 生成锯齿波信号 / Generate Sawtooth Wave Signal
     * <p>
     * 生成指定频率和幅度的锯齿波信号。
     * Generate sawtooth wave signal with specified frequency and amplitude.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param frequency 频率 (Hz) / Frequency (Hz)
     * @param samplingRate 采样率 (Hz) / Sampling rate (Hz)
     * @param amplitude 幅度 / Amplitude
     * @return 锯齿波信号向量 / Sawtooth wave signal vector
     */
    public static IVector<Double> sawtoothWave(int length, double frequency, double samplingRate, double amplitude) {
        if (frequency <= 0 || samplingRate <= 0 || amplitude <= 0) {
            throw new IllegalArgumentException("参数必须为正数");
        }
        
        IVector<Double> time = Linalg.range(length).multiplyScalar(1.0 / samplingRate);
        IVector<Double> signal = Linalg.zeros(length);
        
        double period = 1.0 / frequency;
        
        for (int i = 0; i < length; i++) {
            double t = time.get(i);
            double phase = (t % period) / period;
            signal.set(i, 2 * amplitude * phase - amplitude);
        }
        
        return signal;
    }

    /**
     * 生成白噪声信号 / Generate White Noise Signal
     * <p>
     * 生成指定长度和功率的高斯白噪声信号。
     * Generate Gaussian white noise signal with specified length and power.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param power 噪声功率 / Noise power
     * @return 白噪声信号向量 / White noise signal vector
     */
    public static IVector<Double> whiteNoise(int length, double power) {
        if (length <= 0 || power < 0) {
            throw new IllegalArgumentException("参数必须为正数");
        }
        
        IVector<Double> noise = Linalg.zeros(length);
        double stdDev = Math.sqrt(power);
        
        for (int i = 0; i < length; i++) {
            double value = random.nextGaussian() * stdDev;
            noise.set(i, value);
        }
        
        return noise;
    }

    /**
     * 生成粉红噪声信号 / Generate Pink Noise Signal
     * <p>
     * 生成1/f噪声，功率谱密度与频率成反比。
     * Generate 1/f noise with power spectral density inversely proportional to frequency.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param power 噪声功率 / Noise power
     * @return 粉红噪声信号向量 / Pink noise signal vector
     */
    public static IVector<Double> pinkNoise(int length, double power) {
        if (length <= 0 || power < 0) {
            throw new IllegalArgumentException("参数必须为正数");
        }
        
        // 生成白噪声 / Generate white noise
        IVector<Double> whiteNoise = whiteNoise(length, 1.0);
        
        // 应用1/f滤波器 / Apply 1/f filter
        return applyPinkNoiseFilter(whiteNoise, power);
    }

    /**
     * 生成复合信号 / Generate Composite Signal
     * <p>
     * 生成多个正弦波的叠加信号。
     * Generate composite signal by superimposing multiple sine waves.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param frequencies 频率数组 (Hz) / Frequency array (Hz)
     * @param amplitudes 幅度数组 / Amplitude array
     * @param phases 相位数组 (弧度) / Phase array (radians)
     * @param samplingRate 采样率 (Hz) / Sampling rate (Hz)
     * @return 复合信号向量 / Composite signal vector
     */
    public static IVector<Double> compositeSignal(int length, double[] frequencies, double[] amplitudes, 
                                                double[] phases, double samplingRate) {
        if (frequencies.length != amplitudes.length || frequencies.length != phases.length) {
            throw new IllegalArgumentException("频率、幅度和相位数组长度必须相同");
        }
        
        IVector<Double> composite = Linalg.zeros(length);
        
        for (int i = 0; i < frequencies.length; i++) {
            IVector<Double> component = sineWave(length, frequencies[i], samplingRate, 
                                               amplitudes[i], phases[i]);
            composite = composite.add(component);
        }
        
        return composite;
    }

    /**
     * 生成调频信号 (FM) / Generate Frequency Modulated Signal (FM)
     * <p>
     * 生成频率调制的正弦波信号。
     * Generate frequency modulated sine wave signal.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param carrierFreq 载波频率 (Hz) / Carrier frequency (Hz)
     * @param modFreq 调制频率 (Hz) / Modulation frequency (Hz)
     * @param modIndex 调制指数 / Modulation index
     * @param samplingRate 采样率 (Hz) / Sampling rate (Hz)
     * @param amplitude 幅度 / Amplitude
     * @return 调频信号向量 / FM signal vector
     */
    public static IVector<Double> frequencyModulated(int length, double carrierFreq, double modFreq, 
                                                   double modIndex, double samplingRate, double amplitude) {
        if (carrierFreq <= 0 || modFreq <= 0 || samplingRate <= 0 || amplitude <= 0) {
            throw new IllegalArgumentException("参数必须为正数");
        }
        
        IVector<Double> time = Linalg.range(length).multiplyScalar(1.0 / samplingRate);
        IVector<Double> signal = Linalg.zeros(length);
        
        for (int i = 0; i < length; i++) {
            double t = time.get(i);
            double instantaneousFreq = carrierFreq + modIndex * Math.sin(2 * Math.PI * modFreq * t);
            double phase = 2 * Math.PI * instantaneousFreq * t;
            signal.set(i, amplitude * Math.sin(phase));
        }
        
        return signal;
    }

    /**
     * 生成调幅信号 (AM) / Generate Amplitude Modulated Signal (AM)
     * <p>
     * 生成幅度调制的正弦波信号。
     * Generate amplitude modulated sine wave signal.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param carrierFreq 载波频率 (Hz) / Carrier frequency (Hz)
     * @param modFreq 调制频率 (Hz) / Modulation frequency (Hz)
     * @param modDepth 调制深度 (0-1) / Modulation depth (0-1)
     * @param samplingRate 采样率 (Hz) / Sampling rate (Hz)
     * @param amplitude 幅度 / Amplitude
     * @return 调幅信号向量 / AM signal vector
     */
    public static IVector<Double> amplitudeModulated(int length, double carrierFreq, double modFreq, 
                                                   double modDepth, double samplingRate, double amplitude) {
        if (carrierFreq <= 0 || modFreq <= 0 || samplingRate <= 0 || amplitude <= 0 || 
            modDepth < 0 || modDepth > 1) {
            throw new IllegalArgumentException("参数无效");
        }
        
        IVector<Double> time = Linalg.range(length).multiplyScalar(1.0 / samplingRate);
        IVector<Double> signal = Linalg.zeros(length);
        
        for (int i = 0; i < length; i++) {
            double t = time.get(i);
            double carrier = Math.sin(2 * Math.PI * carrierFreq * t);
            double modulation = 1 + modDepth * Math.sin(2 * Math.PI * modFreq * t);
            signal.set(i, amplitude * carrier * modulation);
        }
        
        return signal;
    }

    /**
     * 生成脉冲信号 / Generate Impulse Signal
     * <p>
     * 生成指定位置和幅度的脉冲信号。
     * Generate impulse signal with specified position and amplitude.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param position 脉冲位置 / Impulse position
     * @param amplitude 脉冲幅度 / Impulse amplitude
     * @return 脉冲信号向量 / Impulse signal vector
     */
    public static IVector<Double> impulse(int length, int position, double amplitude) {
        if (position < 0 || position >= length) {
            throw new IllegalArgumentException("脉冲位置超出范围");
        }
        
        IVector<Double> signal = Linalg.zeros(length);
        signal.set(position, amplitude);
        return signal;
    }

    /**
     * 生成阶跃信号 / Generate Step Signal
     * <p>
     * 生成指定位置和幅度的阶跃信号。
     * Generate step signal with specified position and amplitude.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param position 阶跃位置 / Step position
     * @param amplitude 阶跃幅度 / Step amplitude
     * @return 阶跃信号向量 / Step signal vector
     */
    public static IVector<Double> step(int length, int position, double amplitude) {
        if (position < 0 || position >= length) {
            throw new IllegalArgumentException("阶跃位置超出范围");
        }
        
        IVector<Double> signal = Linalg.zeros(length);
        for (int i = position; i < length; i++) {
            signal.set(i, amplitude);
        }
        return signal;
    }

    // ========== 辅助方法 / Helper Methods ==========

    /**
     * 应用粉红噪声滤波器 / Apply pink noise filter
     */
    private static IVector<Double> applyPinkNoiseFilter(IVector<Double> whiteNoise, double power) {
        int length = whiteNoise.length();
        IVector<Double> pinkNoise = Linalg.zeros(length);
        
        // 简化的粉红噪声生成算法 / Simplified pink noise generation algorithm
        double[] b = {0.049922035, -0.095993537, 0.050612699, -0.004408786};
        double[] a = {1, -2.494956002, 2.017265875, -0.522189400};
        
        // 应用IIR滤波器 / Apply IIR filter
        for (int i = 0; i < length; i++) {
            double y = 0;
            
            // 前向项 / Forward terms
            for (int j = 0; j < b.length && i - j >= 0; j++) {
                y += b[j] * whiteNoise.get(i - j);
            }
            
            // 反馈项 / Feedback terms
            for (int j = 1; j < a.length && i - j >= 0; j++) {
                y -= a[j] * pinkNoise.get(i - j);
            }
            
            pinkNoise.set(i, y / a[0]);
        }
        
        // 归一化功率 / Normalize power
        double currentPower = pinkNoise.multiply(pinkNoise).mean();
        if (currentPower > 0) {
            double scaleFactor = Math.sqrt(power / currentPower);
            pinkNoise = pinkNoise.multiplyScalar(scaleFactor);
        }
        
        return pinkNoise;
    }
}
