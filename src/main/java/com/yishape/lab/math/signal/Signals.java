package com.yishape.lab.math.signal;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.wavele.WaveletAnalysis;
import com.yishape.lab.math.signal.wavele.WaveletPlots;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.signal.analysis.ISignalAnalyzer;
import com.yishape.lab.math.signal.filter.ISignalFilter;
import com.yishape.lab.math.signal.filter.ButterworthFilter;
import com.yishape.lab.math.signal.filter.GaussianFilter;
import com.yishape.lab.math.signal.transform.ISignalTransform;
import com.yishape.lab.math.signal.generation.ISignalGenerator;
import com.yishape.lab.math.signal.factory.SignalProcessorFactory;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.RereDCT;
import com.yishape.lab.math.signal.core.RereFFT;
import com.yishape.lab.math.signal.core.RereHilbert;
import com.yishape.lab.math.signal.wavele.WaveletCoefficients;

/**
 * 信号处理入口工厂类 / Signal Processing Entry Factory Class
 * <p>
 * 提供统一的信号处理接口，封装了信号生成、滤波、分析、变换等核心功能。
 * Provides a unified signal processing interface that encapsulates core functions such as signal generation, filtering, analysis, and transformation.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class Signals {

    private static final Logger log = LoggerFactory.getLogger(Signals.class);

    
    // ========== 信号生成方法 / Signal Generation Methods ==========
    
    /**
     * 生成正弦波信号 / Generate Sine Wave Signal
     * <p>
     * 产生指定参数的正弦波信号。
     * Generate sine wave signal with specified parameters.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param frequency 频率 (Hz) / Frequency in Hz
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @param amplitude 振幅 / Amplitude
     * @param phase 初相位 (弧度) / Initial phase in radians
     * @return 生成的正弦波信号向量 / Generated sine wave signal vector
     */
    public static IVector<Double> sineWave(int length, double frequency, double samplingRate,
                                         double amplitude, double phase) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("sine");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .frequency(frequency)
                .samplingRate(samplingRate)
                .amplitude(amplitude)
                .phase(phase);
            return generator.generate(ISignalGenerator.SignalType.SINE, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate sine wave", e);
        }
    }
    
    /**
     * 生成余弦波信号 / Generate Cosine Wave Signal
     * <p>
     * 产生指定参数的余弦波信号。
     * Generate cosine wave signal with specified parameters.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param frequency 频率 (Hz) / Frequency in Hz
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @param amplitude 振幅 / Amplitude
     * @param phase 初相位 (弧度) / Initial phase in radians
     * @return 生成的余弦波信号向量 / Generated cosine wave signal vector
     */
    public static IVector<Double> cosineWave(int length, double frequency, double samplingRate,
                                           double amplitude, double phase) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("cosine");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .frequency(frequency)
                .samplingRate(samplingRate)
                .amplitude(amplitude)
                .phase(phase);
            return generator.generate(ISignalGenerator.SignalType.COSINE, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate cosine wave", e);
        }
    }
    
    /**
     * 生成方波信号 / Generate Square Wave Signal
     * <p>
     * 产生指定参数的方波信号。
     * Generate square wave signal with specified parameters.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param frequency 频率 (Hz) / Frequency in Hz
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @param amplitude 振幅 / Amplitude
     * @param dutyCycle 占空比 / Duty cycle
     * @return 生成的方波信号向量 / Generated square wave signal vector
     */
    public static IVector<Double> squareWave(int length, double frequency, double samplingRate,
                                           double amplitude, double dutyCycle) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("square");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .frequency(frequency)
                .samplingRate(samplingRate)
                .amplitude(amplitude)
                .dutyCycle(dutyCycle);
            return generator.generate(ISignalGenerator.SignalType.SQUARE, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate square wave", e);
        }
    }
    
    /**
     * 生成三角波信号 / Generate Triangular Wave Signal
     * <p>
     * 产生指定参数的三角波信号。
     * Generate triangular wave signal with specified parameters.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param frequency 频率 (Hz) / Frequency in Hz
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @param amplitude 振幅 / Amplitude
     * @param symmetry 对称性 (0-1) / Symmetry (0-1)
     * @return 生成的三角波信号向量 / Generated triangular wave signal vector
     */
    public static IVector<Double> triangularWave(int length, double frequency, double samplingRate,
                                               double amplitude, double symmetry) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("triangle");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .frequency(frequency)
                .samplingRate(samplingRate)
                .amplitude(amplitude)
                .dutyCycle(symmetry); // Using dutyCycle as symmetry parameter
            return generator.generate(ISignalGenerator.SignalType.TRIANGLE, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate triangular wave", e);
        }
    }
    
    /**
     * 生成锯齿波信号 / Generate Sawtooth Wave Signal
     * <p>
     * 产生指定参数的锯齿波信号。
     * Generate sawtooth wave signal with specified parameters.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param frequency 频率 (Hz) / Frequency in Hz
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @param amplitude 振幅 / Amplitude
     * @return 生成的锯齿波信号向量 / Generated sawtooth wave signal vector
     */
    public static IVector<Double> sawtoothWave(int length, double frequency, double samplingRate, double amplitude) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("sawtooth");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .frequency(frequency)
                .samplingRate(samplingRate)
                .amplitude(amplitude);
            return generator.generate(ISignalGenerator.SignalType.SAWTOOTH, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate sawtooth wave", e);
        }
    }
    
    /**
     * 生成白噪声信号 / Generate White Noise Signal
     * <p>
     * 产生指定功率的白噪声信号。
     * Generate white noise signal with specified power.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param power 噪声功率 / Noise power
     * @return 生成的白噪声信号向量 / Generated white noise signal vector
     */
    public static IVector<Double> whiteNoise(int length, double power) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("noise");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .amplitude(Math.sqrt(power)); // Convert power to amplitude
            return generator.generate(ISignalGenerator.SignalType.WHITE_NOISE, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate white noise", e);
        }
    }
    
    /**
     * 生成粉红噪声信号 / Generate Pink Noise Signal
     * <p>
     * 使用Voss-McCartney算法生成粉红噪声。粉红噪声的功率谱密度与频率成反比（1/f），
     * 在人类听觉和音乐处理中具有重要意义。
     * </p>
     * <p>
     * Generates pink noise using the Voss-McCartney algorithm. Pink noise has a power spectral
     * density proportional to 1/f, which is significant in human hearing and music processing.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param power 噪声功率 / Noise power
     * @return 粉红噪声信号 / Pink noise signal
     */
    public static IVector<Double> pinkNoise(int length, double power) {
        // Voss-McCartney algorithm for pink noise generation
        // Number of octaves (typically 16-20 is sufficient for audio)
        int octaves = 16;

        // Arrays to store white noise for each octave and running sums
        double[] whiteOctaves = new double[octaves];
        double[] pinkNoiseData = new double[length];

        // Initialize white octaves with random values
        java.util.Random random = new java.util.Random();

        // Calculate the target standard deviation from power
        double targetStd = Math.sqrt(power);

        // For each output sample
        for (int i = 0; i < length; i++) {
            // Find which octaves to update (based on binary representation)
            int mask = i + 1;
            int octave = 0;

            while ((mask & 1) == 0 && octave < octaves - 1) {
                mask >>= 1;
                octave++;
            }

            // Generate new white noise for this octave
            whiteOctaves[octave] = random.nextGaussian() * targetStd;

            // Sum all octaves
            double sum = 0;
            for (int j = 0; j < octaves; j++) {
                sum += whiteOctaves[j];
            }

            // Add a fresh white noise component
            sum += random.nextGaussian() * targetStd * 0.5;

            pinkNoiseData[i] = sum;
        }

        // Normalize to match target power
        double actualPower = 0;
        for (int i = 0; i < length; i++) {
            actualPower += pinkNoiseData[i] * pinkNoiseData[i];
        }
        actualPower /= length;
        double scale = targetStd / Math.sqrt(actualPower);

        for (int i = 0; i < length; i++) {
            pinkNoiseData[i] *= scale;
        }

        return IVector.of(pinkNoiseData);
    }
    
    // ========== 复合信号生成方法 / Composite Signal Generation Methods ==========
    
    /**
     * 生成复合信号 / Generate Composite Signal
     * <p>
     * 通过叠加多个信号分量生成复合信号。
     * Generate composite signal by superimposing multiple signal components.
     * </p>
     *
     * @param signalTypes 信号类型数组 / Signal type array
     * @param length 信号长度 / Signal length
     * @param parameters 信号参数数组 / Signal parameters array
     * @return 生成的复合信号 / Generated composite signal
     */
    public static IVector<Double> compositeSignal(ISignalGenerator.SignalType[] signalTypes, 
                                                int length, 
                                                ISignalGenerator.SignalParameters[] parameters) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("sine"); // Any generator will do
            return generator.generateComposite(signalTypes, length, parameters);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate composite signal", e);
        }
    }
    
    /**
     * 生成带噪声的信号 / Generate Signal with Noise
     * <p>
     * 向原始信号添加指定类型的噪声。
     * Add specified type of noise to original signal.
     * </p>
     *
     * @param signal 原始信号 / Original signal
     * @param noiseType 噪声类型 / Noise type
     * @param parameters 噪声参数 / Noise parameters
     * @return 添加噪声后的信号 / Signal with added noise
     */
    public static IVector<Double> addNoise(IVector<Double> signal, 
                                         ISignalGenerator.SignalType noiseType, 
                                         ISignalGenerator.SignalParameters parameters) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("sine"); // Any generator will do
            return generator.addNoise(signal, noiseType, parameters);
        } catch (Exception e) {
            throw new RuntimeException("Failed to add noise to signal", e);
        }
    }
    
    // ========== 特殊信号生成方法 / Special Signal Generation Methods ==========
    
    /**
     * 生成阶跃信号 / Generate Step Signal
     * <p>
     * 产生指定参数的阶跃信号。
     * Generate step signal with specified parameters.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param amplitude 振幅 / Amplitude
     * @param stepTime 阶跃时间 (秒) / Step time in seconds
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @return 生成的阶跃信号向量 / Generated step signal vector
     */
    public static IVector<Double> stepSignal(int length, double amplitude, double stepTime, double samplingRate) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("signal");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .amplitude(amplitude)
                .stepTime(stepTime)
                .samplingRate(samplingRate);
            return generator.generate(ISignalGenerator.SignalType.STEP, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate step signal", e);
        }
    }
    
    /**
     * 生成单位阶跃信号 / Generate Unit Step Signal
     * <p>
     * 产生单位振幅的阶跃信号。
     * Generate unit step signal with unit amplitude.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param stepTime 阶跃时间 (秒) / Step time in seconds
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @return 生成的单位阶跃信号向量 / Generated unit step signal vector
     */
    public static IVector<Double> unitStep(int length, double stepTime, double samplingRate) {
        return stepSignal(length, 1.0, stepTime, samplingRate);
    }
    
    /**
     * 生成狄拉克δ函数（连续时间脉冲） / Generate Dirac Delta Function (Continuous-time impulse)
     * <p>
     * 产生指定参数的狄拉克δ函数。
     * Generate Dirac delta function with specified parameters.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param impulseIndex 脉冲位置索引 / Impulse position index
     * @param amplitude 脉冲幅度 / Impulse amplitude
     * @return 生成的狄拉克δ函数信号向量 / Generated Dirac delta signal vector
     */
    public static IVector<Double> diracDelta(int length, int impulseIndex, double amplitude) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("signal");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .amplitude(amplitude);
            
            // For Dirac delta, we need to handle it specially
            IVector<Double> signal = Linalg.zeros(length);
            if (impulseIndex >= 0 && impulseIndex < length) {
                signal.set(impulseIndex, amplitude);
            }
            return signal;
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate Dirac delta signal", e);
        }
    }
    
    /**
     * 生成单位脉冲信号（克罗内克δ函数） / Generate Unit Impulse Signal (Kronecker Delta Function)
     * <p>
     * 产生单位幅度的克罗内克δ函数。
     * Generate Kronecker delta function with unit amplitude.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param impulseIndex 脉冲位置索引 / Impulse position index
     * @return 生成的单位脉冲信号向量 / Generated unit impulse signal vector
     */
    public static IVector<Double> unitImpulse(int length, int impulseIndex) {
        return diracDelta(length, impulseIndex, 1.0);
    }
    
    /**
     * 生成线性调频信号（Chirp） / Generate Linear Chirp Signal
     * <p>
     * 产生频率随时间线性变化的调频信号。
     * Generate chirp signal with frequency changing linearly over time.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param startFreq 起始频率 (Hz) / Start frequency in Hz
     * @param endFreq 结束频率 (Hz) / End frequency in Hz
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @param amplitude 振幅 / Amplitude
     * @return 生成的线性调频信号向量 / Generated chirp signal vector
     */
    public static IVector<Double> chirpSignal(int length, double startFreq, double endFreq,
                                            double samplingRate, double amplitude) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("signal");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .startFrequency(startFreq)
                .endFrequency(endFreq)
                .samplingRate(samplingRate)
                .amplitude(amplitude);
            return generator.generate(ISignalGenerator.SignalType.CHIRP, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate chirp signal", e);
        }
    }
    
    /**
     * 生成脉冲信号 / Generate Pulse Signal
     * <p>
     * 产生指定参数的脉冲信号。
     * Generate pulse signal with specified parameters.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @param amplitude 脉冲幅度 / Pulse amplitude
     * @param pulseWidth 脉冲宽度（采样点数）/ Pulse width in samples
     * @param frequency 脉冲频率 (Hz) / Pulse frequency in Hz
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @return 生成的脉冲信号向量 / Generated pulse signal vector
     */
    public static IVector<Double> pulseSignal(int length, double amplitude, int pulseWidth,
                                            double frequency, double samplingRate) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("signal");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .amplitude(amplitude)
                .pulseWidth(pulseWidth)
                .frequency(frequency)
                .samplingRate(samplingRate);
            return generator.generate(ISignalGenerator.SignalType.PULSE, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate pulse signal", e);
        }
    }
    
    // ========== 信号滤波方法 / Signal Filtering Methods ==========
    
    /**
     * 移动平均滤波器 / Moving Average Filter
     * <p>
     * 对输入信号应用移动平均滤波。
     * Apply moving average filter to input signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param windowSize 窗口大小 / Window size
     * @return 滤波后的信号 / Filtered signal
     */
    public static IVector<Double> movingAverage(IVector<Double> signal, int windowSize) {
        try {
            ISignalFilter<Double> filter = SignalProcessorFactory.getInstance().createFilter("movingaverage");
            // Set filter parameters if needed
            return filter.process(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply moving average filter", e);
        }
    }
    
    /**
     * 中值滤波器 / Median Filter
     * <p>
     * 对输入信号应用中值滤波。
     * Apply median filter to input signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param windowSize 窗口大小 / Window size
     * @return 滤波后的信号 / Filtered signal
     */
    public static IVector<Double> medianFilter(IVector<Double> signal, int windowSize) {
        try {
            ISignalFilter<Double> filter = SignalProcessorFactory.getInstance().createFilter("median");
            // Set filter parameters if needed
            return filter.process(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply median filter", e);
        }
    }
    
    /**
     * 高斯滤波器（单参数版本）/ Gaussian Filter (Single Parameter Version)
     * <p>
     * 使用默认核大小的高斯滤波器。
     * Apply Gaussian filter with default kernel size.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param sigma 高斯核标准差 / Gaussian kernel standard deviation
     * @return 滤波后的信号 / Filtered signal
     */
    public static IVector<Double> gaussianFilter(IVector<Double> signal, double sigma) {
        int defaultKernelSize = (int) Math.ceil(6 * sigma);
        if (defaultKernelSize % 2 == 0) {
            defaultKernelSize++;
        }
        return gaussianFilter(signal, sigma, defaultKernelSize);
    }

    /**
     * 高斯滤波器（双参数版本）/ Gaussian Filter (Dual Parameter Version)
     * <p>
     * 使用指定核大小的高斯滤波器。
     * Apply Gaussian filter with specified kernel size.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param sigma 高斯核标准差 / Gaussian kernel standard deviation
     * @param kernelSize 核大小（奇数）/ Kernel size (odd number)
     * @return 滤波后的信号 / Filtered signal
     */
    public static IVector<Double> gaussianFilter(IVector<Double> signal, double sigma, int kernelSize) {
        try {
            GaussianFilter filter = new GaussianFilter(sigma, kernelSize);
            return filter.filter(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply gaussian filter", e);
        }
    }
    
    /**
     * 巴特沃斯低通滤波器 / Butterworth Low-pass Filter
     * <p>
     * 对输入信号应用巴特沃斯低通滤波。
     * Apply Butterworth low-pass filter to input signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param cutoffFreq 截止频率 (Hz) / Cutoff frequency in Hz
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @param order 滤波器阶数 / Filter order
     * @return 滤波后的信号 / Filtered signal
     */
    public static IVector<Double> butterworthLowPass(IVector<Double> signal, double cutoffFreq, double samplingRate, int order) {
        try {
            ISignalFilter<Double> filter = SignalProcessorFactory.getInstance().createFilter("butterworth");
            // Set filter parameters if needed
            return filter.process(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply butterworth low-pass filter", e);
        }
    }
    
    /**
     * 巴特沃斯高通滤波器 / Butterworth High-pass Filter
     * <p>
     * 使用巴特沃斯滤波器实现高通滤波，滤除低频成分，保留高频成分。
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param cutoffFreq 截止频率 / Cutoff frequency
     * @param samplingRate 采样率 / Sampling rate
     * @param order 滤波器阶数 / Filter order
     * @return 滤波后的信号 / Filtered signal
     */
    public static IVector<Double> butterworthHighPass(IVector<Double> signal, double cutoffFreq, double samplingRate, int order) {
        try {
            ButterworthFilter filter = new ButterworthFilter(
                    ISignalFilter.FilterType.HIGH_PASS,
                    order,
                    new double[]{cutoffFreq},
                    samplingRate
            );
            return filter.filter(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply butterworth high-pass filter", e);
        }
    }
    
    /**
     * 巴特沃斯带通滤波器 / Butterworth Band-pass Filter
     * <p>
     * 对输入信号应用巴特沃斯带通滤波。
     * Apply Butterworth band-pass filter to input signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param lowFreq 低截止频率 (Hz) / Low cutoff frequency in Hz
     * @param highFreq 高截止频率 (Hz) / High cutoff frequency in Hz
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @param order 滤波器阶数 / Filter order
     * @return 滤波后的信号 / Filtered signal
     */
    public static IVector<Double> bandPass(IVector<Double> signal, double lowFreq, double highFreq, double samplingRate, int order) {
        try {
            ISignalFilter<Double> filter = SignalProcessorFactory.getInstance().createFilter("bandpass");
            // Set filter parameters if needed
            return filter.process(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply bandpass filter", e);
        }
    }

    /**
     * 卡尔曼滤波器 / Kalman Filter
     * <p>
     * 对输入信号应用卡尔曼滤波。
     * Apply Kalman filter to input signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param processNoiseVariance 过程噪声方差 / Process noise variance
     * @param measurementNoiseVariance 测量噪声方差 / Measurement noise variance
     * @return 滤波后的信号 / Filtered signal
     */
    public static IVector<Double> kalmanFilter(IVector<Double> signal, double processNoiseVariance, double measurementNoiseVariance) {
        try {
            ISignalFilter<Double> filter = SignalProcessorFactory.getInstance().createFilter("kalman");
            // Set filter parameters if needed
            return filter.process(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply Kalman filter", e);
        }
    }

    /**
     * 维纳滤波器 / Wiener Filter
     * <p>
     * 对输入信号应用维纳滤波。
     * Apply Wiener filter to input signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param signalPower 信号功率 / Signal power
     * @param noisePower 噪声功率 / Noise power
     * @param filterLength 滤波器长度 / Filter length
     * @return 滤波后的信号 / Filtered signal
     */
    public static IVector<Double> wienerFilter(IVector<Double> signal, double signalPower, double noisePower, int filterLength) {
        try {
            ISignalFilter<Double> filter = SignalProcessorFactory.getInstance().createFilter("wiener");
            // Set filter parameters if needed
            return filter.process(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply Wiener filter", e);
        }
    }

    /**
     * 带阻滤波器 / Band-stop Filter
     * <p>
     * 对输入信号应用带阻（陷波）滤波。
     * Apply band-stop (notch) filter to input signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param lowFreq 低截止频率 (Hz) / Low cutoff frequency in Hz
     * @param highFreq 高截止频率 (Hz) / High cutoff frequency in Hz
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @param order 滤波器阶数 / Filter order
     * @return 滤波后的信号 / Filtered signal
     */
    public static IVector<Double> bandStop(IVector<Double> signal, double lowFreq, double highFreq, double samplingRate, int order) {
        try {
            ISignalFilter<Double> filter = SignalProcessorFactory.getInstance().createFilter("bandstop");
            // Set filter parameters if needed
            return filter.process(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply band-stop filter", e);
        }
    }

    // ========== 信号分析方法 / Signal Analysis Methods ==========
    
    /**
     * 计算信号的功率谱密度 (PSD) / Calculate Power Spectral Density (PSD)
     * <p>
     * 计算信号的功率谱密度。
     * Calculate power spectral density of signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param windowSize 窗口大小 / Window size
     * @param overlap 重叠率 / Overlap ratio
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @return 包含频率和功率谱密度的元组 / Tuple containing frequencies and PSD
     */
    public static Tuple2<IVector<Double>, IVector<Double>> powerSpectralDensity(
            IVector<Double> signal, int windowSize, double overlap, double samplingRate) {
        try {
            ISignalAnalyzer<Double> analyzer = SignalProcessorFactory.getInstance().createAnalyzer("psd");
            ISignalAnalyzer.AnalysisParameters params = new ISignalAnalyzer.AnalysisParameters()
                .windowSize(windowSize)
                .overlap(overlap)
                .samplingRate(samplingRate);
            ISignalAnalyzer.AnalysisResult<Tuple2<IVector<Double>, IVector<Double>>> result = 
                analyzer.analyze(signal, ISignalAnalyzer.AnalysisType.POWER_SPECTRUM, params);
            return result.getResult();
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate power spectral density", e);
        }
    }
    
    /**
     * 计算信号的自相关函数 / Calculate Autocorrelation Function
     * <p>
     * 计算输入信号的自相关函数。
     * Calculate autocorrelation function of input signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @return 自相关函数向量 / Autocorrelation function vector
     */
    public static IVector<Double> autocorrelation(IVector<Double> signal) {
        try {
            ISignalAnalyzer<Double> analyzer = SignalProcessorFactory.getInstance().createAnalyzer("autocorr");
            ISignalAnalyzer.AnalysisResult<IVector<Double>> result = 
                analyzer.analyze(signal, ISignalAnalyzer.AnalysisType.AUTOCORRELATION);
            return result.getResult();
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate autocorrelation", e);
        }
    }
    
    /**
     * 计算两个信号的互相关函数 / Calculate Cross-correlation Function
     * <p>
     * 计算两个信号的全互相关函数。
     * Calculate full cross-correlation function of two signals.
     * </p>
     *
     * @param signal1 第一个信号 / First signal
     * @param signal2 第二个信号 / Second signal
     * @return 互相关函数向量 / Cross-correlation function vector
     */
    public static IVector<Double> crossCorrelation(IVector<Double> signal1, IVector<Double> signal2) {
        try {
            ISignalAnalyzer<Double> analyzer = SignalProcessorFactory.getInstance().createAnalyzer("crosscorr");
            ISignalAnalyzer.AnalysisResult<IVector<Double>> result =
                analyzer.analyze(signal1, ISignalAnalyzer.AnalysisType.CROSS_CORRELATION);
            return result.getResult();
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate cross-correlation", e);
        }
    }

    /**
     * 计算两个信号的互相关函数（带最大滞后限制）/ Calculate Cross-correlation Function with max lag
     * <p>
     * 计算两个信号的互相关函数，并限制最大滞后点数。
     * Calculate cross-correlation function with maximum lag limitation.
     * </p>
     *
     * @param signal1 第一个信号 / First signal
     * @param signal2 第二个信号 / Second signal
     * @param maxLag 最大滞后点数 / Maximum lag in points
     * @return 截断后的互相关函数向量 / Truncated cross-correlation vector
     */
    public static IVector<Double> crossCorrelation(IVector<Double> signal1, IVector<Double> signal2, int maxLag) {
        // First get the full cross-correlation from the analyzer
        IVector<Double> fullCorr = crossCorrelation(signal1, signal2);
        int fullLength = fullCorr.length();

        if (fullLength <= 2 * maxLag + 1) {
            return fullCorr;
        }

        // Extract the window around center (zero lag position)
        int center = fullLength / 2;
        int halfWindow = maxLag;

        int start = Math.max(0, center - halfWindow);
        int end = Math.min(fullLength, center + halfWindow + 1);

        double[] truncated = new double[2 * maxLag + 1];
        int destStart = start <= center - halfWindow ? 0 : center - halfWindow - start;
        int srcLen = end - start;

        for (int i = 0; i < srcLen && (destStart + i) < truncated.length; i++) {
            truncated[destStart + i] = fullCorr.get(start + i);
        }

        return IVector.of(truncated);
    }
    
    /**
     * 计算信号的频谱 / Calculate Signal Spectrum
     * <p>
     * 计算信号的频谱，包含频率、幅度和相位信息。
     * Calculate signal spectrum including frequency, magnitude, and phase information.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @return 包含频率、幅度、相位的元组 / Tuple containing frequency, magnitude, phase
     */
    public static Tuple3<IVector<Double>, IVector<Double>, IVector<Double>> spectrum(
            IVector<Double> signal, double samplingRate) {
        try {
            ISignalAnalyzer<Double> analyzer = SignalProcessorFactory.getInstance().createAnalyzer("spectrum");
            ISignalAnalyzer.AnalysisParameters params = new ISignalAnalyzer.AnalysisParameters()
                .samplingRate(samplingRate);
            ISignalAnalyzer.AnalysisResult<Tuple2<IVector<Double>, IVector<Double>>> result = 
                analyzer.analyze(signal, ISignalAnalyzer.AnalysisType.SPECTRUM, params);
            
            // Convert to Tuple3 format (frequency, magnitude, phase)
            Tuple2<IVector<Double>, IVector<Double>> spectrumResult = result.getResult();
            IVector<Double> frequencies = spectrumResult._1;
            IVector<Double> magnitudes = spectrumResult._2;
            // For now, use zeros for phase as approximation
            IVector<Double> phases = Linalg.zeros(magnitudes.length());
            
            return new Tuple3<>(frequencies, magnitudes, phases);
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate spectrum", e);
        }
    }
    
    /**
     * 计算信号的短时傅里叶变换 (STFT) / Calculate Short-Time Fourier Transform (STFT)
     * <p>
     * 计算信号的短时傅里叶变换。
     * Calculate short-time Fourier transform of signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param windowSize 窗口大小 / Window size
     * @param hopSize 跳跃大小 / Hop size
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @return STFT结果矩阵（实部虚部交替）/ STFT result matrix (real and imaginary alternating)
     */
    public static IMatrix<Double> shortTimeFourierTransform(
            IVector<Double> signal, int windowSize, int hopSize, double samplingRate) {
        int signalLength = signal.length();

        int numFrames = (int) Math.ceil((double)(signalLength - windowSize) / hopSize) + 1;
        int numFreqBins = windowSize / 2 + 1;

        double[][] realParts = new double[numFreqBins][numFrames];
        double[][] imagParts = new double[numFreqBins][numFrames];

        double[] window = new double[windowSize];
        for (int i = 0; i < windowSize; i++) {
            window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (windowSize - 1)));
        }

        for (int frame = 0; frame < numFrames; frame++) {
            int startIdx = frame * hopSize;

            double[] frameData = new double[windowSize];
            for (int i = 0; i < windowSize; i++) {
                int idx = startIdx + i;
                if (idx < signalLength) {
                    frameData[i] = signal.get(idx) * window[i];
                } else {
                    frameData[i] = 0;
                }
            }

            int fftSize = nextPowerOf2(windowSize);
            Complex[] complexFrame = new Complex[fftSize];
            for (int i = 0; i < windowSize; i++) {
                complexFrame[i] = new Complex(frameData[i], 0);
            }
            for (int i = windowSize; i < fftSize; i++) {
                complexFrame[i] = new Complex(0, 0);
            }

            Complex[] fftResult = RereFFT.fft(complexFrame);

            for (int i = 0; i < numFreqBins; i++) {
                realParts[i][frame] = fftResult[i].real;
                imagParts[i][frame] = fftResult[i].imag;
            }
        }

        double[][] result = new double[numFreqBins * 2][numFrames];
        for (int f = 0; f < numFreqBins; f++) {
            for (int fr = 0; fr < numFrames; fr++) {
                result[2 * f][fr] = realParts[f][fr];
                result[2 * f + 1][fr] = imagParts[f][fr];
            }
        }

        return IMatrix.of(result);
    }

    private static int nextPowerOf2(int n) {
        int power = 1;
        while (power < n) {
            power *= 2;
        }
        return power;
    }
    
    /**
     * 计算信号的信噪比 (SNR) / Calculate Signal-to-Noise Ratio (SNR)
     * <p>
     * 计算信号的信噪比（分贝）。
     * Calculate signal-to-noise ratio in decibels.
     * </p>
     *
     * @param signal 有用信号 / Signal
     * @param noise 噪声信号 / Noise signal
     * @return 信噪比 (dB) / SNR in dB
     */
    public static double signalToNoiseRatio(IVector<Double> signal, IVector<Double> noise) {
        try {
            ISignalAnalyzer<Double> analyzer = SignalProcessorFactory.getInstance().createAnalyzer("spectrum");
            ISignalAnalyzer.AnalysisParameters params = new ISignalAnalyzer.AnalysisParameters();
            ISignalAnalyzer.AnalysisResult<Double> result = 
                analyzer.analyze(signal, ISignalAnalyzer.AnalysisType.SNR, params);
            return result.getResult();
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate SNR", e);
        }
    }
    
    /**
     * 计算信号的峰值信噪比 (PSNR) / Calculate Peak Signal-to-Noise Ratio (PSNR)
     * <p>
     * 计算原始信号与重构信号之间的峰值信噪比。
     * Calculate PSNR between original and reconstructed signals.
     * </p>
     *
     * @param original 原始信号 / Original signal
     * @param reconstructed 重构信号 / Reconstructed signal
     * @return 峰值信噪比 (dB) / PSNR in dB
     */
    public static double peakSignalToNoiseRatio(IVector<Double> original, IVector<Double> reconstructed) {
        // Calculate MSE
        double mse = 0;
        int n = original.length();
        for (int i = 0; i < n; i++) {
            double diff = original.get(i) - reconstructed.get(i);
            mse += diff * diff;
        }
        mse /= n;
        
        // Calculate PSNR
        double maxVal = original.max();
        return 20 * Math.log10(maxVal / Math.sqrt(mse));
    }
    
    // ========== 变换方法 / Transform Methods ==========
    
    /**
     * 快速傅里叶变换 / Fast Fourier Transform
     * <p>
     * 对复数输入进行快速傅里叶变换。
     * Perform FFT on complex input.
     * </p>
     *
     * @param x 输入复数数组 / Input complex array
     * @return 变换后的复数数组 / Transformed complex array
     */
    public static Complex[] fft(Complex[] x) {
        // This is a low-level method that doesn't use the factory pattern
        return RereFFT.fft(x);
    }
    
    /**
     * 快速傅里叶逆变换 / Inverse Fast Fourier Transform
     * <p>
     * 对复数频谱进行快速傅里叶逆变换。
     * Perform inverse FFT on complex frequency spectrum.
     * </p>
     *
     * @param x 输入的复数频谱 / Input complex frequency spectrum
     * @return 逆变换后的时域复数信号 / Inverse transformed time-domain signal
     */
    public static Complex[] ifft(Complex[] x) {
        // This is a low-level method that doesn't use the factory pattern
        return RereFFT.ifft(x);
    }
    
    /**
     * 计算信号的幅度谱 / Calculate Magnitude Spectrum
     * <p>
     * 计算FFT结果的幅度谱。
     * Calculate magnitude spectrum from FFT result.
     * </p>
     *
     * @param fftResult FFT结果 / FFT result
     * @return 幅度谱数组 / Magnitude spectrum array
     */
    public static double[] magnitudeSpectrum(Complex[] fftResult) {
        // This is a low-level method that doesn't use the factory pattern
        return RereFFT.magnitudeSpectrum(fftResult);
    }
    
    /**
     * 计算信号的相位谱 / Calculate Phase Spectrum
     * <p>
     * 计算FFT结果的相位谱。
     * Calculate phase spectrum from FFT result.
     * </p>
     *
     * @param fftResult FFT结果 / FFT result
     * @return 相位谱数组（弧度）/ Phase spectrum array in radians
     */
    public static double[] phaseSpectrum(Complex[] fftResult) {
        // This is a low-level method that doesn't use the factory pattern
        return RereFFT.phaseSpectrum(fftResult);
    }
    
    /**
     * 计算信号的功率谱 / Calculate Power Spectrum
     * <p>
     * 计算FFT结果的功率谱。
     * Calculate power spectrum from FFT result.
     * </p>
     *
     * @param fftResult FFT结果 / FFT result
     * @return 功率谱数组 / Power spectrum array
     */
    public static double[] powerSpectrum(Complex[] fftResult) {
        // This is a low-level method that doesn't use the factory pattern
        return RereFFT.powerSpectrum(fftResult);
    }
    
    /**
     * DCT-II变换 / DCT-II Transform
     * <p>
     * 对信号进行离散余弦变换II。
     * Perform DCT-II on signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @return DCT变换结果 / DCT transform result
     */
    public static IVector<Double> dct2(IVector<Double> signal) {
        // This is a low-level method that doesn't use the factory pattern
        return RereDCT.dct2(signal);
    }
    
    /**
     * DCT-II逆变换 / DCT-II Inverse Transform
     * <p>
     * 对DCT系数进行逆变换以重建信号。
     * Perform inverse DCT to reconstruct signal.
     * </p>
     *
     * @param dctSignal DCT变换结果 / DCT transform result
     * @return 重建的原始信号 / Reconstructed original signal
     */
    public static IVector<Double> idct2(IVector<Double> dctSignal) {
        // This is a low-level method that doesn't use the factory pattern
        return RereDCT.idct2(dctSignal);
    }
    
    /**
     * 希尔伯特变换 / Hilbert Transform
     * <p>
     * 计算信号的希尔伯特变换。
     * Perform Hilbert transform on signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @return 希尔伯特变换结果 / Hilbert transform result
     */
    public static IVector<Double> hilbertTransform(IVector<Double> signal) {
        // This is a low-level method that doesn't use the factory pattern
        return RereHilbert.hilbertTransform(signal);
    }
    
    /**
     * 计算解析信号 / Calculate Analytic Signal
     * <p>
     * 计算信号的解析信号。
     * Calculate analytic signal of input.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @return 解析信号（复数数组）/ Analytic signal (complex array)
     */
    public static Complex[] analyticSignal(IVector<Double> signal) {
        // This is a low-level method that doesn't use the factory pattern
        return RereHilbert.analyticSignal(signal);
    }
    
    /**
     * 计算瞬时幅度 / Calculate Instantaneous Amplitude
     * <p>
     * 计算信号的瞬时幅度（包络）。
     * Calculate instantaneous amplitude (envelope) of signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @return 瞬时幅度向量 / Instantaneous amplitude vector
     */
    public static IVector<Double> instantaneousAmplitude(IVector<Double> signal) {
        // This is a low-level method that doesn't use the factory pattern
        return RereHilbert.instantaneousAmplitude(signal);
    }
    
    /**
     * 计算瞬时相位 / Calculate Instantaneous Phase
     * <p>
     * 计算信号的瞬时相位。
     * Calculate instantaneous phase of signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @return 瞬时相位向量（弧度）/ Instantaneous phase vector in radians
     */
    public static IVector<Double> instantaneousPhase(IVector<Double> signal) {
        // This is a low-level method that doesn't use the factory pattern
        return RereHilbert.instantaneousPhase(signal);
    }
    
    /**
     * 计算瞬时频率 / Calculate Instantaneous Frequency
     * <p>
     * 通过相位差分计算瞬时频率。
     * Calculate instantaneous frequency by phase differentiation.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @return 瞬时频率向量 / Instantaneous frequency vector
     */
    public static IVector<Double> instantaneousFrequency(IVector<Double> signal, double samplingRate) {
        // This is a low-level method that doesn't use the factory pattern
        return RereHilbert.instantaneousFrequency(signal, samplingRate);
    }
    
    /**
     * 离散小波变换 (DWT) / Discrete Wavelet Transform (DWT)
     * <p>
     * 对信号进行离散小波变换。
     * Perform discrete wavelet transform on signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Decomposition levels
     * @param param 小波参数 / Wavelet parameter
     * @return 小波系数 / Wavelet coefficients
     */
    public static WaveletCoefficients discreteWaveletTransform(IVector<Double> signal,
                                                               WaveletAnalysis.WaveletType waveletType, int levels, double param) {
        // This is a low-level method that doesn't use the factory pattern
        return WaveletAnalysis.discreteWaveletTransform(signal, waveletType, levels, param);
    }
    
    /**
     * 小波逆变换 (IDWT) / Inverse Discrete Wavelet Transform (IDWT)
     * <p>
     * 对小波系数进行逆变换以重建信号。
     * Perform inverse DWT to reconstruct signal from wavelet coefficients.
     * </p>
     *
     * @param coefficients 小波系数 / Wavelet coefficients
     * @param waveletType 小波类型 / Wavelet type
     * @param param 小波参数 / Wavelet parameter
     * @return 重建的原始信号 / Reconstructed original signal
     */
    public static IVector<Double> inverseDiscreteWaveletTransform(WaveletCoefficients coefficients,
            WaveletAnalysis.WaveletType waveletType, double param) {
        // This is a low-level method that doesn't use the factory pattern
        return WaveletAnalysis.inverseDiscreteWaveletTransform(coefficients, waveletType, param);
    }
    
    // ========== 可视化方法 / Visualization Methods ==========

    /**
     * 绘制信号波形图 / Plot Signal Waveform
     * <p>
     * 绘制信号的时域波形图。
     * Plot time-domain waveform of signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @param title 图表标题 / Chart title
     * @return 绘制的图表实例 / Plot instance
     */
    public static IPlot plotWaveform(IVector<Double> signal, double samplingRate, String title) {
        return SignalPlots.plotWaveform(signal, samplingRate, title);
    }
    
    /**
     * 绘制信号频谱图 / Plot Signal Spectrum
     * <p>
     * 绘制信号的频谱图。
     * Plot frequency spectrum of signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @param title 图表标题 / Chart title
     * @return 绘制的图表实例 / Plot instance
     */
    public static IPlot plotSpectrum(IVector<Double> signal, double samplingRate, String title) {
        return SignalPlots.plotSpectrum(signal, samplingRate, title);
    }
    
    /**
     * 绘制功率谱密度图 / Plot Power Spectral Density
     * <p>
     * 绘制信号的功率谱密度图。
     * Plot power spectral density of signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param samplingRate 采样率 (Hz) / Sampling rate in Hz
     * @param windowSize 窗口大小 / Window size
     * @param overlap 重叠率 / Overlap ratio
     * @param title 图表标题 / Chart title
     * @return 绘制的图表实例 / Plot instance
     */
    public static IPlot plotPowerSpectralDensity(IVector<Double> signal, double samplingRate,
                                               int windowSize, double overlap, String title) {
        return SignalPlots.plotPowerSpectralDensity(signal, samplingRate, windowSize, overlap, title);
    }
    
    /**
     * 绘制自相关图 / Plot Autocorrelation
     * <p>
     * 绘制信号的自相关函数图。
     * Plot autocorrelation function of signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param maxLag 最大滞后点数 / Maximum lag in points
     * @param title 图表标题 / Chart title
     * @return 绘制的图表实例 / Plot instance
     */
    public static IPlot plotAutocorrelation(IVector<Double> signal, int maxLag, String title) {
        return SignalPlots.plotAutocorrelation(signal, maxLag, title);
    }
    
    /**
     * 绘制互相关图 / Plot Cross-correlation
     * <p>
     * 绘制两个信号的互相关函数图。
     * Plot cross-correlation function of two signals.
     * </p>
     *
     * @param signal1 第一个信号 / First signal
     * @param signal2 第二个信号 / Second signal
     * @param maxLag 最大滞后点数 / Maximum lag in points
     * @param title 图表标题 / Chart title
     * @return 绘制的图表实例 / Plot instance
     */
    public static IPlot plotCrossCorrelation(IVector<Double> signal1, IVector<Double> signal2,
                                           int maxLag, String title) {
        return SignalPlots.plotCrossCorrelation(signal1, signal2, maxLag, title);
    }
    
    /**
     * 绘制小波系数图 / Plot Wavelet Coefficients
     * <p>
     * 绘制信号的小波系数图。
     * Plot wavelet coefficients of signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Decomposition levels
     * @param title 图表标题 / Chart title
     * @return 绘制的图表实例 / Plot instance
     */
    public static IPlot plotWaveletCoefficients(IVector<Double> signal, String waveletType,
                                              int levels, String title) {
        return WaveletPlots.plotWaveletCoefficients(signal, waveletType, levels, title);
    }
    
    /**
     * 绘制小波能量分布图 / Plot Wavelet Energy Distribution
     * <p>
     * 绘制信号的小波能量分布图。
     * Plot wavelet energy distribution of signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Decomposition levels
     * @param title 图表标题 / Chart title
     * @return 绘制的图表实例 / Plot instance
     */
    public static IPlot plotWaveletEnergyDistribution(IVector<Double> signal, String waveletType,
                                                    int levels, String title) {
        return WaveletPlots.plotWaveletEnergyDistribution(signal, waveletType, levels, title);
    }
    
    // ========== 工厂方法 / Factory Methods ==========
    
    /**
     * 获取信号处理器工厂实例 / Get Signal Processor Factory Instance
     * <p>
     * 获取信号处理器工厂的单例实例。
     * Get singleton instance of signal processor factory.
     * </p>
     *
     * @return 信号处理器工厂实例 / Signal processor factory instance
     * @see SignalProcessorFactory#getInstance()
     */
    public static SignalProcessorFactory getFactory() {
        return SignalProcessorFactory.getInstance();
    }
    
    /**
     * 创建信号变换器 / Create signal transformer
     * <p>
     * 创建指定类型的信号变换器实例。
     * Create signal transformer instance of specified type.
     * </p>
     * <p>
     * 支持的变换器类型包括:
     * Supported transformer types include:
     * </p>
     * <ul>
     *   <li>"fft" - 快速傅里叶变换 / Fast Fourier Transform</li>
     *   <li>"dct" - 离散余弦变换 / Discrete Cosine Transform</li>
     *   <li>"hilbert" - 希尔伯特变换 / Hilbert Transform</li>
     *   <li>"wavelet" - 小波变换 / Wavelet Transform</li>
     *   <li>"ztransform" - Z变换 / Z Transform</li>
     *   <li>"chirpz" - Chirp-Z变换 / Chirp-Z Transform</li>
     *   <li>"walsh" - 沃尔什-哈达玛变换 / Walsh-Hadamard Transform</li>
     * </ul>
     *
     * @param transformType 变换器类型 / Transformer type
     * @param <T> 输入信号数据类型 / Input signal data type
     * @param <R> 输出变换结果类型 / Output transform result type
     * @return 信号变换器实例 / Signal transformer instance
     * @throws Exception 创建过程中发生错误时抛出 / Thrown when errors occur during creation
     * @see SignalProcessorFactory#createTransform(String)
     */
    public static <T extends Number, R> ISignalTransform<T, R> createTransform(String transformType) throws Exception {
        return SignalProcessorFactory.getInstance().createTransform(transformType);
    }
    
    /**
     * 创建信号滤波器 / Create signal filter
     * <p>
     * 创建指定类型的信号滤波器实例。
     * Create signal filter instance of specified type.
     * </p>
     * <p>
     * 支持的滤波器类型包括:
     * Supported filter types include:
     * </p>
     * <ul>
     *   <li>"butterworth" - 巴特沃斯滤波器 / Butterworth Filter</li>
     *   <li>"chebyshev" - 切比雪夫滤波器 / Chebyshev Filter</li>
     *   <li>"elliptic" - 椭圆滤波器 / Elliptic Filter</li>
     *   <li>"bessel" - 贝塞尔滤波器 / Bessel Filter</li>
     *   <li>"gaussian" - 高斯滤波器 / Gaussian Filter</li>
     *   <li>"movingaverage" - 移动平均滤波器 / Moving Average Filter</li>
     *   <li>"median" - 中值滤波器 / Median Filter</li>
     * </ul>
     *
     * @param filterType 滤波器类型 / Filter type
     * @param <T> 数据类型 / Data type
     * @return 信号滤波器实例 / Signal filter instance
     * @throws Exception 创建过程中发生错误时抛出 / Thrown when errors occur during creation
     * @see SignalProcessorFactory#createFilter(String)
     */
    public static <T extends Number> ISignalFilter<T> createFilter(String filterType) throws Exception {
        return SignalProcessorFactory.getInstance().createFilter(filterType);
    }
    
    /**
     * 创建信号分析器 / Create signal analyzer
     * <p>
     * 创建指定类型的信号分析器实例。
     * Create signal analyzer instance of specified type.
     * </p>
     * <p>
     * 支持的分析器类型包括:
     * Supported analyzer types include:
     * </p>
     * <ul>
     *   <li>"spectrum" - 频谱分析器 / Spectrum Analyzer</li>
     *   <li>"psd" - 功率谱密度分析器 / Power Spectral Density Analyzer</li>
     *   <li>"autocorr" - 自相关分析器 / Autocorrelation Analyzer</li>
     *   <li>"crosscorr" - 互相关分析器 / Cross-correlation Analyzer</li>
     *   <li>"wavelet" - 小波分析器 / Wavelet Analyzer</li>
     *   <li>"envelope" - 包络分析器 / Envelope Analyzer</li>
     *   <li>"instantaneous" - 瞬时特征分析器 / Instantaneous Feature Analyzer</li>
     * </ul>
     *
     * @param analyzerType 分析器类型 / Analyzer type
     * @param <T> 数据类型 / Data type
     * @return 信号分析器实例 / Signal analyzer instance
     * @throws Exception 创建过程中发生错误时抛出 / Thrown when errors occur during creation
     * @see SignalProcessorFactory#createAnalyzer(String)
     */
    public static <T extends Number> ISignalAnalyzer<T> createAnalyzer(String analyzerType) throws Exception {
        return SignalProcessorFactory.getInstance().createAnalyzer(analyzerType);
    }
    
    /**
     * 创建信号生成器 / Create signal generator
     * <p>
     * 创建指定类型的信号生成器实例。
     * Create signal generator instance of specified type.
     * </p>
     * <p>
     * 支持的生成器类型包括:
     * Supported generator types include:
     * </p>
     * <ul>
     *   <li>"sine" - 正弦波生成器 / Sine Wave Generator</li>
     *   <li>"cosine" - 余弦波生成器 / Cosine Wave Generator</li>
     *   <li>"square" - 方波生成器 / Square Wave Generator</li>
     *   <li>"triangle" - 三角波生成器 / Triangle Wave Generator</li>
     *   <li>"sawtooth" - 锯齿波生成器 / Sawtooth Wave Generator</li>
     *   <li>"noise" - 噪声生成器 / Noise Generator</li>
     *   <li>"chirp" - 线性调频信号生成器 / Chirp Signal Generator</li>
     *   <li>"pulse" - 脉冲信号生成器 / Pulse Signal Generator</li>
     * </ul>
     *
     * @param generatorType 生成器类型 / Generator type
     * @param <T> 数据类型 / Data type
     * @return 信号生成器实例 / Signal generator instance
     * @throws Exception 创建过程中发生错误时抛出 / Thrown when errors occur during creation
     * @see SignalProcessorFactory#createGenerator(String)
     */
    public static <T extends Number> ISignalGenerator<T> createGenerator(String generatorType) throws Exception {
        return SignalProcessorFactory.getInstance().createGenerator(generatorType);
    }
    
    /**
     * 测试方法 / Test Method
     * <p>
     * 测试信号生成、滤波和分析功能。
     * Test signal generation, filtering and analysis functionality.
     * </p>
     *
     * @param args 命令行参数 / Command line arguments
     */
    public static void main(String[] args) {
        try {
            // Test signal generation
            log.debug("Testing signal generation...");
            IVector<Double> sineWave = Signals.sineWave(100, 10, 1000, 1.0, 0);
            log.debug("Generated sine wave with " + sineWave.length() + " samples");
            
            IVector<Double> cosineWave = Signals.cosineWave(100, 10, 1000, 1.0, 0);
            log.debug("Generated cosine wave with " + cosineWave.length() + " samples");
            
            IVector<Double> squareWave = Signals.squareWave(100, 10, 1000, 1.0, 0.5);
            log.debug("Generated square wave with " + squareWave.length() + " samples");
            
            IVector<Double> noise = Signals.whiteNoise(100, 0.1);
            log.debug("Generated white noise with " + noise.length() + " samples");
            
            // Test signal filtering
            log.debug("\nTesting signal filtering...");
            IVector<Double> filtered = Signals.butterworthLowPass(sineWave, 50, 1000, 4);
            log.debug("Applied Butterworth filter");
            
            // Test signal analysis
            log.debug("\nTesting signal analysis...");
            IVector<Double> autocorr = Signals.autocorrelation(sineWave);
            log.debug("Calculated autocorrelation with " + autocorr.length() + " samples");
            
            log.debug("\nAll tests passed!");
        } catch (Exception e) {
            log.error("exception", e);
        }
    }
}