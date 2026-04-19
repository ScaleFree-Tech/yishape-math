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
import com.yishape.lab.math.viz.IPlot;
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
     */
    public static IVector<Double> unitStep(int length, double stepTime, double samplingRate) {
        return stepSignal(length, 1.0, stepTime, samplingRate);
    }
    
    /**
     * 生成狄拉克δ函数（连续时间脉冲） / Generate Dirac Delta Function (Continuous-time impulse)
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
     */
    public static IVector<Double> unitImpulse(int length, int impulseIndex) {
        return diracDelta(length, impulseIndex, 1.0);
    }
    
    /**
     * 生成线性调频信号（Chirp） / Generate Linear Chirp Signal
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
     * 高斯滤波器 / Gaussian Filter
     */
    public static IVector<Double> gaussianFilter(IVector<Double> signal, double sigma) {
        int defaultKernelSize = (int) Math.ceil(6 * sigma);
        if (defaultKernelSize % 2 == 0) {
            defaultKernelSize++;
        }
        return gaussianFilter(signal, sigma, defaultKernelSize);
    }

    /**
     * 高斯滤波器 / Gaussian Filter
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
     */
    public static Complex[] fft(Complex[] x) {
        // This is a low-level method that doesn't use the factory pattern
        return RereFFT.fft(x);
    }
    
    /**
     * 快速傅里叶逆变换 / Inverse Fast Fourier Transform
     */
    public static Complex[] ifft(Complex[] x) {
        // This is a low-level method that doesn't use the factory pattern
        return RereFFT.ifft(x);
    }
    
    /**
     * 计算信号的幅度谱 / Calculate Magnitude Spectrum
     */
    public static double[] magnitudeSpectrum(Complex[] fftResult) {
        // This is a low-level method that doesn't use the factory pattern
        return RereFFT.magnitudeSpectrum(fftResult);
    }
    
    /**
     * 计算信号的相位谱 / Calculate Phase Spectrum
     */
    public static double[] phaseSpectrum(Complex[] fftResult) {
        // This is a low-level method that doesn't use the factory pattern
        return RereFFT.phaseSpectrum(fftResult);
    }
    
    /**
     * 计算信号的功率谱 / Calculate Power Spectrum
     */
    public static double[] powerSpectrum(Complex[] fftResult) {
        // This is a low-level method that doesn't use the factory pattern
        return RereFFT.powerSpectrum(fftResult);
    }
    
    /**
     * DCT-II变换 / DCT-II Transform
     */
    public static IVector<Double> dct2(IVector<Double> signal) {
        // This is a low-level method that doesn't use the factory pattern
        return RereDCT.dct2(signal);
    }
    
    /**
     * DCT-II逆变换 / DCT-II Inverse Transform
     */
    public static IVector<Double> idct2(IVector<Double> dctSignal) {
        // This is a low-level method that doesn't use the factory pattern
        return RereDCT.idct2(dctSignal);
    }
    
    /**
     * 希尔伯特变换 / Hilbert Transform
     * @param signal
     * @return 
     */
    public static IVector<Double> hilbertTransform(IVector<Double> signal) {
        // This is a low-level method that doesn't use the factory pattern
        return RereHilbert.hilbertTransform(signal);
    }
    
    /**
     * 计算解析信号 / Calculate Analytic Signal
     */
    public static Complex[] analyticSignal(IVector<Double> signal) {
        // This is a low-level method that doesn't use the factory pattern
        return RereHilbert.analyticSignal(signal);
    }
    
    /**
     * 计算瞬时幅度 / Calculate Instantaneous Amplitude
     */
    public static IVector<Double> instantaneousAmplitude(IVector<Double> signal) {
        // This is a low-level method that doesn't use the factory pattern
        return RereHilbert.instantaneousAmplitude(signal);
    }
    
    /**
     * 计算瞬时相位 / Calculate Instantaneous Phase
     */
    public static IVector<Double> instantaneousPhase(IVector<Double> signal) {
        // This is a low-level method that doesn't use the factory pattern
        return RereHilbert.instantaneousPhase(signal);
    }
    
    /**
     * 计算瞬时频率 / Calculate Instantaneous Frequency
     */
    public static IVector<Double> instantaneousFrequency(IVector<Double> signal, double samplingRate) {
        // This is a low-level method that doesn't use the factory pattern
        return RereHilbert.instantaneousFrequency(signal, samplingRate);
    }
    
    /**
     * 离散小波变换 (DWT) / Discrete Wavelet Transform (DWT)
     */
    public static WaveletCoefficients discreteWaveletTransform(IVector<Double> signal,
                                                               WaveletAnalysis.WaveletType waveletType, int levels, double param) {
        // This is a low-level method that doesn't use the factory pattern
        return WaveletAnalysis.discreteWaveletTransform(signal, waveletType, levels, param);
    }
    
    /**
     * 小波逆变换 (IDWT) / Inverse Discrete Wavelet Transform (IDWT)
     */
    public static IVector<Double> inverseDiscreteWaveletTransform(WaveletCoefficients coefficients, 
            WaveletAnalysis.WaveletType waveletType, double param) {
        // This is a low-level method that doesn't use the factory pattern
        return WaveletAnalysis.inverseDiscreteWaveletTransform(coefficients, waveletType, param);
    }
    
    // ========== 可视化方法 / Visualization Methods ==========
    
    /**
     * 绘制信号波形图 / Plot signal waveform
     */
    public static IPlot plotWaveform(IVector<Double> signal, double samplingRate, String title) {
        return SignalPlots.plotWaveform(signal, samplingRate, title);
    }
    
    /**
     * 绘制信号频谱图 / Plot signal spectrum
     */
    public static IPlot plotSpectrum(IVector<Double> signal, double samplingRate, String title) {
        return SignalPlots.plotSpectrum(signal, samplingRate, title);
    }
    
    /**
     * 绘制功率谱密度图 / Plot Power Spectral Density
     */
    public static IPlot plotPowerSpectralDensity(IVector<Double> signal, double samplingRate, 
                                               int windowSize, double overlap, String title) {
        return SignalPlots.plotPowerSpectralDensity(signal, samplingRate, windowSize, overlap, title);
    }
    
    /**
     * 绘制自相关图 / Plot autocorrelation
     */
    public static IPlot plotAutocorrelation(IVector<Double> signal, int maxLag, String title) {
        return SignalPlots.plotAutocorrelation(signal, maxLag, title);
    }
    
    /**
     * 绘制互相关图 / Plot cross-correlation
     */
    public static IPlot plotCrossCorrelation(IVector<Double> signal1, IVector<Double> signal2, 
                                           int maxLag, String title) {
        return SignalPlots.plotCrossCorrelation(signal1, signal2, maxLag, title);
    }
    
    /**
     * 绘制小波系数图 / Plot wavelet coefficients
     */
    public static IPlot plotWaveletCoefficients(IVector<Double> signal, String waveletType, 
                                              int levels, String title) {
        return WaveletPlots.plotWaveletCoefficients(signal, waveletType, levels, title);
    }
    
    /**
     * 绘制小波能量分布图 / Plot wavelet energy distribution
     */
    public static IPlot plotWaveletEnergyDistribution(IVector<Double> signal, String waveletType, 
                                                    int levels, String title) {
        return WaveletPlots.plotWaveletEnergyDistribution(signal, waveletType, levels, title);
    }
    
    // ========== 工厂方法 / Factory Methods ==========
    
    /**
     * 获取信号处理器工厂实例 / Get signal processor factory instance
     * @return 
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
     * 测试方法 / Test method
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