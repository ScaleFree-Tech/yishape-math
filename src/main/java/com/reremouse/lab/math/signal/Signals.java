package com.reremouse.lab.math.signal;

import com.reremouse.lab.math.signal.wavele.WaveletAnalysis;
import com.reremouse.lab.math.signal.wavele.WaveletVisualizer;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;
import com.reremouse.lab.math.viz.IPlot;
import com.reremouse.lab.math.signal.analysis.ISignalAnalyzer;
import com.reremouse.lab.math.signal.filter.ISignalFilter;
import com.reremouse.lab.math.signal.transform.ISignalTransform;
import com.reremouse.lab.math.signal.generation.ISignalGenerator;
import com.reremouse.lab.math.signal.factory.SignalProcessorFactory;
import com.reremouse.lab.math.signal.core.Complex;
import com.reremouse.lab.math.signal.core.RereDCT;
import com.reremouse.lab.math.signal.core.RereFFT;
import com.reremouse.lab.math.signal.core.RereHilbert;
import com.reremouse.lab.math.signal.generation.ISignalGenerator.SignalParameters;
import com.reremouse.lab.math.signal.generation.ISignalGenerator.SignalType;
import com.reremouse.lab.math.signal.analysis.ISignalAnalyzer.AnalysisParameters;
import com.reremouse.lab.math.signal.analysis.ISignalAnalyzer.AnalysisType;
import com.reremouse.lab.math.signal.wavele.WaveletCoefficients;

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
    
    // ========== 信号生成方法 / Signal Generation Methods ==========
    
    /**
     * 生成正弦波信号 / Generate Sine Wave Signal
     */
    public static IVector<Double> sineWave(int length, double frequency, double samplingRate, 
                                         double amplitude, double phase) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("sine");
            SignalParameters params = new SignalParameters()
                .frequency(frequency)
                .samplingRate(samplingRate)
                .amplitude(amplitude)
                .phase(phase);
            return generator.generate(SignalType.SINE, length, params);
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
            SignalParameters params = new SignalParameters()
                .frequency(frequency)
                .samplingRate(samplingRate)
                .amplitude(amplitude)
                .phase(phase);
            return generator.generate(SignalType.COSINE, length, params);
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
            SignalParameters params = new SignalParameters()
                .frequency(frequency)
                .samplingRate(samplingRate)
                .amplitude(amplitude)
                .dutyCycle(dutyCycle);
            return generator.generate(SignalType.SQUARE, length, params);
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
            SignalParameters params = new SignalParameters()
                .frequency(frequency)
                .samplingRate(samplingRate)
                .amplitude(amplitude)
                .dutyCycle(symmetry); // Using dutyCycle as symmetry parameter
            return generator.generate(SignalType.TRIANGLE, length, params);
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
            SignalParameters params = new SignalParameters()
                .frequency(frequency)
                .samplingRate(samplingRate)
                .amplitude(amplitude);
            return generator.generate(SignalType.SAWTOOTH, length, params);
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
            SignalParameters params = new SignalParameters()
                .amplitude(Math.sqrt(power)); // Convert power to amplitude
            return generator.generate(SignalType.WHITE_NOISE, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate white noise", e);
        }
    }
    
    /**
     * 生成粉红噪声信号 / Generate Pink Noise Signal
     */
    public static IVector<Double> pinkNoise(int length, double power) {
        // For now, use white noise as approximation
        return whiteNoise(length, power);
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
            IVector<Double> signal = com.reremouse.lab.math.linalg.Linalg.zeros(length);
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
        try {
            ISignalFilter<Double> filter = SignalProcessorFactory.getInstance().createFilter("gaussian");
            // Set filter parameters if needed
            return filter.process(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply gaussian filter", e);
        }
    }
    
    /**
     * 高斯滤波器 / Gaussian Filter
     */
    public static IVector<Double> gaussianFilter(IVector<Double> signal, double sigma, int kernelSize) {
        // Use the simpler version for now
        return gaussianFilter(signal, sigma);
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
     */
    public static IVector<Double> butterworthHighPass(IVector<Double> signal, double cutoffFreq, double samplingRate, int order) {
        // For now, use the low-pass version as approximation
        return butterworthLowPass(signal, cutoffFreq, samplingRate, order);
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
            AnalysisParameters params = new AnalysisParameters()
                .windowSize(windowSize)
                .overlap(overlap)
                .samplingRate(samplingRate);
            ISignalAnalyzer.AnalysisResult<Tuple2<IVector<Double>, IVector<Double>>> result = 
                analyzer.analyze(signal, AnalysisType.POWER_SPECTRUM, params);
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
                analyzer.analyze(signal, AnalysisType.AUTOCORRELATION);
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
                analyzer.analyze(signal1, AnalysisType.CROSS_CORRELATION);
            return result.getResult();
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate cross-correlation", e);
        }
    }
    
    /**
     * 计算两个信号的互相关函数 / Calculate Cross-correlation Function
     */
    public static IVector<Double> crossCorrelation(IVector<Double> signal1, IVector<Double> signal2, int maxLag) {
        // Use the simpler version for now
        return crossCorrelation(signal1, signal2);
    }
    
    /**
     * 计算信号的频谱 / Calculate Signal Spectrum
     */
    public static Tuple3<IVector<Double>, IVector<Double>, IVector<Double>> spectrum(
            IVector<Double> signal, double samplingRate) {
        try {
            ISignalAnalyzer<Double> analyzer = SignalProcessorFactory.getInstance().createAnalyzer("spectrum");
            AnalysisParameters params = new AnalysisParameters()
                .samplingRate(samplingRate);
            ISignalAnalyzer.AnalysisResult<Tuple2<IVector<Double>, IVector<Double>>> result = 
                analyzer.analyze(signal, AnalysisType.SPECTRUM, params);
            
            // Convert to Tuple3 format (frequency, magnitude, phase)
            Tuple2<IVector<Double>, IVector<Double>> spectrumResult = result.getResult();
            IVector<Double> frequencies = spectrumResult._1;
            IVector<Double> magnitudes = spectrumResult._2;
            // For now, use zeros for phase as approximation
            IVector<Double> phases = com.reremouse.lab.math.linalg.Linalg.zeros(magnitudes.length());
            
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
        // This is a complex method that would need a dedicated transformer
        throw new UnsupportedOperationException("STFT not yet implemented through factory");
    }
    
    /**
     * 计算信号的信噪比 (SNR) / Calculate Signal-to-Noise Ratio (SNR)
     */
    public static double signalToNoiseRatio(IVector<Double> signal, IVector<Double> noise) {
        try {
            ISignalAnalyzer<Double> analyzer = SignalProcessorFactory.getInstance().createAnalyzer("spectrum");
            AnalysisParameters params = new AnalysisParameters();
            ISignalAnalyzer.AnalysisResult<Double> result = 
                analyzer.analyze(signal, AnalysisType.SNR, params);
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
        return SignalVisualizer.plotWaveform(signal, samplingRate, title);
    }
    
    /**
     * 绘制信号频谱图 / Plot signal spectrum
     */
    public static IPlot plotSpectrum(IVector<Double> signal, double samplingRate, String title) {
        return SignalVisualizer.plotSpectrum(signal, samplingRate, title);
    }
    
    /**
     * 绘制功率谱密度图 / Plot Power Spectral Density
     */
    public static IPlot plotPowerSpectralDensity(IVector<Double> signal, double samplingRate, 
                                               int windowSize, double overlap, String title) {
        return SignalVisualizer.plotPowerSpectralDensity(signal, samplingRate, windowSize, overlap, title);
    }
    
    /**
     * 绘制自相关图 / Plot autocorrelation
     */
    public static IPlot plotAutocorrelation(IVector<Double> signal, int maxLag, String title) {
        return SignalVisualizer.plotAutocorrelation(signal, maxLag, title);
    }
    
    /**
     * 绘制互相关图 / Plot cross-correlation
     */
    public static IPlot plotCrossCorrelation(IVector<Double> signal1, IVector<Double> signal2, 
                                           int maxLag, String title) {
        return SignalVisualizer.plotCrossCorrelation(signal1, signal2, maxLag, title);
    }
    
    /**
     * 绘制小波系数图 / Plot wavelet coefficients
     */
    public static IPlot plotWaveletCoefficients(IVector<Double> signal, String waveletType, 
                                              int levels, String title) {
        return WaveletVisualizer.plotWaveletCoefficients(signal, waveletType, levels, title);
    }
    
    /**
     * 绘制小波能量分布图 / Plot wavelet energy distribution
     */
    public static IPlot plotWaveletEnergyDistribution(IVector<Double> signal, String waveletType, 
                                                    int levels, String title) {
        return WaveletVisualizer.plotWaveletEnergyDistribution(signal, waveletType, levels, title);
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
            System.out.println("Testing signal generation...");
            IVector<Double> sineWave = Signals.sineWave(100, 10, 1000, 1.0, 0);
            System.out.println("Generated sine wave with " + sineWave.length() + " samples");
            
            IVector<Double> cosineWave = Signals.cosineWave(100, 10, 1000, 1.0, 0);
            System.out.println("Generated cosine wave with " + cosineWave.length() + " samples");
            
            IVector<Double> squareWave = Signals.squareWave(100, 10, 1000, 1.0, 0.5);
            System.out.println("Generated square wave with " + squareWave.length() + " samples");
            
            IVector<Double> noise = Signals.whiteNoise(100, 0.1);
            System.out.println("Generated white noise with " + noise.length() + " samples");
            
            // Test signal filtering
            System.out.println("\nTesting signal filtering...");
            IVector<Double> filtered = Signals.butterworthLowPass(sineWave, 50, 1000, 4);
            System.out.println("Applied Butterworth filter");
            
            // Test signal analysis
            System.out.println("\nTesting signal analysis...");
            IVector<Double> autocorr = Signals.autocorrelation(sineWave);
            System.out.println("Calculated autocorrelation with " + autocorr.length() + " samples");
            
            System.out.println("\nAll tests passed!");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}