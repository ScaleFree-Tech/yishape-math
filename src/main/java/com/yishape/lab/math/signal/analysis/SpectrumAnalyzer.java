package com.yishape.lab.math.signal.analysis;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.core.AbstractSignalProcessor;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.RereFFT;
import com.yishape.lab.math.signal.core.SignalProcessingException;
import com.yishape.lab.util.Tuple2;

/**
 * 频谱分析器实现类 / Spectrum Analyzer Implementation Class
 * <p>
 * 实现频谱分析功能，包括功率谱密度计算、自相关分析等。
 * Implements spectrum analysis functions including power spectral density calculation, autocorrelation analysis, etc.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class SpectrumAnalyzer extends AbstractSignalProcessor<Double> implements ISignalAnalyzer<Double> {
    
    /**
     * 默认构造函数 / Default constructor
     */
    public SpectrumAnalyzer() {
        super("Spectrum Analyzer", "1.0.0");
    }

    /**
     * 分析信号 / Analyze signal
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param analysisType 分析类型 / Analysis type
     * @param parameters 分析参数 / Analysis parameters
     * @param <R> 返回结果类型 / Return result type
     * @return 分析结果 / Analysis result
     * @throws SignalProcessingException 分析过程中发生错误时抛出 / Thrown when errors occur during analysis
     */
    @Override
    public <R> AnalysisResult<R> analyze(IVector<Double> signal, AnalysisType analysisType, AnalysisParameters parameters) throws SignalProcessingException {
        switch (analysisType) {
            case SPECTRUM:
                return (AnalysisResult<R>) analyzeSpectrum(signal, parameters);
            case POWER_SPECTRUM:
                return (AnalysisResult<R>) analyzePowerSpectrum(signal, parameters);
            case AUTOCORRELATION:
                return (AnalysisResult<R>) analyzeAutocorrelation(signal, parameters);
            case CROSS_CORRELATION:
                return (AnalysisResult<R>) analyzeCrossCorrelation(signal, parameters);
            case COHERENCE:
                return (AnalysisResult<R>) analyzeCoherence(signal, parameters);
            case ENVELOPE:
                return (AnalysisResult<R>) analyzeEnvelope(signal, parameters);
            case INSTANTANEOUS_FREQUENCY:
                return (AnalysisResult<R>) analyzeInstantaneousFrequency(signal, parameters);
            case TIME_FREQUENCY:
                return (AnalysisResult<R>) analyzeTimeFrequency(signal, parameters);
            case STATISTICAL:
                return (AnalysisResult<R>) analyzeStatistical(signal, parameters);
            case PEAK_DETECTION:
                return (AnalysisResult<R>) analyzePeakDetection(signal, parameters);
            case ENERGY:
                return (AnalysisResult<R>) analyzeEnergy(signal, parameters);
            case SNR:
                return (AnalysisResult<R>) analyzeSNR(signal, parameters);
            case THD:
                return (AnalysisResult<R>) analyzeTHD(signal, parameters);
            case CREST_FACTOR:
                return (AnalysisResult<R>) analyzeCrestFactor(signal, parameters);
            default:
                throw new SignalProcessingException("不支持的分析类型 / Unsupported analysis type: " + analysisType);
        }
    }
    
    /**
     * 频谱分析 / Spectrum analysis
     */
    private AnalysisResult<Tuple2<IVector<Double>, IVector<Double>>> analyzeSpectrum(IVector<Double> signal, AnalysisParameters parameters) throws SignalProcessingException {
        try {
            // 计算FFT / Calculate FFT
            double[] signalArray = signal.toDoubleArray();
            Complex[] complexSignal = new Complex[signalArray.length];
            for (int i = 0; i < signalArray.length; i++) {
                complexSignal[i] = new Complex(signalArray[i], 0);
            }
            // 零填充确保长度为2的幂
            Complex[] paddedSignal = RereFFT.zeroPadToPowerOfTwo(complexSignal);
            Complex[] fftResult = RereFFT.fft(paddedSignal);
            
            // 计算频率轴 / Calculate frequency axis
            int n = signal.length();
            double samplingRate = parameters.getSamplingRate();
            IVector<Double> frequencies = Linalg.linspace(0.0, samplingRate/2, n/2);
            
            // 计算幅度谱 / Calculate magnitude spectrum
            IVector<Double> magnitude = Linalg.zeros(n/2);
            for (int i = 0; i < n/2; i++) {
                magnitude.set(i, fftResult[i].magnitude());
            }
            
            Tuple2<IVector<Double>, IVector<Double>> result = new Tuple2<>(frequencies, magnitude);
            String[] resultNames = {"频率 / Frequency", "幅度 / Magnitude"};
            
            return new AnalysisResult<>(AnalysisType.SPECTRUM, result, resultNames, "频谱分析结果 / Spectrum analysis result", 0.95);
        } catch (Exception e) {
            throw new SignalProcessingException("频谱分析失败 / Spectrum analysis failed", e);
        }
    }
    
    /**
     * 功率谱分析 / Power spectrum analysis
     */
    private AnalysisResult<IVector<Double>> analyzePowerSpectrum(IVector<Double> signal, AnalysisParameters parameters) throws SignalProcessingException {
        try {
            // 计算FFT / Calculate FFT
            double[] signalArray = signal.toDoubleArray();
            Complex[] complexSignal = new Complex[signalArray.length];
            for (int i = 0; i < signalArray.length; i++) {
                complexSignal[i] = new Complex(signalArray[i], 0);
            }
            // 零填充确保长度为2的幂
            Complex[] paddedSignal = RereFFT.zeroPadToPowerOfTwo(complexSignal);
            Complex[] fftResult = RereFFT.fft(paddedSignal);
            
            // 计算功率谱 / Calculate power spectrum
            int n = signal.length();
            IVector<Double> powerSpectrum = Linalg.zeros(n/2);
            for (int i = 0; i < n/2; i++) {
                double magnitude = fftResult[i].magnitude();
                powerSpectrum.set(i, magnitude * magnitude);
            }
            
            String[] resultNames = {"功率谱 / Power Spectrum"};
            
            return new AnalysisResult<>(AnalysisType.POWER_SPECTRUM, powerSpectrum, resultNames, "功率谱分析结果 / Power spectrum analysis result", 0.95);
        } catch (Exception e) {
            throw new SignalProcessingException("功率谱分析失败 / Power spectrum analysis failed", e);
        }
    }
    
    /**
     * 自相关分析 / Autocorrelation analysis
     */
    private AnalysisResult<IVector<Double>> analyzeAutocorrelation(IVector<Double> signal, AnalysisParameters parameters) throws SignalProcessingException {
        try {
            int n = signal.length();
            IVector<Double> autocorr = Linalg.zeros(n);
            
            // 计算信号均值 / Calculate signal mean
            double mean = 0;
            for (int i = 0; i < n; i++) {
                mean += signal.get(i);
            }
            mean /= n;
            
            // 计算自相关 / Calculate autocorrelation
            for (int lag = 0; lag < n; lag++) {
                double sum = 0;
                int count = 0;
                for (int i = 0; i < n - lag; i++) {
                    sum += (signal.get(i) - mean) * (signal.get(i + lag) - mean);
                    count++;
                }
                autocorr.set(lag, sum / count);
            }
            
            String[] resultNames = {"自相关 / Autocorrelation"};
            
            return new AnalysisResult<>(AnalysisType.AUTOCORRELATION, autocorr, resultNames, "自相关分析结果 / Autocorrelation analysis result", 0.95);
        } catch (Exception e) {
            throw new SignalProcessingException("自相关分析失败 / Autocorrelation analysis failed", e);
        }
    }
    
    /**
     * 互相关分析 / Cross-correlation analysis
     * <p>需要两个信号，请使用 {@link #compareAnalyze} 方法。 / Requires two signals, use {@link #compareAnalyze} instead.</p>
     */
    private AnalysisResult<IVector<Double>> analyzeCrossCorrelation(IVector<Double> signal, AnalysisParameters parameters) throws SignalProcessingException {
        throw new SignalProcessingException("互相关分析需要两个信号，请使用 compareAnalyze 方法 / Cross-correlation requires two signals, use compareAnalyze");
    }
    
    /**
     * 相干性分析 / Coherence analysis
     */
    private AnalysisResult<IVector<Double>> analyzeCoherence(IVector<Double> signal, AnalysisParameters parameters) throws SignalProcessingException {
        throw new SignalProcessingException("相干性分析需要两个信号，请使用 compareAnalyze 方法 / Coherence requires two signals, use compareAnalyze");
    }
    
    /**
     * 包络分析 / Envelope analysis
     */
    private AnalysisResult<IVector<Double>> analyzeEnvelope(IVector<Double> signal, AnalysisParameters parameters) throws SignalProcessingException {
        try {
            int n = signal.length();
            IVector<Double> envelope = Linalg.zeros(n);
            
            // 简单的包络检测：取绝对值后低通滤波 / Simple envelope detection: take absolute value then low-pass filter
            for (int i = 0; i < n; i++) {
                envelope.set(i, Math.abs(signal.get(i)));
            }
            
            // 简单的移动平均滤波 / Simple moving average filtering
            int windowSize = Math.max(1, n / 50);
            for (int i = 0; i < n; i++) {
                double sum = 0;
                int count = 0;
                for (int j = Math.max(0, i - windowSize/2); j <= Math.min(n-1, i + windowSize/2); j++) {
                    sum += envelope.get(j);
                    count++;
                }
                envelope.set(i, sum / count);
            }
            
            String[] resultNames = {"包络 / Envelope"};
            
            return new AnalysisResult<>(AnalysisType.ENVELOPE, envelope, resultNames, "包络分析结果 / Envelope analysis result", 0.95);
        } catch (Exception e) {
            throw new SignalProcessingException("包络分析失败 / Envelope analysis failed", e);
        }
    }
    
    /**
     * 瞬时频率分析 / Instantaneous frequency analysis
     * <p>通过 Hilbert 变换计算解析信号，从解析信号的相位导数得到瞬时频率。</p>
     */
    private AnalysisResult<IVector<Double>> analyzeInstantaneousFrequency(IVector<Double> signal, AnalysisParameters parameters) throws SignalProcessingException {
        try {
            int n = signal.length();
            double samplingRate = parameters.getSamplingRate();
            double[] signalArray = signal.toDoubleArray();

            // Build complex signal and compute FFT
            Complex[] complexSignal = new Complex[signalArray.length];
            for (int i = 0; i < signalArray.length; i++) {
                complexSignal[i] = new Complex(signalArray[i], 0);
            }
            Complex[] paddedSignal = RereFFT.zeroPadToPowerOfTwo(complexSignal);
            int fftLen = paddedSignal.length;
            Complex[] fftResult = RereFFT.fft(paddedSignal);

            // Hilbert transform: zero negative frequencies, double positive
            Complex[] hilbert = new Complex[fftLen];
            hilbert[0] = fftResult[0];
            for (int i = 1; i < fftLen / 2; i++) {
                hilbert[i] = new Complex(2.0 * fftResult[i].real, 2.0 * fftResult[i].imag);
            }
            for (int i = fftLen / 2; i < fftLen; i++) {
                hilbert[i] = new Complex(0, 0);
            }

            Complex[] analyticSignal = RereFFT.ifft(hilbert);

            // Compute unwrapped phase
            double[] phase = new double[n];
            for (int i = 0; i < n; i++) {
                phase[i] = Math.atan2(analyticSignal[i].imag, analyticSignal[i].real);
            }

            // Unwrap phase
            double[] unwrapped = new double[n];
            unwrapped[0] = phase[0];
            double cumulativeShift = 0;
            for (int i = 1; i < n; i++) {
                double diff = phase[i] - phase[i - 1];
                if (diff > Math.PI) cumulativeShift -= 2 * Math.PI;
                else if (diff < -Math.PI) cumulativeShift += 2 * Math.PI;
                unwrapped[i] = phase[i] + cumulativeShift;
            }

            // Instantaneous frequency from phase derivative
            IVector<Double> frequency = Linalg.zeros(n);
            for (int i = 1; i < n; i++) {
                double freq = (unwrapped[i] - unwrapped[i - 1]) * samplingRate / (2.0 * Math.PI);
                frequency.set(i, freq);
            }
            if (n > 1) {
                frequency.set(0, frequency.get(1));
            }

            String[] resultNames = {"瞬时频率 / Instantaneous Frequency"};
            return new AnalysisResult<>(AnalysisType.INSTANTANEOUS_FREQUENCY, frequency, resultNames,
                "瞬时频率分析结果 / Instantaneous frequency analysis result", 0.95);
        } catch (Exception e) {
            throw new SignalProcessingException("瞬时频率分析失败 / Instantaneous frequency analysis failed", e);
        }
    }
    
    /**
     * 时频分析 / Time-frequency analysis
     */
    private AnalysisResult<IVector<Double>> analyzeTimeFrequency(IVector<Double> signal, AnalysisParameters parameters) throws SignalProcessingException {
        try {
            // 简化的时频分析：返回信号的短时能量
            int n = signal.length();
            IVector<Double> tfAnalysis = Linalg.zeros(n);
            
            int windowSize = Math.min(parameters.getWindowSize(), n);
            for (int i = 0; i < n; i++) {
                double energy = 0;
                int count = 0;
                for (int j = Math.max(0, i - windowSize/2); j <= Math.min(n-1, i + windowSize/2); j++) {
                    energy += signal.get(j) * signal.get(j);
                    count++;
                }
                tfAnalysis.set(i, energy / count);
            }
            
            String[] resultNames = {"时频特征 / Time-Frequency Features"};
            
            return new AnalysisResult<>(AnalysisType.TIME_FREQUENCY, tfAnalysis, resultNames, "时频分析结果 / Time-frequency analysis result", 0.95);
        } catch (Exception e) {
            throw new SignalProcessingException("时频分析失败 / Time-frequency analysis failed", e);
        }
    }
    
    /**
     * 统计分析 / Statistical analysis
     */
    private AnalysisResult<IVector<Double>> analyzeStatistical(IVector<Double> signal, AnalysisParameters parameters) throws SignalProcessingException {
        try {
            int n = signal.length();
            
            // 计算基本统计特征
            double mean = 0, variance = 0, min = signal.get(0), max = signal.get(0);
            for (int i = 0; i < n; i++) {
                double value = signal.get(i);
                mean += value;
                if (value < min) min = value;
                if (value > max) max = value;
            }
            mean /= n;
            
            for (int i = 0; i < n; i++) {
                double diff = signal.get(i) - mean;
                variance += diff * diff;
            }
            variance /= n;
            double std = Math.sqrt(variance);
            
            // 创建特征向量 [均值, 方差, 标准差, 最小值, 最大值, 峰值因子]
            IVector<Double> features = Linalg.vector(new double[]{mean, variance, std, min, max, max/Math.abs(min)});
            
            String[] resultNames = {"统计特征 / Statistical Features"};
            
            return new AnalysisResult<>(AnalysisType.STATISTICAL, features, resultNames, "统计分析结果 / Statistical analysis result", 0.95);
        } catch (Exception e) {
            throw new SignalProcessingException("统计分析失败 / Statistical analysis failed", e);
        }
    }
    
    /**
     * 峰值检测 / Peak detection
     */
    private AnalysisResult<IVector<Double>> analyzePeakDetection(IVector<Double> signal, AnalysisParameters parameters) throws SignalProcessingException {
        try {
            int n = signal.length();
            double threshold = parameters.getPeakThreshold();
            double minDistance = parameters.getMinPeakDistance();
            
            // 简化的峰值检测
            IVector<Double> peaks = Linalg.zeros(Math.min(10, n/10)); // 预分配空间
            
            int peakCount = 0;
            for (int i = 1; i < n - 1; i++) {
                if (signal.get(i) > signal.get(i-1) && signal.get(i) > signal.get(i+1) && 
                    Math.abs(signal.get(i)) > threshold) {
                    if (peakCount < peaks.length()) {
                        peaks.set(peakCount, (double) i); // 存储峰值位置
                        peakCount++;
                    }
                }
            }
            
            // 调整向量大小
            IVector<Double> resultPeaks = Linalg.zeros(peakCount);
            for (int i = 0; i < peakCount; i++) {
                resultPeaks.set(i, peaks.get(i));
            }
            
            String[] resultNames = {"峰值位置 / Peak Positions"};
            
            return new AnalysisResult<>(AnalysisType.PEAK_DETECTION, resultPeaks, resultNames, "峰值检测结果 / Peak detection result", 0.95);
        } catch (Exception e) {
            throw new SignalProcessingException("峰值检测失败 / Peak detection failed", e);
        }
    }
    
    /**
     * 能量分析 / Energy analysis
     */
    private AnalysisResult<Double> analyzeEnergy(IVector<Double> signal, AnalysisParameters parameters) throws SignalProcessingException {
        try {
            int n = signal.length();
            double energy = 0;
            
            for (int i = 0; i < n; i++) {
                energy += signal.get(i) * signal.get(i);
            }
            
            String[] resultNames = {"信号能量 / Signal Energy"};
            
            return new AnalysisResult<>(AnalysisType.ENERGY, energy, resultNames, "能量分析结果 / Energy analysis result", 0.95);
        } catch (Exception e) {
            throw new SignalProcessingException("能量分析失败 / Energy analysis failed", e);
        }
    }
    
    /**
     * 信噪比分析 / SNR analysis
     */
    private AnalysisResult<Double> analyzeSNR(IVector<Double> signal, AnalysisParameters parameters) throws SignalProcessingException {
        try {
            int n = signal.length();
            // Compute total signal power
            double totalPower = 0;
            for (int i = 0; i < n; i++) {
                double v = signal.get(i);
                totalPower += v * v;
            }
            totalPower /= n;

            // Estimate noise floor from high-frequency FFT bins
            double[] signalArray = signal.toDoubleArray();
            Complex[] complexSignal = new Complex[signalArray.length];
            for (int i = 0; i < signalArray.length; i++) {
                complexSignal[i] = new Complex(signalArray[i], 0);
            }
            Complex[] paddedSignal = RereFFT.zeroPadToPowerOfTwo(complexSignal);
            Complex[] fftResult = RereFFT.fft(paddedSignal);
            int fftLen = fftResult.length;

            double noisePower = 0;
            int noiseBins = 0;
            for (int i = fftLen / 4; i < fftLen / 2; i++) {
                double mag = fftResult[i].magnitude();
                noisePower += mag * mag;
                noiseBins++;
            }
            noisePower = noiseBins > 0 ? noisePower / (noiseBins * fftLen) : 0;
            double signalPower = Math.max(0, totalPower - noisePower);

            double snr = noisePower > 0 ? 10.0 * Math.log10(signalPower / noisePower) : Double.POSITIVE_INFINITY;

            String[] resultNames = {"信噪比 / SNR (dB)"};

            return new AnalysisResult<>(AnalysisType.SNR, snr, resultNames, "信噪比分析结果 / SNR analysis result", 0.95);
        } catch (Exception e) {
            throw new SignalProcessingException("信噪比分析失败 / SNR analysis failed", e);
        }
    }

    /**
     * 总谐波失真分析 / THD analysis
     */
    private AnalysisResult<Double> analyzeTHD(IVector<Double> signal, AnalysisParameters parameters) throws SignalProcessingException {
        try {
            int n = signal.length();
            double[] signalArray = signal.toDoubleArray();
            Complex[] complexSignal = new Complex[signalArray.length];
            for (int i = 0; i < signalArray.length; i++) {
                complexSignal[i] = new Complex(signalArray[i], 0);
            }
            Complex[] paddedSignal = RereFFT.zeroPadToPowerOfTwo(complexSignal);
            Complex[] fftResult = RereFFT.fft(paddedSignal);
            int fftLen = fftResult.length;
            double samplingRate = parameters.getSamplingRate();

            // Find fundamental frequency (largest peak, excluding DC)
            int fundBin = 1;
            double fundMag = 0;
            for (int i = 1; i < fftLen / 2; i++) {
                double mag = fftResult[i].magnitude();
                if (mag > fundMag) {
                    fundMag = mag;
                    fundBin = i;
                }
            }

            // Sum harmonic magnitudes squared (harmonics 2 through 10)
            double harmonicPower = 0;
            int maxHarmonic = Math.min(10, (fftLen / 2) / fundBin);
            for (int h = 2; h <= maxHarmonic; h++) {
                int hBin = h * fundBin;
                if (hBin < fftLen / 2) {
                    double mag = fftResult[hBin].magnitude();
                    harmonicPower += mag * mag;
                }
            }

            double thd = fundMag > 0 ? Math.sqrt(harmonicPower) / fundMag : 0;
            if (thd > 1.0) thd = 1.0; // cap at 100%

            String[] resultNames = {"总谐波失真 / THD"};

            return new AnalysisResult<>(AnalysisType.THD, thd, resultNames, "总谐波失真分析结果 / THD analysis result", 0.95);
        } catch (Exception e) {
            throw new SignalProcessingException("总谐波失真分析失败 / THD analysis failed", e);
        }
    }
    
    /**
     * 峰值因子分析 / Crest factor analysis
     */
    private AnalysisResult<Double> analyzeCrestFactor(IVector<Double> signal, AnalysisParameters parameters) throws SignalProcessingException {
        try {
            int n = signal.length();
            double max = Math.abs(signal.get(0));
            double rms = 0;
            
            for (int i = 0; i < n; i++) {
                double absValue = Math.abs(signal.get(i));
                if (absValue > max) max = absValue;
                rms += signal.get(i) * signal.get(i);
            }
            rms = Math.sqrt(rms / n);
            
            double crestFactor = (rms > 0) ? max / rms : 0;
            
            String[] resultNames = {"峰值因子 / Crest Factor"};
            
            return new AnalysisResult<>(AnalysisType.CREST_FACTOR, crestFactor, resultNames, "峰值因子分析结果 / Crest factor analysis result", 0.95);
        } catch (Exception e) {
            throw new SignalProcessingException("峰值因子分析失败 / Crest factor analysis failed", e);
        }
    }
    
    /**
     * 批量分析信号 / Batch analyze signals
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param analysisTypes 分析类型数组 / Analysis type array
     * @param parameters 分析参数 / Analysis parameters
     * @return 分析结果数组 / Array of analysis results
     * @throws SignalProcessingException 分析过程中发生错误时抛出 / Thrown when errors occur during analysis
     */
    @Override
    public AnalysisResult<?>[] batchAnalyze(IVector<Double> signal, AnalysisType[] analysisTypes, AnalysisParameters parameters) throws SignalProcessingException {
        AnalysisResult<?>[] results = new AnalysisResult[analysisTypes.length];
        for (int i = 0; i < analysisTypes.length; i++) {
            results[i] = analyze(signal, analysisTypes[i], parameters);
        }
        return results;
    }
    
    /**
     * 比较分析两个信号 / Compare and analyze two signals
     *
     * @param signal1 第一个信号向量 / First signal vector
     * @param signal2 第二个信号向量 / Second signal vector
     * @param analysisType 分析类型 / Analysis type
     * @param parameters 分析参数 / Analysis parameters
     * @param <R> 返回结果类型 / Return result type
     * @return 比较分析结果 / Comparative analysis result
     * @throws SignalProcessingException 分析过程中发生错误时抛出 / Thrown when errors occur during analysis
     */
    @Override
    public <R> AnalysisResult<R> compareAnalyze(IVector<Double> signal1, IVector<Double> signal2, AnalysisType analysisType, AnalysisParameters parameters) throws SignalProcessingException {
        // 简化的比较分析 / Simplified comparative analysis
        throw new UnsupportedOperationException("比较分析功能尚未实现 / Comparative analysis not yet implemented");
    }
    
    /**
     * 获取支持的信号分析类型列表 / Get list of supported signal analysis types
     *
     * @return 支持的分析类型数组 / Array of supported analysis types
     */
    @Override
    public AnalysisType[] getSupportedAnalysisTypes() {
        return new AnalysisType[] {
            AnalysisType.SPECTRUM,
            AnalysisType.POWER_SPECTRUM,
            AnalysisType.AUTOCORRELATION,
            AnalysisType.CROSS_CORRELATION,
            AnalysisType.COHERENCE,
            AnalysisType.ENVELOPE,
            AnalysisType.INSTANTANEOUS_FREQUENCY,
            AnalysisType.TIME_FREQUENCY,
            AnalysisType.STATISTICAL,
            AnalysisType.PEAK_DETECTION,
            AnalysisType.ENERGY,
            AnalysisType.SNR,
            AnalysisType.THD,
            AnalysisType.CREST_FACTOR
        };
    }
    
    /**
     * 验证分析参数是否有效 / Validate if analysis parameters are valid
     *
     * @param analysisType 分析类型 / Analysis type
     * @param parameters 分析参数 / Analysis parameters
     * @return 参数是否有效 / Whether parameters are valid
     */
    @Override
    public boolean validateParameters(AnalysisType analysisType, AnalysisParameters parameters) {
        // 基本验证 / Basic validation
        if (parameters.getSamplingRate() <= 0) {
            return false;
        }
        if (parameters.getWindowSize() <= 0) {
            return false;
        }
        return true;
    }
    
    /**
     * 获取推荐的分析参数 / Get recommended analysis parameters
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param analysisType 分析类型 / Analysis type
     * @return 推荐的分析参数 / Recommended analysis parameters
     */
    @Override
    public AnalysisParameters getRecommendedParameters(IVector<Double> signal, AnalysisType analysisType) {
        // 返回推荐参数 / Return recommended parameters
        return new AnalysisParameters()
            .samplingRate(1000.0)
            .windowSize(Math.min(256, signal.length()))
            .overlap(0.5)
            .nfft(512);
    }
    
    /**
     * 执行默认的信号处理流程 / Execute default signal processing flow
     * <p>
     * 默认处理：计算功率谱 / Default processing: calculate power spectrum
     * </p>
     *
     * @param input 输入信号向量 / Input signal vector
     * @return 处理后的信号向量 / Processed signal vector
     * @throws SignalProcessingException 处理过程中发生错误时抛出 / Thrown when errors occur during processing
     */
    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        // 默认处理：计算功率谱 / Default processing: calculate power spectrum
        AnalysisResult<IVector<Double>> result = analyzePowerSpectrum(input, new AnalysisParameters());
        return result.getResult();
    }
    
    /**
     * 创建当前对象的副本 / Create a copy of the current object
     *
     * @return SpectrumAnalyzer 实例副本 / Copy of SpectrumAnalyzer instance
     */
    @Override
    public SpectrumAnalyzer clone() {
        return new SpectrumAnalyzer();
    }
}