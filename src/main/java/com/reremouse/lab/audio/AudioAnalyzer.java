package com.reremouse.lab.audio;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.signal.SignalAnalysis;
import com.reremouse.lab.math.signal.RereFFT;
import com.reremouse.lab.math.signal.Complex;
import com.reremouse.lab.util.Tuple2;

/**
 * 音频分析器类 / Audio Analyzer Class
 * <p>
 * 提供音频分析功能，包括频谱分析、特征提取、音调检测等。
 * 使用项目现有的signal包和linalg包功能进行分析。
 * </p>
 * <p>
 * Provides audio analysis functionality including spectral analysis, feature extraction, pitch detection, etc.
 * Uses existing signal and linalg package functionality for analysis.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AudioAnalyzer {
    
    /**
     * 计算音频的频谱 / Calculate audio spectrum
     * <p>
     * 使用FFT计算音频的频谱，返回频率和幅度信息。
     * Use FFT to calculate audio spectrum, returning frequency and magnitude information.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 包含频率和幅度的元组 / Tuple containing frequencies and magnitudes
     */
    public static Tuple2<IVector<Double>, IVector<Double>> calculateSpectrum(AudioData audioData) {
        return calculateSpectrum(audioData, 1024, 0.5);
    }
    
    /**
     * 计算音频的频谱（指定参数） / Calculate audio spectrum (with specified parameters)
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param windowSize 窗函数大小 / Window size
     * @param overlap 重叠比例 / Overlap ratio
     * @return 包含频率和幅度的元组 / Tuple containing frequencies and magnitudes
     */
    public static Tuple2<IVector<Double>, IVector<Double>> calculateSpectrum(AudioData audioData, int windowSize, double overlap) {
        // 使用单声道数据进行分析 / Use mono data for analysis
        IVector<Double> samples = audioData.isMono() ? 
            audioData.getSamples() : 
            AudioProcessor.stereoToMono(audioData).getSamples();
        
        // 使用Welch方法计算功率谱密度 / Use Welch's method to calculate power spectral density
        Tuple2<IVector<Double>, IVector<Double>> psdResult = SignalAnalysis.powerSpectralDensity(
            samples, windowSize, overlap, audioData.getSampleRate());
        
        IVector<Double> frequencies = psdResult._1;
        IVector<Double> psd = psdResult._2;
        
        // 计算幅度谱 / Calculate magnitude spectrum
        IVector<Double> magnitudes = psd.apply(x -> Math.sqrt(x));
        
        return new Tuple2<>(frequencies, magnitudes);
    }
    
    /**
     * 计算音频的短时傅里叶变换 (STFT) / Calculate Short-Time Fourier Transform (STFT)
     * <p>
     * 将音频信号分解为时频表示，用于分析音频的时变特性。
     * Decompose audio signal into time-frequency representation for analyzing time-varying characteristics.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param windowSize 窗函数大小 / Window size
     * @param hopSize 跳跃大小 / Hop size
     * @return STFT结果矩阵，行为频率，列为时间 / STFT result matrix, rows are frequencies, columns are time
     */
    public static IMatrix<Double> calculateSTFT(AudioData audioData, int windowSize, int hopSize) {
        IVector<Double> samples = audioData.isMono() ? 
            audioData.getSamples() : 
            AudioProcessor.stereoToMono(audioData).getSamples();
        
        int numFrames = (samples.length() - windowSize) / hopSize + 1;
        int numBins = windowSize / 2 + 1;
        
        IMatrix<Double> stftMatrix = Linalg.zeros(numBins, numFrames);
        
        // 创建汉宁窗 / Create Hanning window
        IVector<Double> window = createHanningWindow(windowSize);
        
        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopSize;
            int end = Math.min(start + windowSize, samples.length());
            
            // 提取窗口信号 / Extract window signal
            IVector<Double> windowSignal = samples.slice(start, end);
            
            // 应用窗函数 / Apply window function
            IVector<Double> windowedSignal = windowSignal.multiply(window);
            
            // 计算FFT / Calculate FFT
            Complex[] fftResult = RereFFT.fft(convertToComplex(windowedSignal));
            
            // 存储幅度谱 / Store magnitude spectrum
            for (int bin = 0; bin < numBins; bin++) {
                double magnitude = fftResult[bin].magnitude();
                stftMatrix.set(bin, frame, magnitude);
            }
        }
        
        return stftMatrix;
    }
    
    /**
     * 提取音频特征 / Extract audio features
     * <p>
     * 提取音频的多种特征，包括MFCC、频谱质心、带宽等。
     * Extract various audio features including MFCC, spectral centroid, bandwidth, etc.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 音频特征对象 / Audio features object
     */
    public static AudioFeatures extractFeatures(AudioData audioData) {
        IVector<Double> samples = audioData.isMono() ? 
            audioData.getSamples() : 
            AudioProcessor.stereoToMono(audioData).getSamples();
        
        // 计算频谱 / Calculate spectrum
        var spectrumResult = calculateSpectrum(audioData);
        IVector<Double> frequencies = spectrumResult._1;
        IVector<Double> magnitudes = spectrumResult._2;
        
        // 提取各种特征 / Extract various features
        double spectralCentroid = calculateSpectralCentroid(frequencies, magnitudes);
        double spectralBandwidth = calculateSpectralBandwidth(frequencies, magnitudes, spectralCentroid);
        double spectralRolloff = calculateSpectralRolloff(frequencies, magnitudes);
        double zeroCrossingRate = calculateZeroCrossingRate(samples);
        double[] mfcc = calculateMFCC(samples, audioData.getSampleRate());
        double[] spectralContrast = calculateSpectralContrast(magnitudes);
        
        return new AudioFeatures(
            spectralCentroid, spectralBandwidth, spectralRolloff, zeroCrossingRate,
            mfcc, spectralContrast, audioData.getSampleRate()
        );
    }
    
    /**
     * 检测音调 / Detect pitch
     * <p>
     * 使用自相关方法检测音频的主频率（音调）。
     * Use autocorrelation method to detect main frequency (pitch) of audio.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 音调频率 (Hz) / Pitch frequency (Hz)
     */
    public static double detectPitch(AudioData audioData) {
        IVector<Double> samples = audioData.isMono() ? 
            audioData.getSamples() : 
            AudioProcessor.stereoToMono(audioData).getSamples();
        
        // 使用自相关方法 / Use autocorrelation method
        IVector<Double> autocorr = calculateAutocorrelation(samples);
        
        // 寻找峰值 / Find peaks
        double maxCorr = 0;
        int maxLag = 0;
        
        // 在合理的音调范围内搜索 / Search in reasonable pitch range
        int minLag = (int) (audioData.getSampleRate() / 800); // 最高音调 / Highest pitch
        int maxLagLimit = (int) (audioData.getSampleRate() / 80); // 最低音调 / Lowest pitch
        
        for (int lag = minLag; lag < Math.min(maxLagLimit, autocorr.length()); lag++) {
            if (autocorr.get(lag) > maxCorr) {
                maxCorr = autocorr.get(lag);
                maxLag = lag;
            }
        }
        
        return maxLag > 0 ? audioData.getSampleRate() / maxLag : 0;
    }
    
    /**
     * 计算频谱质心 / Calculate spectral centroid
     *
     * @param frequencies 频率向量 / Frequency vector
     * @param magnitudes 幅度向量 / Magnitude vector
     * @return 频谱质心 / Spectral centroid
     */
    private static double calculateSpectralCentroid(IVector<Double> frequencies, IVector<Double> magnitudes) {
        double weightedSum = 0;
        double magnitudeSum = 0;
        
        for (int i = 0; i < frequencies.length(); i++) {
            double freq = frequencies.get(i);
            double mag = magnitudes.get(i);
            weightedSum += freq * mag;
            magnitudeSum += mag;
        }
        
        return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0;
    }
    
    /**
     * 计算频谱带宽 / Calculate spectral bandwidth
     *
     * @param frequencies 频率向量 / Frequency vector
     * @param magnitudes 幅度向量 / Magnitude vector
     * @param spectralCentroid 频谱质心 / Spectral centroid
     * @return 频谱带宽 / Spectral bandwidth
     */
    private static double calculateSpectralBandwidth(IVector<Double> frequencies, IVector<Double> magnitudes, double spectralCentroid) {
        double weightedSum = 0;
        double magnitudeSum = 0;
        
        for (int i = 0; i < frequencies.length(); i++) {
            double freq = frequencies.get(i);
            double mag = magnitudes.get(i);
            double diff = freq - spectralCentroid;
            weightedSum += diff * diff * mag;
            magnitudeSum += mag;
        }
        
        return magnitudeSum > 0 ? Math.sqrt(weightedSum / magnitudeSum) : 0;
    }
    
    /**
     * 计算频谱滚降点 / Calculate spectral rolloff
     *
     * @param frequencies 频率向量 / Frequency vector
     * @param magnitudes 幅度向量 / Magnitude vector
     * @return 频谱滚降点 / Spectral rolloff
     */
    private static double calculateSpectralRolloff(IVector<Double> frequencies, IVector<Double> magnitudes) {
        double totalEnergy = magnitudes.sum();
        double threshold = totalEnergy * 0.85; // 85%的能量 / 85% of energy
        
        double cumulativeEnergy = 0;
        for (int i = 0; i < frequencies.length(); i++) {
            cumulativeEnergy += magnitudes.get(i);
            if (cumulativeEnergy >= threshold) {
                return frequencies.get(i);
            }
        }
        
        return frequencies.get(frequencies.length() - 1);
    }
    
    /**
     * 计算零交叉率 / Calculate zero crossing rate
     *
     * @param samples 音频样本 / Audio samples
     * @return 零交叉率 / Zero crossing rate
     */
    private static double calculateZeroCrossingRate(IVector<Double> samples) {
        int crossings = 0;
        for (int i = 1; i < samples.length(); i++) {
            if ((samples.get(i) >= 0) != (samples.get(i - 1) >= 0)) {
                crossings++;
            }
        }
        return (double) crossings / (samples.length() - 1);
    }
    
    /**
     * 计算MFCC特征 / Calculate MFCC features
     *
     * @param samples 音频样本 / Audio samples
     * @param sampleRate 采样率 / Sample rate
     * @return MFCC特征数组 / MFCC feature array
     */
    private static double[] calculateMFCC(IVector<Double> samples, double sampleRate) {
        // 简化的MFCC计算 / Simplified MFCC calculation
        int numCoeffs = 13;
        double[] mfcc = new double[numCoeffs];
        
        // 计算频谱 / Calculate spectrum
        var spectrumResult = calculateSpectrum(
            new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV), 1024, 0.5);
        
        IVector<Double> frequencies = spectrumResult._1;
        IVector<Double> magnitudes = spectrumResult._2;
        
        // 应用梅尔滤波器组 / Apply Mel filter bank
        double[] melFilters = applyMelFilterBank(frequencies, magnitudes, sampleRate, numCoeffs);
        
        // 计算对数能量 / Calculate log energy
        for (int i = 0; i < numCoeffs; i++) {
            mfcc[i] = Math.log(melFilters[i] + 1e-10);
        }
        
        // 应用DCT / Apply DCT
        // 这里使用简化的DCT实现 / Here uses simplified DCT implementation
        for (int i = 0; i < numCoeffs; i++) {
            double sum = 0;
            for (int j = 0; j < numCoeffs; j++) {
                sum += mfcc[j] * Math.cos(Math.PI * i * (2 * j + 1) / (2 * numCoeffs));
            }
            mfcc[i] = sum * Math.sqrt(2.0 / numCoeffs);
        }
        
        return mfcc;
    }
    
    /**
     * 应用梅尔滤波器组 / Apply Mel filter bank
     *
     * @param frequencies 频率向量 / Frequency vector
     * @param magnitudes 幅度向量 / Magnitude vector
     * @param sampleRate 采样率 / Sample rate
     * @param numFilters 滤波器数量 / Number of filters
     * @return 滤波器输出 / Filter outputs
     */
    private static double[] applyMelFilterBank(IVector<Double> frequencies, IVector<Double> magnitudes, 
                                            double sampleRate, int numFilters) {
        double[] filterOutputs = new double[numFilters];
        
        // 梅尔频率范围 / Mel frequency range
        double melLow = 2595 * Math.log10(1 + 300 / 700.0);
        double melHigh = 2595 * Math.log10(1 + sampleRate / 2 / 700.0);
        
        // 创建梅尔滤波器 / Create Mel filters
        for (int i = 0; i < numFilters; i++) {
            double melCenter = melLow + (melHigh - melLow) * i / (numFilters + 1);
            double freqCenter = 700 * (Math.pow(10, melCenter / 2595) - 1);
            
            double sum = 0;
            for (int j = 0; j < frequencies.length(); j++) {
                double freq = frequencies.get(j);
                double magnitude = magnitudes.get(j);
                
                // 简化的三角形滤波器 / Simplified triangular filter
                double filterValue = 1 - Math.abs(freq - freqCenter) / (freqCenter * 0.5);
                filterValue = Math.max(0, filterValue);
                
                sum += magnitude * filterValue;
            }
            
            filterOutputs[i] = sum;
        }
        
        return filterOutputs;
    }
    
    /**
     * 计算频谱对比度 / Calculate spectral contrast
     *
     * @param magnitudes 幅度向量 / Magnitude vector
     * @return 频谱对比度 / Spectral contrast
     */
    private static double[] calculateSpectralContrast(IVector<Double> magnitudes) {
        // 简化的频谱对比度计算 / Simplified spectral contrast calculation
        int numBands = 6;
        double[] contrast = new double[numBands];
        
        int bandSize = magnitudes.length() / numBands;
        
        for (int band = 0; band < numBands; band++) {
            int start = band * bandSize;
            int end = Math.min(start + bandSize, magnitudes.length());
            
            double maxVal = 0;
            double meanVal = 0;
            
            for (int i = start; i < end; i++) {
                double mag = magnitudes.get(i);
                maxVal = Math.max(maxVal, mag);
                meanVal += mag;
            }
            
            meanVal /= (end - start);
            contrast[band] = maxVal - meanVal;
        }
        
        return contrast;
    }
    
    /**
     * 计算自相关 / Calculate autocorrelation
     *
     * @param samples 音频样本 / Audio samples
     * @return 自相关向量 / Autocorrelation vector
     */
    private static IVector<Double> calculateAutocorrelation(IVector<Double> samples) {
        int maxLag = Math.min(samples.length() / 2, 2000);
        IVector<Double> autocorr = Linalg.zeros(maxLag);
        
        for (int lag = 0; lag < maxLag; lag++) {
            double sum = 0;
            int count = 0;
            
            for (int i = 0; i < samples.length() - lag; i++) {
                sum += samples.get(i) * samples.get(i + lag);
                count++;
            }
            
            autocorr.set(lag, count > 0 ? sum / count : 0);
        }
        
        return autocorr;
    }
    
    /**
     * 创建汉宁窗 / Create Hanning window
     *
     * @param size 窗函数大小 / Window size
     * @return 汉宁窗向量 / Hanning window vector
     */
    private static IVector<Double> createHanningWindow(int size) {
        IVector<Double> window = Linalg.zeros(size);
        for (int i = 0; i < size; i++) {
            double value = 0.5 * (1 - Math.cos(2 * Math.PI * i / (size - 1)));
            window.set(i, value);
        }
        return window;
    }
    
    /**
     * 转换为复数数组 / Convert to complex array
     *
     * @param samples 实数样本 / Real samples
     * @return 复数数组 / Complex array
     */
    private static Complex[] convertToComplex(IVector<Double> samples) {
        Complex[] complex = new Complex[samples.length()];
        for (int i = 0; i < samples.length(); i++) {
            complex[i] = new Complex(samples.get(i), 0);
        }
        return complex;
    }
}
