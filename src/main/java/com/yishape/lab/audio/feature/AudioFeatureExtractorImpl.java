package com.yishape.lab.audio.feature;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioUtil;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.signal.core.Complex;

import java.util.Map;
import java.util.HashMap;
import java.util.Arrays;

/**
 * 音频特征提取器实现 / Audio Feature Extractor Implementation
 * <p>
 * 实现音频特征提取接口，专注于音频层面的特征提取，包括时域、频域和谱特征。
 * Implements audio feature extraction interface, focusing on audio-level features including 
 * time-domain, frequency-domain and spectral features.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public class AudioFeatureExtractorImpl implements IAudioFeatureExtractor {

    // 默认参数 / Default parameters
    private static final int DEFAULT_WINDOW_SIZE = 4096;
    private static final int DEFAULT_HOP_SIZE = 2048;
    private static final int DEFAULT_FRAME_SIZE = 1024;
    private static final int DEFAULT_MFCC_COUNT = 13;

    // 特征名称常量 / Feature name constants
    private static final String RMS_ENERGY = "rms_energy";
    private static final String ZERO_CROSSING_RATE = "zero_crossing_rate";
    private static final String SPECTRAL_CENTROID = "spectral_centroid";
    private static final String SPECTRAL_BANDWIDTH = "spectral_bandwidth";
    private static final String SPECTRAL_ROLLOFF = "spectral_rolloff";
    private static final String MFCC = "mfcc";

    /**
     * 默认构造函数 / Default constructor
     */
    public AudioFeatureExtractorImpl() {
        // 默认构造函数 / Default constructor
    }

    @Override
    public AudioFeatureResult extractAudioFeatures(AudioData audioData) throws AudioProcessingException {
        return extractAudioFeatures(audioData, getDefaultParameters());
    }

    @Override
    public AudioFeatureResult extractAudioFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }

        try {
            // 提取各类音频特征 / Extract various audio features
            TimeDomainFeatureResult timeDomainFeatures = extractTimeDomainFeatures(audioData, parameters);
            FrequencyDomainFeatureResult frequencyDomainFeatures = extractFrequencyDomainFeatures(audioData, parameters);
            SpectralFeatureResult spectralFeatures = extractSpectralFeatures(audioData, parameters);

            // 创建综合结果 / Create comprehensive result
            AudioFeatureResult result = new AudioFeatureResult(
                timeDomainFeatures, frequencyDomainFeatures, spectralFeatures
            );

            // 添加元数据 / Add metadata
            result.addMetadata("sampleRate", audioData.getSampleRate());
            result.addMetadata("audioLength", audioData.getSamples().length());
            result.addMetadata("duration", (double) audioData.getSamples().length() / audioData.getSampleRate());
            result.addMetadata("extractorVersion", getVersion());

            return result;

        } catch (Exception e) {
            throw new AudioProcessingException("Audio feature extraction failed: " + e.getMessage(), e);
        }
    }

    @Override
    public TimeDomainFeatureResult extractTimeDomainFeatures(AudioData audioData) throws AudioProcessingException {
        return extractTimeDomainFeatures(audioData, getDefaultParameters());
    }

    @Override
    public TimeDomainFeatureResult extractTimeDomainFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }

        try {
            // 计算时域特征 / Calculate time-domain features
            double rmsEnergy = AudioUtil.calculateRMS(audioData.getSamples());
            double zeroCrossingRate = AudioUtil.calculateZeroCrossingRate(audioData.getSamples());
            double energy = calculateEnergy(audioData.getSamples());
            double[] amplitudeEnvelope = calculateAmplitudeEnvelope(audioData.getSamples(), parameters);
            
            // 计算置信度 / Calculate confidence
            double confidence = calculateTimeDomainConfidence(rmsEnergy, zeroCrossingRate, energy);

            // 创建时域特征结果 / Create time-domain feature result
            TimeDomainFeatureResult result = new TimeDomainFeatureResult(
                rmsEnergy, zeroCrossingRate, energy, amplitudeEnvelope, confidence
            );

            return result;

        } catch (Exception e) {
            throw new AudioProcessingException("Time-domain feature extraction failed: " + e.getMessage(), e);
        }
    }

    @Override
    public FrequencyDomainFeatureResult extractFrequencyDomainFeatures(AudioData audioData) throws AudioProcessingException {
        return extractFrequencyDomainFeatures(audioData, getDefaultParameters());
    }

    @Override
    public FrequencyDomainFeatureResult extractFrequencyDomainFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }

        try {
            // 计算频域特征 / Calculate frequency-domain features
            Complex[] spectrum = AudioUtil.processFFT(audioData);
            double sampleRate = audioData.getSampleRate();
            int windowSize = getIntValue(parameters, "windowSize", DEFAULT_WINDOW_SIZE);
            
            double spectralCentroid = AudioUtil.calculateSpectralCentroid(spectrum, sampleRate, windowSize);
            double spectralBandwidth = AudioUtil.calculateSpectralBandwidth(spectrum, sampleRate, windowSize, spectralCentroid);
            double spectralRolloff = AudioUtil.calculateSpectralRolloff(spectrum, sampleRate, windowSize);
            double[] spectralContrast = calculateSpectralContrast(spectrum);
            double spectralFlatness = AudioUtil.calculateSpectralFlatness(spectrum);
            double spectralFlux = AudioUtil.calculateSpectralFlux(audioData, parameters);
            
            // 计算置信度 / Calculate confidence
            double confidence = calculateFrequencyDomainConfidence(spectralCentroid, spectralBandwidth, spectralRolloff);

            // 创建频域特征结果 / Create frequency-domain feature result
            FrequencyDomainFeatureResult result = new FrequencyDomainFeatureResult(
                spectralCentroid, spectralBandwidth, spectralRolloff, 
                spectralContrast, spectralFlatness, spectralFlux, confidence
            );

            return result;

        } catch (Exception e) {
            throw new AudioProcessingException("Frequency-domain feature extraction failed: " + e.getMessage(), e);
        }
    }

    @Override
    public SpectralFeatureResult extractSpectralFeatures(AudioData audioData) throws AudioProcessingException {
        return extractSpectralFeatures(audioData, getDefaultParameters());
    }

    @Override
    public SpectralFeatureResult extractSpectralFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }

        try {
            // 计算谱特征 / Calculate spectral features
            int mfccCount = getIntValue(parameters, "mfccCount", DEFAULT_MFCC_COUNT);
            double[][] mfccFeatures = AudioUtil.calculateMFCCFrames(audioData, parameters, mfccCount);
            
            // 计算MFCC的均值作为代表 / Calculate mean of MFCC as representative
            double[] mfcc = new double[mfccFeatures.length];
            for (int i = 0; i < mfccFeatures.length; i++) {
                mfcc[i] = calculateMean(mfccFeatures[i]);
            }
            
            // 计算色度特征 / Calculate chroma features
            // For now, use a placeholder for chroma features
            double[] chroma = new double[12]; // 12 chroma bins
            
            // 计算其他谱特征 / Calculate other spectral features
            double[] spectralFeatures = new double[0]; // 可以扩展 / Can be extended
            
            // 计算MFCC差分 / Calculate MFCC delta and delta-delta
            double[] mfccDelta = calculateDelta(mfcc);
            double[] mfccDeltaDelta = calculateDelta(mfccDelta);
            
            // 计算置信度 / Calculate confidence
            double confidence = calculateSpectralConfidence(mfcc, chroma);

            // 创建谱特征结果 / Create spectral feature result
            SpectralFeatureResult result = new SpectralFeatureResult(
                mfcc, chroma, spectralFeatures, mfccDelta, mfccDeltaDelta, confidence
            );

            return result;

        } catch (Exception e) {
            throw new AudioProcessingException("Spectral feature extraction failed: " + e.getMessage(), e);
        }
    }

    @Override
    public String[] getSupportedFeatureTypes() {
        return new String[]{
            "time_domain", "frequency_domain", "spectral",
            "rms_energy", "zero_crossing_rate", "energy",
            "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
            "mfcc", "chroma", "spectral_contrast"
        };
    }

    @Override
    public boolean isFeatureTypeSupported(String featureType) {
        return Arrays.asList(getSupportedFeatureTypes()).contains(featureType.toLowerCase());
    }

    @Override
    public String getExtractorName() {
        return "AudioFeatureExtractorImpl";
    }

    @Override
    public String getVersion() {
        return "2.0";
    }

    @Override
    public String[] getSupportedParameters() {
        return new String[]{
            "windowSize", "hopSize", "frameSize", "mfccCount"
        };
    }

    @Override
    public Map<String, Object> getDefaultParameters() {
        Map<String, Object> params = new HashMap<>();
        params.put("windowSize", DEFAULT_WINDOW_SIZE);
        params.put("hopSize", DEFAULT_HOP_SIZE);
        params.put("frameSize", DEFAULT_FRAME_SIZE);
        params.put("mfccCount", DEFAULT_MFCC_COUNT);
        return params;
    }

    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        if (parameters == null) {
            return false;
        }

        // 验证窗口大小
        Object windowSizeObj = parameters.get("windowSize");
        if (windowSizeObj != null && windowSizeObj instanceof Integer) {
            int windowSize = (Integer) windowSizeObj;
            if (windowSize <= 0 || windowSize > 16384) {
                return false;
            }
        }

        // 验证跳跃大小
        Object hopSizeObj = parameters.get("hopSize");
        if (hopSizeObj != null && hopSizeObj instanceof Integer) {
            int hopSize = (Integer) hopSizeObj;
            if (hopSize <= 0 || hopSize > 8192) {
                return false;
            }
        }

        return true;
    }

    // 辅助方法 / Helper methods

    /**
     * 计算能量 / Calculate energy
     *
     * @param samples 音频样本 / Audio samples
     * @return 能量值 / Energy value
     */
    private double calculateEnergy(IVector<Double> samples) {
        double sum = 0.0;
        int length = samples.length();
        
        for (int i = 0; i < length; i++) {
            double sample = samples.get(i);
            sum += sample * sample;
        }
        
        return sum;
    }

    /**
     * 计算幅度包络 / Calculate amplitude envelope
     *
     * @param samples 音频样本 / Audio samples
     * @param parameters 参数映射 / Parameter map
     * @return 幅度包络数组 / Amplitude envelope array
     */
    private double[] calculateAmplitudeEnvelope(IVector<Double> samples, Map<String, Object> parameters) {
        int frameSize = getIntValue(parameters, "frameSize", DEFAULT_FRAME_SIZE);
        int hopSize = getIntValue(parameters, "hopSize", DEFAULT_HOP_SIZE);
        
        int sampleCount = samples.length();
        int frameCount = (sampleCount - frameSize) / hopSize + 1;
        
        double[] envelope = new double[frameCount];
        
        for (int i = 0; i < frameCount; i++) {
            int start = i * hopSize;
            int end = Math.min(start + frameSize, sampleCount);
            
            double maxAmplitude = 0.0;
            for (int j = start; j < end; j++) {
                double amplitude = Math.abs(samples.get(j));
                if (amplitude > maxAmplitude) {
                    maxAmplitude = amplitude;
                }
            }
            
            envelope[i] = maxAmplitude;
        }
        
        return envelope;
    }

    /**
     * 计算频谱对比度 / Calculate spectral contrast
     *
     * @param spectrum 频谱数据 / Spectrum data
     * @return 频谱对比度数组 / Spectral contrast array
     */
    private double[] calculateSpectralContrast(Complex[] spectrum) {
        // 简化实现：返回固定长度的数组 / Simplified implementation: return fixed-length array
        return new double[6]; // 通常使用6个频段 / Typically use 6 frequency bands
    }

    /**
     * 计算数组均值 / Calculate array mean
     *
     * @param array 输入数组 / Input array
     * @return 均值 / Mean value
     */
    private double calculateMean(double[] array) {
        if (array == null || array.length == 0) {
            return 0.0;
        }
        double sum = 0.0;
        for (double value : array) {
            sum += value;
        }
        return sum / array.length;
    }

    /**
     * 计算差分 / Calculate delta
     *
     * @param array 输入数组 / Input array
     * @return 差分数组 / Delta array
     */
    private double[] calculateDelta(double[] array) {
        if (array == null || array.length < 2) {
            return new double[array != null ? array.length : 0];
        }
        
        double[] delta = new double[array.length];
        delta[0] = array[1] - array[0];
        
        for (int i = 1; i < array.length - 1; i++) {
            delta[i] = (array[i + 1] - array[i - 1]) / 2.0;
        }
        
        delta[array.length - 1] = array[array.length - 1] - array[array.length - 2];
        
        return delta;
    }

    /**
     * 从参数中获取整数值 / Get integer value from parameters
     *
     * @param parameters 参数映射 / Parameter map
     * @param key 参数键 / Parameter key
     * @param defaultValue 默认值 / Default value
     * @return 整数值 / Integer value
     */
    private int getIntValue(Map<String, Object> parameters, String key, int defaultValue) {
        if (parameters != null && parameters.containsKey(key)) {
            Object value = parameters.get(key);
            if (value instanceof Integer) {
                return (Integer) value;
            }
        }
        return defaultValue;
    }

    /**
     * 计算时域特征置信度 / Calculate time-domain feature confidence
     *
     * @param rmsEnergy 均方根能量 / RMS energy
     * @param zeroCrossingRate 过零率 / Zero crossing rate
     * @param energy 能量 / Energy
     * @return 置信度值 / Confidence value
     */
    private double calculateTimeDomainConfidence(double rmsEnergy, double zeroCrossingRate, double energy) {
        // 基于特征值的有效性计算置信度 / Calculate confidence based on validity of feature values
        double confidence = 1.0;
        
        // 检查RMS能量是否在合理范围内 / Check if RMS energy is within reasonable range
        if (rmsEnergy < 0.0 || rmsEnergy > 1.0) {
            confidence *= 0.8;
        }
        
        // 检查过零率是否在合理范围内 / Check if zero crossing rate is within reasonable range
        if (zeroCrossingRate < 0.0 || zeroCrossingRate > 1.0) {
            confidence *= 0.8;
        }
        
        // 检查能量是否为非负值 / Check if energy is non-negative
        if (energy < 0.0) {
            confidence *= 0.5;
        }
        
        return Math.max(0.0, Math.min(1.0, confidence));
    }

    /**
     * 计算频域特征置信度 / Calculate frequency-domain feature confidence
     *
     * @param spectralCentroid 频谱质心 / Spectral centroid
     * @param spectralBandwidth 频谱带宽 / Spectral bandwidth
     * @param spectralRolloff 频谱滚降点 / Spectral rolloff
     * @return 置信度值 / Confidence value
     */
    private double calculateFrequencyDomainConfidence(double spectralCentroid, double spectralBandwidth, double spectralRolloff) {
        // 基于特征值的有效性计算置信度 / Calculate confidence based on validity of feature values
        double confidence = 1.0;
        
        // 检查频谱质心是否为非负值 / Check if spectral centroid is non-negative
        if (spectralCentroid < 0.0) {
            confidence *= 0.5;
        }
        
        // 检查频谱带宽是否为非负值 / Check if spectral bandwidth is non-negative
        if (spectralBandwidth < 0.0) {
            confidence *= 0.5;
        }
        
        // 检查频谱滚降点是否为非负值 / Check if spectral rolloff is non-negative
        if (spectralRolloff < 0.0) {
            confidence *= 0.5;
        }
        
        return Math.max(0.0, Math.min(1.0, confidence));
    }

    /**
     * 计算谱特征置信度 / Calculate spectral feature confidence
     *
     * @param mfcc MFCC特征 / MFCC features
     * @param chroma 色度特征 / Chroma features
     * @return 置信度值 / Confidence value
     */
    private double calculateSpectralConfidence(double[] mfcc, double[] chroma) {
        // 基于特征数组的有效性计算置信度 / Calculate confidence based on validity of feature arrays
        double confidence = 1.0;
        
        // 检查MFCC数组是否为空 / Check if MFCC array is empty
        if (mfcc == null || mfcc.length == 0) {
            confidence *= 0.5;
        }
        
        // 检查色度数组是否为空 / Check if chroma array is empty
        if (chroma == null || chroma.length == 0) {
            confidence *= 0.5;
        }
        
        return Math.max(0.0, Math.min(1.0, confidence));
    }
}