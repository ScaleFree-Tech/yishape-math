package com.yishape.lab.audio.preprocessing;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * 音频预处理器 - 音频数据预处理和标准化 / Audio Preprocessor - Audio data preprocessing and normalization
 * <p>
 * 提供音频数据预处理功能，包括采样率标准化、音量归一化、降噪等。
 * Provides audio data preprocessing functions, including sample rate normalization, volume normalization, noise reduction, etc.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class AudioPreprocessor {
    
    /** 默认目标采样率 / Default target sample rate */
    private static final double DEFAULT_TARGET_SAMPLE_RATE = 44100.0;
    
    /** 默认目标位深度 / Default target bit depth */
    private static final int DEFAULT_TARGET_BIT_DEPTH = 16;
    
    /** 默认归一化目标值 / Default normalization target */
    private static final double DEFAULT_NORMALIZATION_TARGET = 0.95;
    
    /** 默认降噪阈值 / Default noise reduction threshold */
    private static final double DEFAULT_NOISE_THRESHOLD = 0.01;
    
    /**
     * 预处理音频数据 / Preprocess audio data
     */
    public AudioData preprocess(AudioData audioData) throws AudioProcessingException {
        return preprocess(audioData, null);
    }
    
    /**
     * 预处理音频数据（带参数）/ Preprocess audio data with parameters
     */
    public AudioData preprocess(AudioData audioData, AudioPreprocessingOptions options) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("音频数据不能为null / Audio data cannot be null");
        }
        
        // 使用默认选项 / Use default options
        if (options == null) {
            options = AudioPreprocessingOptions.getDefault();
        }
        
        AudioData result = audioData;
        
        try {
            // 1. 采样率标准化 / Sample rate normalization
            if (options.isNormalizeSampleRate()) {
                result = normalizeSampleRate(result, options.getTargetSampleRate());
            }
            
            // 2. 音量归一化 / Volume normalization
            if (options.isNormalizeVolume()) {
                result = normalizeVolume(result, options.getNormalizationTarget());
            }
            
            // 3. 降噪处理 / Noise reduction
            if (options.isApplyNoiseReduction()) {
                result = applyNoiseReduction(result, options.getNoiseThreshold());
            }
            
            // 4. 立体声转单声道（如果需要）/ Stereo to mono conversion (if needed)
            if (options.isConvertToMono() && result.getChannels() > 1) {
                result = convertToMono(result);
            }
            
            // 5. 直流分量去除 / DC offset removal
            if (options.isRemoveDCOffset()) {
                result = removeDCOffset(result);
            }
            
            // 6. 预加重滤波 / Pre-emphasis filtering
            if (options.isApplyPreEmphasis()) {
                result = applyPreEmphasis(result, options.getPreEmphasisCoefficient());
            }
            
            return result;
            
        } catch (Exception e) {
            throw new AudioProcessingException("音频预处理失败 / Audio preprocessing failed", e);
        }
    }
    
    /**
     * 采样率标准化 / Sample rate normalization
     */
    private AudioData normalizeSampleRate(AudioData audioData, double targetSampleRate) throws AudioProcessingException {
        double currentSampleRate = audioData.getSampleRate();
        
        // 如果采样率已经匹配，直接返回 / If sample rate already matches, return directly
        if (Math.abs(currentSampleRate - targetSampleRate) < 0.1) {
            return audioData;
        }
        
        // 简单的重采样实现：线性插值 / Simple resampling implementation: linear interpolation
        IVector<Double> originalSamples = audioData.getSamples();
        double resampleRatio = targetSampleRate / currentSampleRate;
        int newLength = (int) (originalSamples.length() * resampleRatio);
        
        IVector<Double> resampledSamples = Linalg.zeros(newLength);
        
        for (int i = 0; i < newLength; i++) {
            double originalIndex = i / resampleRatio;
            int index0 = (int) Math.floor(originalIndex);
            int index1 = Math.min(index0 + 1, originalSamples.length() - 1);
            double fraction = originalIndex - index0;
            
            double value0 = originalSamples.get(index0);
            double value1 = originalSamples.get(index1);
            
            // 线性插值 / Linear interpolation
            resampledSamples.set(i, value0 + fraction * (value1 - value0));
        }
        
        return new AudioData(
            resampledSamples,
            targetSampleRate,
            audioData.getChannels(),
            audioData.getBitDepth(),
            audioData.getFormat()
        );
    }
    
    /**
     * 音量归一化 / Volume normalization
     */
    private AudioData normalizeVolume(AudioData audioData, double targetLevel) throws AudioProcessingException {
        IVector<Double> samples = audioData.getSamples();
        
        // 找到最大幅度 / Find maximum amplitude
        double maxAmplitude = 0.0;
        for (int i = 0; i < samples.length(); i++) {
            double absValue = Math.abs(samples.get(i));
            if (absValue > maxAmplitude) {
                maxAmplitude = absValue;
            }
        }
        
        // 如果最大幅度为0或已经小于目标值，不需要归一化 / If max amplitude is 0 or already less than target, no normalization needed
        if (maxAmplitude == 0.0 || maxAmplitude <= targetLevel) {
            return audioData;
        }
        
        // 计算归一化因子 / Calculate normalization factor
        double normalizationFactor = targetLevel / maxAmplitude;
        
        // 应用归一化 / Apply normalization
        IVector<Double> normalizedSamples = Linalg.zeros(samples.length());
        for (int i = 0; i < samples.length(); i++) {
            normalizedSamples.set(i, samples.get(i) * normalizationFactor);
        }
        
        return new AudioData(
            normalizedSamples,
            audioData.getSampleRate(),
            audioData.getChannels(),
            audioData.getBitDepth(),
            audioData.getFormat()
        );
    }
    
    /**
     * 降噪处理 / Noise reduction
     */
    private AudioData applyNoiseReduction(AudioData audioData, double threshold) throws AudioProcessingException {
        IVector<Double> samples = audioData.getSamples();
        IVector<Double> denoisedSamples = Linalg.zeros(samples.length());
        
        // 简单的阈值降噪 / Simple threshold noise reduction
        for (int i = 0; i < samples.length(); i++) {
            double sample = samples.get(i);
            if (Math.abs(sample) < threshold) {
                denoisedSamples.set(i, 0.0);
            } else {
                denoisedSamples.set(i, sample);
            }
        }
        
        return new AudioData(
            denoisedSamples,
            audioData.getSampleRate(),
            audioData.getChannels(),
            audioData.getBitDepth(),
            audioData.getFormat()
        );
    }
    
    /**
     * 立体声转单声道 / Stereo to mono conversion
     */
    private AudioData convertToMono(AudioData audioData) throws AudioProcessingException {
        if (audioData.getChannels() == 1) {
            return audioData;
        }
        
        IVector<Double> samples = audioData.getSamples();
        
        // 简单的平均处理 / Simple averaging
        int monoLength = samples.length() / audioData.getChannels();
        IVector<Double> monoSamples = Linalg.zeros(monoLength);
        
        for (int i = 0; i < monoLength; i++) {
            double sum = 0.0;
            for (int ch = 0; ch < audioData.getChannels(); ch++) {
                sum += samples.get(i * audioData.getChannels() + ch);
            }
            monoSamples.set(i, sum / audioData.getChannels());
        }
        
        return new AudioData(
            monoSamples,
            audioData.getSampleRate(),
            1, // 单声道 / Mono
            audioData.getBitDepth(),
            audioData.getFormat()
        );
    }
    
    /**
     * 去除直流分量 / Remove DC offset
     */
    private AudioData removeDCOffset(AudioData audioData) throws AudioProcessingException {
        IVector<Double> samples = audioData.getSamples();
        
        // 计算平均值 / Calculate mean value
        double mean = 0.0;
        for (int i = 0; i < samples.length(); i++) {
            mean += samples.get(i);
        }
        mean /= samples.length();
        
        // 去除直流分量 / Remove DC offset
        IVector<Double> correctedSamples = Linalg.zeros(samples.length());
        for (int i = 0; i < samples.length(); i++) {
            correctedSamples.set(i, samples.get(i) - mean);
        }
        
        return new AudioData(
            correctedSamples,
            audioData.getSampleRate(),
            audioData.getChannels(),
            audioData.getBitDepth(),
            audioData.getFormat()
        );
    }
    
    /**
     * 应用预加重滤波器 / Apply pre-emphasis filter
     */
    private AudioData applyPreEmphasis(AudioData audioData, double coefficient) throws AudioProcessingException {
        IVector<Double> samples = audioData.getSamples();
        IVector<Double> emphasizedSamples = Linalg.zeros(samples.length());
        
        // 预加重滤波：y[n] = x[n] - α * x[n-1] / Pre-emphasis filter: y[n] = x[n] - α * x[n-1]
        emphasizedSamples.set(0, samples.get(0));
        for (int i = 1; i < samples.length(); i++) {
            emphasizedSamples.set(i, samples.get(i) - coefficient * samples.get(i - 1));
        }
        
        return new AudioData(
            emphasizedSamples,
            audioData.getSampleRate(),
            audioData.getChannels(),
            audioData.getBitDepth(),
            audioData.getFormat()
        );
    }
}