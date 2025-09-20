package com.reremouse.lab.audio.core;

import com.reremouse.lab.audio.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.util.Tuple2;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 抽象音频分析器基类 / Abstract Audio Analyzer Base Class
 * <p>
 * 实现IAudioAnalyzer接口的抽象基类，提供通用的参数管理和特征提取框架。
 * 使用Strategy模式允许子类实现不同的分析算法。
 * </p>
 * <p>
 * Abstract base class implementing IAudioAnalyzer interface, providing common parameter management and feature extraction framework.
 * Uses Strategy pattern to allow subclasses to implement different analysis algorithms.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public abstract class AbstractAudioAnalyzer implements IAudioAnalyzer {
    
    /** 分析器参数存储 / Analyzer parameter storage */
    protected final Map<String, Object> parameters = new ConcurrentHashMap<>();
    
    /** 分析器名称 / Analyzer name */
    protected final String name;
    
    /** 支持的特征类型 / Supported feature types */
    protected final String[] supportedFeatureTypes;
    
    /**
     * 构造函数 / Constructor
     *
     * @param name 分析器名称 / Analyzer name
     * @param supportedFeatureTypes 支持的特征类型 / Supported feature types
     */
    protected AbstractAudioAnalyzer(String name, String[] supportedFeatureTypes) {
        this.name = name;
        this.supportedFeatureTypes = supportedFeatureTypes.clone();
        initializeDefaultParameters();
    }
    
    @Override
    public final IVector<Double> extractFeatures(AudioData audioData) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }
        
        try {
            // 预处理音频数据 / Pre-process audio data
            AudioData preprocessed = preprocessAudio(audioData);
            
            // 执行特征提取 (Strategy pattern) / Perform feature extraction (Strategy pattern)
            return doExtractFeatures(preprocessed);
            
        } catch (Exception e) {
            throw new AudioProcessingException("Feature extraction failed", e);
        }
    }
    
    /**
     * 核心特征提取方法 - 子类必须实现 / Core feature extraction method - subclasses must implement
     * <p>
     * 这是Strategy模式的关键部分，不同的子类实现不同的特征提取算法。
     * This is the key part of Strategy pattern, different subclasses implement different feature extraction algorithms.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 特征向量 / Feature vector
     * @throws AudioProcessingException 特征提取过程中发生错误 / Error during feature extraction
     */
    protected abstract IVector<Double> doExtractFeatures(AudioData audioData) throws AudioProcessingException;
    
    @Override
    public final Tuple2<IVector<Double>, IVector<Double>> calculateSpectrum(AudioData audioData) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }
        
        try {
            // 预处理音频数据 / Pre-process audio data
            AudioData preprocessed = preprocessAudio(audioData);
            
            // 执行频谱计算 / Perform spectrum calculation
            return doCalculateSpectrum(preprocessed);
            
        } catch (Exception e) {
            throw new AudioProcessingException("Spectrum calculation failed", e);
        }
    }
    
    /**
     * 核心频谱计算方法 - 子类可以重写 / Core spectrum calculation method - subclasses can override
     * <p>
     * 提供默认的频谱计算实现，子类可以重写以提供特定的频谱计算方法。
     * Provides default spectrum calculation implementation, subclasses can override for specific spectrum calculation methods.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 频率和幅度的元组 / Tuple of frequencies and magnitudes
     * @throws AudioProcessingException 频谱计算过程中发生错误 / Error during spectrum calculation
     */
    protected abstract Tuple2<IVector<Double>, IVector<Double>> doCalculateSpectrum(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 音频预处理 - 子类可以重写 / Audio pre-processing - subclasses can override
     * <p>
     * 在特征提取之前对音频进行预处理，如单声道转换、归一化等。
     * Pre-process audio before feature extraction, such as mono conversion, normalization, etc.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 预处理后的音频数据 / Pre-processed audio data
     * @throws AudioProcessingException 预处理过程中发生错误 / Error during pre-processing
     */
    protected AudioData preprocessAudio(AudioData audioData) throws AudioProcessingException {
        // 默认实现：如果是立体声，转换为单声道 / Default implementation: convert to mono if stereo
        if (audioData.getChannels() > 1) {
            // 这里需要实现立体声到单声道的转换 / Here need to implement stereo to mono conversion
            // 为了简化，暂时返回原始数据 / For simplification, temporarily return original data
            return audioData;
        }
        return audioData;
    }
    
    /**
     * 初始化默认参数 - 子类可以重写 / Initialize default parameters - subclasses can override
     */
    protected void initializeDefaultParameters() {
        parameters.put("window_size", 1024);
        parameters.put("hop_size", 512);
        parameters.put("sample_rate", 44100.0);
    }
    
    @Override
    public void setParameter(String key, Object value) throws IllegalArgumentException {
        if (key == null) {
            throw new IllegalArgumentException("Parameter key cannot be null");
        }
        
        // 验证参数 / Validate parameter
        validateParameter(key, value);
        
        parameters.put(key, value);
    }
    
    /**
     * 验证参数 - 子类可以重写 / Validate parameter - subclasses can override
     *
     * @param key 参数键 / Parameter key
     * @param value 参数值 / Parameter value
     * @throws IllegalArgumentException 参数无效时抛出 / Thrown when parameter is invalid
     */
    protected void validateParameter(String key, Object value) throws IllegalArgumentException {
        switch (key) {
            case "window_size":
                if (!(value instanceof Integer) || (Integer) value <= 0) {
                    throw new IllegalArgumentException("window_size must be a positive integer");
                }
                break;
            case "hop_size":
                if (!(value instanceof Integer) || (Integer) value <= 0) {
                    throw new IllegalArgumentException("hop_size must be a positive integer");
                }
                break;
            case "sample_rate":
                if (!(value instanceof Number) || ((Number) value).doubleValue() <= 0) {
                    throw new IllegalArgumentException("sample_rate must be a positive number");
                }
                break;
            default:
                // 子类可以处理其他参数 / Subclasses can handle other parameters
                break;
        }
    }
    
    @Override
    public Object getParameter(String key) throws IllegalArgumentException {
        if (!parameters.containsKey(key)) {
            throw new IllegalArgumentException("Unknown parameter: " + key);
        }
        return parameters.get(key);
    }
    
    @Override
    public void reset() {
        parameters.clear();
        initializeDefaultParameters();
    }
    
    @Override
    public String getName() {
        return name;
    }
    
    @Override
    public String[] getSupportedFeatureTypes() {
        return supportedFeatureTypes.clone();
    }
    
    @Override
    public boolean supportsFeatureType(String featureType) {
        if (featureType == null) {
            return false;
        }
        
        for (String supportedType : supportedFeatureTypes) {
            if (supportedType.equalsIgnoreCase(featureType)) {
                return true;
            }
        }
        return false;
    }
    
    @Override
    public int getFeatureDimension(String featureType) {
        if (!supportsFeatureType(featureType)) {
            throw new IllegalArgumentException("Unsupported feature type: " + featureType);
        }
        
        // 子类应该重写此方法以提供准确的特征维度 / Subclasses should override this method to provide accurate feature dimension
        return getDefaultFeatureDimension(featureType);
    }
    
    /**
     * 获取默认特征维度 - 子类应该重写 / Get default feature dimension - subclasses should override
     *
     * @param featureType 特征类型 / Feature type
     * @return 特征维度 / Feature dimension
     */
    protected int getDefaultFeatureDimension(String featureType) {
        // 提供一些常见特征的默认维度 / Provide default dimensions for common features
        switch (featureType.toLowerCase()) {
            case "mfcc":
                return 13;
            case "chroma":
                return 12;
            case "spectral_centroid":
                return 1;
            case "spectral_bandwidth":
                return 1;
            case "spectral_rolloff":
                return 1;
            case "zero_crossing_rate":
                return 1;
            case "spectral_contrast":
                return 6;
            default:
                return 1; // 默认维度 / Default dimension
        }
    }
    
    /**
     * 获取窗口大小参数 / Get window size parameter
     *
     * @return 窗口大小 / Window size
     */
    protected int getWindowSize() {
        return (Integer) parameters.getOrDefault("window_size", 1024);
    }
    
    /**
     * 获取跳跃大小参数 / Get hop size parameter
     *
     * @return 跳跃大小 / Hop size
     */
    protected int getHopSize() {
        return (Integer) parameters.getOrDefault("hop_size", 512);
    }
    
    /**
     * 获取采样率参数 / Get sample rate parameter
     *
     * @return 采样率 / Sample rate
     */
    protected double getSampleRate() {
        Object sampleRate = parameters.getOrDefault("sample_rate", 44100.0);
        if (sampleRate instanceof Number) {
            return ((Number) sampleRate).doubleValue();
        }
        return 44100.0;
    }
    
    @Override
    public String toString() {
        return String.format("%s{name='%s', supportedFeatures=%s}", 
                           getClass().getSimpleName(), name, java.util.Arrays.toString(supportedFeatureTypes));
    }
}