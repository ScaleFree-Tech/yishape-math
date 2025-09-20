package com.reremouse.lab.audio.core;

import com.reremouse.lab.audio.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.util.Tuple2;

/**
 * 音频分析器接口 / Audio Analyzer Interface
 * <p>
 * 定义音频分析器的基本操作，包括特征提取、频谱分析、时频分析等。
 * 所有音频分析器都应该实现此接口，确保一致的API设计。
 * </p>
 * <p>
 * Defines basic operations for audio analyzers, including feature extraction, spectral analysis, time-frequency analysis, etc.
 * All audio analyzers should implement this interface to ensure consistent API design.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IAudioAnalyzer {
    
    /**
     * 提取音频特征 / Extract audio features
     * <p>
     * 从音频数据中提取特征向量。特征类型由具体实现决定。
     * Extract feature vector from audio data. Feature type is determined by specific implementation.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 特征向量 / Feature vector
     * @throws AudioProcessingException 当分析过程中发生错误时抛出 / Thrown when error occurs during analysis
     */
    IVector<Double> extractFeatures(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 计算频谱 / Calculate spectrum
     * <p>
     * 计算音频的频域表示。
     * Calculate frequency domain representation of audio.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 频率和幅度的元组 / Tuple of frequencies and magnitudes
     * @throws AudioProcessingException 当分析过程中发生错误时抛出 / Thrown when error occurs during analysis
     */
    Tuple2<IVector<Double>, IVector<Double>> calculateSpectrum(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 设置分析参数 / Set analysis parameters
     * <p>
     * 设置分析器的配置参数。
     * Set configuration parameters for the analyzer.
     * </p>
     *
     * @param key 参数键 / Parameter key
     * @param value 参数值 / Parameter value
     * @throws IllegalArgumentException 当参数无效时抛出 / Thrown when parameter is invalid
     */
    void setParameter(String key, Object value) throws IllegalArgumentException;
    
    /**
     * 获取分析参数 / Get analysis parameter
     * <p>
     * 获取指定参数的当前值。
     * Get current value of specified parameter.
     * </p>
     *
     * @param key 参数键 / Parameter key
     * @return 参数值 / Parameter value
     * @throws IllegalArgumentException 当参数键不存在时抛出 / Thrown when parameter key doesn't exist
     */
    Object getParameter(String key) throws IllegalArgumentException;
    
    /**
     * 重置分析器参数 / Reset analyzer parameters
     * <p>
     * 将所有参数重置为默认值。
     * Reset all parameters to default values.
     * </p>
     */
    void reset();
    
    /**
     * 获取分析器名称 / Get analyzer name
     * <p>
     * 返回分析器的唯一标识名称。
     * Return unique identifier name of the analyzer.
     * </p>
     *
     * @return 分析器名称 / Analyzer name
     */
    String getName();
    
    /**
     * 获取支持的特征类型 / Get supported feature types
     * <p>
     * 返回分析器支持的所有特征类型列表。
     * Return list of all feature types supported by the analyzer.
     * </p>
     *
     * @return 特征类型数组 / Feature type array
     */
    String[] getSupportedFeatureTypes();
    
    /**
     * 检查是否支持指定特征类型 / Check if supports specified feature type
     * <p>
     * 检查分析器是否支持提取指定类型的特征。
     * Check if analyzer supports extracting specified type of feature.
     * </p>
     *
     * @param featureType 特征类型 / Feature type
     * @return 如果支持返回true / Return true if supported
     */
    boolean supportsFeatureType(String featureType);
    
    /**
     * 获取特征维度 / Get feature dimension
     * <p>
     * 返回提取的特征向量的维度。
     * Return dimension of extracted feature vector.
     * </p>
     *
     * @param featureType 特征类型 / Feature type
     * @return 特征维度 / Feature dimension
     */
    int getFeatureDimension(String featureType);
}