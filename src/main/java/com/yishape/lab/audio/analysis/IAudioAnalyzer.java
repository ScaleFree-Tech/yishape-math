package com.yishape.lab.audio.analysis;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.util.Tuple2;

import java.util.Map;

/**
 * 标准化音频分析器接口 / Standardized Audio Analyzer Interface
 * <p>
 * 定义音频分析操作的统一接口，支持特征提取和频谱分析。
 * 所有音频分析算法都应实现此接口以确保一致的行为和可扩展性。
 * </p>
 * <p>
 * Defines unified interface for audio analysis operations, supporting feature extraction and spectral analysis.
 * All audio analysis algorithms should implement this interface to ensure consistent behavior and extensibility.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public interface IAudioAnalyzer {
    
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
    

    /**
     * 计算频谱 / Calculate Spectrum
     * <p>
     * 计算输入音频的频谱表示。
     * Calculate spectral representation of input audio.
     * </p>
     * 
     * @param input 输入音频 / Input audio
     * @return 频率和幅度的元组 / Tuple of frequencies and magnitudes
     * @throws AudioProcessingException 分析过程中发生错误 / Error occurred during analysis
     */
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
     * 使用参数计算频谱 / Calculate Spectrum with Parameters
     * <p>
     * 计算输入音频的频谱表示，使用指定的参数。
     * Calculate spectral representation of input audio with specified parameters.
     * </p>
     * 
     * @param input 输入音频 / Input audio
     * @param parameters 分析参数 / Analysis parameters
     * @return 频率和幅度的元组 / Tuple of frequencies and magnitudes
     * @throws AudioProcessingException 分析过程中发生错误 / Error occurred during analysis
     */
    Tuple2<IVector<Double>, IVector<Double>> calculateSpectrum(AudioData input, Map<String, Object> parameters) throws AudioProcessingException;
    
    /**
     * 获取分析器名称 / Get Analyzer Name
     * 
     * @return 分析器名称 / Analyzer name
     */
    String getName();
    
    /**
     * 获取分析器描述 / Get Analyzer Description
     * 
     * @return 分析器描述 / Analyzer description
     */
    String getDescription();
    
    /**
     * 获取支持的参数 / Get Supported Parameters
     * 
     * @return 支持的参数名称列表 / List of supported parameter names
     */
    java.util.Set<String> getSupportedParameters();
    
    /**
     * 获取默认参数 / Get Default Parameters
     * 
     * @return 默认参数映射 / Default parameter mapping
     */
    Map<String, Object> getDefaultParameters();
    
    /**
     * 验证输入音频 / Validate Input Audio
     * <p>
     * 验证输入音频是否符合分析器的要求。
     * Validates if input audio meets analyzer requirements.
     * </p>
     * 
     * @param input 输入音频 / Input audio
     * @return 验证结果 / Validation result
     */
    boolean validateInput(AudioData input);
    
    /**
     * 验证参数 / Validate Parameters
     * <p>
     * 验证参数是否有效。
     * Validates if parameters are valid.
     * </p>
     * 
     * @param parameters 参数映射 / Parameter mapping
     * @return 验证结果 / Validation result
     */
    boolean validateParameters(Map<String, Object> parameters);
    
    /**
     * 克隆分析器 / Clone Analyzer
     * <p>
     * 创建分析器的副本。
     * Creates a copy of analyzer.
     * </p>
     * 
     * @return 分析器副本 / Analyzer copy
     */
    IAudioAnalyzer clone();
    
    /**
     * 获取分析器版本 / Get Analyzer Version
     * 
     * @return 版本字符串 / Version string
     */
    default String getVersion() {
        return "1.0";
    }
}