package com.yishape.lab.music.analysis.advanced;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.music.analysis.MusicDetectionResult;
import java.util.Map;

/**
 * 高级音乐分析器接口 / Advanced Music Analyzer Interface
 * <p>
 * 定义高级音乐分析功能，包括情感分析、风格识别、复杂度分析等。
 * Defines advanced music analysis functionality including emotion analysis, genre recognition, complexity analysis, etc.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public interface IAdvancedAnalyzer {
    
    /**
     * 分析音频 / Analyze audio
     *
     * @param audioData 音频数据 / Audio data
     * @return 分析结果 / Analysis result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    Map<String, Object> analyze(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 分析音频（带参数） / Analyze audio with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 分析参数 / Analysis parameters
     * @return 分析结果 / Analysis result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    Map<String, Object> analyze(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;
    
    /**
     * 获取分析器名称 / Get analyzer name
     *
     * @return 分析器名称 / Analyzer name
     */
    String getAnalyzerName();
    
    /**
     * 获取支持的参数名称 / Get supported parameter names
     *
     * @return 参数名称数组 / Array of parameter names
     */
    String[] getSupportedParameters();
    
    /**
     * 获取默认参数 / Get default parameters
     *
     * @return 默认参数映射 / Default parameters map
     */
    Map<String, Object> getDefaultParameters();
    
    /**
     * 获取分析器版本 / Get analyzer version
     *
     * @return 版本字符串 / Version string
     */
    default String getVersion() {
        return "1.0";
    }
    
    /**
     * 检查是否支持指定的音频格式 / Check if specified audio format is supported
     *
     * @param audioData 音频数据 / Audio data
     * @return 是否支持 / Whether supported
     */
    default boolean isSupported(AudioData audioData) {
        return audioData != null && audioData.getSamples() != null && audioData.getSamples().length() > 0;
    }
    
    
    
    /**
     * 执行高级音乐分析 / Perform advanced music analysis
     *
     * @param audioData 音频数据 / Audio data
     * @return 音乐检测结果 / Music detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    MusicDetectionResult analyzeAdvancedMusic(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 执行带参数的高级音乐分析 / Perform advanced music analysis with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 分析参数 / Analysis parameters
     * @return 音乐检测结果 / Music detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    MusicDetectionResult analyzeAdvancedMusic(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;
    

    
    /**
     * 验证分析参数 / Validate analysis parameters
     * 
     * @param parameters 要验证的参数 / Parameters to validate
     * @return 参数是否有效 / Whether parameters are valid
     */
    boolean validateParameters(Map<String, Object> parameters);
    

    
    /**
     * 设置分析器参数 / Set analyzer parameters
     * 
     * @param parameters 要设置的参数 / Parameters to set
     * @throws AudioProcessingException 参数无效时抛出异常 / Thrown when parameters are invalid
     */
    void setParameters(Map<String, Object> parameters) throws AudioProcessingException;
    
    /**
     * 获取当前分析器参数 / Get current analyzer parameters
     * 
     * @return 当前参数映射 / Current parameters map
     */
    Map<String, Object> getCurrentParameters();
    
    /**
     * 重置分析器参数为默认值 / Reset analyzer parameters to default values
     */
    void resetParameters();
    

    
    /**
     * 获取分析器名称 / Get analyzer name
     * 
     * @return 分析器名称 / Analyzer name
     */
    String getName();
    
    /**
     * 获取分析器描述 / Get analyzer description
     * 
     * @return 分析器描述 / Analyzer description
     */
    String getDescription();
    
    /**
     * 检查分析器是否支持指定的音频格式 / Check if analyzer supports specified audio format
     * 
     * @param sampleRate 采样率 / Sample rate
     * @param channels 声道数 / Number of channels
     * @param bitDepth 位深度 / Bit depth
     * @return 是否支持 / Whether supported
     */
    boolean supportsAudioFormat(double sampleRate, int channels, int bitDepth);
    
    /**
     * 获取分析器支持的最小音频长度 / Get minimum audio length supported by analyzer
     * 
     * @return 最小长度（秒）/ Minimum length in seconds
     */
    double getMinimumAudioLength();
    
    /**
     * 获取分析器支持的最大音频长度 / Get maximum audio length supported by analyzer
     * 
     * @return 最大长度（秒）/ Maximum length in seconds
     */
    double getMaximumAudioLength();
    
    /**
     * 获取分析器的计算复杂度估计 / Get computational complexity estimate of analyzer
     * 
     * @param audioLength 音频长度（秒）/ Audio length in seconds
     * @return 复杂度估计值 / Complexity estimate
     */
    double getComplexityEstimate(double audioLength);
    
    /**
     * 预热分析器 / Warm up analyzer
     * <p>
     * 执行一些初始化操作以提高后续分析的性能。
     * Performs initialization operations to improve performance of subsequent analyses.
     * </p>
     * 
     * @throws AudioProcessingException 预热失败时抛出异常 / Thrown when warm-up fails
     */
    void warmUp() throws AudioProcessingException;
    
    /**
     * 清理分析器资源 / Clean up analyzer resources
     * <p>
     * 释放分析器占用的资源，如内存、临时文件等。
     * Releases resources used by analyzer, such as memory, temporary files, etc.
     * </p>
     */
    void cleanup();
    
    /**
     * 获取分析器状态 / Get analyzer status
     * 
     * @return 状态字符串 / Status string
     */
    String getStatus();
    
    /**
     * 检查分析器是否就绪 / Check if analyzer is ready
     * 
     * @return 是否就绪 / Whether ready
     */
    boolean isReady();
    
    /**
     * 获取最后一次分析的统计信息 / Get statistics of last analysis
     * 
     * @return 统计信息映射 / Statistics map
     */
    Map<String, Object> getLastAnalysisStatistics();
    
    /**
     * 获取分析器的性能指标 / Get performance metrics of analyzer
     * 
     * @return 性能指标映射 / Performance metrics map
     */
    Map<String, Object> getPerformanceMetrics();
    
    /**
     * 启用或禁用详细日志 / Enable or disable verbose logging
     * 
     * @param enabled 是否启用 / Whether to enable
     */
    void setVerboseLogging(boolean enabled);
    
    /**
     * 检查是否启用了详细日志 / Check if verbose logging is enabled
     * 
     * @return 是否启用 / Whether enabled
     */
    boolean isVerboseLoggingEnabled();
}