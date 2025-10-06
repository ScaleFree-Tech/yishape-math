package com.yishape.lab.audio.processing;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;

import java.util.Map;

/**
 * 高级音频处理器接口 / Advanced Audio Processor Interface
 * <p>
 * 继承基础音频处理功能，并提供高级音频处理能力，如时间段处理、流处理、批量处理等。
 * Extends basic audio processing functionality and provides advanced audio processing capabilities
 * such as time range processing, stream processing, batch processing, etc.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IAdvancedAudioProcessor extends IBaseAudioProcessor {
    
    /**
     * 处理指定时间段的音频 / Process audio in specified time range
     * 
     * @param audioData 输入音频数据 / Input audio data
     * @param startTime 开始时间（秒）/ Start time in seconds
     * @param endTime 结束时间（秒）/ End time in seconds
     * @return 处理后的音频数据 / Processed audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData processTimeRange(AudioData audioData, double startTime, double endTime) throws AudioProcessingException;
    
    /**
     * 处理指定时间段的音频（带参数）/ Process audio in specified time range with parameters
     * 
     * @param audioData 输入音频数据 / Input audio data
     * @param startTime 开始时间（秒）/ Start time in seconds
     * @param endTime 结束时间（秒）/ End time in seconds
     * @param parameters 处理参数 / Processing parameters
     * @return 处理后的音频数据 / Processed audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData processTimeRange(AudioData audioData, double startTime, double endTime, Map<String, Object> parameters) throws AudioProcessingException;
    
    /**
     * 实时处理音频流 / Process audio stream in real-time
     * 
     * @param audioData 输入音频数据 / Input audio data
     * @param windowSize 窗口大小（秒）/ Window size in seconds
     * @param hopSize 跳跃大小（秒）/ Hop size in seconds
     * @return 处理后的音频数据 / Processed audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData processStream(AudioData audioData, double windowSize, double hopSize) throws AudioProcessingException;
    
    /**
     * 实时处理音频流（带参数）/ Process audio stream in real-time with parameters
     * 
     * @param audioData 输入音频数据 / Input audio data
     * @param windowSize 窗口大小（秒）/ Window size in seconds
     * @param hopSize 跳跃大小（秒）/ Hop size in seconds
     * @param parameters 处理参数 / Processing parameters
     * @return 处理后的音频数据 / Processed audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData processStream(AudioData audioData, double windowSize, double hopSize, Map<String, Object> parameters) throws AudioProcessingException;
    
    /**
     * 批量处理多个音频文件 / Batch process multiple audio files
     * 
     * @param audioDataArray 音频数据数组 / Array of audio data
     * @return 处理后的音频数据数组 / Array of processed audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData[] processBatch(AudioData[] audioDataArray) throws AudioProcessingException;
    
    /**
     * 批量处理多个音频文件（带参数）/ Batch process multiple audio files with parameters
     * 
     * @param audioDataArray 音频数据数组 / Array of audio data
     * @param parameters 处理参数 / Processing parameters
     * @return 处理后的音频数据数组 / Array of processed audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData[] processBatch(AudioData[] audioDataArray, Map<String, Object> parameters) throws AudioProcessingException;
    
    /**
     * 获取处理器支持的最小音频长度 / Get minimum audio length supported by processor
     * 
     * @return 最小长度（秒）/ Minimum length in seconds
     */
    double getMinimumAudioLength();
    
    /**
     * 获取处理器支持的最大音频长度 / Get maximum audio length supported by processor
     * 
     * @return 最大长度（秒）/ Maximum length in seconds
     */
    double getMaximumAudioLength();
    
    /**
     * 获取处理器的计算复杂度估计 / Get computational complexity estimate of processor
     * 
     * @param audioLength 音频长度（秒）/ Audio length in seconds
     * @return 复杂度估计 / Complexity estimate
     */
    double getComplexityEstimate(double audioLength);
    
    /**
     * 预热处理器 / Warm up processor
     * <p>
     * 执行一些初始化操作以提高后续处理的性能。
     * Performs initialization operations to improve performance of subsequent processing.
     * </p>
     * 
     * @throws AudioProcessingException 预热失败时抛出异常 / Thrown when warm-up fails
     */
    void warmUp() throws AudioProcessingException;
    
    /**
     * 清理处理器资源 / Clean up processor resources
     * <p>
     * 释放处理器占用的资源，如内存、临时文件等。
     * Releases resources used by processor, such as memory, temporary files, etc.
     * </p>
     */
    void cleanup();
    
    /**
     * 获取最后一次处理的统计信息 / Get statistics of last processing
     * 
     * @return 统计信息映射 / Statistics map
     */
    Map<String, Object> getLastProcessingStatistics();
    
    /**
     * 获取处理器的性能指标 / Get performance metrics of processor
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
    
    /**
     * 获取处理器支持的处理类型 / Get processing types supported by processor
     * 
     * @return 处理类型数组 / Array of processing types
     */
    String[] getSupportedProcessingTypes();
    
    /**
     * 检查是否支持指定的处理类型 / Check if specified processing type is supported
     * 
     * @param processingType 处理类型 / Processing type
     * @return 是否支持 / Whether supported
     */
    boolean supportsProcessingType(String processingType);
    
    /**
     * 获取处理质量级别 / Get processing quality levels
     * 
     * @return 质量级别数组 / Array of quality levels
     */
    String[] getQualityLevels();
    
    /**
     * 设置处理质量级别 / Set processing quality level
     * 
     * @param qualityLevel 质量级别 / Quality level
     * @throws AudioProcessingException 质量级别无效时抛出异常 / Thrown when quality level is invalid
     */
    void setQualityLevel(String qualityLevel) throws AudioProcessingException;
}