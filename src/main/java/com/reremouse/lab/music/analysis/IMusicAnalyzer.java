package com.reremouse.lab.music.analysis;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.music.analysis.basic.BeatDetectionResult;
import com.reremouse.lab.music.analysis.basic.KeyDetectionResult;
import com.reremouse.lab.music.analysis.basic.ChordDetectionResult;

import java.util.Map;

/**
 * 音乐分析器主接口 / Main Music Analyzer Interface
 * <p>
 * 结合基本音乐分析功能的主要接口，提供统一的音乐分析入口。
 * Main interface combining basic music analysis functionality, providing unified music analysis entry point.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IMusicAnalyzer {

    /**
     * 执行完整的音乐分析 / Perform complete music analysis
     * <p>
     * 对音频数据执行全面的音乐分析，包括节拍、调性、和弦等基本分析。
     * Performs comprehensive music analysis on audio data, including beat, key, chord and other basic analyses.
     * </p>
     * 
     * @param audioData 音频数据 / Audio data to analyze
     * @return 音乐检测结果 / Music detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    MusicDetectionResult analyzeMusic(AudioData audioData) throws AudioProcessingException;

    /**
     * 执行带参数的完整音乐分析 / Perform complete music analysis with parameters
     * 
     * @param audioData 音频数据 / Audio data to analyze
     * @param parameters 分析参数 / Analysis parameters
     * @return 音乐检测结果 / Music detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    MusicDetectionResult analyzeMusic(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 分析音频的节拍信息 / Analyze beat information of audio
     * 
     * @param audioData 音频数据 / Audio data to analyze
     * @return 节拍检测结果 / Beat detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    BeatDetectionResult analyzeBeat(AudioData audioData) throws AudioProcessingException;

    /**
     * 分析音频的节拍信息（带参数）/ Analyze beat information with parameters
     * 
     * @param audioData 音频数据 / Audio data to analyze
     * @param parameters 分析参数 / Analysis parameters
     * @return 节拍检测结果 / Beat detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    BeatDetectionResult analyzeBeat(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 分析音频的调性信息 / Analyze key information of audio
     * 
     * @param audioData 音频数据 / Audio data to analyze
     * @return 调性检测结果 / Key detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    KeyDetectionResult analyzeKey(AudioData audioData) throws AudioProcessingException;

    /**
     * 分析音频的调性信息（带参数）/ Analyze key information with parameters
     * 
     * @param audioData 音频数据 / Audio data to analyze
     * @param parameters 分析参数 / Analysis parameters
     * @return 调性检测结果 / Key detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    KeyDetectionResult analyzeKey(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 分析音频的和弦信息 / Analyze chord information of audio
     * 
     * @param audioData 音频数据 / Audio data to analyze
     * @return 和弦检测结果 / Chord detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    ChordDetectionResult analyzeChord(AudioData audioData) throws AudioProcessingException;

    /**
     * 分析音频的和弦信息（带参数）/ Analyze chord information with parameters
     * 
     * @param audioData 音频数据 / Audio data to analyze
     * @param parameters 分析参数 / Analysis parameters
     * @return 和弦检测结果 / Chord detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    ChordDetectionResult analyzeChord(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 分析指定时间段的音频 / Analyze audio in specified time range
     * 
     * @param audioData 音频数据 / Audio data to analyze
     * @param startTime 开始时间（秒）/ Start time in seconds
     * @param endTime 结束时间（秒）/ End time in seconds
     * @return 音乐检测结果 / Music detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    MusicDetectionResult analyzeTimeRange(AudioData audioData, double startTime, double endTime) throws AudioProcessingException;

    /**
     * 分析指定时间段的音频（带参数）/ Analyze audio in specified time range with parameters
     * 
     * @param audioData 音频数据 / Audio data to analyze
     * @param startTime 开始时间（秒）/ Start time in seconds
     * @param endTime 结束时间（秒）/ End time in seconds
     * @param parameters 分析参数 / Analysis parameters
     * @return 音乐检测结果 / Music detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    MusicDetectionResult analyzeTimeRange(AudioData audioData, double startTime, double endTime, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 分析音频的实时流 / Analyze real-time audio stream
     * 
     * @param audioData 音频数据 / Audio data to analyze
     * @param windowSize 窗口大小（秒）/ Window size in seconds
     * @param hopSize 跳跃大小（秒）/ Hop size in seconds
     * @return 音乐检测结果数组 / Array of music detection results
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    MusicDetectionResult[] analyzeStream(AudioData audioData, double windowSize, double hopSize) throws AudioProcessingException;

    /**
     * 分析音频的实时流（带参数）/ Analyze real-time audio stream with parameters
     * 
     * @param audioData 音频数据 / Audio data to analyze
     * @param windowSize 窗口大小（秒）/ Window size in seconds
     * @param hopSize 跳跃大小（秒）/ Hop size in seconds
     * @param parameters 分析参数 / Analysis parameters
     * @return 音乐检测结果数组 / Array of music detection results
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    MusicDetectionResult[] analyzeStream(AudioData audioData, double windowSize, double hopSize, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 获取分析器支持的参数 / Get supported parameters by the analyzer
     * 
     * @return 支持的参数名称数组 / Array of supported parameter names
     */
    String[] getSupportedParameters();

    /**
     * 验证分析参数 / Validate analysis parameters
     * 
     * @param parameters 要验证的参数 / Parameters to validate
     * @return 参数是否有效 / Whether parameters are valid
     */
    boolean validateParameters(Map<String, Object> parameters);

    /**
     * 获取默认分析参数 / Get default analysis parameters
     * 
     * @return 默认参数映射 / Default parameters map
     */
    Map<String, Object> getDefaultParameters();

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
     * 获取分析器版本信息 / Get analyzer version information
     * 
     * @return 版本字符串 / Version string
     */
    String getVersion();

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
     * @return 复杂度估计 / Complexity estimate
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