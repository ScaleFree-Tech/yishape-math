package com.reremouse.lab.music.analysis.basic;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import java.util.Map;
import java.util.List;

/**
 * 和弦分析器接口 / Chord Analyzer Interface
 * <p>
 * 定义和弦检测和分析的基本功能
 * Defines basic functionality for chord detection and analysis.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IChordAnalyzer {
    
    /**
     * 检测音频中的和弦 / Detect chords in audio
     *
     * @param audioData 音频数据 / Audio data
     * @return 和弦检测结果列表 / List of chord detection results
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    List<ChordDetectionResult> detectChords(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 检测音频中的和弦（带参数） / Detect chords in audio with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 分析参数 / Analysis parameters
     * @return 和弦检测结果列表 / List of chord detection results
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    List<ChordDetectionResult> detectChords(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;
    
    /**
     * 检测指定时间段的和弦 / Detect chord in specific time segment
     *
     * @param audioData 音频数据 / Audio data
     * @param startTime 开始时间（秒） / Start time in seconds
     * @param endTime 结束时间（秒） / End time in seconds
     * @return 和弦检测结果 / Chord detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    ChordDetectionResult detectChordInSegment(AudioData audioData, double startTime, double endTime) throws AudioProcessingException;
    
    /**
     * 检测指定时间段的和弦（带参数） / Detect chord in specific time segment with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param startTime 开始时间（秒） / Start time in seconds
     * @param endTime 结束时间（秒） / End time in seconds
     * @param parameters 分析参数 / Analysis parameters
     * @return 和弦检测结果 / Chord detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    ChordDetectionResult detectChordInSegment(AudioData audioData, double startTime, double endTime, Map<String, Object> parameters) throws AudioProcessingException;
    
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
}