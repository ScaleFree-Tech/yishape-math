package com.reremouse.lab.music.analysis.basic;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import java.util.Map;

/**
 * 节拍分析器接口 / Beat Analyzer Interface
 * <p>
 * 定义节拍检测和分析的基本功能
 * Defines basic functionality for beat detection and analysis.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IBeatAnalyzer {
    
    /**
     * 检测音频中的节拍 / Detect beats in audio
     *
     * @param audioData 音频数据 / Audio data
     * @return 节拍检测结果 / Beat detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    BeatDetectionResult detectBeats(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 检测音频中的节拍（带参数） / Detect beats in audio with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 分析参数 / Analysis parameters
     * @return 节拍检测结果 / Beat detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    BeatDetectionResult detectBeats(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;
    
    /**
     * 估算音频的节拍速度 / Estimate tempo of audio
     *
     * @param audioData 音频数据 / Audio data
     * @return 节拍速度（BPM） / Tempo in BPM
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    double estimateTempo(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 估算音频的节拍速度（带参数） / Estimate tempo of audio with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 分析参数 / Analysis parameters
     * @return 节拍速度（BPM） / Tempo in BPM
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    double estimateTempo(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;
    
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