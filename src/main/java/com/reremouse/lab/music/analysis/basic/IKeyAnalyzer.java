package com.reremouse.lab.music.analysis.basic;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import java.util.Map;

/**
 * 调性分析器接口 / Key Analyzer Interface
 * <p>
 * 定义音乐调性检测和分析的基本功能。
 * Defines basic functionality for musical key detection and analysis.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IKeyAnalyzer {

    /**
     * 检测音频的调性 / Detect key of audio
     *
     * @param audioData 音频数据 / Audio data
     * @return 调性检测结果 / Key detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    KeyDetectionResult detectKey(AudioData audioData) throws AudioProcessingException;

    /**
     * 检测音频的调性（带参数） / Detect key of audio with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 分析参数 / Analysis parameters
     * @return 调性检测结果 / Key detection result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    KeyDetectionResult detectKey(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 分析音频的色度特征 / Analyze chroma features of audio
     *
     * @param audioData 音频数据 / Audio data
     * @return 色度特征向量 / Chroma feature vector
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    double[] analyzeChromaFeatures(AudioData audioData) throws AudioProcessingException;

    /**
     * 分析音频的色度特征（带参数） / Analyze chroma features of audio with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 分析参数 / Analysis parameters
     * @return 色度特征向量 / Chroma feature vector
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    double[] analyzeChromaFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;

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