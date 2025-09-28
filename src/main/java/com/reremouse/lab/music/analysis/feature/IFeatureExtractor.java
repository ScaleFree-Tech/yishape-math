package com.reremouse.lab.music.analysis.feature;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;

import java.util.Map;

/**
 * 音乐特征提取器接口 / Music Feature Extractor Interface
 * <p>
 * 专注于音乐层面的特征提取，包括节拍模式、调性稳定性、和声复杂度、音乐结构等高层次音乐特征。
 * 与audio.features包的底层音频特征（频谱、MFCC等）形成互补，专注于音乐理论和感知层面的特征。
 * Focuses on music-level feature extraction, including rhythm patterns, tonal stability, harmonic complexity, 
 * musical structure and other high-level musical features. Complements the low-level audio features 
 * (spectrum, MFCC, etc.) in the audio.features package, focusing on music theory and perceptual aspects.
 * </p>
 *
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public interface IFeatureExtractor {

    /**
     * 提取完整的音乐特征集合 / Extract complete set of music features
     *
     * @param audioData 音频数据 / Audio data
     * @return 结构化的音乐特征结果 / Structured music feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    MusicFeatureResult extractMusicFeatures(AudioData audioData) throws AudioProcessingException;

    /**
     * 提取完整的音乐特征集合（带参数） / Extract complete set of music features with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 提取参数 / Extraction parameters
     * @return 结构化的音乐特征结果 / Structured music feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    MusicFeatureResult extractMusicFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 提取节拍和律动特征 / Extract rhythm and beat features
     *
     * @param audioData 音频数据 / Audio data
     * @return 节拍特征结果 / Rhythm feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    RhythmFeatureResult extractRhythmFeatures(AudioData audioData) throws AudioProcessingException;

    /**
     * 提取节拍和律动特征（带参数） / Extract rhythm and beat features with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 提取参数 / Extraction parameters
     * @return 节拍特征结果 / Rhythm feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    RhythmFeatureResult extractRhythmFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 提取调性和和声特征 / Extract tonal and harmonic features
     *
     * @param audioData 音频数据 / Audio data
     * @return 调性特征结果 / Tonal feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    TonalFeatureResult extractTonalFeatures(AudioData audioData) throws AudioProcessingException;

    /**
     * 提取调性和和声特征（带参数） / Extract tonal and harmonic features with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 提取参数 / Extraction parameters
     * @return 调性特征结果 / Tonal feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    TonalFeatureResult extractTonalFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 提取音乐结构特征 / Extract musical structure features
     *
     * @param audioData 音频数据 / Audio data
     * @return 结构特征结果 / Structure feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    StructureFeatureResult extractStructureFeatures(AudioData audioData) throws AudioProcessingException;

    /**
     * 提取音乐结构特征（带参数） / Extract musical structure features with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 提取参数 / Extraction parameters
     * @return 结构特征结果 / Structure feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    StructureFeatureResult extractStructureFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 提取音乐表现力特征 / Extract musical expressiveness features
     *
     * @param audioData 音频数据 / Audio data
     * @return 表现力特征结果 / Expressiveness feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    ExpressivenessFeatureResult extractExpressivenessFeatures(AudioData audioData) throws AudioProcessingException;

    /**
     * 提取音乐表现力特征（带参数） / Extract musical expressiveness features with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 提取参数 / Extraction parameters
     * @return 表现力特征结果 / Expressiveness feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    ExpressivenessFeatureResult extractExpressivenessFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 获取支持的音乐特征类型 / Get supported music feature types
     *
     * @return 支持的特征类型数组 / Array of supported feature types
     */
    String[] getSupportedFeatureTypes();

    /**
     * 检查是否支持指定特征类型 / Check if a specific feature type is supported
     *
     * @param featureType 特征类型名称 / Feature type name
     * @return 是否支持该特征类型 / Whether the feature type is supported
     */
    boolean isFeatureTypeSupported(String featureType);

    /**
     * 获取特征提取器名称 / Get feature extractor name
     *
     * @return 特征提取器名称 / Feature extractor name
     */
    String getExtractorName();

    /**
     * 获取特征提取器版本 / Get feature extractor version
     *
     * @return 版本号 / Version number
     */
    String getVersion();

    /**
     * 获取支持的参数列表 / Get list of supported parameters
     *
     * @return 支持的参数名称数组 / Array of supported parameter names
     */
    String[] getSupportedParameters();

    /**
     * 获取默认参数 / Get default parameters
     *
     * @return 默认参数映射 / Default parameter map
     */
    Map<String, Object> getDefaultParameters();

    /**
     * 验证参数有效性 / Validate parameter validity
     *
     * @param parameters 要验证的参数 / Parameters to validate
     * @return 参数是否有效 / Whether parameters are valid
     */
    boolean validateParameters(Map<String, Object> parameters);
}