package com.reremouse.lab.audio.feature;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;

import java.util.Map;

import com.reremouse.lab.audio.feature.AudioFeatureResult;
import com.reremouse.lab.audio.feature.TimeDomainFeatureResult;
import com.reremouse.lab.audio.feature.FrequencyDomainFeatureResult;
import com.reremouse.lab.audio.feature.SpectralFeatureResult;

/**
 * 音频特征提取器接口 / Audio Feature Extractor Interface
 * <p>
 * 定义音频特征提取的标准接口，包括时域特征、频域特征等。
 * Defines the standard interface for audio feature extraction, including time-domain features, frequency-domain features, etc.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public interface IAudioFeatureExtractor {

    /**
     * 提取完整的音频特征集合 / Extract complete set of audio features
     *
     * @param audioData 音频数据 / Audio data
     * @return 结构化的音频特征结果 / Structured audio feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioFeatureResult extractAudioFeatures(AudioData audioData) throws AudioProcessingException;

    /**
     * 提取完整的音频特征集合（带参数） / Extract complete set of audio features with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 提取参数 / Extraction parameters
     * @return 结构化的音频特征结果 / Structured audio feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioFeatureResult extractAudioFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 提取时域特征 / Extract time-domain features
     *
     * @param audioData 音频数据 / Audio data
     * @return 时域特征结果 / Time-domain feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    TimeDomainFeatureResult extractTimeDomainFeatures(AudioData audioData) throws AudioProcessingException;

    /**
     * 提取时域特征（带参数） / Extract time-domain features with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 提取参数 / Extraction parameters
     * @return 时域特征结果 / Time-domain feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    TimeDomainFeatureResult extractTimeDomainFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 提取频域特征 / Extract frequency-domain features
     *
     * @param audioData 音频数据 / Audio data
     * @return 频域特征结果 / Frequency-domain feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    FrequencyDomainFeatureResult extractFrequencyDomainFeatures(AudioData audioData) throws AudioProcessingException;

    /**
     * 提取频域特征（带参数） / Extract frequency-domain features with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 提取参数 / Extraction parameters
     * @return 频域特征结果 / Frequency-domain feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    FrequencyDomainFeatureResult extractFrequencyDomainFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 提取谱特征 / Extract spectral features
     *
     * @param audioData 音频数据 / Audio data
     * @return 谱特征结果 / Spectral feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    SpectralFeatureResult extractSpectralFeatures(AudioData audioData) throws AudioProcessingException;

    /**
     * 提取谱特征（带参数） / Extract spectral features with parameters
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 提取参数 / Extraction parameters
     * @return 谱特征结果 / Spectral feature result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    SpectralFeatureResult extractSpectralFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 获取支持的音频特征类型 / Get supported audio feature types
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