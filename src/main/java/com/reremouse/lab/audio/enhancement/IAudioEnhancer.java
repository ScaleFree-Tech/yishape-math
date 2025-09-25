package com.reremouse.lab.audio.enhancement;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.IAudioComponentStandard;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import java.util.Map;

/**
 * 标准化音频增强器接口 / Standardized Audio Enhancer Interface
 * <p>
 * 定义音频增强器的标准接口，所有音频增强器实现都应遵循此接口。
 * Defines the standard interface for audio enhancers, all audio enhancer implementations should follow this interface.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public interface IAudioEnhancer extends IAudioComponentStandard {
    
    /**
     * 增强音频数据 / Enhance audio data
     *
     * @param input 输入音频数据 / Input audio data
     * @return 增强后的音频数据 / Enhanced audio data
     * @throws AudioProcessingException 处理异常 / Processing exception
     */
    AudioData enhance(AudioData input) throws AudioProcessingException;
    
    /**
     * 增强音频数据（带参数） / Enhance audio data (with parameters)
     *
     * @param input 输入音频数据 / Input audio data
     * @param parameters 处理参数 / Processing parameters
     * @return 增强后的音频数据 / Enhanced audio data
     * @throws AudioProcessingException 处理异常 / Processing exception
     */
    AudioData enhance(AudioData input, Map<String, Object> parameters) throws AudioProcessingException;
    
    /**
     * 获取增强器类型 / Get enhancer type
     *
     * @return 增强器类型 / Enhancer type
     */
    EnhancerType getEnhancerType();
    
    /**
     * 增强器类型枚举 / Enhancer Type Enum
     */
    enum EnhancerType {
        NOISE_REDUCTION,    // 降噪 / Noise reduction
        EQUALIZATION,       // 均衡 / Equalization
        COMPRESSION,        // 压缩 / Compression
        REVERB,             // 混响 / Reverb
        NORMALIZATION,      // 标准化 / Normalization
        DYNAMIC_RANGE,      // 动态范围 / Dynamic range
        SPECTRAL_PROCESSING // 频谱处理 / Spectral processing
    }
}