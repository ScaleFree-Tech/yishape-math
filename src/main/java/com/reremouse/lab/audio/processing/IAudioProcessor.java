package com.reremouse.lab.audio.processing;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;

/**
 * 音频处理器接口 / Audio Processor Interface
 * <p>
 * 定义音频处理器的基本操作，包括音频处理和参数设置。
 * 所有音频处理器都应该实现此接口，确保一致的API设计。
 * </p>
 * <p>
 * Defines basic operations for audio processors, including audio processing and parameter settings.
 * All audio processors should implement this interface to ensure consistent API design.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IAudioProcessor {
    
    /**
     * 处理音频数据 / Process audio data
     * <p>
     * 对输入的音频数据执行特定的处理操作。
     * Perform specific processing operation on input audio data.
     * </p>
     *
     * @param input 输入音频数据 / Input audio data
     * @return 处理后的音频数据 / Processed audio data
     * @throws AudioProcessingException 当处理过程中发生错误时抛出 / Thrown when error occurs during processing
     */
    AudioData process(AudioData input) throws AudioProcessingException;
    
    /**
     * 设置处理器参数 / Set processor parameters
     * <p>
     * 设置处理器的配置参数。参数格式由具体实现决定。
     * Set configuration parameters for the processor. Parameter format is determined by specific implementation.
     * </p>
     *
     * @param key 参数键 / Parameter key
     * @param value 参数值 / Parameter value
     * @throws IllegalArgumentException 当参数无效时抛出 / Thrown when parameter is invalid
     */
    void setParameter(String key, Object value) throws IllegalArgumentException;
    
    /**
     * 获取处理器参数 / Get processor parameter
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
     * 重置处理器参数 / Reset processor parameters
     * <p>
     * 将所有参数重置为默认值。
     * Reset all parameters to default values.
     * </p>
     */
    void reset();
    
    /**
     * 获取处理器名称 / Get processor name
     * <p>
     * 返回处理器的唯一标识名称。
     * Return unique identifier name of the processor.
     * </p>
     *
     * @return 处理器名称 / Processor name
     */
    String getName();
    
    /**
     * 获取处理器版本 / Get processor version
     * <p>
     * 返回处理器的版本信息。
     * Return version information of the processor.
     * </p>
     *
     * @return 版本信息 / Version information
     */
    String getVersion();
    
    /**
     * 克隆处理器 / Clone processor
     * <p>
     * 创建处理器的深度克隆，包括所有参数设置。
     * Create deep clone of the processor, including all parameter settings.
     * </p>
     *
     * @return 处理器克隆 / Processor clone
     */
    IAudioProcessor clone();
    
    /**
     * 检查是否支持指定格式 / Check if supports specified format
     * <p>
     * 检查处理器是否支持处理指定格式的音频数据。
     * Check if processor supports processing audio data of specified format.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @return 如果支持返回true / Return true if supported
     */
    boolean supportsFormat(AudioData audioData);
    
    /**
     * 获取处理延迟 / Get processing latency
     * <p>
     * 返回处理器引入的延迟（以样本数为单位）。
     * Return latency introduced by processor (in number of samples).
     * </p>
     *
     * @return 延迟样本数 / Latency in samples
     */
    int getLatency();
}