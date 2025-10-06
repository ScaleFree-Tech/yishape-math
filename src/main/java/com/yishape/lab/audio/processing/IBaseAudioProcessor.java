package com.yishape.lab.audio.processing;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;

import java.util.Map;

/**
 * 基础音频处理器接口 / Base Audio Processor Interface
 * <p>
 * 定义音频处理器的基础操作，包括基本的处理器信息和状态管理。
 * Defines basic operations for audio processors, including basic processor information and state management.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IBaseAudioProcessor extends IAudioProcessor {
    
    /**
     * 处理音频数据 / Process audio data
     * <p>
     * 对输入的音频数据执行特定的处理操作。
     * Perform specific processing operation on input audio data.
     * </p>
     *
     * @param input 输入音频数据 / Input audio data
     * @param parameters 处理参数 / Processing parameters
     * @return 处理后的音频数据 / Processed audio data
     * @throws AudioProcessingException 当处理过程中发生错误时抛出 / Thrown when error occurs during processing
     */
    AudioData process(AudioData input, Map<String, Object> parameters) throws AudioProcessingException;
    
    /**
     * 获取处理器描述 / Get processor description
     * <p>
     * 返回处理器的详细描述信息。
     * Return detailed description information of the processor.
     * </p>
     *
     * @return 处理器描述 / Processor description
     */
    String getDescription();
    
    /**
     * 获取支持的参数列表 / Get supported parameter list
     * <p>
     * 返回处理器支持的所有参数名称。
     * Return all parameter names supported by the processor.
     * </p>
     *
     * @return 支持的参数名称数组 / Array of supported parameter names
     */
    String[] getSupportedParameters();
    
    /**
     * 获取默认参数 / Get default parameters
     * <p>
     * 返回处理器的默认参数映射。
     * Return default parameter mapping of the processor.
     * </p>
     *
     * @return 默认参数映射 / Default parameter mapping
     */
    Map<String, Object> getDefaultParameters();
    
    /**
     * 检查处理器是否就绪 / Check if processor is ready
     * <p>
     * 检查处理器是否已准备好进行处理操作。
     * Check if processor is ready for processing operations.
     * </p>
     *
     * @return 如果就绪返回true / Return true if ready
     */
    boolean isReady();
    
    /**
     * 验证参数 / Validate parameters
     * <p>
     * 验证提供的参数是否有效。
     * Validate if provided parameters are valid.
     * </p>
     *
     * @param parameters 参数映射 / Parameter mapping
     * @return 如果参数有效返回true / Return true if parameters are valid
     */
    boolean validateParameters(Map<String, Object> parameters);
    
    /**
     * 设置参数 / Set parameters
     * <p>
     * 设置处理器的参数。
     * Set parameters of the processor.
     * </p>
     *
     * @param parameters 参数映射 / Parameter mapping
     * @throws AudioProcessingException 当参数设置失败时抛出 / Thrown when parameter setting fails
     */
    void setParameters(Map<String, Object> parameters) throws AudioProcessingException;
    
    /**
     * 获取当前参数 / Get current parameters
     * <p>
     * 获取处理器的当前参数设置。
     * Get current parameter settings of the processor.
     * </p>
     *
     * @return 当前参数映射 / Current parameter mapping
     */
    Map<String, Object> getCurrentParameters();
    
    /**
     * 重置参数 / Reset parameters
     * <p>
     * 将所有参数重置为默认值。
     * Reset all parameters to default values.
     * </p>
     */
    void resetParameters();
    
    /**
     * 检查是否支持指定的音频格式 / Check if supports specified audio format
     * <p>
     * 检查处理器是否支持处理指定采样率、声道数和位深度的音频。
     * Check if processor supports processing audio with specified sample rate, number of channels, and bit depth.
     * </p>
     *
     * @param sampleRate 采样率 / Sample rate
     * @param channels 声道数 / Number of channels
     * @param bitDepth 位深度 / Bit depth
     * @return 如果支持返回true / Return true if supported
     */
    boolean supportsAudioFormat(double sampleRate, int channels, int bitDepth);
    
    /**
     * 获取状态 / Get status
     * <p>
     * 返回处理器的当前状态。
     * Return current status of the processor.
     * </p>
     *
     * @return 处理器状态 / Processor status
     */
    String getStatus();
}