package com.yishape.lab.audio.core;

import java.util.Map;
import java.util.Set;

/**
 * 标准化音频组件接口 / Standardized Audio Component Interface
 * <p>
 * 定义音频组件的标准接口，所有音频组件实现都应遵循此接口。
 * Defines the standard interface for audio components, all audio component implementations should follow this interface.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public interface IAudioComponentStandard {
    
    /**
     * 获取组件名称 / Get Component Name
     * 
     * @return 组件名称 / Component name
     */
    String getName();
    
    /**
     * 获取组件描述 / Get Component Description
     * 
     * @return 组件描述 / Component description
     */
    String getDescription();
    
    /**
     * 获取支持的参数 / Get Supported Parameters
     * 
     * @return 支持的参数名称列表 / List of supported parameter names
     */
    Set<String> getSupportedParameters();
    
    /**
     * 获取默认参数 / Get Default Parameters
     * 
     * @return 默认参数映射 / Default parameter mapping
     */
    Map<String, Object> getDefaultParameters();
    
    /**
     * 验证输入音频 / Validate Input Audio
     * <p>
     * 验证输入音频是否符合组件的要求。
     * Validates if input audio meets component requirements.
     * </p>
     * 
     * @param input 输入音频 / Input audio
     * @return 验证结果 / Validation result
     */
    boolean validateInput(AudioData input);
    
    /**
     * 验证参数 / Validate Parameters
     * <p>
     * 验证参数是否有效。
     * Validates if parameters are valid.
     * </p>
     * 
     * @param parameters 参数映射 / Parameter mapping
     * @return 验证结果 / Validation result
     */
    boolean validateParameters(Map<String, Object> parameters);
    
    /**
     * 克隆组件 / Clone Component
     * <p>
     * 创建组件的副本，用于并行处理。
     * Creates a copy of component for parallel processing.
     * </p>
     * 
     * @return 组件副本 / Component copy
     */
    IAudioComponentStandard clone();
    
    /**
     * 获取组件版本 / Get Component Version
     * 
     * @return 版本字符串 / Version string
     */
    default String getVersion() {
        return "1.0";
    }
    
    /**
     * 是否支持并行处理 / Supports Parallel Processing
     * 
     * @return 是否支持并行处理 / Whether parallel processing is supported
     */
    default boolean supportsParallel() {
        return true;
    }
    
    /**
     * 是否支持GPU加速 / Supports GPU Acceleration
     * 
     * @return 是否支持GPU加速 / Whether GPU acceleration is supported
     */
    default boolean supportsGPU() {
        return false;
    }
    
    /**
     * 获取估计的处理时间复杂度 / Get Estimated Time Complexity
     * 
     * @param audioSize 音频尺寸 / Audio size
     * @return 时间复杂度描述 / Time complexity description
     */
    default String getTimeComplexity(int audioSize) {
        return "O(n)";
    }
    
    /**
     * 获取内存使用估计 / Get Memory Usage Estimate
     * 
     * @param audioSize 音频尺寸 / Audio size
     * @return 内存使用估计（字节） / Memory usage estimate in bytes
     */
    default long getMemoryUsage(int audioSize) {
        return audioSize * 8L; // Default: 8 bytes per sample
    }
}