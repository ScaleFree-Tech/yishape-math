package com.reremouse.lab.audio.enhancement;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.IAudioComponentStandard;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Collections;

/**
 * 标准化音频增强器抽象基类 / Standardized Audio Enhancer Abstract Base Class
 * <p>
 * 提供音频增强器的通用实现，包括参数管理、验证和克隆功能。
 * Provides common implementation for audio enhancers, including parameter management, validation, and cloning functionality.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public abstract class AbstractAudioEnhancer implements IAudioEnhancer {
    
    // 基本属性 / Basic properties
    protected String name;
    protected String description;
    protected String version;
    protected EnhancerType enhancerType;
    
    // 参数管理 / Parameter management
    protected Set<String> supportedParameters = new HashSet<>();
    protected Map<String, Object> defaultParameters = new HashMap<>();
    
    /**
     * 构造函数 / Constructor
     *
     * @param name 增强器名称 / Enhancer name
     * @param description 增强器描述 / Enhancer description
     * @param enhancerType 增强器类型 / Enhancer type
     */
    protected AbstractAudioEnhancer(String name, String description, EnhancerType enhancerType) {
        this.name = name;
        this.description = description;
        this.enhancerType = enhancerType;
        this.version = "1.0";
    }
    
    @Override
    public AudioData enhance(AudioData input) throws AudioProcessingException {
        return enhance(input, null);
    }
    
    @Override
    public AudioData enhance(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        // 验证输入 / Validate input
        if (!validateInput(input)) {
            throw new AudioProcessingException("Invalid input audio data");
        }
        
        // 验证参数 / Validate parameters
        if (parameters != null && !validateParameters(parameters)) {
            throw new AudioProcessingException("Invalid parameters provided");
        }
        
        // 合并参数 / Merge parameters
        Map<String, Object> mergedParameters = new HashMap<>();
        if (defaultParameters != null) {
            mergedParameters.putAll(defaultParameters);
        }
        if (parameters != null) {
            mergedParameters.putAll(parameters);
        }
        
        // 执行增强 / Execute enhancement
        return doEnhance(input, mergedParameters);
    }
    
    /**
     * 执行增强操作（子类实现） / Execute enhancement operation (implemented by subclasses)
     *
     * @param input 输入音频数据 / Input audio data
     * @param parameters 处理参数 / Processing parameters
     * @return 增强后的音频数据 / Enhanced audio data
     * @throws AudioProcessingException 处理异常 / Processing exception
     */
    protected abstract AudioData doEnhance(AudioData input, Map<String, Object> parameters) throws AudioProcessingException;
    
    /**
     * 添加支持的参数 / Add supported parameter
     *
     * @param name 参数名称 / Parameter name
     * @param defaultValue 默认值 / Default value
     */
    protected void addSupportedParameter(String name, Object defaultValue) {
        supportedParameters.add(name);
        if (defaultValue != null) {
            defaultParameters.put(name, defaultValue);
        }
    }
    
    @Override
    public String getName() {
        return name;
    }
    
    @Override
    public String getDescription() {
        return description;
    }
    
    @Override
    public Set<String> getSupportedParameters() {
        return new HashSet<>(supportedParameters);
    }
    
    @Override
    public Map<String, Object> getDefaultParameters() {
        return new HashMap<>(defaultParameters);
    }
    
    @Override
    public boolean validateInput(AudioData input) {
        return input != null && input.getSamples() != null && input.getSamples().length() > 0;
    }
    
    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        if (parameters == null) {
            return true;
        }
        
        for (String paramName : parameters.keySet()) {
            if (!supportedParameters.contains(paramName)) {
                return false;
            }
        }
        
        return true;
    }
    
    @Override
    public EnhancerType getEnhancerType() {
        return enhancerType;
    }
    
    @Override
    public IAudioComponentStandard clone() {
        try {
            AbstractAudioEnhancer cloned = (AbstractAudioEnhancer) super.clone();
            // 深拷贝参数映射 / Deep copy parameter maps
            cloned.defaultParameters = new HashMap<>(this.defaultParameters);
            cloned.supportedParameters = new HashSet<>(this.supportedParameters);
            return cloned;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException("Cloning not supported", e);
        }
    }
    
    @Override
    public String getVersion() {
        return version;
    }
    
    /**
     * 设置版本 / Set version
     *
     * @param version 版本字符串 / Version string
     */
    protected void setVersion(String version) {
        this.version = version;
    }
    
    /**
     * 设置增强器类型 / Set enhancer type
     *
     * @param enhancerType 增强器类型 / Enhancer type
     */
    protected void setEnhancerType(EnhancerType enhancerType) {
        this.enhancerType = enhancerType;
    }
}