package com.reremouse.lab.audio.effect;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;

import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Collections;

/**
 * 统一音频效果器抽象基类 / Unified Audio Effect Abstract Base Class
 * <p>
 * 提供统一音频效果器接口的基本实现，包括参数管理、验证和克隆功能。
 * 所有具体的音频效果器实现都应该继承此类。
 * </p>
 * <p>
 * Provides basic implementation of unified audio effect interface, including parameter management, validation, and cloning.
 * All concrete audio effect implementations should extend this class.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public abstract class AbstractAudioEffect implements IAudioEffect {
    
    protected String name;
    protected String description;
    protected EffectType effectType;
    protected Map<String, Object> defaultParameters;
    protected Set<String> supportedParameters;
    protected double dryWetMix = 1.0; // 默认全湿声 / Default all wet signal
    protected double intensity = 1.0; // 默认强度 / Default intensity
    protected boolean enabled = true; // 默认启用 / Default enabled
    protected String[] presets = new String[0]; // 默认无预设 / Default no presets
    
    /**
     * 构造函数 / Constructor
     * 
     * @param name 效果器名称 / Effect name
     * @param description 效果器描述 / Effect description
     * @param effectType 效果器类型 / Effect type
     */
    public AbstractAudioEffect(String name, String description, EffectType effectType) {
        this.name = name;
        this.description = description;
        this.effectType = effectType;
        this.defaultParameters = new HashMap<>();
        this.supportedParameters = new HashSet<>();
        
        // 添加默认支持的参数 / Add default supported parameters
        addSupportedParameter("effectType", this.effectType);
        addSupportedParameter("dryWetMix", this.dryWetMix);
        addSupportedParameter("intensity", this.intensity);
        addSupportedParameter("enabled", this.enabled);
    }
    
    @Override
    public AudioData process(AudioData input) throws AudioProcessingException {
        return applyEffect(input);
    }
    
    @Override
    public AudioData applyEffect(AudioData input) throws AudioProcessingException {
        return applyEffect(input, getDefaultParameters());
    }
    
    @Override
    public AudioData applyEffect(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        // 验证输入 / Validate input
        if (!validateInput(input)) {
            throw new AudioProcessingException("Invalid input audio data");
        }
        
        // 验证参数 / Validate parameters
        if (!validateParameters(parameters)) {
            throw new AudioProcessingException("Invalid parameters");
        }
        
        // 更新参数 / Update parameters
        if (parameters != null) {
            if (parameters.containsKey("effectType")) {
                this.effectType = (EffectType) parameters.get("effectType");
            }
            if (parameters.containsKey("dryWetMix")) {
                setDryWetMix((Double) parameters.get("dryWetMix"));
            }
            if (parameters.containsKey("intensity")) {
                setIntensity((Double) parameters.get("intensity"));
            }
            if (parameters.containsKey("enabled")) {
                setEnabled((Boolean) parameters.get("enabled"));
            }
        }
        
        // 如果禁用则直接返回原音频 / If disabled, return original audio directly
        if (!isEnabled()) {
            return input;
        }
        
        // 执行效果应用 / Perform effect application
        return doApplyEffect(input);
    }
    
    /**
     * 执行实际的效果应用操作 / Perform actual effect application operation
     * <p>
     * 子类必须实现此方法以提供具体的效果应用逻辑。
     * Subclasses must implement this method to provide specific effect application logic.
     * </p>
     * 
     * @param input 输入音频 / Input audio
     * @return 应用效果后的音频 / Audio with applied effect
     * @throws AudioProcessingException 效果应用过程中发生错误 / Error occurred during effect application
     */
    protected abstract AudioData doApplyEffect(AudioData input) throws AudioProcessingException;
    
    @Override
    public EffectType getEffectType() {
        return effectType;
    }
    
    @Override
    public void setEffectType(EffectType effectType) {
        this.effectType = effectType;
    }
    
    @Override
    public void setDryWetMix(double mix) throws IllegalArgumentException {
        if (mix < 0.0 || mix > 1.0) {
            throw new IllegalArgumentException("Dry/wet mix must be between 0.0 and 1.0");
        }
        this.dryWetMix = mix;
    }
    
    @Override
    public double getDryWetMix() {
        return dryWetMix;
    }
    
    @Override
    public void setIntensity(double intensity) throws IllegalArgumentException {
        if (intensity < 0.0 || intensity > 1.0) {
            throw new IllegalArgumentException("Intensity must be between 0.0 and 1.0");
        }
        this.intensity = intensity;
    }
    
    @Override
    public double getIntensity() {
        return intensity;
    }
    
    @Override
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }
    
    @Override
    public boolean isEnabled() {
        return enabled;
    }
    
    @Override
    public String[] getPresets() {
        return presets.clone();
    }
    
    @Override
    public void loadPreset(String presetName) throws IllegalArgumentException {
        // 默认实现为空 / Default implementation is empty
        // 具体效果器应重写此方法 / Specific effects should override this method
    }
    
    @Override
    public void savePreset(String presetName) throws IllegalArgumentException {
        // 默认实现为空 / Default implementation is empty
        // 具体效果器应重写此方法 / Specific effects should override this method
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
    public String[] getSupportedParameters() {
        return supportedParameters.toArray(new String[0]);
    }
    
    @Override
    public Map<String, Object> getDefaultParameters() {
        return Collections.unmodifiableMap(defaultParameters);
    }
    
    @Override
    public boolean validateInput(AudioData input) {
        // 基本验证 / Basic validation
        return input != null && input.getSamples() != null && input.getSamples().length() > 0;
    }
    
    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        // 基本验证 / Basic validation
        if (parameters == null) {
            return true; // null参数被视为使用默认参数 / null parameters are treated as using default parameters
        }
        
        // 检查是否所有参数都在支持的参数列表中 / Check if all parameters are in the supported parameter list
        for (String param : parameters.keySet()) {
            if (!supportedParameters.contains(param)) {
                return false;
            }
        }
        
        return true;
    }
    
    @Override
    public IAudioEffect clone() {
        try {
            AbstractAudioEffect cloned = (AbstractAudioEffect) super.clone();
            // 深拷贝参数映射 / Deep copy parameter maps
            cloned.defaultParameters = new HashMap<>(this.defaultParameters);
            cloned.supportedParameters = new HashSet<>(this.supportedParameters);
            return cloned;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException("Cloning not supported", e);
        }
    }
    
    /**
     * 添加支持的参数 / Add supported parameter
     * 
     * @param paramName 参数名称 / Parameter name
     * @param defaultValue 默认值 / Default value
     */
    protected void addSupportedParameter(String paramName, Object defaultValue) {
        supportedParameters.add(paramName);
        if (defaultValue != null) {
            defaultParameters.put(paramName, defaultValue);
        }
    }
    
    /**
     * 设置默认参数 / Set default parameter
     * 
     * @param paramName 参数名称 / Parameter name
     * @param defaultValue 默认值 / Default value
     */
    protected void setDefaultParameter(String paramName, Object defaultValue) {
        defaultParameters.put(paramName, defaultValue);
    }
    
    @Override
    public String toString() {
        return String.format("%s{name='%s', description='%s', effectType=%s}", 
                           getClass().getSimpleName(), name, description, effectType);
    }
}