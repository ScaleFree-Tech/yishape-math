package com.yishape.lab.audio.filter;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;

import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Collections;

/**
 * 标准化音频滤波器抽象基类 / Standardized Audio Filter Abstract Base Class
 * <p>
 * 提供音频滤波器接口的基本实现，包括参数管理、验证和克隆功能。
 * 所有具体的音频滤波器实现都应该继承此类。
 * </p>
 * <p>
 * Provides basic implementation of audio filter interface, including parameter management, validation, and cloning.
 * All concrete audio filter implementations should extend this class.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public abstract class AbstractAudioFilterStandard implements IBaseAudioFilter {

    /** 滤波器名称 / Filter name */
    protected String name;
    /** 滤波器描述 / Filter description */
    protected String description;
    /** 滤波器类型 / Filter type */
    protected FilterType filterType;
    /** 截止频率 (Hz) / Cutoff frequency (Hz) */
    protected double cutoffFrequency;
    /** 滤波器阶数 / Filter order */
    protected int order;
    /** 默认参数映射 / Default parameter map */
    protected Map<String, Object> defaultParameters;
    /** 支持的参数集合 / Supported parameter set */
    protected Set<String> supportedParameters;

    /**
     * 构造函数 / Constructor
     *
     * @param name 滤波器名称 / Filter name
     * @param description 滤波器描述 / Filter description
     */
    public AbstractAudioFilterStandard(String name, String description) {
        this.name = name;
        this.description = description;
        this.filterType = FilterType.LOW_PASS;
        this.cutoffFrequency = 1000.0;
        this.order = 4;
        this.defaultParameters = new HashMap<>();
        this.supportedParameters = new HashSet<>();
        
        // 添加默认支持的参数 / Add default supported parameters
        addSupportedParameter("filterType", this.filterType);
        addSupportedParameter("cutoffFrequency", this.cutoffFrequency);
        addSupportedParameter("order", this.order);
    }
    
    @Override
    public AudioData filter(AudioData input) throws AudioProcessingException {
        return filter(input, getDefaultParameters());
    }
    
    @Override
    public AudioData filter(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
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
            if (parameters.containsKey("filterType")) {
                this.filterType = (FilterType) parameters.get("filterType");
            }
            if (parameters.containsKey("cutoffFrequency")) {
                this.cutoffFrequency = (Double) parameters.get("cutoffFrequency");
            }
            if (parameters.containsKey("order")) {
                this.order = (Integer) parameters.get("order");
            }
        }
        
        // 执行滤波 / Perform filtering
        return doFilter(input);
    }
    
    /**
     * 执行实际的滤波操作 / Perform actual filtering operation
     * <p>
     * 子类必须实现此方法以提供具体的滤波逻辑。
     * Subclasses must implement this method to provide specific filtering logic.
     * </p>
     *
     * @param input 输入音频数据 / Input audio data
     * @return 滤波后的音频 / Filtered audio
     * @throws AudioProcessingException 滤波过程中发生错误 / Error occurred during filtering
     */
    protected abstract AudioData doFilter(AudioData input) throws AudioProcessingException;
    
    @Override
    public FilterType getFilterType() {
        return filterType;
    }
    
    @Override
    public void setFilterType(FilterType filterType) {
        this.filterType = filterType;
    }
    
    @Override
    public double getCutoffFrequency() {
        return cutoffFrequency;
    }
    
    @Override
    public void setCutoffFrequency(double cutoffFrequency) {
        this.cutoffFrequency = cutoffFrequency;
    }
    
    @Override
    public int getOrder() {
        return order;
    }
    
    @Override
    public void setOrder(int order) {
        this.order = order;
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
        return Collections.unmodifiableSet(supportedParameters);
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
        
        // 特定参数验证 / Specific parameter validation
        if (parameters.containsKey("cutoffFrequency")) {
            Double freq = (Double) parameters.get("cutoffFrequency");
            if (freq <= 0) {
                return false;
            }
        }
        
        if (parameters.containsKey("order")) {
            Integer ord = (Integer) parameters.get("order");
            if (ord <= 0) {
                return false;
            }
        }
        
        return true;
    }
    
    @Override
    public IBaseAudioFilter clone() {
        try {
            AbstractAudioFilterStandard cloned = (AbstractAudioFilterStandard) super.clone();
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
        return String.format("%s{name='%s', description='%s', filterType=%s, cutoffFrequency=%.2f, order=%d}", 
                           getClass().getSimpleName(), name, description, filterType, cutoffFrequency, order);
    }
    
    // ================ IAudioProcessor 方法实现 / IAudioProcessor Method Implementations ================
    
    @Override
    public AudioData process(AudioData input) throws AudioProcessingException {
        return filter(input);
    }
    
    @Override
    public void setParameter(String key, Object value) throws IllegalArgumentException {
        switch (key) {
            case "filterType":
                if (value instanceof FilterType) {
                    setFilterType((FilterType) value);
                } else {
                    throw new IllegalArgumentException("filterType must be of type FilterType");
                }
                break;
            case "cutoffFrequency":
                if (value instanceof Number) {
                    setCutoffFrequency(((Number) value).doubleValue());
                } else {
                    throw new IllegalArgumentException("cutoffFrequency must be a number");
                }
                break;
            case "order":
                if (value instanceof Number) {
                    setOrder(((Number) value).intValue());
                } else {
                    throw new IllegalArgumentException("order must be a number");
                }
                break;
            default:
                throw new IllegalArgumentException("Unsupported parameter: " + key);
        }
    }
    
    @Override
    public Object getParameter(String key) throws IllegalArgumentException {
        switch (key) {
            case "filterType":
                return getFilterType();
            case "cutoffFrequency":
                return getCutoffFrequency();
            case "order":
                return getOrder();
            default:
                throw new IllegalArgumentException("Unsupported parameter: " + key);
        }
    }
    
    @Override
    public void reset() {
        // Reset to default values
        this.filterType = FilterType.LOW_PASS;
        this.cutoffFrequency = 1000.0;
        this.order = 4;
    }
    
    @Override
    public String getVersion() {
        return "1.0";
    }
    
    @Override
    public boolean supportsFormat(AudioData audioData) {
        // Basic validation - most filters should support standard audio formats
        return audioData != null && audioData.getSamples() != null;
    }
    
    @Override
    public int getLatency() {
        // Most basic filters have minimal latency
        return 0;
    }
}