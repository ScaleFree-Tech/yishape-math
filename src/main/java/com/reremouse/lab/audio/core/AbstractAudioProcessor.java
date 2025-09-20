package com.reremouse.lab.audio.core;

import com.reremouse.lab.audio.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 抽象音频处理器基类 / Abstract Audio Processor Base Class
 * <p>
 * 实现IAudioProcessor接口的抽象基类，提供通用的参数管理、事件通知等功能。
 * 使用Strategy模式允许子类实现不同的处理算法。
 * </p>
 * <p>
 * Abstract base class implementing IAudioProcessor interface, providing common parameter management, event notification, etc.
 * Uses Strategy pattern to allow subclasses to implement different processing algorithms.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public abstract class AbstractAudioProcessor implements IAudioProcessor {
    
    /** 处理器参数存储 / Processor parameter storage */
    protected final Map<String, Object> parameters = new ConcurrentHashMap<>();
    
    /** 处理器名称 / Processor name */
    protected final String name;
    
    /** 处理器版本 / Processor version */
    protected final String version;
    
    /** 事件监听器列表 / Event listener list */
    protected final java.util.List<IAudioListener> listeners = new java.util.ArrayList<>();
    
    /** 是否启用处理器 / Whether processor is enabled */
    protected boolean enabled = true;
    
    /** 处理延迟（样本数） / Processing latency (in samples) */
    protected int latency = 0;
    
    /**
     * 构造函数 / Constructor
     *
     * @param name 处理器名称 / Processor name
     * @param version 处理器版本 / Processor version
     */
    protected AbstractAudioProcessor(String name, String version) {
        this.name = name;
        this.version = version;
        initializeDefaultParameters();
    }
    
    /**
     * 处理音频数据的模板方法 / Template method for processing audio data
     * <p>
     * 实现了处理流程的骨架，包括前处理、核心处理、后处理等步骤。
     * 子类只需要实现核心处理逻辑。
     * </p>
     */
    @Override
    public final AudioData process(AudioData input) throws AudioProcessingException {
        if (!enabled) {
            return input; // 如果禁用，直接返回原始数据 / If disabled, return original data directly
        }
        
        if (!supportsFormat(input)) {
            throw new AudioProcessingException("Unsupported audio format: " + input.getFormat());
        }
        
        try {
            // 通知处理开始 / Notify processing started
            notifyProcessingStarted(input);
            
            // 前处理 / Pre-processing
            AudioData preprocessed = preProcess(input);
            
            // 核心处理 (Strategy pattern) / Core processing (Strategy pattern)
            AudioData processed = doProcess(preprocessed);
            
            // 后处理 / Post-processing
            AudioData postprocessed = postProcess(processed);
            
            // 通知处理完成 / Notify processing completed
            notifyProcessingCompleted(postprocessed);
            
            return postprocessed;
            
        } catch (Exception e) {
            // 通知处理失败 / Notify processing failed
            notifyProcessingFailed(input, e);
            throw new AudioProcessingException("Processing failed", e);
        }
    }
    
    /**
     * 核心处理方法 - 子类必须实现 / Core processing method - subclasses must implement
     * <p>
     * 这是Strategy模式的关键部分，不同的子类实现不同的处理算法。
     * This is the key part of Strategy pattern, different subclasses implement different processing algorithms.
     * </p>
     *
     * @param input 输入音频数据 / Input audio data
     * @return 处理后的音频数据 / Processed audio data
     * @throws AudioProcessingException 处理过程中发生错误 / Error during processing
     */
    protected abstract AudioData doProcess(AudioData input) throws AudioProcessingException;
    
    /**
     * 前处理 - 子类可以重写 / Pre-processing - subclasses can override
     * <p>
     * 在核心处理之前执行的操作，如格式检查、数据预处理等。
     * Operations performed before core processing, such as format checking, data preprocessing, etc.
     * </p>
     *
     * @param input 输入音频数据 / Input audio data
     * @return 预处理后的音频数据 / Pre-processed audio data
     * @throws AudioProcessingException 预处理过程中发生错误 / Error during pre-processing
     */
    protected AudioData preProcess(AudioData input) throws AudioProcessingException {
        return input; // 默认实现不做任何处理 / Default implementation does nothing
    }
    
    /**
     * 后处理 - 子类可以重写 / Post-processing - subclasses can override
     * <p>
     * 在核心处理之后执行的操作，如数据清理、格式转换等。
     * Operations performed after core processing, such as data cleanup, format conversion, etc.
     * </p>
     *
     * @param input 处理后的音频数据 / Processed audio data
     * @return 后处理后的音频数据 / Post-processed audio data
     * @throws AudioProcessingException 后处理过程中发生错误 / Error during post-processing
     */
    protected AudioData postProcess(AudioData input) throws AudioProcessingException {
        return input; // 默认实现不做任何处理 / Default implementation does nothing
    }
    
    /**
     * 初始化默认参数 - 子类可以重写 / Initialize default parameters - subclasses can override
     */
    protected void initializeDefaultParameters() {
        parameters.put("enabled", true);
        parameters.put("bypass", false);
    }
    
    @Override
    public void setParameter(String key, Object value) throws IllegalArgumentException {
        if (key == null) {
            throw new IllegalArgumentException("Parameter key cannot be null");
        }
        
        // 验证参数 / Validate parameter
        validateParameter(key, value);
        
        Object oldValue = parameters.get(key);
        parameters.put(key, value);
        
        // 处理特殊参数 / Handle special parameters
        handleSpecialParameter(key, value);
        
        // 通知参数改变 / Notify parameter changed
        notifyParameterChanged(key, oldValue, value);
    }
    
    /**
     * 验证参数 - 子类可以重写 / Validate parameter - subclasses can override
     *
     * @param key 参数键 / Parameter key
     * @param value 参数值 / Parameter value
     * @throws IllegalArgumentException 参数无效时抛出 / Thrown when parameter is invalid
     */
    protected void validateParameter(String key, Object value) throws IllegalArgumentException {
        // 默认实现进行基本验证 / Default implementation performs basic validation
        if ("enabled".equals(key) && !(value instanceof Boolean)) {
            throw new IllegalArgumentException("Parameter 'enabled' must be a boolean");
        }
        if ("bypass".equals(key) && !(value instanceof Boolean)) {
            throw new IllegalArgumentException("Parameter 'bypass' must be a boolean");
        }
    }
    
    /**
     * 处理特殊参数 - 子类可以重写 / Handle special parameter - subclasses can override
     *
     * @param key 参数键 / Parameter key
     * @param value 参数值 / Parameter value
     */
    protected void handleSpecialParameter(String key, Object value) {
        if ("enabled".equals(key)) {
            this.enabled = (Boolean) value;
        }
    }
    
    @Override
    public Object getParameter(String key) throws IllegalArgumentException {
        if (!parameters.containsKey(key)) {
            throw new IllegalArgumentException("Unknown parameter: " + key);
        }
        return parameters.get(key);
    }
    
    @Override
    public void reset() {
        parameters.clear();
        initializeDefaultParameters();
        this.enabled = true;
    }
    
    @Override
    public String getName() {
        return name;
    }
    
    @Override
    public String getVersion() {
        return version;
    }
    
    @Override
    public IAudioProcessor clone() {
        try {
            AbstractAudioProcessor cloned = (AbstractAudioProcessor) super.clone();
            cloned.parameters.putAll(this.parameters);
            return cloned;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException("Clone not supported", e);
        }
    }
    
    @Override
    public boolean supportsFormat(AudioData audioData) {
        // 默认支持所有格式，子类可以重写 / Default supports all formats, subclasses can override
        return audioData != null && audioData.getSamples() != null;
    }
    
    @Override
    public int getLatency() {
        return latency;
    }
    
    /**
     * 设置处理延迟 / Set processing latency
     *
     * @param latency 延迟样本数 / Latency in samples
     */
    protected void setLatency(int latency) {
        this.latency = Math.max(0, latency);
    }
    
    // ================ 事件通知方法 / Event Notification Methods ================
    
    /**
     * 添加事件监听器 / Add event listener
     *
     * @param listener 事件监听器 / Event listener
     */
    public void addListener(IAudioListener listener) {
        if (listener != null && !listeners.contains(listener)) {
            listeners.add(listener);
        }
    }
    
    /**
     * 移除事件监听器 / Remove event listener
     *
     * @param listener 事件监听器 / Event listener
     */
    public void removeListener(IAudioListener listener) {
        listeners.remove(listener);
    }
    
    /**
     * 通知处理开始 / Notify processing started
     */
    protected void notifyProcessingStarted(AudioData audioData) {
        for (IAudioListener listener : listeners) {
            listener.onProcessingStarted(this, audioData);
        }
    }
    
    /**
     * 通知处理完成 / Notify processing completed
     */
    protected void notifyProcessingCompleted(AudioData audioData) {
        for (IAudioListener listener : listeners) {
            listener.onProcessingCompleted(this, audioData);
        }
    }
    
    /**
     * 通知处理失败 / Notify processing failed
     */
    protected void notifyProcessingFailed(AudioData audioData, Throwable error) {
        for (IAudioListener listener : listeners) {
            listener.onProcessingFailed(this, audioData, error);
        }
    }
    
    /**
     * 通知参数改变 / Notify parameter changed
     */
    protected void notifyParameterChanged(String parameterName, Object oldValue, Object newValue) {
        for (IAudioListener listener : listeners) {
            listener.onParameterChanged(this, parameterName, oldValue, newValue);
        }
    }
    
    /**
     * 通知进度更新 / Notify progress update
     */
    protected void notifyProgressUpdate(AudioData audioData, double progress) {
        for (IAudioListener listener : listeners) {
            listener.onProgressUpdate(this, audioData, progress);
        }
    }
    
    @Override
    public String toString() {
        return String.format("%s{name='%s', version='%s', enabled=%s, latency=%d}", 
                           getClass().getSimpleName(), name, version, enabled, latency);
    }
}