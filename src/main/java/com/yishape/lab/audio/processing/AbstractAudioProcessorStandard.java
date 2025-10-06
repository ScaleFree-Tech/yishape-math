package com.yishape.lab.audio.processing;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Collections;

/**
 * 标准化音频处理器抽象基类 / Standardized Audio Processor Abstract Base Class
 * <p>
 * 提供音频处理器接口的基本实现，包括参数管理、验证和克隆功能。
 * 所有具体的音频处理器实现都应该继承此类。
 * </p>
 * <p>
 * Provides basic implementation of audio processor interface, including parameter management, validation, and cloning.
 * All concrete audio processor implementations should extend this class.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public abstract class AbstractAudioProcessorStandard implements IAdvancedAudioProcessor {
    
    protected String name;
    protected String description;
    protected Map<String, Object> defaultParameters;
    protected Set<String> supportedParameters;
    
    /**
     * 构造函数 / Constructor
     * 
     * @param name 处理器名称 / Processor name
     * @param description 处理器描述 / Processor description
     */
    public AbstractAudioProcessorStandard(String name, String description) {
        this.name = name;
        this.description = description;
        this.defaultParameters = new HashMap<>();
        this.supportedParameters = new HashSet<>();
    }
    
    @Override
    public AudioData process(AudioData input) throws AudioProcessingException {
        return process(input, getDefaultParameters());
    }
    
    @Override
    public AudioData process(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        // 验证输入 / Validate input
        if (!validateInput(input)) {
            throw new AudioProcessingException("Invalid input audio data");
        }
        
        // 验证参数 / Validate parameters
        if (!validateParameters(parameters)) {
            throw new AudioProcessingException("Invalid parameters");
        }
        
        // 执行处理 / Perform processing
        return doProcess(input, parameters);
    }
    
    /**
     * 执行实际的处理操作 / Perform actual processing operation
     * <p>
     * 子类必须实现此方法以提供具体的处理逻辑。
     * Subclasses must implement this method to provide specific processing logic.
     * </p>
     * 
     * @param input 输入音频 / Input audio
     * @param parameters 处理参数 / Processing parameters
     * @return 处理后的音频 / Processed audio
     * @throws AudioProcessingException 处理过程中发生错误 / Error occurred during processing
     */
    protected abstract AudioData doProcess(AudioData input, Map<String, Object> parameters) throws AudioProcessingException;
    
    @Override
    public String getName() {
        return name;
    }
    
    @Override
    public String getDescription() {
        return description;
    }
    

    
    @Override
    public Map<String, Object> getDefaultParameters() {
        return Collections.unmodifiableMap(defaultParameters);
    }
    
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
    public IAdvancedAudioProcessor clone() {
        try {
            AbstractAudioProcessorStandard cloned = (AbstractAudioProcessorStandard) super.clone();
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
        return String.format("%s{name='%s', description='%s'}", 
                           getClass().getSimpleName(), name, description);
    }
    
    // ========== IAdvancedAudioProcessor 缺失方法的默认实现 / Default Implementation of Missing IAdvancedAudioProcessor Methods ==========
    
    @Override
    public AudioData processTimeRange(AudioData audioData, double startTime, double endTime) throws AudioProcessingException {
        // 默认实现：处理整个音频数据
        // Default implementation: process entire audio data
        return process(audioData);
    }
    
    @Override
    public AudioData processTimeRange(AudioData audioData, double startTime, double endTime, Map<String, Object> parameters) throws AudioProcessingException {
        // 默认实现：处理整个音频数据
        // Default implementation: process entire audio data
        return process(audioData, parameters);
    }
    
    @Override
    public AudioData processStream(AudioData audioData, double windowSize, double hopSize) throws AudioProcessingException {
        // 默认实现：简单处理
        // Default implementation: simple processing
        return process(audioData);
    }

    @Override
    public AudioData processStream(AudioData audioData, double windowSize, double hopSize, Map<String, Object> parameters) throws AudioProcessingException {
        // 默认实现：简单处理
        // Default implementation: simple processing
        return process(audioData, parameters);
    }
    
    @Override
    public AudioData[] processBatch(AudioData[] audioDataArray) throws AudioProcessingException {
        if (audioDataArray == null) {
            throw new AudioProcessingException("Audio data array cannot be null");
        }
        
        AudioData[] results = new AudioData[audioDataArray.length];
        for (int i = 0; i < audioDataArray.length; i++) {
            results[i] = process(audioDataArray[i]);
        }
        return results;
    }
    
    @Override
    public AudioData[] processBatch(AudioData[] audioDataArray, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioDataArray == null) {
            throw new AudioProcessingException("Audio data array cannot be null");
        }
        
        AudioData[] results = new AudioData[audioDataArray.length];
        for (int i = 0; i < audioDataArray.length; i++) {
            results[i] = process(audioDataArray[i], parameters);
        }
        return results;
    }
    
    @Override
    public String[] getSupportedParameters() {
        return supportedParameters.toArray(new String[0]);
    }
    
    @Override
    public void setParameters(Map<String, Object> parameters) throws AudioProcessingException {
        if (!validateParameters(parameters)) {
            throw new AudioProcessingException("Invalid parameters provided");
        }
        // 子类应该重写此方法以实际设置参数
        // Subclasses should override this method to actually set parameters
    }
    
    @Override
    public Map<String, Object> getCurrentParameters() {
        return new HashMap<>(defaultParameters);
    }
    
    @Override
    public void resetParameters() {
        // 子类应该重写此方法以重置参数到默认值
        // Subclasses should override this method to reset parameters to default values
    }
    
    @Override
    public String getVersion() {
        return "1.0";
    }
    
    @Override
    public boolean supportsAudioFormat(double sampleRate, int channels, int bitDepth) {
        return sampleRate > 0 && channels > 0 && bitDepth > 0;
    }
    
    @Override
    public double getMinimumAudioLength() {
        return 0.001; // 1ms
    }
    
    @Override
    public double getMaximumAudioLength() {
        return Double.MAX_VALUE;
    }
    
    @Override
    public double getComplexityEstimate(double audioLength) {
        return audioLength; // Linear complexity by default
    }
    
    @Override
    public Map<String, Object> getPerformanceMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        metrics.put("processingTime", 0.0);
        metrics.put("memoryUsage", 0.0);
        metrics.put("cpuUsage", 0.0);
        return metrics;
    }
    
    @Override
    public void warmUp() throws AudioProcessingException {
        // 默认实现：什么都不做
        // Default implementation: do nothing
    }
    
    @Override
    public void cleanup() {
        // 默认实现：什么都不做
        // Default implementation: do nothing
    }
    
    @Override
    public String[] getSupportedProcessingTypes() {
        return new String[]{"basic"};
    }
    
    @Override
    public boolean supportsProcessingType(String processingType) {
        String[] supported = getSupportedProcessingTypes();
        for (String type : supported) {
            if (type.equalsIgnoreCase(processingType)) {
                return true;
            }
        }
        return false;
    }
    
    @Override
    public String[] getQualityLevels() {
        return new String[]{"low", "medium", "high"};
    }
    
    @Override
    public void setQualityLevel(String qualityLevel) throws AudioProcessingException {
        // 默认实现：什么都不做
        // Default implementation: do nothing
    }
    
    // ========== IBaseAudioProcessor 缺失方法的默认实现 / Default Implementation of Missing IBaseAudioProcessor Methods ==========
    
    public Object getParameter(String parameterName) {
        return defaultParameters.get(parameterName);
    }
    
    public void setParameter(String parameterName, Object value) throws IllegalArgumentException {
        if (!supportedParameters.contains(parameterName)) {
            throw new IllegalArgumentException("Unsupported parameter: " + parameterName);
        }
        // 子类应该重写此方法以实际设置参数
        // Subclasses should override this method to actually set parameters
    }
    
    
    
    @Override
    public boolean isReady() {
        return true; // 默认实现：总是就绪
    }
    
    @Override
    public String getStatus() {
        return "ready"; // 默认实现：总是就绪状态
    }
    
    // ========== IAdvancedAudioProcessor 缺失方法的默认实现 / Default Implementation of Missing IAdvancedAudioProcessor Methods ==========

    private boolean verboseLogging = false;
    
    @Override
    public Map<String, Object> getLastProcessingStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("lastProcessingTime", 0.0);
        stats.put("lastProcessingStatus", getStatus());
        return stats;
    }
    
    @Override
    public void setVerboseLogging(boolean enabled) {
        this.verboseLogging = enabled;
    }
    
    @Override
    public boolean isVerboseLoggingEnabled() {
        return verboseLogging;
    }
}