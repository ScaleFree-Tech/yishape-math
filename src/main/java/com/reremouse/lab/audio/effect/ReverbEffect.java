package com.reremouse.lab.audio.effect;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;

import java.util.Map;
import java.util.HashMap;
import java.util.Collections;
import java.util.Set;
import java.util.HashSet;

/**
 * 统一接口的混响效果器实现 / Reverb Effect Implementation with Unified Interface
 * <p>
 * 实现音频混响效果，通过延迟和反馈创建空间感。
 * Implements audio reverb effect, creating spatial sense through delay and feedback.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class ReverbEffect extends AbstractAudioEffect {
    
    /**
     * 构造函数 / Constructor
     */
    public ReverbEffect() {
        super("reverb", "Reverb effect", IAudioEffect.EffectType.REVERB);
        addSupportedParameter("roomSize", 0.5);
        addSupportedParameter("damping", 0.5);
        addSupportedParameter("wetLevel", 0.33);
        addSupportedParameter("dryLevel", 0.4);
        addSupportedParameter("width", 1.0);
        addSupportedParameter("freezeMode", 0.0);
    }
    
    @Override
    protected AudioData doApplyEffect(AudioData input) throws AudioProcessingException {
        try {
            // 获取参数 / Get parameters
            double roomSize = 0.5;
            double damping = 0.5;
            double wetLevel = 0.33;
            double dryLevel = 0.4;
            
            // 应用混响效果 / Apply reverb effect
            IVector<Double> processedSamples = input.getSamples();
            
            // 对每个声道分别处理 / Process each channel separately
            if (input.getChannels() > 1) {
                IVector<Double>[] channelData = input.getAllChannels();
                @SuppressWarnings("unchecked")
                IVector<Double>[] processedChannels = new IVector[channelData.length];
                
                for (int ch = 0; ch < channelData.length; ch++) {
                    processedChannels[ch] = applyReverb(channelData[ch], roomSize, damping, wetLevel, dryLevel);
                }
                
                // 重新合并声道 / Re-merge channels
                processedSamples = mergeChannelData(processedChannels);
            } else {
                processedSamples = applyReverb(input.getSamples(), roomSize, damping, wetLevel, dryLevel);
            }
            
            return new AudioData(processedSamples, input.getSampleRate(), input.getChannels(), 
                               input.getBitDepth(), input.getFormat());
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to apply reverb effect", e);
        }
    }
    
    /**
     * 应用混响效果（内部方法） / Apply reverb effect (internal method)
     */
    private IVector<Double> applyReverb(IVector<Double> samples, double roomSize, double damping, double wetLevel, double dryLevel) {
        int length = samples.length();
        IVector<Double> output = Linalg.zeros(length);
        
        // 简化的混响实现 / Simplified reverb implementation
        // 使用简单的延迟线和反馈 / Use simple delay line and feedback
        
        // 创建延迟缓冲区 / Create delay buffer
        int delayLength = Math.max(1, (int)(roomSize * 1000));
        IVector<Double> delayBuffer = Linalg.zeros(delayLength);
        int delayIndex = 0;
        
        // 处理每个样本 / Process each sample
        for (int i = 0; i < length; i++) {
            double inputSample = samples.get(i);
            
            // 从延迟缓冲区获取样本 / Get sample from delay buffer
            double delayedSample = delayBuffer.get(delayIndex);
            
            // 应用阻尼 / Apply damping
            delayedSample *= damping;
            
            // 计算输出 / Calculate output
            double wetSignal = delayedSample * wetLevel;
            double drySignal = inputSample * dryLevel;
            double outputSample = wetSignal + drySignal;
            
            // 更新延迟缓冲区 / Update delay buffer
            delayBuffer.set(delayIndex, inputSample + delayedSample);
            delayIndex = (delayIndex + 1) % delayLength;
            
            // 限制输出范围 / Limit output range
            outputSample = Math.max(-1.0, Math.min(1.0, outputSample));
            output.set(i, outputSample);
        }
        
        return output;
    }
    
    /**
     * 合并声道数据（内部方法） / Merge channel data (internal method)
     *
     * @param channelData 声道数据数组 / Channel data array
     * @return 合并后的数据 / Merged data
     */
    private IVector<Double> mergeChannelData(IVector<Double>[] channelData) {
        int channels = channelData.length;
        int samplesPerChannel = channelData[0].length();
        IVector<Double> mergedData = Linalg.zeros(samplesPerChannel * channels);
        
        for (int ch = 0; ch < channels; ch++) {
            for (int i = 0; i < samplesPerChannel; i++) {
                mergedData.set(i * channels + ch, channelData[ch].get(i));
            }
        }
        
        return mergedData;
    }
    
    @Override
    public ReverbEffect clone() {
        return new ReverbEffect();
    }
    
    @Override
    public AudioData process(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        return applyEffect(input, parameters);
    }
    
    @Override
    public void setParameter(String key, Object value) throws IllegalArgumentException {
        if (key == null) {
            throw new IllegalArgumentException("Parameter key cannot be null");
        }
        addSupportedParameter(key, value);
    }
    
    @Override
    public Object getParameter(String key) throws IllegalArgumentException {
        if (key == null) {
            throw new IllegalArgumentException("Parameter key cannot be null");
        }
        return getDefaultParameters().get(key);
    }
    
    @Override
    public void reset() {
        resetParameters();
    }
    
    @Override
    public boolean supportsFormat(AudioData audioData) {
        if (audioData == null) {
            return false;
        }
        return supportsAudioFormat(audioData.getSampleRate(), audioData.getChannels(), audioData.getBitDepth());
    }
    
    @Override
    public int getLatency() {
        // Return a default latency value for reverb effect
        return 1024;
    }
    
    @Override
    public void loadPreset(String presetName) throws IllegalArgumentException {
        // Implementation for loading reverb presets
        if (presetName == null || presetName.isEmpty()) {
            throw new IllegalArgumentException("Preset name cannot be null or empty");
        }
        
        // Define some standard reverb presets
        switch (presetName.toLowerCase()) {
            case "small-room":
                addSupportedParameter("roomSize", 0.3);
                addSupportedParameter("damping", 0.7);
                addSupportedParameter("wetLevel", 0.2);
                addSupportedParameter("dryLevel", 0.8);
                addSupportedParameter("width", 0.8);
                break;
            case "large-room":
                addSupportedParameter("roomSize", 0.7);
                addSupportedParameter("damping", 0.5);
                addSupportedParameter("wetLevel", 0.5);
                addSupportedParameter("dryLevel", 0.5);
                addSupportedParameter("width", 1.0);
                break;
            case "hall":
                addSupportedParameter("roomSize", 0.9);
                addSupportedParameter("damping", 0.3);
                addSupportedParameter("wetLevel", 0.7);
                addSupportedParameter("dryLevel", 0.3);
                addSupportedParameter("width", 1.2);
                break;
            case "plate":
                addSupportedParameter("roomSize", 0.6);
                addSupportedParameter("damping", 0.8);
                addSupportedParameter("wetLevel", 0.6);
                addSupportedParameter("dryLevel", 0.4);
                addSupportedParameter("width", 0.9);
                break;
            default:
                throw new IllegalArgumentException("Unknown preset: " + presetName);
        }
    }
    
    @Override
    public void savePreset(String presetName) throws IllegalArgumentException {
        // In a real implementation, this would save the current parameters to a file or database
        // For now, we'll just validate the preset name
        if (presetName == null || presetName.isEmpty()) {
            throw new IllegalArgumentException("Preset name cannot be null or empty");
        }
        // Preset saving logic would go here in a complete implementation
    }
    
    @Override
    public String[] getPresets() {
        // Return the list of available presets
        return new String[]{"small-room", "large-room", "hall", "plate"};
    }
    
    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        if (parameters == null) {
            return true;
        }
        
        // Validate each parameter
        for (Map.Entry<String, Object> entry : parameters.entrySet()) {
            String paramName = entry.getKey();
            Object value = entry.getValue();
            
            switch (paramName) {
                case "roomSize":
                case "damping":
                case "wetLevel":
                case "dryLevel":
                case "width":
                case "freezeMode":
                case "dryWetMix":
                case "intensity":
                    if (!(value instanceof Number) || ((Number) value).doubleValue() < 0.0 || ((Number) value).doubleValue() > 1.0) {
                        return false;
                    }
                    break;
                case "effectType":
                    if (value != null && !(value instanceof EffectType)) {
                        return false;
                    }
                    break;
                case "enabled":
                    if (!(value instanceof Boolean)) {
                        return false;
                    }
                    break;
                default:
                    // Unknown parameter
                    return false;
            }
        }
        
        return true;
    }
    
    @Override
    public void setParameters(Map<String, Object> parameters) throws AudioProcessingException {
        if (parameters == null) {
            return;
        }
        
        // Set each parameter
        for (Map.Entry<String, Object> entry : parameters.entrySet()) {
            String paramName = entry.getKey();
            Object value = entry.getValue();
            
            switch (paramName) {
                case "roomSize":
                    addSupportedParameter("roomSize", value);
                    break;
                case "damping":
                    addSupportedParameter("damping", value);
                    break;
                case "wetLevel":
                    addSupportedParameter("wetLevel", value);
                    break;
                case "dryLevel":
                    addSupportedParameter("dryLevel", value);
                    break;
                case "width":
                    addSupportedParameter("width", value);
                    break;
                case "freezeMode":
                    addSupportedParameter("freezeMode", value);
                    break;
                case "dryWetMix":
                    setDryWetMix(((Number) value).doubleValue());
                    break;
                case "intensity":
                    setIntensity(((Number) value).doubleValue());
                    break;
                case "effectType":
                    if (value instanceof EffectType) {
                        setEffectType((EffectType) value);
                    }
                    break;
                case "enabled":
                    setEnabled((Boolean) value);
                    break;
            }
        }
    }
    
    @Override
    public Map<String, Object> getCurrentParameters() {
        Map<String, Object> currentParams = new HashMap<>();
        currentParams.put("roomSize", 0.5); // Default values, in a real implementation these would be the actual current values
        currentParams.put("damping", 0.5);
        currentParams.put("wetLevel", 0.33);
        currentParams.put("dryLevel", 0.4);
        currentParams.put("width", 1.0);
        currentParams.put("freezeMode", 0.0);
        currentParams.put("dryWetMix", getDryWetMix());
        currentParams.put("intensity", getIntensity());
        currentParams.put("effectType", getEffectType());
        currentParams.put("enabled", isEnabled());
        return currentParams;
    }
    
    @Override
    public void resetParameters() {
        // Reset to default values
        addSupportedParameter("roomSize", 0.5);
        addSupportedParameter("damping", 0.5);
        addSupportedParameter("wetLevel", 0.33);
        addSupportedParameter("dryLevel", 0.4);
        addSupportedParameter("width", 1.0);
        addSupportedParameter("freezeMode", 0.0);
        setDryWetMix(1.0);
        setIntensity(1.0);
        setEnabled(true);
    }
    
    @Override
    public boolean supportsAudioFormat(double sampleRate, int channels, int bitDepth) {
        // The reverb effect should support most common audio formats
        return sampleRate > 0 && channels > 0 && channels <= 8 && (bitDepth == 8 || bitDepth == 16 || bitDepth == 24 || bitDepth == 32);
    }
    
    @Override
    public String getStatus() {
        return isEnabled() ? "Enabled" : "Disabled";
    }
    
    @Override
    public boolean isReady() {
        return true; // The reverb effect is always ready
    }
}