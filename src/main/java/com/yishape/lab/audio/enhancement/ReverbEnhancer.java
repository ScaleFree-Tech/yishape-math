package com.yishape.lab.audio.enhancement;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.Map;

/**
 * 混响增强器实现 / Reverb Enhancer Implementation
 * <p>
 * 实现音频混响功能，使用简单的延迟和反馈网络模拟混响效果。
 * Implements audio reverb functionality, using simple delay and feedback network to simulate reverb effect.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class ReverbEnhancer extends AbstractAudioEnhancer {
    
    /**
     * 构造函数 / Constructor
     */
    public ReverbEnhancer() {
        super("reverb_enhancer", "Reverb enhancer", EnhancerType.REVERB);
        addSupportedParameter("roomSize", 0.5);
        addSupportedParameter("damping", 0.5);
        addSupportedParameter("wetLevel", 0.33);
        addSupportedParameter("dryLevel", 0.4);
        addSupportedParameter("width", 1.0);
        addSupportedParameter("freezeMode", 0.0);
    }
    
    @Override
    protected AudioData doEnhance(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 获取参数 / Get parameters
            double roomSize = 0.5;
            double damping = 0.5;
            double wetLevel = 0.33;
            double dryLevel = 0.4;
            double width = 1.0;
            double freezeMode = 0.0;
            
            if (parameters != null) {
                if (parameters.containsKey("roomSize")) {
                    roomSize = (Double) parameters.get("roomSize");
                }
                if (parameters.containsKey("damping")) {
                    damping = (Double) parameters.get("damping");
                }
                if (parameters.containsKey("wetLevel")) {
                    wetLevel = (Double) parameters.get("wetLevel");
                }
                if (parameters.containsKey("dryLevel")) {
                    dryLevel = (Double) parameters.get("dryLevel");
                }
                if (parameters.containsKey("width")) {
                    width = (Double) parameters.get("width");
                }
                if (parameters.containsKey("freezeMode")) {
                    freezeMode = (Double) parameters.get("freezeMode");
                }
            }
            
            IVector<Double> samples = input.getSamples();
            IVector<Double> reverbSamples = Linalg.zeros(samples.length());
            
            // 对每个声道分别处理 / Process each channel separately
            if (input.getChannels() > 1) {
                IVector<Double>[] channelData = input.getAllChannels();
                @SuppressWarnings("unchecked")
                IVector<Double>[] reverbChannels = new IVector[channelData.length];
                
                for (int ch = 0; ch < channelData.length; ch++) {
                    reverbChannels[ch] = applyReverb(channelData[ch], roomSize, damping, wetLevel, dryLevel, width, freezeMode);
                }
                
                // 重新合并声道 / Re-merge channels
                reverbSamples = mergeChannelData(reverbChannels);
            } else {
                reverbSamples = applyReverb(samples, roomSize, damping, wetLevel, dryLevel, width, freezeMode);
            }
            
            return new AudioData(reverbSamples, input.getSampleRate(), input.getChannels(), 
                               input.getBitDepth(), input.getFormat());
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to apply reverb", e);
        }
    }
    
    /**
     * 应用混响效果 / Apply reverb effect
     *
     * @param samples 音频样本 / Audio samples
     * @param roomSize 房间大小 / Room size
     * @param damping 阻尼 / Damping
     * @param wetLevel 湿信号级别 / Wet level
     * @param dryLevel 干信号级别 / Dry level
     * @param width 宽度 / Width
     * @param freezeMode 冻结模式 / Freeze mode
     * @return 添加混响后的样本 / Samples with reverb effect
     */
    private IVector<Double> applyReverb(IVector<Double> samples, double roomSize, double damping, 
                                     double wetLevel, double dryLevel, double width, double freezeMode) {
        int length = samples.length();
        IVector<Double> output = Linalg.zeros(length);
        
        // 简化的混响实现 / Simplified reverb implementation
        // 使用简单的延迟线和反馈 / Use simple delay line and feedback
        
        // 创建延迟缓冲区 / Create delay buffer
        int delayLength = Math.max(1, (int)(roomSize * 10000));
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
            delayBuffer.set(delayIndex, inputSample + delayedSample * (1.0 - freezeMode));
            delayIndex = (delayIndex + 1) % delayLength;
            
            // 限制输出范围 / Limit output range
            outputSample = Math.max(-1.0, Math.min(1.0, outputSample));
            output.set(i, outputSample);
        }
        
        return output;
    }
    
    /**
     * 合并声道数据 / Merge channel data
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
    public ReverbEnhancer clone() {
        return new ReverbEnhancer();
    }
}