package com.yishape.lab.audio.processing;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;

import java.util.Map;

/**
 * 音量处理器实现 / Volume Processor Implementation
 * <p>
 * 调节音频的音量增益。
 * Adjusts the volume gain of audio.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class VolumeProcessor extends AbstractAudioProcessorStandard {
    
    /**
     * 构造函数 / Constructor
     */
    public VolumeProcessor() {
        super("volume", "Volume adjustment processor");
        addSupportedParameter("gain", 1.0);
    }
    
    @Override
    protected AudioData doProcess(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 获取增益参数 / Get gain parameter
            double gain = 1.0;
            if (parameters != null && parameters.containsKey("gain")) {
                gain = (Double) parameters.get("gain");
            }
            
            // 应用增益 / Apply gain
            IVector<Double> samples = input.getSamples();
            IVector<Double> adjustedSamples = samples.multiplyScalar(gain);
            
            // 防止削波 / Prevent clipping
            adjustedSamples = normalizeAudio(adjustedSamples);
            
            return new AudioData(adjustedSamples, input.getSampleRate(), 
                               input.getChannels(), input.getBitDepth(), input.getFormat());
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to adjust volume", e);
        }
    }
    
    /**
     * 音频归一化（内部方法） / Normalize audio (internal method)
     *
     * @param samples 音频样本 / Audio samples
     * @return 归一化后的样本 / Normalized samples
     */
    private IVector<Double> normalizeAudio(IVector<Double> samples) {
        double maxAbs = Math.max(Math.abs(samples.max()), Math.abs(samples.min()));
        
        if (maxAbs == 0) {
            return samples; // 静音信号 / Silent signal
        }
        
        // 留5%余量 / Leave 5% margin
        return samples.multiplyScalar(0.95 / maxAbs);
    }
    
    @Override
    public VolumeProcessor clone() {
        return new VolumeProcessor();
    }
    
    @Override
    public void reset() {
        // Default implementation: do nothing
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
        return 0; // No additional latency
    }
}