package com.yishape.lab.audio.processing;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;

import java.util.Map;

/**
 * 音频归一化处理器实现 / Audio Normalization Processor Implementation
 * <p>
 * 将音频信号归一化到[-1, 1]范围内，防止削波。
 * Normalize audio signal to [-1, 1] range to prevent clipping.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class NormalizeProcessor extends AbstractAudioProcessorStandard {
    
    /**
     * 构造函数 / Constructor
     */
    public NormalizeProcessor() {
        super("normalize", "Audio normalization processor");
        addSupportedParameter("margin", 0.05);
    }
    
    @Override
    protected AudioData doProcess(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 获取余量参数 / Get margin parameter
            double margin = 0.05;
            if (parameters != null && parameters.containsKey("margin")) {
                margin = (Double) parameters.get("margin");
            }
            
            // 归一化音频 / Normalize audio
            IVector<Double> normalizedSamples = normalizeAudio(input.getSamples(), margin);
            
            return new AudioData(normalizedSamples, input.getSampleRate(), 
                               input.getChannels(), input.getBitDepth(), input.getFormat());
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to normalize audio", e);
        }
    }
    
    /**
     * 音频归一化 / Normalize audio
     *
     * @param samples 音频样本 / Audio samples
     * @param margin 余量 / Margin
     * @return 归一化后的样本 / Normalized samples
     */
    private IVector<Double> normalizeAudio(IVector<Double> samples, double margin) {
        double maxAbs = Math.max(Math.abs(samples.max()), Math.abs(samples.min()));
        
        if (maxAbs == 0) {
            return samples; // 静音信号 / Silent signal
        }
        
        // 应用余量 / Apply margin
        return samples.multiplyScalar((1.0 - margin) / maxAbs);
    }
    
    @Override
    public NormalizeProcessor clone() {
        return new NormalizeProcessor();
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