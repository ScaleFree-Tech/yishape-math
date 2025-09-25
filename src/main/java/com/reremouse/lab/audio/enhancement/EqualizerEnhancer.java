package com.reremouse.lab.audio.enhancement;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.Signals;
import java.util.Map;

/**
 * 均衡增强器实现 / Equalizer Enhancer Implementation
 * <p>
 * 实现音频均衡功能，使用多频段均衡器调整音频的频率响应。
 * Implements audio equalization functionality, using multi-band equalizer to adjust audio frequency response.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class EqualizerEnhancer extends AbstractAudioEnhancer {
    
    /**
     * 构造函数 / Constructor
     */
    public EqualizerEnhancer() {
        super("equalizer", "Equalizer enhancer", EnhancerType.EQUALIZATION);
        addSupportedParameter("lowGain", 0.0);
        addSupportedParameter("midGain", 0.0);
        addSupportedParameter("highGain", 0.0);
        addSupportedParameter("lowFrequency", 200.0);
        addSupportedParameter("highFrequency", 4000.0);
    }
    
    @Override
    protected AudioData doEnhance(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 获取参数 / Get parameters
            double lowGain = 0.0;
            double midGain = 0.0;
            double highGain = 0.0;
            double lowFrequency = 200.0;
            double highFrequency = 4000.0;
            
            if (parameters != null) {
                if (parameters.containsKey("lowGain")) {
                    lowGain = (Double) parameters.get("lowGain");
                }
                if (parameters.containsKey("midGain")) {
                    midGain = (Double) parameters.get("midGain");
                }
                if (parameters.containsKey("highGain")) {
                    highGain = (Double) parameters.get("highGain");
                }
                if (parameters.containsKey("lowFrequency")) {
                    lowFrequency = (Double) parameters.get("lowFrequency");
                }
                if (parameters.containsKey("highFrequency")) {
                    highFrequency = (Double) parameters.get("highFrequency");
                }
            }
            
            IVector<Double> samples = input.getSamples();
            IVector<Double> equalizedSamples = Linalg.zeros(samples.length());
            
            // 对每个声道分别处理 / Process each channel separately
            if (input.getChannels() > 1) {
                IVector<Double>[] channelData = input.getAllChannels();
                @SuppressWarnings("unchecked")
                IVector<Double>[] equalizedChannels = new IVector[channelData.length];
                
                for (int ch = 0; ch < channelData.length; ch++) {
                    equalizedChannels[ch] = applyEqualizer(channelData[ch], lowGain, midGain, highGain, 
                                                         lowFrequency, highFrequency, input.getSampleRate());
                }
                
                // 重新合并声道 / Re-merge channels
                equalizedSamples = mergeChannelData(equalizedChannels);
            } else {
                equalizedSamples = applyEqualizer(samples, lowGain, midGain, highGain, 
                                               lowFrequency, highFrequency, input.getSampleRate());
            }
            
            return new AudioData(equalizedSamples, input.getSampleRate(), input.getChannels(), 
                               input.getBitDepth(), input.getFormat());
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to apply equalization", e);
        }
    }
    
    /**
     * 应用均衡器 / Apply equalizer
     *
     * @param samples 音频样本 / Audio samples
     * @param lowGain 低频增益 / Low frequency gain
     * @param midGain 中频增益 / Mid frequency gain
     * @param highGain 高频增益 / High frequency gain
     * @param lowFrequency 低频分界点 / Low frequency boundary
     * @param highFrequency 高频分界点 / High frequency boundary
     * @param sampleRate 采样率 / Sample rate
     * @return 均衡后的样本 / Equalized samples
     */
    private IVector<Double> applyEqualizer(IVector<Double> samples, double lowGain, double midGain, double highGain,
                                         double lowFrequency, double highFrequency, double sampleRate) {
        IVector<Double> equalizedSamples = samples.copy();
        
        // 应用低频增益 / Apply low frequency gain
        if (lowGain != 0.0) {
            double gain = Math.pow(10, lowGain / 20.0); // 转换为线性增益 / Convert to linear gain
            equalizedSamples = applyFrequencyGain(equalizedSamples, 0, lowFrequency, gain, sampleRate);
        }
        
        // 应用中频增益 / Apply mid frequency gain
        if (midGain != 0.0) {
            double gain = Math.pow(10, midGain / 20.0); // 转换为线性增益 / Convert to linear gain
            equalizedSamples = applyFrequencyGain(equalizedSamples, lowFrequency, highFrequency, gain, sampleRate);
        }
        
        // 应用高频增益 / Apply high frequency gain
        if (highGain != 0.0) {
            double gain = Math.pow(10, highGain / 20.0); // 转换为线性增益 / Convert to linear gain
            equalizedSamples = applyFrequencyGain(equalizedSamples, highFrequency, sampleRate / 2, gain, sampleRate);
        }
        
        return equalizedSamples;
    }
    
    /**
     * 应用频率增益 / Apply frequency gain
     *
     * @param samples 音频样本 / Audio samples
     * @param lowFreq 低频边界 / Low frequency boundary
     * @param highFreq 高频边界 / High frequency boundary
     * @param gain 增益 / Gain
     * @param sampleRate 采样率 / Sample rate
     * @return 处理后的样本 / Processed samples
     */
    private IVector<Double> applyFrequencyGain(IVector<Double> samples, double lowFreq, double highFreq, 
                                             double gain, double sampleRate) {
        // 简化的频域处理 / Simplified frequency domain processing
        // 这里使用简单的滤波器实现 / Here uses simple filter implementation
        int windowSize = Math.max(32, (int)(sampleRate / highFreq));
        return Signals.movingAverage(samples, windowSize).multiplyScalar(gain);
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
    public EqualizerEnhancer clone() {
        return new EqualizerEnhancer();
    }
}