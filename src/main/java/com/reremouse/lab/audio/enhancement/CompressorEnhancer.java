package com.reremouse.lab.audio.enhancement;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import java.util.Map;

/**
 * 压缩增强器实现 / Compressor Enhancer Implementation
 * <p>
 * 实现音频压缩功能，使用动态范围压缩器控制音频的动态范围。
 * Implements audio compression functionality, using dynamic range compressor to control audio dynamic range.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class CompressorEnhancer extends AbstractAudioEnhancer {
    
    /**
     * 构造函数 / Constructor
     */
    public CompressorEnhancer() {
        super("compressor", "Compressor enhancer", EnhancerType.COMPRESSION);
        addSupportedParameter("threshold", -20.0);
        addSupportedParameter("ratio", 4.0);
        addSupportedParameter("attack", 5.0);
        addSupportedParameter("release", 50.0);
        addSupportedParameter("makeupGain", 0.0);
    }
    
    @Override
    protected AudioData doEnhance(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 获取参数 / Get parameters
            double threshold = -20.0;
            double ratio = 4.0;
            double attack = 5.0;
            double release = 50.0;
            double makeupGain = 0.0;
            
            if (parameters != null) {
                if (parameters.containsKey("threshold")) {
                    threshold = (Double) parameters.get("threshold");
                }
                if (parameters.containsKey("ratio")) {
                    ratio = (Double) parameters.get("ratio");
                }
                if (parameters.containsKey("attack")) {
                    attack = (Double) parameters.get("attack");
                }
                if (parameters.containsKey("release")) {
                    release = (Double) parameters.get("release");
                }
                if (parameters.containsKey("makeupGain")) {
                    makeupGain = (Double) parameters.get("makeupGain");
                }
            }
            
            IVector<Double> samples = input.getSamples();
            IVector<Double> compressedSamples = Linalg.zeros(samples.length());
            
            // 对每个声道分别处理 / Process each channel separately
            if (input.getChannels() > 1) {
                IVector<Double>[] channelData = input.getAllChannels();
                @SuppressWarnings("unchecked")
                IVector<Double>[] compressedChannels = new IVector[channelData.length];
                
                for (int ch = 0; ch < channelData.length; ch++) {
                    compressedChannels[ch] = applyCompression(channelData[ch], threshold, ratio, attack, release, makeupGain);
                }
                
                // 重新合并声道 / Re-merge channels
                compressedSamples = mergeChannelData(compressedChannels);
            } else {
                compressedSamples = applyCompression(samples, threshold, ratio, attack, release, makeupGain);
            }
            
            return new AudioData(compressedSamples, input.getSampleRate(), input.getChannels(), 
                               input.getBitDepth(), input.getFormat());
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to apply compression", e);
        }
    }
    
    /**
     * 应用压缩器 / Apply compressor
     *
     * @param samples 音频样本 / Audio samples
     * @param threshold 阈值 / Threshold
     * @param ratio 压缩比 / Ratio
     * @param attack 启动时间 / Attack time
     * @param release 释放时间 / Release time
     * @param makeupGain 补偿增益 / Makeup gain
     * @return 压缩后的样本 / Compressed samples
     */
    private IVector<Double> applyCompression(IVector<Double> samples, double threshold, double ratio, 
                                          double attack, double release, double makeupGain) {
        IVector<Double> compressedSamples = Linalg.zeros(samples.length());
        
        // 将阈值从dB转换为线性值 / Convert threshold from dB to linear value
        double thresholdLinear = Math.pow(10, threshold / 20.0);
        double makeupGainLinear = Math.pow(10, makeupGain / 20.0);
        
        double envelope = 0;
        
        for (int i = 0; i < samples.length(); i++) {
            double input = samples.get(i);
            double inputLevel = Math.abs(input);
            
            // 包络检测 / Envelope detection
            if (inputLevel > envelope) {
                envelope = inputLevel + (envelope - inputLevel) * Math.exp(-1.0 / (attack * 0.001 * 44100));
            } else {
                envelope = inputLevel + (envelope - inputLevel) * Math.exp(-1.0 / (release * 0.001 * 44100));
            }
            
            // 压缩计算 / Compression calculation
            double output;
            if (envelope > thresholdLinear) {
                double overThreshold = envelope - thresholdLinear;
                double compressedOverThreshold = overThreshold / ratio;
                output = input * (thresholdLinear + compressedOverThreshold) / envelope;
            } else {
                output = input;
            }
            
            // 应用补偿增益 / Apply makeup gain
            output *= makeupGainLinear;
            
            // 限制输出范围 / Limit output range
            output = Math.max(-1.0, Math.min(1.0, output));
            
            compressedSamples.set(i, output);
        }
        
        return compressedSamples;
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
    public CompressorEnhancer clone() {
        return new CompressorEnhancer();
    }
}