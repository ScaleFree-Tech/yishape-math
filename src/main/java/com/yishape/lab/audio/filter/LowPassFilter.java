package com.yishape.lab.audio.filter;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.Signals;

/**
 * 低通滤波器实现 / Low-pass Filter Implementation
 * <p>
 * 实现低通滤波功能，允许低频信号通过，衰减高频信号。
 * Implements low-pass filtering functionality, allowing low-frequency signals to pass while attenuating high-frequency signals.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class LowPassFilter extends AbstractAudioFilterStandard {
    
    /**
     * 构造函数 / Constructor（名称与描述由父类 {@code super} 固定为 lowpass / Low-pass filter）
     */
    public LowPassFilter() {
        super("lowpass", "Low-pass filter");
        setFilterType(FilterType.LOW_PASS);
        setDefaultParameter("cutoffFrequency", 1000.0);
        setDefaultParameter("order", 4);
    }
    
    @Override
    protected AudioData doFilter(AudioData input) throws AudioProcessingException {
        try {
            // 对于简化实现，使用移动平均滤波器 / For simplified implementation, use moving average filter
            IVector<Double> filteredSamples = input.getSamples();
            
            // 对每个声道分别滤波 / Filter each channel separately
            if (input.getChannels() > 1) {
                IVector<Double>[] channelData = input.getAllChannels();
                @SuppressWarnings("unchecked")
                IVector<Double>[] filteredChannels = new IVector[channelData.length];
                
                for (int ch = 0; ch < channelData.length; ch++) {
                    filteredChannels[ch] = applyFilter(channelData[ch]);
                }
                
                // 重新合并声道 / Re-merge channels
                filteredSamples = mergeChannelData(filteredChannels);
            } else {
                filteredSamples = applyFilter(input.getSamples());
            }
            
            return new AudioData(filteredSamples, input.getSampleRate(), input.getChannels(), 
                               input.getBitDepth(), input.getFormat());
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to apply low-pass filter", e);
        }
    }
    
    /**
     * 应用滤波器（内部方法） / Apply filter (internal method)
     *
     * @param samples 音频样本 / Audio samples
     * @return 滤波后的样本 / Filtered samples
     */
    private IVector<Double> applyFilter(IVector<Double> samples) {
        // 使用简单的移动平均滤波器 / Use simple moving average filter
        int windowSize = (int) (getSampleRate() / getCutoffFrequency());
        windowSize = Math.max(3, Math.min(windowSize, 100)); // 限制窗口大小 / Limit window size
        
        return Signals.movingAverage(samples, windowSize);
    }
    
    /**
     * 获取采样率（辅助方法） / Get sample rate (helper method)
     *
     * @return 采样率 (Hz) / Sample rate (Hz)
     */
    private double getSampleRate() {
        // 在实际实现中，这可能需要通过其他方式获取 / In actual implementation, this may need to be obtained through other means
        return 44100.0; // 默认采样率 / Default sample rate
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
    
    /**
     * 创建此滤波器的克隆 / Create clone of this filter
     *
     * @return 低通滤波器克隆 / Low-pass filter clone
     */
    @Override
    public LowPassFilter clone() {
        LowPassFilter cloned = new LowPassFilter();
        cloned.setCutoffFrequency(getCutoffFrequency());
        cloned.setOrder(getOrder());
        return cloned;
    }
}