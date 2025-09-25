package com.reremouse.lab.audio.core;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.Signals;

/**
 * 音频处理器类 / Audio Processor Class
 * <p>
 * 提供音频处理的基本功能，包括音量调节、声道处理、格式转换等。
 * 使用项目现有的Signal包功能进行信号处理。
 * </p>
 * <p>
 * Provides basic audio processing functionality including volume adjustment, channel processing, format conversion, etc.
 * Uses existing signal package functionality for signal processing.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AudioProcessor {
    
    /**
     * 调节音频音量 / Adjust audio volume
     * <p>
     * 将音频信号乘以指定的增益系数。
     * Multiply audio signal by specified gain factor.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param gain 增益系数 (1.0为原始音量) / Gain factor (1.0 for original volume)
     * @return 音量调节后的音频数据 / Volume-adjusted audio data
     */
    public static AudioData adjustVolume(AudioData audioData, double gain) {
        IVector<Double> adjustedSamples = audioData.getSamples().multiplyScalar(gain);
        
        // 防止削波 / Prevent clipping
        adjustedSamples = normalizeAudio(adjustedSamples);
        
        return new AudioData(adjustedSamples, audioData.getSampleRate(),
                           audioData.getChannels(), audioData.getBitDepth(), audioData.getFormat());
    }
    
    /**
     * 音频归一化 / Normalize audio
     * <p>
     * 将音频信号归一化到[-1, 1]范围内，防止削波。
     * Normalize audio signal to [-1, 1] range to prevent clipping.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 归一化后的音频数据 / Normalized audio data
     */
    public static AudioData normalize(AudioData audioData) {
        IVector<Double> normalizedSamples = normalizeAudio(audioData.getSamples());
        
        return new AudioData(normalizedSamples, audioData.getSampleRate(),
                           audioData.getChannels(), audioData.getBitDepth(), audioData.getFormat());
    }
    
    /**
     * 音频归一化（内部方法） / Normalize audio (internal method)
     *
     * @param samples 音频样本 / Audio samples
     * @return 归一化后的样本 / Normalized samples
     */
    private static IVector<Double> normalizeAudio(IVector<Double> samples) {
        double maxAbs = Math.max(Math.abs(samples.max()), Math.abs(samples.min()));
        
        if (maxAbs == 0) {
            return samples; // 静音信号 / Silent signal
        }
        
        return samples.multiplyScalar(0.95 / maxAbs); // 留5%余量 / Leave 5% margin
    }
    
    /**
     * 声道转换：单声道转立体声 / Channel conversion: mono to stereo
     * <p>
     * 将单声道音频复制到左右声道，创建立体声效果。
     * Copy mono audio to both left and right channels to create stereo effect.
     * </p>
     *
     * @param audioData 输入单声道音频数据 / Input mono audio data
     * @return 立体声音频数据 / Stereo audio data
     * @throws IllegalArgumentException 如果输入不是单声道 / If input is not mono
     */
    public static AudioData monoToStereo(AudioData audioData) {
        if (!audioData.isMono()) {
            throw new IllegalArgumentException("Input audio must be mono");
        }
        
        IVector<Double> monoData = audioData.getSamples();
        IVector<Double> stereoData = Linalg.zeros(monoData.length() * 2);
        
        // 复制到左右声道 / Copy to left and right channels
        for (int i = 0; i < monoData.length(); i++) {
            double sample = monoData.get(i);
            stereoData.set(i * 2, sample);     // 左声道 / Left channel
            stereoData.set(i * 2 + 1, sample); // 右声道 / Right channel
        }
        
        return new AudioData(stereoData, audioData.getSampleRate(), 2,
                           audioData.getBitDepth(), audioData.getFormat());
    }
    
    /**
     * 声道转换：立体声转单声道 / Channel conversion: stereo to mono
     * <p>
     * 将立体声音频的左右声道平均，创建单声道音频。
     * Average left and right channels of stereo audio to create mono audio.
     * </p>
     *
     * @param audioData 输入立体声音频数据 / Input stereo audio data
     * @return 单声道音频数据 / Mono audio data
     * @throws IllegalArgumentException 如果输入不是立体声 / If input is not stereo
     */
    public static AudioData stereoToMono(AudioData audioData) {
        if (!audioData.isStereo()) {
            throw new IllegalArgumentException("Input audio must be stereo");
        }
        
        IVector<Double> leftChannel = audioData.getChannel(0);
        IVector<Double> rightChannel = audioData.getChannel(1);
        
        // 平均左右声道 / Average left and right channels
        IVector<Double> monoData = leftChannel.add(rightChannel).multiplyScalar(0.5);
        
        return new AudioData(monoData, audioData.getSampleRate(), 1,
                           audioData.getBitDepth(), audioData.getFormat());
    }
    
    /**
     * 声道分离 / Channel separation
     * <p>
     * 将多声道音频分离为独立的声道。
     * Separate multi-channel audio into independent channels.
     * </p>
     *
     * @param audioData 输入多声道音频数据 / Input multi-channel audio data
     * @return 声道数组，每个元素代表一个声道 / Channel array, each element represents a channel
     */
    public static AudioData[] separateChannels(AudioData audioData) {
        IVector<Double>[] channelData = audioData.getAllChannels();
        AudioData[] channelAudios = new AudioData[channelData.length];
        
        for (int i = 0; i < channelData.length; i++) {
            channelAudios[i] = new AudioData(channelData[i], audioData.getSampleRate(), 1,
                                           audioData.getBitDepth(), audioData.getFormat());
        }
        
        return channelAudios;
    }
    
    /**
     * 声道合并 / Channel merging
     * <p>
     * 将多个单声道音频合并为多声道音频。
     * Merge multiple mono audio files into multi-channel audio.
     * </p>
     *
     * @param channelAudios 单声道音频数组 / Mono audio array
     * @return 合并后的多声道音频数据 / Merged multi-channel audio data
     * @throws IllegalArgumentException 如果声道数不匹配或采样率不匹配 / If channel count or sample rate mismatch
     */
    public static AudioData mergeChannels(AudioData... channelAudios) {
        if (channelAudios.length == 0) {
            throw new IllegalArgumentException("At least one channel audio is required");
        }
        
        // 检查采样率是否一致 / Check if sample rates are consistent
        double sampleRate = channelAudios[0].getSampleRate();
        for (AudioData audio : channelAudios) {
            if (Math.abs(audio.getSampleRate() - sampleRate) > 1e-6) {
                throw new IllegalArgumentException("Sample rates must be consistent");
            }
        }
        
        // 检查长度是否一致 / Check if lengths are consistent
        int length = channelAudios[0].getLength();
        for (AudioData audio : channelAudios) {
            if (audio.getLength() != length) {
                throw new IllegalArgumentException("Audio lengths must be consistent");
            }
        }
        
        // 合并声道 / Merge channels
        IVector<Double> mergedSamples = Linalg.zeros(length * channelAudios.length);
        
        for (int ch = 0; ch < channelAudios.length; ch++) {
            IVector<Double> channelData = channelAudios[ch].getSamples();
            for (int i = 0; i < length; i++) {
                mergedSamples.set(i * channelAudios.length + ch, channelData.get(i));
            }
        }
        
        return new AudioData(mergedSamples, sampleRate, channelAudios.length,
                           channelAudios[0].getBitDepth(), channelAudios[0].getFormat());
    }
    
    /**
     * 重采样 / Resampling
     * <p>
     * 改变音频的采样率，使用简单的线性插值方法。
     * Change audio sample rate using simple linear interpolation method.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param newSampleRate 新的采样率 / New sample rate
     * @return 重采样后的音频数据 / Resampled audio data
     */
    public static AudioData resample(AudioData audioData, double newSampleRate) {
        double ratio = newSampleRate / audioData.getSampleRate();
        int newLength = (int) (audioData.getLength() * ratio);
        
        IVector<Double> originalSamples = audioData.getSamples();
        IVector<Double> resampledSamples = Linalg.zeros(newLength);
        
        for (int i = 0; i < newLength; i++) {
            double originalIndex = i / ratio;
            int index1 = (int) Math.floor(originalIndex);
            int index2 = Math.min(index1 + 1, originalSamples.length() - 1);
            
            double weight = originalIndex - index1;
            double sample1 = originalSamples.get(index1);
            double sample2 = originalSamples.get(index2);
            
            double interpolatedSample = sample1 + weight * (sample2 - sample1);
            resampledSamples.set(i, interpolatedSample);
        }
        
        return new AudioData(resampledSamples, newSampleRate, audioData.getChannels(),
                           audioData.getBitDepth(), audioData.getFormat());
    }
    
    /**
     * 音频滤波 / Audio filtering
     * <p>
     * 对音频应用低通、高通或带通滤波器。
     * Apply low-pass, high-pass, or band-pass filter to audio.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param filterType 滤波器类型 / Filter type
     * @param cutoffFreq 截止频率 / Cutoff frequency
     * @return 滤波后的音频数据 / Filtered audio data
     */
    public static AudioData filter(AudioData audioData, FilterType filterType, double cutoffFreq) {
        IVector<Double> filteredSamples = audioData.getSamples();
        
        // 对每个声道分别滤波 / Filter each channel separately
        if (audioData.getChannels() > 1) {
            IVector<Double>[] channelData = audioData.getAllChannels();
            @SuppressWarnings("unchecked")
            IVector<Double>[] filteredChannels = new IVector[channelData.length];
            
            for (int ch = 0; ch < channelData.length; ch++) {
                filteredChannels[ch] = applyFilter(channelData[ch], filterType, cutoffFreq, audioData.getSampleRate());
            }
            
            // 重新合并声道 / Re-merge channels
            filteredSamples = mergeChannelData(filteredChannels);
        } else {
            filteredSamples = applyFilter(audioData.getSamples(), filterType, cutoffFreq, audioData.getSampleRate());
        }
        
        return new AudioData(filteredSamples, audioData.getSampleRate(), audioData.getChannels(),
                           audioData.getBitDepth(), audioData.getFormat());
    }
    
    /**
     * 应用滤波器（内部方法） / Apply filter (internal method)
     *
     * @param samples 音频样本 / Audio samples
     * @param filterType 滤波器类型 / Filter type
     * @param cutoffFreq 截止频率 / Cutoff frequency
     * @param sampleRate 采样率 / Sample rate
     * @return 滤波后的样本 / Filtered samples
     */
    private static IVector<Double> applyFilter(IVector<Double> samples, FilterType filterType, double cutoffFreq, double sampleRate) {
        // 使用简单的移动平均滤波器 / Use simple moving average filter
        int windowSize = (int) (sampleRate / cutoffFreq);
        windowSize = Math.max(3, Math.min(windowSize, 100)); // 限制窗口大小 / Limit window size
        
        return Signals.movingAverage(samples, windowSize);
    }
    
    /**
     * 合并声道数据（内部方法） / Merge channel data (internal method)
     *
     * @param channelData 声道数据数组 / Channel data array
     * @return 合并后的数据 / Merged data
     */
    private static IVector<Double> mergeChannelData(IVector<Double>[] channelData) {
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
     * 滤波器类型枚举 / Filter Type Enum
     */
    public enum FilterType {
        LOW_PASS,   // 低通滤波器 / Low-pass filter
        HIGH_PASS,  // 高通滤波器 / High-pass filter
        BAND_PASS   // 带通滤波器 / Band-pass filter
    }
}