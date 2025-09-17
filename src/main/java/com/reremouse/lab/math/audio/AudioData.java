package com.reremouse.lab.math.audio;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;

/**
 * 音频数据类 / Audio Data Class
 * <p>
 * 封装音频数据的基本信息，包括采样率、声道数、位深度、时长等。
 * 使用IVector接口存储音频样本数据，确保与现有代码库的兼容性。
 * </p>
 * <p>
 * Encapsulates basic audio information including sample rate, channels, bit depth, duration, etc.
 * Uses IVector interface to store audio sample data, ensuring compatibility with existing codebase.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AudioData {
    
    /** 音频样本数据 / Audio sample data */
    private final IVector<Double> samples;
    
    /** 采样率 (Hz) / Sample rate (Hz) */
    private final double sampleRate;
    
    /** 声道数 / Number of channels */
    private final int channels;
    
    /** 位深度 (bits) / Bit depth (bits) */
    private final int bitDepth;
    
    /** 时长 (秒) / Duration (seconds) */
    private final double duration;
    
    /** 音频格式 / Audio format */
    private final AudioFormat format;
    
    /**
     * 构造函数 / Constructor
     *
     * @param samples 音频样本数据 / Audio sample data
     * @param sampleRate 采样率 / Sample rate
     * @param channels 声道数 / Number of channels
     * @param bitDepth 位深度 / Bit depth
     * @param format 音频格式 / Audio format
     */
    public AudioData(IVector<Double> samples, double sampleRate, int channels, int bitDepth, AudioFormat format) {
        this.samples = samples;
        this.sampleRate = sampleRate;
        this.channels = channels;
        this.bitDepth = bitDepth;
        this.format = format;
        this.duration = samples.length() / (sampleRate * channels);
    }
    
    /**
     * 获取音频样本数据 / Get audio sample data
     *
     * @return 音频样本向量 / Audio sample vector
     */
    public IVector<Double> getSamples() {
        return samples;
    }
    
    /**
     * 获取采样率 / Get sample rate
     *
     * @return 采样率 (Hz) / Sample rate (Hz)
     */
    public double getSampleRate() {
        return sampleRate;
    }
    
    /**
     * 获取声道数 / Get number of channels
     *
     * @return 声道数 / Number of channels
     */
    public int getChannels() {
        return channels;
    }
    
    /**
     * 获取位深度 / Get bit depth
     *
     * @return 位深度 (bits) / Bit depth (bits)
     */
    public int getBitDepth() {
        return bitDepth;
    }
    
    /**
     * 获取时长 / Get duration
     *
     * @return 时长 (秒) / Duration (seconds)
     */
    public double getDuration() {
        return duration;
    }
    
    /**
     * 获取音频格式 / Get audio format
     *
     * @return 音频格式 / Audio format
     */
    public AudioFormat getFormat() {
        return format;
    }
    
    /**
     * 获取指定声道的数据 / Get data for specific channel
     *
     * @param channel 声道索引 (0-based) / Channel index (0-based)
     * @return 指定声道的样本数据 / Sample data for specified channel
     * @throws IllegalArgumentException 如果声道索引无效 / If channel index is invalid
     */
    public IVector<Double> getChannel(int channel) {
        if (channel < 0 || channel >= channels) {
            throw new IllegalArgumentException("Invalid channel index: " + channel);
        }
        
        int samplesPerChannel = samples.length() / channels;
        IVector<Double> channelSamples = Linalg.zeros(samplesPerChannel);
        
        for (int i = 0; i < samplesPerChannel; i++) {
            channelSamples.set(i, samples.get(i * channels + channel));
        }
        
        return channelSamples;
    }
    
    /**
     * 获取所有声道的数据 / Get data for all channels
     *
     * @return 声道数据矩阵，每行代表一个声道 / Channel data matrix, each row represents a channel
     */
    public IVector<Double>[] getAllChannels() {
        @SuppressWarnings("unchecked")
        IVector<Double>[] channelData = new IVector[channels];
        
        for (int ch = 0; ch < channels; ch++) {
            channelData[ch] = getChannel(ch);
        }
        
        return channelData;
    }
    
    /**
     * 获取音频长度（样本数） / Get audio length (number of samples)
     *
     * @return 样本总数 / Total number of samples
     */
    public int getLength() {
        return samples.length();
    }
    
    /**
     * 获取每个声道的样本数 / Get number of samples per channel
     *
     * @return 每个声道的样本数 / Number of samples per channel
     */
    public int getSamplesPerChannel() {
        return samples.length() / channels;
    }
    
    /**
     * 检查是否为单声道 / Check if mono
     *
     * @return 如果是单声道返回true / True if mono
     */
    public boolean isMono() {
        return channels == 1;
    }
    
    /**
     * 检查是否为立体声 / Check if stereo
     *
     * @return 如果是立体声返回true / True if stereo
     */
    public boolean isStereo() {
        return channels == 2;
    }
    
    /**
     * 获取音频的基本统计信息 / Get basic audio statistics
     *
     * @return 音频统计信息 / Audio statistics
     */
    public AudioStatistics getStatistics() {
        return new AudioStatistics(samples);
    }
    
    @Override
    public String toString() {
        return String.format("AudioData{sampleRate=%.1fHz, channels=%d, bitDepth=%d, duration=%.2fs, format=%s}",
                sampleRate, channels, bitDepth, duration, format);
    }
}
