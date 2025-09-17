package com.reremouse.lab.audio;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.SignalFiltering;
import com.reremouse.lab.math.signal.SignalUtilities;
import com.reremouse.lab.math.stats.Stats;

/**
 * 音频增强器类 / Audio Enhancer Class
 * <p>
 * 提供音频增强功能，包括降噪、均衡、压缩、混响等。
 * 使用项目现有的signal包和stats包功能进行音频处理。
 * </p>
 * <p>
 * Provides audio enhancement functionality including noise reduction, equalization, compression, reverb, etc.
 * Uses existing signal and stats package functionality for audio processing.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AudioEnhancer {
    
    /**
     * 音频降噪 / Audio noise reduction
     * <p>
     * 使用谱减法进行音频降噪，基于噪声谱估计。
     * Use spectral subtraction for audio noise reduction based on noise spectrum estimation.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param noiseProfile 噪声配置文件（可选） / Noise profile (optional)
     * @return 降噪后的音频数据 / Noise-reduced audio data
     */
    public static AudioData reduceNoise(AudioData audioData, NoiseProfile noiseProfile) {
        if (noiseProfile == null) {
            // 自动估计噪声谱 / Automatically estimate noise spectrum
            noiseProfile = estimateNoiseProfile(audioData);
        }
        
        IVector<Double> samples = audioData.getSamples();
        IVector<Double> enhancedSamples = Linalg.zeros(samples.length());
        
        // 分帧处理 / Frame-based processing
        int frameSize = 1024;
        int hopSize = frameSize / 2;
        int numFrames = (samples.length() - frameSize) / hopSize + 1;
        
        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopSize;
            int end = Math.min(start + frameSize, samples.length());
            
            // 提取帧 / Extract frame
            IVector<Double> frameData = samples.slice(start, end);
            
            // 应用窗函数 / Apply window function
            IVector<Double> window = createHanningWindow(frameSize);
            IVector<Double> windowedFrame = frameData.multiply(window);
            
            // 谱减法降噪 / Spectral subtraction noise reduction
            IVector<Double> enhancedFrame = spectralSubtraction(windowedFrame, noiseProfile, audioData.getSampleRate());
            
            // 重叠相加 / Overlap-add
            for (int i = 0; i < enhancedFrame.length(); i++) {
                int idx = start + i;
                if (idx < enhancedSamples.length()) {
                    enhancedSamples.set(idx, enhancedSamples.get(idx) + enhancedFrame.get(i));
                }
            }
        }
        
        return new AudioData(enhancedSamples, audioData.getSampleRate(), audioData.getChannels(), 
                           audioData.getBitDepth(), audioData.getFormat());
    }
    
    /**
     * 音频均衡 / Audio equalization
     * <p>
     * 使用多频段均衡器调整音频的频率响应。
     * Use multi-band equalizer to adjust audio frequency response.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param eqSettings 均衡器设置 / Equalizer settings
     * @return 均衡后的音频数据 / Equalized audio data
     */
    public static AudioData equalize(AudioData audioData, EqualizerSettings eqSettings) {
        IVector<Double> samples = audioData.getSamples();
        IVector<Double> equalizedSamples = Linalg.zeros(samples.length());
        
        // 对每个声道分别处理 / Process each channel separately
        if (audioData.getChannels() > 1) {
            IVector<Double>[] channelData = audioData.getAllChannels();
            @SuppressWarnings("unchecked")
            IVector<Double>[] equalizedChannels = new IVector[channelData.length];
            
            for (int ch = 0; ch < channelData.length; ch++) {
                equalizedChannels[ch] = applyEqualizer(channelData[ch], eqSettings, audioData.getSampleRate());
            }
            
            // 重新合并声道 / Re-merge channels
            equalizedSamples = mergeChannelData(equalizedChannels);
        } else {
            equalizedSamples = applyEqualizer(samples, eqSettings, audioData.getSampleRate());
        }
        
        return new AudioData(equalizedSamples, audioData.getSampleRate(), audioData.getChannels(), 
                           audioData.getBitDepth(), audioData.getFormat());
    }
    
    /**
     * 音频压缩 / Audio compression
     * <p>
     * 使用动态范围压缩器控制音频的动态范围。
     * Use dynamic range compressor to control audio dynamic range.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param compressionSettings 压缩器设置 / Compressor settings
     * @return 压缩后的音频数据 / Compressed audio data
     */
    public static AudioData compress(AudioData audioData, CompressionSettings compressionSettings) {
        IVector<Double> samples = audioData.getSamples();
        IVector<Double> compressedSamples = Linalg.zeros(samples.length());
        
        // 对每个声道分别处理 / Process each channel separately
        if (audioData.getChannels() > 1) {
            IVector<Double>[] channelData = audioData.getAllChannels();
            @SuppressWarnings("unchecked")
            IVector<Double>[] compressedChannels = new IVector[channelData.length];
            
            for (int ch = 0; ch < channelData.length; ch++) {
                compressedChannels[ch] = applyCompression(channelData[ch], compressionSettings);
            }
            
            // 重新合并声道 / Re-merge channels
            compressedSamples = mergeChannelData(compressedChannels);
        } else {
            compressedSamples = applyCompression(samples, compressionSettings);
        }
        
        return new AudioData(compressedSamples, audioData.getSampleRate(), audioData.getChannels(), 
                           audioData.getBitDepth(), audioData.getFormat());
    }
    
    /**
     * 添加混响效果 / Add reverb effect
     * <p>
     * 使用简单的延迟和反馈网络模拟混响效果。
     * Use simple delay and feedback network to simulate reverb effect.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param reverbSettings 混响设置 / Reverb settings
     * @return 添加混响后的音频数据 / Audio data with reverb effect
     */
    public static AudioData addReverb(AudioData audioData, ReverbSettings reverbSettings) {
        IVector<Double> samples = audioData.getSamples();
        IVector<Double> reverbSamples = Linalg.zeros(samples.length());
        
        // 对每个声道分别处理 / Process each channel separately
        if (audioData.getChannels() > 1) {
            IVector<Double>[] channelData = audioData.getAllChannels();
            @SuppressWarnings("unchecked")
            IVector<Double>[] reverbChannels = new IVector[channelData.length];
            
            for (int ch = 0; ch < channelData.length; ch++) {
                reverbChannels[ch] = applyReverb(channelData[ch], reverbSettings, audioData.getSampleRate());
            }
            
            // 重新合并声道 / Re-merge channels
            reverbSamples = mergeChannelData(reverbChannels);
        } else {
            reverbSamples = applyReverb(samples, reverbSettings, audioData.getSampleRate());
        }
        
        return new AudioData(reverbSamples, audioData.getSampleRate(), audioData.getChannels(), 
                           audioData.getBitDepth(), audioData.getFormat());
    }
    
    /**
     * 估计噪声谱 / Estimate noise spectrum
     *
     * @param audioData 音频数据 / Audio data
     * @return 噪声配置文件 / Noise profile
     */
    private static NoiseProfile estimateNoiseProfile(AudioData audioData) {
        IVector<Double> samples = audioData.getSamples();
        
        // 假设前10%的音频为噪声 / Assume first 10% of audio is noise
        int noiseLength = Math.max(1000, samples.length() / 10);
        IVector<Double> noiseSamples = samples.slice(0, noiseLength);
        
        // 计算噪声的统计特性 / Calculate noise statistical properties
        double noiseMean = noiseSamples.mean();
        double noiseStd = Math.sqrt(noiseSamples.var());
        
        return new NoiseProfile(noiseMean, noiseStd);
    }
    
    /**
     * 谱减法降噪 / Spectral subtraction noise reduction
     *
     * @param frame 音频帧 / Audio frame
     * @param noiseProfile 噪声配置文件 / Noise profile
     * @param sampleRate 采样率 / Sample rate
     * @return 降噪后的帧 / Noise-reduced frame
     */
    private static IVector<Double> spectralSubtraction(IVector<Double> frame, NoiseProfile noiseProfile, double sampleRate) {
        // 简化的谱减法实现 / Simplified spectral subtraction implementation
        IVector<Double> enhancedFrame = Linalg.zeros(frame.length());
        
        // 计算噪声阈值 / Calculate noise threshold
        double noiseThreshold = noiseProfile.getMean() + 2 * noiseProfile.getStdDev();
        
        for (int i = 0; i < frame.length(); i++) {
            double sample = frame.get(i);
            
            // 如果样本低于噪声阈值，则衰减 / If sample is below noise threshold, attenuate
            if (Math.abs(sample) < noiseThreshold) {
                enhancedFrame.set(i, sample * 0.1); // 衰减到10% / Attenuate to 10%
            } else {
                enhancedFrame.set(i, sample);
            }
        }
        
        return enhancedFrame;
    }
    
    /**
     * 应用均衡器 / Apply equalizer
     *
     * @param samples 音频样本 / Audio samples
     * @param eqSettings 均衡器设置 / Equalizer settings
     * @param sampleRate 采样率 / Sample rate
     * @return 均衡后的样本 / Equalized samples
     */
    private static IVector<Double> applyEqualizer(IVector<Double> samples, EqualizerSettings eqSettings, double sampleRate) {
        IVector<Double> equalizedSamples = samples.copy();
        
        // 应用各个频段的增益 / Apply gains for each frequency band
        for (EqualizerBand band : eqSettings.getBands()) {
            double gain = Math.pow(10, band.getGain() / 20.0); // 转换为线性增益 / Convert to linear gain
            
            // 简化的频域均衡 / Simplified frequency domain equalization
            // 这里使用简单的滤波器实现 / Here uses simple filter implementation
            equalizedSamples = applyBandGain(equalizedSamples, band.getFrequency(), gain, sampleRate);
        }
        
        return equalizedSamples;
    }
    
    /**
     * 应用频段增益 / Apply band gain
     *
     * @param samples 音频样本 / Audio samples
     * @param frequency 中心频率 / Center frequency
     * @param gain 增益 / Gain
     * @param sampleRate 采样率 / Sample rate
     * @return 处理后的样本 / Processed samples
     */
    private static IVector<Double> applyBandGain(IVector<Double> samples, double frequency, double gain, double sampleRate) {
        // 简化的带通滤波器实现 / Simplified band-pass filter implementation
        double normalizedFreq = frequency / (sampleRate / 2);
        int filterOrder = 4;
        
        // 使用简单的移动平均滤波器 / Use simple moving average filter
        int windowSize = (int) (1.0 / normalizedFreq);
        return SignalFiltering.movingAverage(samples, windowSize);
    }
    
    /**
     * 应用压缩器 / Apply compressor
     *
     * @param samples 音频样本 / Audio samples
     * @param settings 压缩器设置 / Compressor settings
     * @return 压缩后的样本 / Compressed samples
     */
    private static IVector<Double> applyCompression(IVector<Double> samples, CompressionSettings settings) {
        IVector<Double> compressedSamples = Linalg.zeros(samples.length());
        
        double threshold = settings.getThreshold();
        double ratio = settings.getRatio();
        double attack = settings.getAttack();
        double release = settings.getRelease();
        
        double envelope = 0;
        
        for (int i = 0; i < samples.length(); i++) {
            double input = samples.get(i);
            double inputLevel = Math.abs(input);
            
            // 包络检测 / Envelope detection
            if (inputLevel > envelope) {
                envelope = inputLevel + (envelope - inputLevel) * Math.exp(-1.0 / attack);
            } else {
                envelope = inputLevel + (envelope - inputLevel) * Math.exp(-1.0 / release);
            }
            
            // 压缩计算 / Compression calculation
            double output;
            if (envelope > threshold) {
                double overThreshold = envelope - threshold;
                double compressedOverThreshold = overThreshold / ratio;
                output = input * (threshold + compressedOverThreshold) / envelope;
            } else {
                output = input;
            }
            
            compressedSamples.set(i, output);
        }
        
        return compressedSamples;
    }
    
    /**
     * 应用混响效果 / Apply reverb effect
     *
     * @param samples 音频样本 / Audio samples
     * @param settings 混响设置 / Reverb settings
     * @param sampleRate 采样率 / Sample rate
     * @return 添加混响后的样本 / Samples with reverb effect
     */
    private static IVector<Double> applyReverb(IVector<Double> samples, ReverbSettings settings, double sampleRate) {
        int delaySamples = (int) (settings.getDelay() * sampleRate);
        double feedback = settings.getFeedback();
        double wetLevel = settings.getWetLevel();
        double dryLevel = settings.getDryLevel();
        
        IVector<Double> reverbSamples = Linalg.zeros(samples.length() + delaySamples);
        
        // 复制原始信号 / Copy original signal
        for (int i = 0; i < samples.length(); i++) {
            reverbSamples.set(i, samples.get(i) * dryLevel);
        }
        
        // 添加延迟和反馈 / Add delay and feedback
        for (int i = 0; i < samples.length(); i++) {
            double input = samples.get(i);
            double delayed = (i >= delaySamples) ? reverbSamples.get(i - delaySamples) : 0;
            double reverb = input + delayed * feedback;
            
            if (i + delaySamples < reverbSamples.length()) {
                reverbSamples.set(i + delaySamples, reverb * wetLevel);
            }
        }
        
        return reverbSamples.slice(0, samples.length());
    }
    
    /**
     * 合并声道数据 / Merge channel data
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
     * 创建汉宁窗 / Create Hanning window
     *
     * @param size 窗函数大小 / Window size
     * @return 汉宁窗向量 / Hanning window vector
     */
    private static IVector<Double> createHanningWindow(int size) {
        IVector<Double> window = Linalg.zeros(size);
        for (int i = 0; i < size; i++) {
            double value = 0.5 * (1 - Math.cos(2 * Math.PI * i / (size - 1)));
            window.set(i, value);
        }
        return window;
    }
}
