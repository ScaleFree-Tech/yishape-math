package com.yishape.lab.audio.processing;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.Map;

/**
 * 声道处理器实现 / Channel Processor Implementation
 * <p>
 * 处理音频的声道转换操作，如单声道转立体声、立体声转单声道等。
 * Processes audio channel conversion operations, such as mono to stereo, stereo to mono, etc.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class ChannelProcessor extends AbstractAudioProcessorStandard {
    
    /**
     * 声道操作类型枚举 / Channel operation type enum
     */
    public enum ChannelOperation {
        MONO_TO_STEREO,  // 单声道转立体声 / Mono to stereo
        STEREO_TO_MONO,  // 立体声转单声道 / Stereo to mono
        SEPARATE_CHANNELS // 分离声道 / Separate channels
    }
    
    /**
     * 构造函数 / Constructor
     */
    public ChannelProcessor() {
        super("channel", "Audio channel processor");
        addSupportedParameter("operation", ChannelOperation.MONO_TO_STEREO);
    }
    
    @Override
    protected AudioData doProcess(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 获取操作参数 / Get operation parameter
            ChannelOperation operation = ChannelOperation.MONO_TO_STEREO;
            if (parameters != null && parameters.containsKey("operation")) {
                operation = (ChannelOperation) parameters.get("operation");
            }
            
            switch (operation) {
                case MONO_TO_STEREO:
                    return monoToStereo(input);
                case STEREO_TO_MONO:
                    return stereoToMono(input);
                case SEPARATE_CHANNELS:
                    // 对于分离声道操作，我们返回第一个声道 / For channel separation, we return the first channel
                    return separateChannels(input)[0];
                default:
                    throw new AudioProcessingException("Unsupported channel operation: " + operation);
            }
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to process channels", e);
        }
    }
    
    /**
     * 单声道转立体声 / Mono to stereo
     */
    private AudioData monoToStereo(AudioData audioData) throws AudioProcessingException {
        if (!audioData.isMono()) {
            throw new AudioProcessingException("Input audio must be mono");
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
     * 立体声转单声道 / Stereo to mono
     */
    private AudioData stereoToMono(AudioData audioData) throws AudioProcessingException {
        if (!audioData.isStereo()) {
            throw new AudioProcessingException("Input audio must be stereo");
        }
        
        IVector<Double> leftChannel = audioData.getChannel(0);
        IVector<Double> rightChannel = audioData.getChannel(1);
        
        // 平均左右声道 / Average left and right channels
        IVector<Double> monoData = leftChannel.add(rightChannel).multiplyScalar(0.5);
        
        return new AudioData(monoData, audioData.getSampleRate(), 1, 
                           audioData.getBitDepth(), audioData.getFormat());
    }
    
    /**
     * 分离声道 / Separate channels
     */
    private AudioData[] separateChannels(AudioData audioData) throws AudioProcessingException {
        IVector<Double>[] channelData = audioData.getAllChannels();
        AudioData[] channelAudios = new AudioData[channelData.length];
        
        for (int i = 0; i < channelData.length; i++) {
            channelAudios[i] = new AudioData(channelData[i], audioData.getSampleRate(), 1, 
                                          audioData.getBitDepth(), audioData.getFormat());
        }
        
        return channelAudios;
    }
    
    @Override
    public ChannelProcessor clone() {
        return new ChannelProcessor();
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

    @Override
    public Map<String, Object> getLastProcessingStatistics() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    @Override
    public void setVerboseLogging(boolean enabled) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    @Override
    public boolean isVerboseLoggingEnabled() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }
}