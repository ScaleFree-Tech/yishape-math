package com.reremouse.lab.audio.processing;

import com.reremouse.lab.audio.AudioData;
import com.reremouse.lab.audio.AudioProcessor;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.RereFFT;
import com.reremouse.lab.math.signal.Complex;

/**
 * 高级音频处理类 / Advanced Audio Processing Class
 * <p>
 * 提供高级音频处理功能，包括变调、时间拉伸、混响、回声等效果。
 * Provides advanced audio processing including pitch shifting, time stretching, reverb, echo effects.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AdvancedAudioProcessing {
    
    /**
     * 音调转换 / Pitch Shifting
     * 使用PSOLA(Pitch Synchronous Overlap and Add)算法进行音调转换
     * 
     * @param audioData 输入音频 / Input audio
     * @param semitones 半音变化量 / Semitone shift amount
     * @return 变调后的音频 / Pitch-shifted audio
     */
    public static AudioData pitchShift(AudioData audioData, double semitones) throws AudioProcessingException {
        double pitchRatio = Math.pow(2.0, semitones / 12.0);
        
        // 转换为单声道处理 / Convert to mono for processing
        AudioData monoAudio = audioData.isMono() ? audioData : AudioProcessor.stereoToMono(audioData);
        IVector<Double> samples = monoAudio.getSamples();
        
        // 使用简化的频域变调算法 / Use simplified frequency domain pitch shifting
        return pitchShiftFrequencyDomain(samples, pitchRatio, audioData.getSampleRate(), audioData);
    }
    
    /**
     * 时间拉伸 / Time Stretching
     * 改变音频时长而不改变音调
     * 
     * @param audioData 输入音频 / Input audio
     * @param stretchFactor 拉伸因子 / Stretch factor
     * @return 时间拉伸后的音频 / Time-stretched audio
     */
    public static AudioData timeStretch(AudioData audioData, double stretchFactor) throws AudioProcessingException {
        if (stretchFactor <= 0) {
            throw new AudioProcessingException("Stretch factor must be positive");
        }
        
        AudioData monoAudio = audioData.isMono() ? audioData : AudioProcessor.stereoToMono(audioData);
        IVector<Double> samples = monoAudio.getSamples();
        
        return timeStretchPSOLA(samples, stretchFactor, audioData);
    }
    
    /**
     * 高质量混响效果 / High Quality Reverb Effect
     * 
     * @param audioData 输入音频 / Input audio
     * @param roomSize 房间大小 / Room size (0-1)
     * @param damping 阻尼系数 / Damping factor (0-1)
     * @param wetLevel 湿声级别 / Wet level (0-1)
     * @return 添加混响的音频 / Audio with reverb
     */
    public static AudioData addReverb(AudioData audioData, double roomSize, double damping, double wetLevel) 
            throws AudioProcessingException {
        
        IVector<Double> samples = audioData.getSamples();
        IVector<Double> reverbSamples = createReverbEffect(samples, roomSize, damping, wetLevel, audioData.getSampleRate());
        
        return new AudioData(reverbSamples, audioData.getSampleRate(), audioData.getChannels(), 
                           audioData.getBitDepth(), audioData.getFormat());
    }
    
    // ======== 私有实现方法 / Private Implementation Methods ========
    
    private static AudioData pitchShiftFrequencyDomain(IVector<Double> samples, double pitchRatio, 
            double sampleRate, AudioData originalAudio) {
        
        int windowSize = 1024;
        int hopSize = windowSize / 4;
        int numFrames = (samples.length() - windowSize) / hopSize + 1;
        
        IVector<Double> output = Linalg.zeros((int)(samples.length() / pitchRatio));
        
        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopSize;
            int end = Math.min(start + windowSize, samples.length());
            
            IVector<Double> window = samples.slice(start, end);
            Complex[] fft = RereFFT.fft(convertToComplex(window));
            
            // 频域音调转换 / Frequency domain pitch shifting
            Complex[] shiftedFFT = new Complex[fft.length];
            for (int i = 0; i < fft.length; i++) {
                int sourceIndex = (int)(i / pitchRatio);
                shiftedFFT[i] = sourceIndex < fft.length ? fft[sourceIndex] : new Complex(0, 0);
            }
            
            Complex[] ifft = RereFFT.ifft(shiftedFFT);
            
            // 重叠相加到输出 / Overlap-add to output
            int outputStart = (int)(frame * hopSize / pitchRatio);
            for (int i = 0; i < ifft.length && outputStart + i < output.length(); i++) {
                output.set(outputStart + i, output.get(outputStart + i) + ifft[i].getReal());
            }
        }
        
        return new AudioData(output, sampleRate, originalAudio.getChannels(), 
                           originalAudio.getBitDepth(), originalAudio.getFormat());
    }
    
    private static AudioData timeStretchPSOLA(IVector<Double> samples, double stretchFactor, AudioData originalAudio) {
        int outputLength = (int)(samples.length() * stretchFactor);
        IVector<Double> output = Linalg.zeros(outputLength);
        
        int windowSize = 1024;
        double inputPos = 0;
        
        for (int outputPos = 0; outputPos < outputLength - windowSize; outputPos += windowSize / 2) {
            int inputStart = (int)inputPos;
            if (inputStart + windowSize >= samples.length()) break;
            
            // 提取窗口 / Extract window
            IVector<Double> window = samples.slice(inputStart, inputStart + windowSize);
            
            // 应用窗函数 / Apply window function
            for (int i = 0; i < windowSize; i++) {
                double hanningValue = 0.5 * (1 - Math.cos(2 * Math.PI * i / (windowSize - 1)));
                window.set(i, window.get(i) * hanningValue);
            }
            
            // 重叠相加 / Overlap-add
            for (int i = 0; i < windowSize && outputPos + i < outputLength; i++) {
                output.set(outputPos + i, output.get(outputPos + i) + window.get(i));
            }
            
            inputPos += windowSize / (2 * stretchFactor);
        }
        
        return new AudioData(output, originalAudio.getSampleRate(), originalAudio.getChannels(), 
                           originalAudio.getBitDepth(), originalAudio.getFormat());
    }
    
    private static IVector<Double> createReverbEffect(IVector<Double> samples, double roomSize, 
            double damping, double wetLevel, double sampleRate) {
        
        // 简化的混响实现 / Simplified reverb implementation
        int[] delays = {(int)(0.03 * sampleRate), (int)(0.06 * sampleRate), (int)(0.12 * sampleRate)};
        double[] gains = {0.3, 0.2, 0.1};
        
        IVector<Double> output = samples.copy();
        
        for (int d = 0; d < delays.length; d++) {
            int delay = (int)(delays[d] * roomSize);
            double gain = gains[d] * (1.0 - damping);
            
            for (int i = delay; i < samples.length(); i++) {
                double delayed = samples.get(i - delay) * gain * wetLevel;
                output.set(i, output.get(i) + delayed);
            }
        }
        
        return output;
    }
    
    private static Complex[] convertToComplex(IVector<Double> samples) {
        Complex[] complex = new Complex[samples.length()];
        for (int i = 0; i < samples.length(); i++) {
            complex[i] = new Complex(samples.get(i), 0);
        }
        return complex;
    }
}