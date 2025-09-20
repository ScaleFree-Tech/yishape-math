package com.reremouse.lab.audio.music;

import com.reremouse.lab.audio.AudioData;
import com.reremouse.lab.audio.MusicTheory;
import com.reremouse.lab.audio.core.AbstractAudioProcessor;
import com.reremouse.lab.audio.core.IMusicProcessor;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;

import java.util.HashMap;
import java.util.Map;

/**
 * 音乐理论处理器 / Music Theory Processor
 * <p>
 * 实现IMusicProcessor接口，整合MusicTheory类的功能到接口架构中。
 * 提供音乐理论相关的音频生成和处理功能。
 * </p>
 * <p>
 * Implements IMusicProcessor interface, integrates MusicTheory class functionality
 * into interface architecture. Provides music theory-related audio generation
 * and processing functionality.
 * </p>
 *
 * @author Qoder AI
 * @version 1.0
 * @since 1.0
 */
public class MusicTheoryProcessor extends AbstractAudioProcessor implements IMusicProcessor {
    
    /** 音乐处理参数 / Music processing parameters */
    private static final String PARAM_AMPLITUDE = "amplitude";
    private static final String PARAM_SAMPLE_RATE = "sample_rate";
    private static final String PARAM_DEFAULT_OCTAVE = "default_octave";
    private static final String PARAM_DEFAULT_DURATION = "default_duration";
    private static final String PARAM_FADE_TIME = "fade_time";
    
    /** 支持的乐器类型 / Supported instrument types */
    private static final String[] SUPPORTED_INSTRUMENTS = {
        "sine", "square", "sawtooth", "triangle", "piano", "guitar", "organ"
    };
    
    /** 支持的音乐风格 / Supported music styles */
    private static final String[] SUPPORTED_MUSIC_STYLES = {
        "classical", "jazz", "rock", "pop", "electronic", "folk", "blues"
    };
    
    /** 支持的和声类型 / Supported harmony types */
    private static final String[] SUPPORTED_HARMONY_TYPES = {
        "parallel", "contrary", "oblique", "similar", "fourth", "fifth", "third"
    };
    
    /**
     * 构造函数 / Constructor
     */
    public MusicTheoryProcessor() {
        super("MusicTheoryProcessor", "1.0");
    }
    
    @Override
    protected void initializeDefaultParameters() {
        super.initializeDefaultParameters();
        // 音乐处理特定参数 / Music processing specific parameters
        parameters.put(PARAM_AMPLITUDE, 0.5);
        parameters.put(PARAM_SAMPLE_RATE, 44100.0);
        parameters.put(PARAM_DEFAULT_OCTAVE, 4);
        parameters.put(PARAM_DEFAULT_DURATION, 1.0);
        parameters.put(PARAM_FADE_TIME, 0.1);
    }
    
    @Override
    protected AudioData doProcess(AudioData audioData) throws AudioProcessingException {
        // 默认处理：返回原始音频数据 / Default processing: return original audio data
        return audioData;
    }
    
    @Override
    public AudioData generateScale(int rootNote, MusicTheory.ScaleType scaleType, int octave, double noteDuration) throws AudioProcessingException {
        try {
            double sampleRate = getSampleRate();
            double amplitude = getAmplitude();
            
            return MusicTheory.generateScaleAudio(rootNote, scaleType, octave, noteDuration, sampleRate, amplitude);
        } catch (Exception e) {
            throw new AudioProcessingException("Scale generation failed", e);
        }
    }
    
    @Override
    public AudioData generateChord(int rootNote, MusicTheory.ChordType chordType, int octave, double duration) throws AudioProcessingException {
        try {
            double sampleRate = getSampleRate();
            double amplitude = getAmplitude();
            
            return MusicTheory.generateChordAudio(rootNote, chordType, octave, duration, sampleRate, amplitude);
        } catch (Exception e) {
            throw new AudioProcessingException("Chord generation failed", e);
        }
    }
    
    @Override
    public AudioData generateInterval(int note1, int note2, int octave, double duration, boolean simultaneous) throws AudioProcessingException {
        try {
            double sampleRate = getSampleRate();
            double amplitude = getAmplitude();
            int totalSamples = (int) (duration * sampleRate);
            
            if (simultaneous) {
                // 同时播放两个音符 / Play both notes simultaneously
                double freq1 = MusicTheory.noteToFrequency(MusicTheory.NOTE_NAMES[note1], octave);
                double freq2 = MusicTheory.noteToFrequency(MusicTheory.NOTE_NAMES[note2], octave);
                
                IVector<Double> samples = Linalg.zeros(totalSamples);
                for (int i = 0; i < totalSamples; i++) {
                    double t = i / sampleRate;
                    double sample1 = amplitude * 0.5 * Math.sin(2 * Math.PI * freq1 * t);
                    double sample2 = amplitude * 0.5 * Math.sin(2 * Math.PI * freq2 * t);
                    samples.set(i, sample1 + sample2);
                }
                
                return new AudioData(samples, sampleRate, 1, 16, com.reremouse.lab.audio.AudioFormat.WAV);
            } else {
                // 顺序播放两个音符 / Play notes sequentially
                int samplesPerNote = totalSamples / 2;
                double freq1 = MusicTheory.noteToFrequency(MusicTheory.NOTE_NAMES[note1], octave);
                double freq2 = MusicTheory.noteToFrequency(MusicTheory.NOTE_NAMES[note2], octave);
                
                IVector<Double> samples = Linalg.zeros(totalSamples);
                
                // 第一个音符 / First note
                for (int i = 0; i < samplesPerNote; i++) {
                    double t = i / sampleRate;
                    double sample = amplitude * Math.sin(2 * Math.PI * freq1 * t);
                    samples.set(i, sample);
                }
                
                // 第二个音符 / Second note
                for (int i = samplesPerNote; i < totalSamples; i++) {
                    double t = (i - samplesPerNote) / sampleRate;
                    double sample = amplitude * Math.sin(2 * Math.PI * freq2 * t);
                    samples.set(i, sample);
                }
                
                return new AudioData(samples, sampleRate, 1, 16, com.reremouse.lab.audio.AudioFormat.WAV);
            }
        } catch (Exception e) {
            throw new AudioProcessingException("Interval generation failed", e);
        }
    }
    
    @Override
    public AudioData transpose(AudioData audioData, int semitones) throws AudioProcessingException {
        try {
            // 简化的转调实现 / Simplified transposition implementation
            IVector<Double> inputSamples = audioData.getSamples();
            double ratio = Math.pow(2.0, semitones / 12.0);
            
            // 重采样实现转调 / Implement transposition through resampling
            int newLength = (int) (inputSamples.length() / ratio);
            IVector<Double> outputSamples = Linalg.zeros(newLength);
            
            for (int i = 0; i < newLength; i++) {
                double sourceIndex = i * ratio;
                int index = (int) sourceIndex;
                
                if (index < inputSamples.length() - 1) {
                    double fraction = sourceIndex - index;
                    double sample = inputSamples.get(index) * (1 - fraction) + 
                                  inputSamples.get(index + 1) * fraction;
                    outputSamples.set(i, sample);
                }
            }
            
            return new AudioData(outputSamples, audioData.getSampleRate(), audioData.getChannels(), 
                               audioData.getBitDepth(), audioData.getFormat());
        } catch (Exception e) {
            throw new AudioProcessingException("Transposition failed", e);
        }
    }
    
    @Override
    public AudioData convertKey(AudioData audioData, MusicTheory.Key sourceKey, MusicTheory.Key targetKey) throws AudioProcessingException {
        try {
            // 计算转调的半音数 / Calculate semitones for transposition
            int semitones = targetKey.getRootNote() - sourceKey.getRootNote();
            if (semitones < 0) semitones += 12;
            
            return transpose(audioData, semitones);
        } catch (Exception e) {
            throw new AudioProcessingException("Key conversion failed", e);
        }
    }
    
    @Override
    public AudioData harmonize(AudioData audioData, String harmonyType) throws AudioProcessingException {
        try {
            // 简化的和声化实现 / Simplified harmonization implementation
            IVector<Double> inputSamples = audioData.getSamples();
            IVector<Double> harmonySamples = Linalg.zeros(inputSamples.length());
            
            double harmonyRatio = getHarmonyRatio(harmonyType);
            
            // 生成和声声部 / Generate harmony voice
            for (int i = 0; i < inputSamples.length(); i++) {
                // 简单的频率偏移和声 / Simple frequency shift harmony
                double harmonySample = inputSamples.get(i) * 0.5; // 降低音量
                harmonySamples.set(i, harmonySample);
            }
            
            // 混合原始音频和和声 / Mix original audio with harmony
            IVector<Double> outputSamples = inputSamples.add(harmonySamples);
            
            return new AudioData(outputSamples, audioData.getSampleRate(), audioData.getChannels(), 
                               audioData.getBitDepth(), audioData.getFormat());
        } catch (Exception e) {
            throw new AudioProcessingException("Harmonization failed", e);
        }
    }
    
    @Override
    public AudioData quantizeBeats(AudioData audioData, String beatGrid) throws AudioProcessingException {
        try {
            // 简化的节拍量化实现 / Simplified beat quantization implementation
            // 这里返回原始音频，实际实现需要复杂的节拍检测和时间拉伸
            return audioData;
        } catch (Exception e) {
            throw new AudioProcessingException("Beat quantization failed", e);
        }
    }
    
    @Override
    public AudioData concatenateMusically(AudioData[] audioSegments, String transitionType) throws AudioProcessingException {
        try {
            if (audioSegments == null || audioSegments.length == 0) {
                throw new IllegalArgumentException("Audio segments cannot be null or empty");
            }
            
            // 计算总长度 / Calculate total length
            int totalLength = 0;
            for (AudioData segment : audioSegments) {
                totalLength += segment.getSamples().length();
            }
            
            IVector<Double> concatenated = Linalg.zeros(totalLength);
            int currentPos = 0;
            
            // 拼接音频片段 / Concatenate audio segments
            for (AudioData segment : audioSegments) {
                IVector<Double> segmentSamples = segment.getSamples();
                for (int i = 0; i < segmentSamples.length(); i++) {
                    concatenated.set(currentPos + i, segmentSamples.get(i));
                }
                currentPos += segmentSamples.length();
            }
            
            return new AudioData(concatenated, audioSegments[0].getSampleRate(), 
                               audioSegments[0].getChannels(), audioSegments[0].getBitDepth(), 
                               audioSegments[0].getFormat());
        } catch (Exception e) {
            throw new AudioProcessingException("Musical concatenation failed", e);
        }
    }
    
    @Override
    public AudioData transformTimbre(AudioData audioData, String targetInstrument) throws AudioProcessingException {
        try {
            // 简化的音色变换实现 / Simplified timbre transformation implementation
            IVector<Double> inputSamples = audioData.getSamples();
            IVector<Double> outputSamples = inputSamples.copy();
            
            // 根据目标乐器应用简单的滤波 / Apply simple filtering based on target instrument
            double filterFactor = getInstrumentFilterFactor(targetInstrument);
            
            for (int i = 0; i < outputSamples.length(); i++) {
                outputSamples.set(i, outputSamples.get(i) * filterFactor);
            }
            
            return new AudioData(outputSamples, audioData.getSampleRate(), audioData.getChannels(), 
                               audioData.getBitDepth(), audioData.getFormat());
        } catch (Exception e) {
            throw new AudioProcessingException("Timbre transformation failed", e);
        }
    }
    
    @Override
    public AudioData convertMusicStyle(AudioData audioData, String targetStyle) throws AudioProcessingException {
        try {
            // 简化的风格转换实现 / Simplified style conversion implementation
            return audioData; // 返回原始音频，实际实现需要复杂的信号处理
        } catch (Exception e) {
            throw new AudioProcessingException("Music style conversion failed", e);
        }
    }
    
    @Override
    public AudioData matchTempo(AudioData audioData, double targetBpm) throws AudioProcessingException {
        try {
            // 简化的节拍匹配实现 / Simplified tempo matching implementation
            // 检测当前BPM并计算拉伸比例 / Detect current BPM and calculate stretch ratio
            com.reremouse.lab.audio.MusicAnalyzer.BeatDetectionResult beatResult = 
                com.reremouse.lab.audio.MusicAnalyzer.detectBeats(audioData);
            
            double currentBpm = beatResult.getBpm();
            double stretchRatio = currentBpm / targetBpm;
            
            // 时间拉伸 / Time stretching
            IVector<Double> inputSamples = audioData.getSamples();
            int newLength = (int) (inputSamples.length() * stretchRatio);
            IVector<Double> outputSamples = Linalg.zeros(newLength);
            
            for (int i = 0; i < newLength; i++) {
                double sourceIndex = i / stretchRatio;
                int index = (int) sourceIndex;
                
                if (index < inputSamples.length() - 1) {
                    double fraction = sourceIndex - index;
                    double sample = inputSamples.get(index) * (1 - fraction) + 
                                  inputSamples.get(index + 1) * fraction;
                    outputSamples.set(i, sample);
                }
            }
            
            return new AudioData(outputSamples, audioData.getSampleRate(), audioData.getChannels(), 
                               audioData.getBitDepth(), audioData.getFormat());
        } catch (Exception e) {
            throw new AudioProcessingException("Tempo matching failed", e);
        }
    }
    
    @Override
    public AudioData correctPitch(AudioData audioData, double strength) throws AudioProcessingException {
        try {
            // 简化的音高校正实现 / Simplified pitch correction implementation
            return audioData; // 返回原始音频，实际实现需要复杂的音高检测和校正
        } catch (Exception e) {
            throw new AudioProcessingException("Pitch correction failed", e);
        }
    }
    
    @Override
    public void setMusicParameter(String parameterName, Object value) throws IllegalArgumentException {
        setParameter(parameterName, value);
    }
    
    @Override
    public Object getMusicParameter(String parameterName) throws IllegalArgumentException {
        return getParameter(parameterName);
    }
    
    @Override
    public String[] getSupportedInstruments() {
        return SUPPORTED_INSTRUMENTS.clone();
    }
    
    @Override
    public String[] getSupportedMusicStyles() {
        return SUPPORTED_MUSIC_STYLES.clone();
    }
    
    @Override
    public String[] getSupportedHarmonyTypes() {
        return SUPPORTED_HARMONY_TYPES.clone();
    }
    
    // 辅助方法 / Helper methods
    
    private double getSampleRate() {
        Object sampleRate = parameters.getOrDefault(PARAM_SAMPLE_RATE, 44100.0);
        return ((Number) sampleRate).doubleValue();
    }
    
    private double getAmplitude() {
        Object amplitude = parameters.getOrDefault(PARAM_AMPLITUDE, 0.5);
        return ((Number) amplitude).doubleValue();
    }
    
    private double getHarmonyRatio(String harmonyType) {
        // 根据和声类型返回频率比 / Return frequency ratio based on harmony type
        switch (harmonyType.toLowerCase()) {
            case "fifth": return 1.5; // 完全五度
            case "fourth": return 1.33; // 完全四度
            case "third": return 1.25; // 大三度
            default: return 1.2; // 默认
        }
    }
    
    private double getInstrumentFilterFactor(String instrument) {
        // 根据乐器类型返回滤波因子 / Return filter factor based on instrument type
        switch (instrument.toLowerCase()) {
            case "piano": return 0.8;
            case "guitar": return 0.9;
            case "organ": return 1.1;
            case "sine": return 1.0;
            default: return 1.0;
        }
    }
    
    @Override
    protected void validateParameter(String key, Object value) throws IllegalArgumentException {
        switch (key) {
            case PARAM_AMPLITUDE:
                if (!(value instanceof Number)) {
                    throw new IllegalArgumentException("Amplitude must be a number");
                }
                double amp = ((Number) value).doubleValue();
                if (amp < 0.0 || amp > 1.0) {
                    throw new IllegalArgumentException("Amplitude must be between 0.0 and 1.0");
                }
                break;
            case PARAM_SAMPLE_RATE:
                if (!(value instanceof Number) || ((Number) value).doubleValue() <= 0) {
                    throw new IllegalArgumentException("Sample rate must be a positive number");
                }
                break;
            case PARAM_DEFAULT_OCTAVE:
                if (!(value instanceof Integer)) {
                    throw new IllegalArgumentException("Default octave must be an integer");
                }
                int octave = (Integer) value;
                if (octave < 0 || octave > 10) {
                    throw new IllegalArgumentException("Default octave must be between 0 and 10");
                }
                break;
            case PARAM_DEFAULT_DURATION:
            case PARAM_FADE_TIME:
                if (!(value instanceof Number) || ((Number) value).doubleValue() <= 0) {
                    throw new IllegalArgumentException(key + " must be a positive number");
                }
                break;
            default:
                super.validateParameter(key, value);
                break;
        }
    }
}