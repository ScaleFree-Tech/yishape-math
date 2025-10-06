package com.yishape.lab.music.generation;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.*;

/**
 * 和弦生成器 / Chord Generator
 * <p>
 * 用于生成各种类型的和弦音频数据，支持多种和弦类型、转位和音色。
 * Used to generate various types of chord audio data, supporting multiple chord types, inversions, and timbres.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ChordGenerator {
    
    /**
     * 和弦类型枚举 / Chord Type Enumeration
     */
    public enum ChordType {
        MAJOR("大三和弦", new int[]{0, 4, 7}),
        MINOR("小三和弦", new int[]{0, 3, 7}),
        DIMINISHED("减三和弦", new int[]{0, 3, 6}),
        AUGMENTED("增三和弦", new int[]{0, 4, 8}),
        MAJOR_SEVENTH("大七和弦", new int[]{0, 4, 7, 11}),
        MINOR_SEVENTH("小七和弦", new int[]{0, 3, 7, 10}),
        DOMINANT_SEVENTH("属七和弦", new int[]{0, 4, 7, 10}),
        DIMINISHED_SEVENTH("减七和弦", new int[]{0, 3, 6, 9}),
        HALF_DIMINISHED_SEVENTH("半减七和弦", new int[]{0, 3, 6, 10}),
        MAJOR_NINTH("大九和弦", new int[]{0, 4, 7, 11, 14}),
        MINOR_NINTH("小九和弦", new int[]{0, 3, 7, 10, 14}),
        SUSPENDED_SECOND("挂二和弦", new int[]{0, 2, 7}),
        SUSPENDED_FOURTH("挂四和弦", new int[]{0, 5, 7});
        
        private final String chineseName;
        private final int[] intervals;
        
        ChordType(String chineseName, int[] intervals) {
            this.chineseName = chineseName;
            this.intervals = intervals;
        }
        
        public String getChineseName() { return chineseName; }
        public int[] getIntervals() { return intervals.clone(); }
    }
    
    /**
     * 波形类型枚举 / Waveform Type Enumeration
     */
    public enum WaveformType {
        SINE("正弦波"),
        SQUARE("方波"),
        TRIANGLE("三角波"),
        SAWTOOTH("锯齿波");
        
        private final String chineseName;
        
        WaveformType(String chineseName) {
            this.chineseName = chineseName;
        }
        
        public String getChineseName() { return chineseName; }
    }
    
    private static final double[] NOTE_FREQUENCIES = {
        261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88
    };
    
    private WaveformType waveformType = WaveformType.SINE;
    private double amplitude = 0.5;
    private double attackTime = 0.1;
    private double decayTime = 0.1;
    private double sustainLevel = 0.7;
    private double releaseTime = 0.2;
    
    /**
     * 默认构造函数 / Default Constructor
     */
    public ChordGenerator() {
    }
    
    /**
     * 生成和弦 / Generate Chord
     *
     * @param rootNote 根音 / Root note (0-11, C=0, C#=1, D=2, ...)
     * @param chordType 和弦类型 / Chord type
     * @param octave 八度 / Octave (0-8)
     * @param duration 持续时间（秒） / Duration in seconds
     * @param sampleRate 采样率 / Sample rate
     * @return 生成的和弦音频数据 / Generated chord audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public AudioData generateChord(int rootNote, ChordType chordType, int octave, double duration, double sampleRate) 
            throws AudioProcessingException {
        return generateChord(rootNote, chordType, 0, octave, duration, sampleRate);
    }
    
    /**
     * 生成带转位的和弦 / Generate Chord with Inversion
     *
     * @param rootNote 根音 / Root note (0-11)
     * @param chordType 和弦类型 / Chord type
     * @param inversion 转位 / Inversion (0=原位, 1=第一转位, 2=第二转位, ...)
     * @param octave 八度 / Octave
     * @param duration 持续时间（秒） / Duration in seconds
     * @param sampleRate 采样率 / Sample rate
     * @return 生成的和弦音频数据 / Generated chord audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public AudioData generateChord(int rootNote, ChordType chordType, int inversion, int octave, double duration, double sampleRate) 
            throws AudioProcessingException {
        validateParameters(rootNote, octave, duration, sampleRate);
        
        int[] intervals = chordType.getIntervals();
        if (inversion > 0 && inversion < intervals.length) {
            intervals = applyInversion(intervals, inversion);
        }
        
        int numSamples = (int) (duration * sampleRate);
        IVector<Double> samples = Linalg.vector(new double[numSamples]);
        
        // 为每个音符生成波形并混合
        for (int interval : intervals) {
            int noteNumber = (rootNote + interval) % 12;
            double frequency = calculateFrequency(noteNumber, octave + (rootNote + interval) / 12);
            IVector<Double> noteSamples = generateWaveform(frequency, duration, sampleRate);
            
            // 混合到主样本中
            for (int i = 0; i < numSamples; i++) {
                double currentValue = samples.get(i);
                double noteValue = noteSamples.get(i) * amplitude / intervals.length;
                samples.set(i, currentValue + noteValue);
            }
        }
        
        // 应用包络
        applyEnvelope(samples, duration, sampleRate);
        
        // Create AudioData with proper constructor
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成琶音 / Generate Arpeggio
     *
     * @param rootNote 根音 / Root note
     * @param chordType 和弦类型 / Chord type
     * @param octave 八度 / Octave
     * @param totalDuration 总持续时间 / Total duration
     * @param sampleRate 采样率 / Sample rate
     * @param ascending 是否上行 / Whether ascending
     * @return 生成的琶音音频数据 / Generated arpeggio audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public AudioData generateArpeggio(int rootNote, ChordType chordType, int octave, double totalDuration, 
                                    double sampleRate, boolean ascending) throws AudioProcessingException {
        validateParameters(rootNote, octave, totalDuration, sampleRate);
        
        int[] intervals = chordType.getIntervals();
        if (!ascending) {
            // 反转数组用于下行琶音
            for (int i = 0; i < intervals.length / 2; i++) {
                int temp = intervals[i];
                intervals[i] = intervals[intervals.length - 1 - i];
                intervals[intervals.length - 1 - i] = temp;
            }
        }
        
        double noteDuration = totalDuration / intervals.length;
        int totalSamples = (int) (totalDuration * sampleRate);
        IVector<Double> samples = Linalg.vector(new double[totalSamples]);
        
        for (int i = 0; i < intervals.length; i++) {
            int noteNumber = (rootNote + intervals[i]) % 12;
            double frequency = calculateFrequency(noteNumber, octave + (rootNote + intervals[i]) / 12);
            IVector<Double> noteSamples = generateWaveform(frequency, noteDuration, sampleRate);
            
            int startIndex = (int) (i * noteDuration * sampleRate);
            int endIndex = Math.min(startIndex + noteSamples.size(), totalSamples);
            
            for (int j = startIndex; j < endIndex; j++) {
                samples.set(j, noteSamples.get(j - startIndex) * amplitude);
            }
        }
        
        // Create AudioData with proper constructor
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成和弦进行 / Generate Chord Progression
     *
     * @param chordProgression 和弦进行 / Chord progression
     * @param key 调性 / Key (0-11)
     * @param octave 八度 / Octave
     * @param chordDuration 每个和弦的持续时间 / Duration of each chord
     * @param sampleRate 采样率 / Sample rate
     * @return 生成的和弦进行音频数据 / Generated chord progression audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public AudioData generateChordProgression(ChordType[] chordProgression, int key, int octave, 
                                            double chordDuration, double sampleRate) throws AudioProcessingException {
        if (chordProgression == null || chordProgression.length == 0) {
            throw new AudioProcessingException("和弦进行不能为空 / Chord progression cannot be empty");
        }
        
        double totalDuration = chordProgression.length * chordDuration;
        int totalSamples = (int) (totalDuration * sampleRate);
        IVector<Double> samples = Linalg.vector(new double[totalSamples]);
        
        for (int i = 0; i < chordProgression.length; i++) {
            AudioData chordData = generateChord(key, chordProgression[i], octave, chordDuration, sampleRate);
            IVector<Double> chordSamples = chordData.getSamples();
            
            int startIndex = (int) (i * chordDuration * sampleRate);
            int endIndex = Math.min(startIndex + chordSamples.size(), totalSamples);
            
            for (int j = startIndex; j < endIndex; j++) {
                samples.set(j, chordSamples.get(j - startIndex));
            }
        }
        
        // Create AudioData with proper constructor
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 应用转位 / Apply Inversion
     */
    private int[] applyInversion(int[] intervals, int inversion) {
        int[] inverted = new int[intervals.length];
        for (int i = 0; i < intervals.length; i++) {
            int newIndex = (i + inversion) % intervals.length;
            inverted[i] = intervals[newIndex];
            if (newIndex < inversion) {
                inverted[i] += 12; // 提高八度
            }
        }
        return inverted;
    }
    
    /**
     * 计算频率 / Calculate Frequency
     */
    private double calculateFrequency(int noteNumber, int octave) {
        return NOTE_FREQUENCIES[noteNumber] * Math.pow(2, octave - 4);
    }
    
    /**
     * 生成波形 / Generate Waveform
     */
    private IVector<Double> generateWaveform(double frequency, double duration, double sampleRate) {
        int numSamples = (int) (duration * sampleRate);
        IVector<Double> samples = Linalg.vector(new double[numSamples]);
        
        for (int i = 0; i < numSamples; i++) {
            double t = i / sampleRate;
            double value = 0.0;
            
            switch (waveformType) {
                case SINE:
                    value = Math.sin(2 * Math.PI * frequency * t);
                    break;
                case SQUARE:
                    value = Math.sin(2 * Math.PI * frequency * t) >= 0 ? 1.0 : -1.0;
                    break;
                case TRIANGLE:
                    double phase = (frequency * t) % 1.0;
                    value = phase < 0.5 ? 4 * phase - 1 : 3 - 4 * phase;
                    break;
                case SAWTOOTH:
                    value = 2 * ((frequency * t) % 1.0) - 1;
                    break;
            }
            
            samples.set(i, value);
        }
        
        return samples;
    }
    
    /**
     * 应用包络 / Apply Envelope
     */
    private void applyEnvelope(IVector<Double> samples, double duration, double sampleRate) {
        int numSamples = samples.size();
        int attackSamples = (int) (attackTime * sampleRate);
        int decaySamples = (int) (decayTime * sampleRate);
        int releaseSamples = (int) (releaseTime * sampleRate);
        int sustainSamples = numSamples - attackSamples - decaySamples - releaseSamples;
        
        for (int i = 0; i < numSamples; i++) {
            double envelope = 1.0;
            
            if (i < attackSamples) {
                // 起音阶段
                envelope = (double) i / attackSamples;
            } else if (i < attackSamples + decaySamples) {
                // 衰减阶段
                double decayProgress = (double) (i - attackSamples) / decaySamples;
                envelope = 1.0 - decayProgress * (1.0 - sustainLevel);
            } else if (i < attackSamples + decaySamples + sustainSamples) {
                // 延音阶段
                envelope = sustainLevel;
            } else {
                // 释音阶段
                double releaseProgress = (double) (i - attackSamples - decaySamples - sustainSamples) / releaseSamples;
                envelope = sustainLevel * (1.0 - releaseProgress);
            }
            
            samples.set(i, samples.get(i) * envelope);
        }
    }
    
    /**
     * 验证参数 / Validate Parameters
     */
    private void validateParameters(int rootNote, int octave, double duration, double sampleRate) 
            throws AudioProcessingException {
        if (rootNote < 0 || rootNote > 11) {
            throw new AudioProcessingException("根音必须在0-11之间 / Root note must be between 0-11");
        }
        if (octave < 0 || octave > 8) {
            throw new AudioProcessingException("八度必须在0-8之间 / Octave must be between 0-8");
        }
        if (duration <= 0) {
            throw new AudioProcessingException("持续时间必须大于0 / Duration must be greater than 0");
        }
        if (sampleRate <= 0) {
            throw new AudioProcessingException("采样率必须大于0 / Sample rate must be greater than 0");
        }
    }
    
    // Getter和Setter方法 / Getter and Setter Methods
    
    public WaveformType getWaveformType() { return waveformType; }
    public void setWaveformType(WaveformType waveformType) { this.waveformType = waveformType; }
    
    public double getAmplitude() { return amplitude; }
    public void setAmplitude(double amplitude) { this.amplitude = Math.max(0.0, Math.min(1.0, amplitude)); }
    
    public double getAttackTime() { return attackTime; }
    public void setAttackTime(double attackTime) { this.attackTime = Math.max(0.0, attackTime); }
    
    public double getDecayTime() { return decayTime; }
    public void setDecayTime(double decayTime) { this.decayTime = Math.max(0.0, decayTime); }
    
    public double getSustainLevel() { return sustainLevel; }
    public void setSustainLevel(double sustainLevel) { this.sustainLevel = Math.max(0.0, Math.min(1.0, sustainLevel)); }
    
    public double getReleaseTime() { return releaseTime; }
    public void setReleaseTime(double releaseTime) { this.releaseTime = Math.max(0.0, releaseTime); }
    
    /**
     * 获取支持的和弦类型 / Get Supported Chord Types
     *
     * @return 支持的和弦类型数组 / Array of supported chord types
     */
    public static ChordType[] getSupportedChordTypes() {
        return ChordType.values();
    }
    
    /**
     * 获取和弦类型的中文名称 / Get Chinese Name of Chord Type
     *
     * @param chordType 和弦类型 / Chord type
     * @return 中文名称 / Chinese name
     */
    public static String getChordTypeName(ChordType chordType) {
        return chordType.getChineseName();
    }
}