package com.reremouse.lab.music.generation;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioFormat;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;

import java.util.*;

/**
 * 音程生成器 / Interval Generator
 * <p>
 * 用于生成各种音程的音频数据，支持和声音程、旋律音程以及复合音程。
 * Used to generate audio data for various intervals, supporting harmonic intervals, melodic intervals, and compound intervals.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class IntervalGenerator {
    
    /**
     * 音程类型枚举 / Interval Type Enumeration
     */
    public enum IntervalType {
        UNISON("同度", 0),
        MINOR_SECOND("小二度", 1),
        MAJOR_SECOND("大二度", 2),
        MINOR_THIRD("小三度", 3),
        MAJOR_THIRD("大三度", 4),
        PERFECT_FOURTH("纯四度", 5),
        TRITONE("三全音", 6),
        PERFECT_FIFTH("纯五度", 7),
        MINOR_SIXTH("小六度", 8),
        MAJOR_SIXTH("大六度", 9),
        MINOR_SEVENTH("小七度", 10),
        MAJOR_SEVENTH("大七度", 11),
        OCTAVE("八度", 12),
        MINOR_NINTH("小九度", 13),
        MAJOR_NINTH("大九度", 14),
        MINOR_TENTH("小十度", 15),
        MAJOR_TENTH("大十度", 16),
        PERFECT_ELEVENTH("纯十一度", 17),
        TRITONE_ELEVENTH("增十一度", 18),
        PERFECT_TWELFTH("纯十二度", 19),
        MINOR_THIRTEENTH("小十三度", 20),
        MAJOR_THIRTEENTH("大十三度", 21),
        MINOR_FOURTEENTH("小十四度", 22),
        MAJOR_FOURTEENTH("大十四度", 23),
        DOUBLE_OCTAVE("双八度", 24);
        
        private final String chineseName;
        private final int semitones;
        
        IntervalType(String chineseName, int semitones) {
            this.chineseName = chineseName;
            this.semitones = semitones;
        }
        
        public String getChineseName() { return chineseName; }
        public int getSemitones() { return semitones; }
    }
    
    /**
     * 音程播放模式枚举 / Interval Play Mode Enumeration
     */
    public enum PlayMode {
        HARMONIC("和声音程", "同时播放两个音"),
        MELODIC_ASCENDING("旋律音程上行", "先低音后高音"),
        MELODIC_DESCENDING("旋律音程下行", "先高音后低音"),
        MELODIC_BOTH("旋律音程双向", "低音-高音-低音");
        
        private final String chineseName;
        private final String description;
        
        PlayMode(String chineseName, String description) {
            this.chineseName = chineseName;
            this.description = description;
        }
        
        public String getChineseName() { return chineseName; }
        public String getDescription() { return description; }
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
    private double attackTime = 0.05;
    private double decayTime = 0.1;
    private double sustainLevel = 0.8;
    private double releaseTime = 0.15;
    
    /**
     * 默认构造函数 / Default Constructor
     */
    public IntervalGenerator() {
    }
    
    /**
     * 生成音程 / Generate Interval
     *
     * @param rootNote 根音 / Root note (0-11, C=0, C#=1, D=2, ...)
     * @param intervalType 音程类型 / Interval type
     * @param octave 八度 / Octave (0-8)
     * @param duration 持续时间（秒） / Duration in seconds
     * @param sampleRate 采样率 / Sample rate
     * @param playMode 播放模式 / Play mode
     * @return 生成的音程音频数据 / Generated interval audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public AudioData generateInterval(int rootNote, IntervalType intervalType, int octave, 
                                    double duration, double sampleRate, PlayMode playMode) 
            throws AudioProcessingException {
        validateParameters(rootNote, octave, duration, sampleRate);
        
        int upperNote = (rootNote + intervalType.getSemitones()) % 12;
        int upperOctave = octave + (rootNote + intervalType.getSemitones()) / 12;
        
        switch (playMode) {
            case HARMONIC:
                return generateHarmonicInterval(rootNote, upperNote, octave, upperOctave, duration, sampleRate);
            case MELODIC_ASCENDING:
                return generateMelodicInterval(rootNote, upperNote, octave, upperOctave, duration, sampleRate, true);
            case MELODIC_DESCENDING:
                return generateMelodicInterval(upperNote, rootNote, upperOctave, octave, duration, sampleRate, true);
            case MELODIC_BOTH:
                return generateMelodicBothInterval(rootNote, upperNote, octave, upperOctave, duration, sampleRate);
            default:
                throw new AudioProcessingException("不支持的播放模式 / Unsupported play mode: " + playMode);
        }
    }
    
    /**
     * 生成和声音程 / Generate Harmonic Interval
     */
    private AudioData generateHarmonicInterval(int lowerNote, int upperNote, int lowerOctave, int upperOctave,
                                             double duration, double sampleRate) throws AudioProcessingException {
        int numSamples = (int) (duration * sampleRate);
        IVector<Double> samples = Linalg.zeros(numSamples);
        
        double lowerFreq = calculateFrequency(lowerNote, lowerOctave);
        double upperFreq = calculateFrequency(upperNote, upperOctave);
        
        IVector<Double> lowerSamples = generateWaveform(lowerFreq, duration, sampleRate);
        IVector<Double> upperSamples = generateWaveform(upperFreq, duration, sampleRate);
        
        // 混合两个音符
        for (int i = 0; i < numSamples; i++) {
            double mixedValue = (lowerSamples.get(i) + upperSamples.get(i)) * amplitude * 0.5;
            samples.set(i, mixedValue);
        }
        
        // 应用包络
        applyEnvelope(samples, duration, sampleRate);
        
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成旋律音程 / Generate Melodic Interval
     */
    private AudioData generateMelodicInterval(int firstNote, int secondNote, int firstOctave, int secondOctave,
                                            double duration, double sampleRate, boolean ascending) 
            throws AudioProcessingException {
        double noteDuration = duration / 2.0;
        int samplesPerNote = (int) (noteDuration * sampleRate);
        int totalSamples = samplesPerNote * 2;
        
        IVector<Double> samples = Linalg.zeros(totalSamples);
        
        double firstFreq = calculateFrequency(firstNote, firstOctave);
        double secondFreq = calculateFrequency(secondNote, secondOctave);
        
        IVector<Double> firstNoteSamples = generateWaveform(firstFreq, noteDuration, sampleRate);
        IVector<Double> secondNoteSamples = generateWaveform(secondFreq, noteDuration, sampleRate);
        
        // 应用包络到每个音符
        applyEnvelope(firstNoteSamples, noteDuration, sampleRate);
        applyEnvelope(secondNoteSamples, noteDuration, sampleRate);
        
        // 组合两个音符
        for (int i = 0; i < samplesPerNote; i++) {
            samples.set(i, firstNoteSamples.get(i) * amplitude);
        }
        for (int i = 0; i < samplesPerNote && i + samplesPerNote < totalSamples; i++) {
            samples.set(i + samplesPerNote, secondNoteSamples.get(i) * amplitude);
        }
        
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成双向旋律音程 / Generate Melodic Both Interval
     */
    private AudioData generateMelodicBothInterval(int lowerNote, int upperNote, int lowerOctave, int upperOctave,
                                                double duration, double sampleRate) throws AudioProcessingException {
        double noteDuration = duration / 3.0;
        int samplesPerNote = (int) (noteDuration * sampleRate);
        int totalSamples = samplesPerNote * 3;
        
        IVector<Double> samples = Linalg.zeros(totalSamples);
        
        double lowerFreq = calculateFrequency(lowerNote, lowerOctave);
        double upperFreq = calculateFrequency(upperNote, upperOctave);
        
        IVector<Double> lowerSamples1 = generateWaveform(lowerFreq, noteDuration, sampleRate);
        IVector<Double> upperSamples = generateWaveform(upperFreq, noteDuration, sampleRate);
        IVector<Double> lowerSamples2 = generateWaveform(lowerFreq, noteDuration, sampleRate);
        
        // 应用包络
        applyEnvelope(lowerSamples1, noteDuration, sampleRate);
        applyEnvelope(upperSamples, noteDuration, sampleRate);
        applyEnvelope(lowerSamples2, noteDuration, sampleRate);
        
        // 组合三个音符：低音-高音-低音
        for (int i = 0; i < samplesPerNote; i++) {
            samples.set(i, lowerSamples1.get(i) * amplitude);
        }
        for (int i = 0; i < samplesPerNote && i + samplesPerNote < totalSamples; i++) {
            samples.set(i + samplesPerNote, upperSamples.get(i) * amplitude);
        }
        for (int i = 0; i < samplesPerNote && i + 2 * samplesPerNote < totalSamples; i++) {
            samples.set(i + 2 * samplesPerNote, lowerSamples2.get(i) * amplitude);
        }
        
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成音程序列 / Generate Interval Sequence
     *
     * @param rootNote 根音 / Root note
     * @param intervals 音程序列 / Interval sequence
     * @param octave 八度 / Octave
     * @param noteDuration 每个音程的持续时间 / Duration of each interval
     * @param sampleRate 采样率 / Sample rate
     * @param playMode 播放模式 / Play mode
     * @return 生成的音程序列音频数据 / Generated interval sequence audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public AudioData generateIntervalSequence(int rootNote, IntervalType[] intervals, int octave,
                                             double noteDuration, double sampleRate, PlayMode playMode) 
            throws AudioProcessingException {
        if (intervals == null || intervals.length == 0) {
            throw new AudioProcessingException("音程序列不能为空 / Interval sequence cannot be empty");
        }
        
        List<AudioData> intervalAudios = new ArrayList<>();
        
        for (IntervalType interval : intervals) {
            AudioData intervalAudio = generateInterval(rootNote, interval, octave, noteDuration, sampleRate, playMode);
            intervalAudios.add(intervalAudio);
        }
        
        return concatenateAudioData(intervalAudios, sampleRate);
    }
    
    /**
     * 生成音程练习 / Generate Interval Exercise
     *
     * @param intervalTypes 要练习的音程类型 / Interval types to practice
     * @param rootNote 根音 / Root note
     * @param octave 八度 / Octave
     * @param intervalDuration 每个音程的持续时间 / Duration of each interval
     * @param pauseDuration 音程间的停顿时间 / Pause duration between intervals
     * @param sampleRate 采样率 / Sample rate
     * @param playMode 播放模式 / Play mode
     * @return 生成的音程练习音频数据 / Generated interval exercise audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public AudioData generateIntervalExercise(IntervalType[] intervalTypes, int rootNote, int octave,
                                             double intervalDuration, double pauseDuration, double sampleRate,
                                             PlayMode playMode) throws AudioProcessingException {
        if (intervalTypes == null || intervalTypes.length == 0) {
            throw new AudioProcessingException("音程类型数组不能为空 / Interval types array cannot be empty");
        }
        
        List<AudioData> exerciseAudios = new ArrayList<>();
        
        for (int i = 0; i < intervalTypes.length; i++) {
            // 生成音程
            AudioData intervalAudio = generateInterval(rootNote, intervalTypes[i], octave, 
                                                     intervalDuration, sampleRate, playMode);
            exerciseAudios.add(intervalAudio);
            
            // 添加停顿（除了最后一个音程）
            if (i < intervalTypes.length - 1 && pauseDuration > 0) {
                AudioData pauseAudio = generateSilence(pauseDuration, sampleRate);
                exerciseAudios.add(pauseAudio);
            }
        }
        
        return concatenateAudioData(exerciseAudios, sampleRate);
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
        IVector<Double> samples = Linalg.zeros(numSamples);
        
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
        int sustainSamples = Math.max(0, numSamples - attackSamples - decaySamples - releaseSamples);
        
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
     * 连接音频数据 / Concatenate Audio Data
     */
    private AudioData concatenateAudioData(List<AudioData> audioList, double sampleRate) throws AudioProcessingException {
        if (audioList == null || audioList.isEmpty()) {
            return new AudioData(Linalg.zeros(0), sampleRate, 1, 16, AudioFormat.WAV);
        }
        
        // 计算总样本数
        int totalSamples = 0;
        for (AudioData audio : audioList) {
            totalSamples += audio.getSamples().size();
        }
        
        // 创建结果向量
        IVector<Double> concatenatedSamples = Linalg.zeros(totalSamples);
        
        // 复制每个音频数据
        int currentIndex = 0;
        for (AudioData audio : audioList) {
            IVector<Double> samples = audio.getSamples();
            for (int i = 0; i < samples.size(); i++) {
                concatenatedSamples.set(currentIndex + i, samples.get(i));
            }
            currentIndex += samples.size();
        }
        
        return new AudioData(concatenatedSamples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成静音 / Generate Silence
     */
    private AudioData generateSilence(double duration, double sampleRate) {
        int numSamples = (int) (duration * sampleRate);
        IVector<Double> samples = Linalg.zeros(numSamples);
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
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
     * 获取支持的音程类型 / Get Supported Interval Types
     *
     * @return 支持的音程类型数组 / Array of supported interval types
     */
    public static IntervalType[] getSupportedIntervalTypes() {
        return IntervalType.values();
    }
    
    /**
     * 获取音程类型的中文名称 / Get Chinese Name of Interval Type
     *
     * @param intervalType 音程类型 / Interval type
     * @return 中文名称 / Chinese name
     */
    public static String getIntervalTypeName(IntervalType intervalType) {
        return intervalType.getChineseName();
    }
    
    /**
     * 根据半音数获取音程类型 / Get Interval Type by Semitones
     *
     * @param semitones 半音数 / Number of semitones
     * @return 音程类型 / Interval type
     */
    public static IntervalType getIntervalTypeBySemitones(int semitones) {
        for (IntervalType type : IntervalType.values()) {
            if (type.getSemitones() == semitones) {
                return type;
            }
        }
        return null;
    }
}