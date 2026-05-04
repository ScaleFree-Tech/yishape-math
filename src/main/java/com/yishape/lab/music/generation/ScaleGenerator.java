package com.yishape.lab.music.generation;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.*;

/**
 * 音阶生成器 / Scale Generator
 * <p>
 * 用于生成各种音阶的音频数据，支持多种音阶类型、调式和演奏模式。
 * Used to generate audio data for various scales, supporting multiple scale types, modes, and playing patterns.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ScaleGenerator {
    
    /**
     * 音阶类型枚举 / Scale Type Enumeration
     */
    public enum ScaleType {
        MAJOR("大调音阶", new int[]{0, 2, 4, 5, 7, 9, 11}),
        NATURAL_MINOR("自然小调音阶", new int[]{0, 2, 3, 5, 7, 8, 10}),
        HARMONIC_MINOR("和声小调音阶", new int[]{0, 2, 3, 5, 7, 8, 11}),
        MELODIC_MINOR("旋律小调音阶", new int[]{0, 2, 3, 5, 7, 9, 11}),
        DORIAN("多利亚调式", new int[]{0, 2, 3, 5, 7, 9, 10}),
        PHRYGIAN("弗里吉亚调式", new int[]{0, 1, 3, 5, 7, 8, 10}),
        LYDIAN("利底亚调式", new int[]{0, 2, 4, 6, 7, 9, 11}),
        MIXOLYDIAN("混合利底亚调式", new int[]{0, 2, 4, 5, 7, 9, 10}),
        LOCRIAN("洛克里亚调式", new int[]{0, 1, 3, 5, 6, 8, 10}),
        PENTATONIC_MAJOR("大调五声音阶", new int[]{0, 2, 4, 7, 9}),
        PENTATONIC_MINOR("小调五声音阶", new int[]{0, 3, 5, 7, 10}),
        BLUES("布鲁斯音阶", new int[]{0, 3, 5, 6, 7, 10}),
        WHOLE_TONE("全音音阶", new int[]{0, 2, 4, 6, 8, 10}),
        CHROMATIC("半音音阶", new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
        DIMINISHED("减音阶", new int[]{0, 2, 3, 5, 6, 8, 9, 11}),
        AUGMENTED("增音阶", new int[]{0, 3, 4, 7, 8, 11}),
        HUNGARIAN_MINOR("匈牙利小调", new int[]{0, 2, 3, 6, 7, 8, 11}),
        GYPSY("吉普赛音阶", new int[]{0, 1, 4, 5, 7, 8, 10}),
        ARABIC("阿拉伯音阶", new int[]{0, 1, 4, 5, 7, 8, 11}),
        JAPANESE("日本音阶", new int[]{0, 1, 5, 7, 8});
        
        private final String chineseName;
        private final int[] intervals;
        
        ScaleType(String chineseName, int[] intervals) {
            this.chineseName = chineseName;
            this.intervals = intervals;
        }
        
        public String getChineseName() { return chineseName; }
        public int[] getIntervals() { return intervals.clone(); }
        public int getLength() { return intervals.length; }
    }
    
    /**
     * 演奏模式枚举 / Playing Pattern Enumeration
     */
    public enum PlayingPattern {
        ASCENDING("上行", "从低音到高音"),
        DESCENDING("下行", "从高音到低音"),
        ASCENDING_DESCENDING("上行下行", "上行后下行"),
        ARPEGGIO("琶音", "分解演奏"),
        HARMONIC("和声", "同时演奏所有音符"),
        RANDOM("随机", "随机顺序演奏");
        
        private final String chineseName;
        private final String description;
        
        PlayingPattern(String chineseName, String description) {
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
    private Random random = new Random();
    
    /**
     * 默认构造函数 / Default Constructor
     */
    public ScaleGenerator() {
    }
    
    /**
     * 生成音阶 / Generate Scale
     *
     * @param rootNote 根音 / Root note (0-11, C=0, C#=1, D=2, ...)
     * @param scaleType 音阶类型 / Scale type
     * @param octave 八度 / Octave (0-8)
     * @param totalDuration 总持续时间（秒） / Total duration in seconds
     * @param sampleRate 采样率 / Sample rate
     * @param pattern 演奏模式 / Playing pattern
     * @return 生成的音阶音频数据 / Generated scale audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public AudioData generateScale(int rootNote, ScaleType scaleType, int octave, double totalDuration, 
                                 double sampleRate, PlayingPattern pattern) throws AudioProcessingException {
        switch (pattern) {
            case ASCENDING:
                return generateAscendingScale(rootNote, scaleType, octave, totalDuration, sampleRate);
            case DESCENDING:
                return generateDescendingScale(rootNote, scaleType, octave, totalDuration, sampleRate);
            case ASCENDING_DESCENDING:
                return generateAscendingDescendingScale(rootNote, scaleType, octave, totalDuration, sampleRate);
            case ARPEGGIO:
                return generateArpeggioScale(rootNote, scaleType, octave, totalDuration, sampleRate);
            case HARMONIC:
                return generateHarmonicScale(rootNote, scaleType, octave, totalDuration, sampleRate);
            case RANDOM:
                return generateRandomScale(rootNote, scaleType, octave, totalDuration, sampleRate);
            default:
                throw new AudioProcessingException("不支持的演奏模式 / Unsupported playing pattern: " + pattern);
        }
    }
    
    /**
     * 生成上行音阶 / Generate Ascending Scale
     *
     * @param rootNote 根音 / Root note
     * @param scaleType 音阶类型 / Scale type
     * @param octave 八度 / Octave
     * @param totalDuration 总持续时间 / Total duration
     * @param sampleRate 采样率 / Sample rate
     * @return 上行音阶音频数据 / Ascending scale audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    private AudioData generateAscendingScale(int rootNote, ScaleType scaleType, int octave, 
                                           double totalDuration, double sampleRate) throws AudioProcessingException {
        int[] intervals = scaleType.getIntervals();
        double noteDuration = totalDuration / intervals.length;
        
        return generateSequentialScale(rootNote, intervals, octave, noteDuration, sampleRate, false);
    }
    
    /**
     * 生成下行音阶 / Generate Descending Scale
     *
     * @param rootNote 根音 / Root note
     * @param scaleType 音阶类型 / Scale type
     * @param octave 八度 / Octave
     * @param totalDuration 总持续时间 / Total duration
     * @param sampleRate 采样率 / Sample rate
     * @return 下行音阶音频数据 / Descending scale audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    private AudioData generateDescendingScale(int rootNote, ScaleType scaleType, int octave, 
                                            double totalDuration, double sampleRate) throws AudioProcessingException {
        int[] intervals = scaleType.getIntervals();
        // 反转音阶
        int[] reversedIntervals = new int[intervals.length];
        for (int i = 0; i < intervals.length; i++) {
            reversedIntervals[i] = intervals[intervals.length - 1 - i];
        }
        
        double noteDuration = totalDuration / intervals.length;
        return generateSequentialScale(rootNote, reversedIntervals, octave, noteDuration, sampleRate, true);
    }
    
    /**
     * 生成上行下行音阶 / Generate Ascending-Descending Scale
     *
     * @param rootNote 根音 / Root note
     * @param scaleType 音阶类型 / Scale type
     * @param octave 八度 / Octave
     * @param totalDuration 总持续时间 / Total duration
     * @param sampleRate 采样率 / Sample rate
     * @return 上行下行音阶音频数据 / Ascending-descending scale audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    private AudioData generateAscendingDescendingScale(int rootNote, ScaleType scaleType, int octave, 
                                                     double totalDuration, double sampleRate) throws AudioProcessingException {
        int[] intervals = scaleType.getIntervals();
        int totalNotes = intervals.length * 2 - 1; // 避免重复最高音
        double noteDuration = totalDuration / totalNotes;
        
        int totalSamples = (int) (totalDuration * sampleRate);
        IVector<Double> samples = Linalg.zeros(totalSamples);
        
        int currentSample = 0;
        
        // 上行
        for (int i = 0; i < intervals.length; i++) {
            int noteNumber = (rootNote + intervals[i]) % 12;
            int noteOctave = octave + (rootNote + intervals[i]) / 12;
            AudioData noteAudio = generateSingleNote(noteNumber, noteOctave, noteDuration, sampleRate);
            
            addToSamples(samples, noteAudio.getSamples(), currentSample);
            currentSample += noteAudio.getSamples().size();
        }
        
        // 下行（跳过最高音）
        for (int i = intervals.length - 2; i >= 0; i--) {
            int noteNumber = (rootNote + intervals[i]) % 12;
            int noteOctave = octave + (rootNote + intervals[i]) / 12;
            AudioData noteAudio = generateSingleNote(noteNumber, noteOctave, noteDuration, sampleRate);
            
            addToSamples(samples, noteAudio.getSamples(), currentSample);
            currentSample += noteAudio.getSamples().size();
        }
        
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成琶音音阶 / Generate Arpeggio Scale
     *
     * @param rootNote 根音 / Root note
     * @param scaleType 音阶类型 / Scale type
     * @param octave 八度 / Octave
     * @param totalDuration 总持续时间 / Total duration
     * @param sampleRate 采样率 / Sample rate
     * @return 琶音音阶音频数据 / Arpeggio scale audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    private AudioData generateArpeggioScale(int rootNote, ScaleType scaleType, int octave, 
                                          double totalDuration, double sampleRate) throws AudioProcessingException {
        int[] intervals = scaleType.getIntervals();
        double noteDuration = totalDuration / (intervals.length * 2); // 每个音符播放两次
        
        int totalSamples = (int) (totalDuration * sampleRate);
        IVector<Double> samples = Linalg.zeros(totalSamples);
        
        int currentSample = 0;
        
        // 快速琶音，每个音符播放两次
        for (int round = 0; round < 2; round++) {
            for (int i = 0; i < intervals.length; i++) {
                if (currentSample >= totalSamples) break;
                
                int noteNumber = (rootNote + intervals[i]) % 12;
                int noteOctave = octave + (rootNote + intervals[i]) / 12;
                AudioData noteAudio = generateSingleNote(noteNumber, noteOctave, noteDuration, sampleRate);
                
                addToSamples(samples, noteAudio.getSamples(), currentSample);
                currentSample += noteAudio.getSamples().size();
            }
        }
        
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成和声音阶 / Generate Harmonic Scale
     *
     * @param rootNote 根音 / Root note
     * @param scaleType 音阶类型 / Scale type
     * @param octave 八度 / Octave
     * @param totalDuration 总持续时间 / Total duration
     * @param sampleRate 采样率 / Sample rate
     * @return 和声音阶音频数据 / Harmonic scale audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    private AudioData generateHarmonicScale(int rootNote, ScaleType scaleType, int octave, 
                                          double totalDuration, double sampleRate) throws AudioProcessingException {
        int[] intervals = scaleType.getIntervals();
        int totalSamples = (int) (totalDuration * sampleRate);
        IVector<Double> samples = Linalg.zeros(totalSamples);
        
        // 同时播放所有音符
        for (int interval : intervals) {
            int noteNumber = (rootNote + interval) % 12;
            int noteOctave = octave + (rootNote + interval) / 12;
            double frequency = calculateFrequency(noteNumber, noteOctave);
            IVector<Double> noteSamples = generateWaveform(frequency, totalDuration, sampleRate);
            
            // 混合到主样本中
            for (int i = 0; i < totalSamples; i++) {
                double currentValue = samples.get(i);
                double noteValue = noteSamples.get(i) * amplitude / intervals.length;
                samples.set(i, currentValue + noteValue);
            }
        }
        
        // 应用包络
        applyEnvelope(samples, totalDuration, sampleRate);
        
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成随机音阶 / Generate Random Scale
     *
     * @param rootNote 根音 / Root note
     * @param scaleType 音阶类型 / Scale type
     * @param octave 八度 / Octave
     * @param totalDuration 总持续时间 / Total duration
     * @param sampleRate 采样率 / Sample rate
     * @return 随机音阶音频数据 / Random scale audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    private AudioData generateRandomScale(int rootNote, ScaleType scaleType, int octave, 
                                        double totalDuration, double sampleRate) throws AudioProcessingException {
        int[] intervals = scaleType.getIntervals();
        double noteDuration = totalDuration / intervals.length;
        
        // 随机打乱音阶顺序
        List<Integer> shuffledIntervals = new ArrayList<>();
        for (int interval : intervals) {
            shuffledIntervals.add(interval);
        }
        Collections.shuffle(shuffledIntervals, random);
        
        int[] randomIntervals = shuffledIntervals.stream().mapToInt(i -> i).toArray();
        return generateSequentialScale(rootNote, randomIntervals, octave, noteDuration, sampleRate, false);
    }
    
    /**
     * 生成音阶练习 / Generate Scale Exercise
     *
     * @param scaleTypes 要练习的音阶类型 / Scale types to practice
     * @param rootNote 根音 / Root note
     * @param octave 八度 / Octave
     * @param scaleDuration 每个音阶的持续时间 / Duration of each scale
     * @param pauseDuration 音阶间的停顿时间 / Pause duration between scales
     * @param sampleRate 采样率 / Sample rate
     * @param pattern 演奏模式 / Playing pattern
     * @return 生成的音阶练习音频数据 / Generated scale exercise audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public AudioData generateScaleExercise(ScaleType[] scaleTypes, int rootNote, int octave,
                                         double scaleDuration, double pauseDuration, double sampleRate,
                                         PlayingPattern pattern) throws AudioProcessingException {
        if (scaleTypes == null || scaleTypes.length == 0) {
            throw new AudioProcessingException("音阶类型数组不能为空 / Scale types array cannot be empty");
        }
        
        List<AudioData> exerciseAudios = new ArrayList<>();
        
        for (int i = 0; i < scaleTypes.length; i++) {
            // 生成音阶
            AudioData scaleAudio = generateScale(rootNote, scaleTypes[i], octave, scaleDuration, sampleRate, pattern);
            exerciseAudios.add(scaleAudio);
            
            // 添加停顿（除了最后一个音阶）
            if (i < scaleTypes.length - 1 && pauseDuration > 0) {
                AudioData pauseAudio = generateSilence(pauseDuration, sampleRate);
                exerciseAudios.add(pauseAudio);
            }
        }
        
        return concatenateAudioData(exerciseAudios, sampleRate);
    }
    
    /**
     * 生成多八度音阶 / Generate Multi-Octave Scale
     *
     * @param rootNote 根音 / Root note
     * @param scaleType 音阶类型 / Scale type
     * @param startOctave 起始八度 / Start octave
     * @param octaveCount 八度数量 / Number of octaves
     * @param totalDuration 总持续时间 / Total duration
     * @param sampleRate 采样率 / Sample rate
     * @param pattern 演奏模式 / Playing pattern
     * @return 生成的多八度音阶音频数据 / Generated multi-octave scale audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public AudioData generateMultiOctaveScale(int rootNote, ScaleType scaleType, int startOctave, int octaveCount,
                                             double totalDuration, double sampleRate, PlayingPattern pattern) 
            throws AudioProcessingException {
        double noteDuration = totalDuration / (scaleType.getIntervals().length * octaveCount);
        
        int totalSamples = (int) (totalDuration * sampleRate);
        IVector<Double> samples = Linalg.zeros(totalSamples);
        
        int currentSample = 0;
        
        for (int octaveOffset = 0; octaveOffset < octaveCount; octaveOffset++) {
            int currentOctave = startOctave + octaveOffset;
            int[] intervals = scaleType.getIntervals();
            
            for (int interval : intervals) {
                if (currentSample >= totalSamples) break;
                
                int noteNumber = (rootNote + interval) % 12;
                int noteOctave = currentOctave + (rootNote + interval) / 12;
                AudioData noteAudio = generateSingleNote(noteNumber, noteOctave, noteDuration, sampleRate);
                
                addToSamples(samples, noteAudio.getSamples(), currentSample);
                currentSample += noteAudio.getSamples().size();
            }
        }
        
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成顺序音阶 / Generate Sequential Scale
     *
     * @param rootNote 根音 / Root note
     * @param intervals 音程序列 / Interval sequence
     * @param octave 八度 / Octave
     * @param noteDuration 每个音符的持续时间 / Duration of each note
     * @param sampleRate 采样率 / Sample rate
     * @param descending 是否下行 / Whether descending
     * @return 顺序音阶音频数据 / Sequential scale audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    private AudioData generateSequentialScale(int rootNote, int[] intervals, int octave, 
                                            double noteDuration, double sampleRate, boolean descending) 
            throws AudioProcessingException {
        int totalSamples = (int) (noteDuration * intervals.length * sampleRate);
        IVector<Double> samples = Linalg.zeros(totalSamples);
        
        int currentSample = 0;
        
        for (int interval : intervals) {
            int noteNumber = (rootNote + interval) % 12;
            int noteOctave = octave + (rootNote + interval) / 12;
            AudioData noteAudio = generateSingleNote(noteNumber, noteOctave, noteDuration, sampleRate);
            
            addToSamples(samples, noteAudio.getSamples(), currentSample);
            currentSample += noteAudio.getSamples().size();
        }
        
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成单个音符 / Generate Single Note
     *
     * @param noteNumber 音符编号 / Note number
     * @param octave 八度 / Octave
     * @param duration 持续时间 / Duration
     * @param sampleRate 采样率 / Sample rate
     * @return 单个音符音频数据 / Single note audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    private AudioData generateSingleNote(int noteNumber, int octave, double duration, double sampleRate) 
            throws AudioProcessingException {
        double frequency = calculateFrequency(noteNumber, octave);
        IVector<Double> samples = generateWaveform(frequency, duration, sampleRate);
        
        // 应用包络
        applyEnvelope(samples, duration, sampleRate);
        
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 添加样本到主向量 / Add samples to main vector
     *
     * @param mainSamples 主样本向量 / Main samples vector
     * @param addSamples 要添加的样本向量 / Samples to add
     * @param startIndex 起始索引 / Start index
     */
    private void addToSamples(IVector<Double> mainSamples, IVector<Double> addSamples, int startIndex) {
        int endIndex = Math.min(startIndex + addSamples.size(), mainSamples.size());
        for (int i = startIndex; i < endIndex; i++) {
            mainSamples.set(i, addSamples.get(i - startIndex));
        }
    }
    
    /**
     * 连接音频数据 / Concatenate Audio Data
     *
     * @param audioList 音频数据列表 / Audio data list
     * @param sampleRate 采样率 / Sample rate
     * @return 连接后的音频数据 / Concatenated audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
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
     *
     * @param duration 持续时间 / Duration
     * @param sampleRate 采样率 / Sample rate
     * @return 静音音频数据 / Silence audio data
     */
    private AudioData generateSilence(double duration, double sampleRate) {
        int numSamples = (int) (duration * sampleRate);
        IVector<Double> samples = Linalg.zeros(numSamples);
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成波形 / Generate Waveform
     *
     * @param frequency 频率 / Frequency
     * @param duration 持续时间 / Duration
     * @param sampleRate 采样率 / Sample rate
     * @return 波形样本向量 / Waveform samples vector
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
     * 计算频率 / Calculate Frequency
     *
     * @param noteNumber 音符编号（0-11）/ Note number (0-11)
     * @param octave 八度 / Octave
     * @return 频率（Hz）/ Frequency in Hz
     */
    private double calculateFrequency(int noteNumber, int octave) {
        return NOTE_FREQUENCIES[noteNumber] * Math.pow(2, octave - 4);
    }
    
    /**
     * 应用包络 / Apply Envelope
     *
     * @param samples 样本向量 / Samples vector
     * @param duration 持续时间 / Duration
     * @param sampleRate 采样率 / Sample rate
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
     * 获取支持的音阶类型 / Get Supported Scale Types
     *
     * @return 支持的音阶类型数组 / Array of supported scale types
     */
    public static ScaleType[] getSupportedScaleTypes() {
        return ScaleType.values();
    }
    
    /**
     * 获取音阶类型的中文名称 / Get Chinese Name of Scale Type
     *
     * @param scaleType 音阶类型 / Scale type
     * @return 中文名称 / Chinese name
     */
    public static String getScaleTypeName(ScaleType scaleType) {
        return scaleType.getChineseName();
    }
    
    /**
     * 根据音程模式获取音阶类型 / Get Scale Type by Interval Pattern
     *
     * @param intervals 音程模式 / Interval pattern
     * @return 匹配的音阶类型，如果没有匹配则返回null / Matching scale type, or null if no match
     */
    public static ScaleType getScaleTypeByIntervals(int[] intervals) {
        for (ScaleType type : ScaleType.values()) {
            if (Arrays.equals(type.getIntervals(), intervals)) {
                return type;
            }
        }
        return null;
    }
}