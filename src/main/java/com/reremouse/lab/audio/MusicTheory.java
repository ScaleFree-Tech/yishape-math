package com.reremouse.lab.audio;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.Signals;

/**
 * 音乐理论类 / Music Theory Class
 * <p>
 * 提供音乐理论相关功能，包括音阶、和弦、调性、音程等。
 * 使用项目现有的signal包和linalg包功能进行音乐理论计算。
 * </p>
 * <p>
 * Provides music theory functionality including scales, chords, keys, intervals, etc.
 * Uses existing signal and linalg package functionality for music theory calculations.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class MusicTheory {
    
    /** 标准音A4的频率 (Hz) / Standard A4 frequency (Hz) */
    public static final double A4_FREQUENCY = 440.0;
    
    /** 十二平均律的半音频率比 / Semitone frequency ratio in equal temperament */
    public static final double SEMITONE_RATIO = Math.pow(2, 1.0/12);
    
    /** 音名数组 / Note names array */
    public static final String[] NOTE_NAMES = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    
    /** 大调音阶的半音间隔 / Major scale semitone intervals */
    public static final int[] MAJOR_SCALE_INTERVALS = {0, 2, 4, 5, 7, 9, 11};
    
    /** 小调音阶的半音间隔 / Minor scale semitone intervals */
    public static final int[] MINOR_SCALE_INTERVALS = {0, 2, 3, 5, 7, 8, 10};
    
    /** 自然小调音阶的半音间隔 / Natural minor scale semitone intervals */
    public static final int[] NATURAL_MINOR_SCALE_INTERVALS = {0, 2, 3, 5, 7, 8, 10};
    
    /** 和声小调音阶的半音间隔 / Harmonic minor scale semitone intervals */
    public static final int[] HARMONIC_MINOR_SCALE_INTERVALS = {0, 2, 3, 5, 7, 8, 11};
    
    /** 旋律小调音阶的半音间隔 / Melodic minor scale semitone intervals */
    public static final int[] MELODIC_MINOR_SCALE_INTERVALS = {0, 2, 3, 5, 7, 9, 11};
    
    /**
     * 音阶类型枚举 / Scale Type Enum
     */
    public enum ScaleType {
        MAJOR("大调", "Major"),
        MINOR("小调", "Minor"),
        NATURAL_MINOR("自然小调", "Natural Minor"),
        HARMONIC_MINOR("和声小调", "Harmonic Minor"),
        MELODIC_MINOR("旋律小调", "Melodic Minor"),
        PENTATONIC_MAJOR("大调五声音阶", "Major Pentatonic"),
        PENTATONIC_MINOR("小调五声音阶", "Minor Pentatonic"),
        BLUES("布鲁斯音阶", "Blues Scale"),
        DORIAN("多利亚调式", "Dorian Mode"),
        PHRYGIAN("弗里吉亚调式", "Phrygian Mode"),
        LYDIAN("利底亚调式", "Lydian Mode"),
        MIXOLYDIAN("混合利底亚调式", "Mixolydian Mode"),
        LOCRIAN("洛克里亚调式", "Locrian Mode");
        
        private final String chineseName;
        private final String englishName;
        
        ScaleType(String chineseName, String englishName) {
            this.chineseName = chineseName;
            this.englishName = englishName;
        }
        
        public String getChineseName() { return chineseName; }
        public String getEnglishName() { return englishName; }
    }
    
    /**
     * 和弦类型枚举 / Chord Type Enum
     */
    public enum ChordType {
        MAJOR("大三和弦", "Major Triad", new int[]{0, 4, 7}),
        MINOR("小三和弦", "Minor Triad", new int[]{0, 3, 7}),
        DIMINISHED("减三和弦", "Diminished Triad", new int[]{0, 3, 6}),
        AUGMENTED("增三和弦", "Augmented Triad", new int[]{0, 4, 8}),
        MAJOR_7TH("大七和弦", "Major 7th", new int[]{0, 4, 7, 11}),
        MINOR_7TH("小七和弦", "Minor 7th", new int[]{0, 3, 7, 10}),
        DOMINANT_7TH("属七和弦", "Dominant 7th", new int[]{0, 4, 7, 10}),
        DIMINISHED_7TH("减七和弦", "Diminished 7th", new int[]{0, 3, 6, 9}),
        HALF_DIMINISHED_7TH("半减七和弦", "Half Diminished 7th", new int[]{0, 3, 6, 10}),
        MAJOR_6TH("大六和弦", "Major 6th", new int[]{0, 4, 7, 9}),
        MINOR_6TH("小六和弦", "Minor 6th", new int[]{0, 3, 7, 9}),
        SUSPENDED_2ND("挂二和弦", "Suspended 2nd", new int[]{0, 2, 7}),
        SUSPENDED_4TH("挂四和弦", "Suspended 4th", new int[]{0, 5, 7});
        
        private final String chineseName;
        private final String englishName;
        private final int[] intervals;
        
        ChordType(String chineseName, String englishName, int[] intervals) {
            this.chineseName = chineseName;
            this.englishName = englishName;
            this.intervals = intervals;
        }
        
        public String getChineseName() { return chineseName; }
        public String getEnglishName() { return englishName; }
        public int[] getIntervals() { return intervals.clone(); }
    }
    
    /**
     * 调性类 / Key Class
     */
    public static class Key {
        private final int rootNote; // 根音 (0-11) / Root note (0-11)
        private final ScaleType scaleType; // 音阶类型 / Scale type
        
        public Key(int rootNote, ScaleType scaleType) {
            this.rootNote = rootNote % 12;
            this.scaleType = scaleType;
        }
        
        public int getRootNote() { return rootNote; }
        public ScaleType getScaleType() { return scaleType; }
        
        public String getRootNoteName() {
            return NOTE_NAMES[rootNote];
        }
        
        public String getKeyName() {
            return getRootNoteName() + " " + scaleType.getChineseName();
        }
        
        public String getKeyNameEnglish() {
            return getRootNoteName() + " " + scaleType.getEnglishName();
        }
        
        @Override
        public String toString() {
            return getKeyName() + " / " + getKeyNameEnglish();
        }
    }
    
    /**
     * 和弦类 / Chord Class
     */
    public static class Chord {
        private final int rootNote; // 根音 (0-11) / Root note (0-11)
        private final ChordType chordType; // 和弦类型 / Chord type
        
        public Chord(int rootNote, ChordType chordType) {
            this.rootNote = rootNote % 12;
            this.chordType = chordType;
        }
        
        public int getRootNote() { return rootNote; }
        public ChordType getChordType() { return chordType; }
        
        public String getRootNoteName() {
            return NOTE_NAMES[rootNote];
        }
        
        public String getChordName() {
            return getRootNoteName() + chordType.getChineseName();
        }
        
        public String getChordNameEnglish() {
            return getRootNoteName() + chordType.getEnglishName();
        }
        
        /**
         * 获取和弦的所有音符 / Get all notes in the chord
         * @return 音符数组 / Notes array
         */
        public int[] getNotes() {
            int[] intervals = chordType.getIntervals();
            int[] notes = new int[intervals.length];
            for (int i = 0; i < intervals.length; i++) {
                notes[i] = (rootNote + intervals[i]) % 12;
            }
            return notes;
        }
        
        @Override
        public String toString() {
            return getChordName() + " / " + getChordNameEnglish();
        }
    }
    
    /**
     * 根据半音数计算频率 / Calculate frequency from semitone number
     * <p>
     * 使用十二平均律计算指定半音数对应的频率。
     * Uses equal temperament to calculate frequency for given semitone number.
     * </p>
     *
     * @param semitones 半音数，A4为0 / Semitone number, A4 is 0
     * @return 频率 (Hz) / Frequency (Hz)
     */
    public static double semitonesToFrequency(int semitones) {
        return A4_FREQUENCY * Math.pow(SEMITONE_RATIO, semitones);
    }
    
    /**
     * 根据频率计算半音数 / Calculate semitone number from frequency
     * <p>
     * 使用十二平均律计算指定频率对应的半音数。
     * Uses equal temperament to calculate semitone number for given frequency.
     * </p>
     *
     * @param frequency 频率 (Hz) / Frequency (Hz)
     * @return 半音数，A4为0 / Semitone number, A4 is 0
     */
    public static int frequencyToSemitones(double frequency) {
        return (int) Math.round(12 * Math.log(frequency / A4_FREQUENCY) / Math.log(2));
    }
    
    /**
     * 根据音名和八度计算频率 / Calculate frequency from note name and octave
     *
     * @param noteName 音名 (C, C#, D, etc.) / Note name (C, C#, D, etc.)
     * @param octave 八度 / Octave
     * @return 频率 (Hz) / Frequency (Hz)
     */
    public static double noteToFrequency(String noteName, int octave) {
        int noteIndex = getNoteIndex(noteName);
        int semitones = (octave - 4) * 12 + noteIndex;
        return semitonesToFrequency(semitones);
    }
    
    /**
     * 根据频率计算音名和八度 / Calculate note name and octave from frequency
     *
     * @param frequency 频率 (Hz) / Frequency (Hz)
     * @return 包含音名和八度的数组 / Array containing note name and octave
     */
    public static String[] frequencyToNote(double frequency) {
        int semitones = frequencyToSemitones(frequency);
        int octave = 4 + semitones / 12;
        int noteIndex = ((semitones % 12) + 12) % 12;
        return new String[]{NOTE_NAMES[noteIndex], String.valueOf(octave)};
    }
    
    /**
     * 生成音阶 / Generate scale
     * <p>
     * 根据根音和音阶类型生成音阶的所有音符。
     * Generate all notes in a scale based on root note and scale type.
     * </p>
     *
     * @param rootNote 根音 (0-11) / Root note (0-11)
     * @param scaleType 音阶类型 / Scale type
     * @return 音阶音符数组 / Scale notes array
     */
    public static int[] generateScale(int rootNote, ScaleType scaleType) {
        int[] intervals = getScaleIntervals(scaleType);
        int[] scale = new int[intervals.length];
        
        for (int i = 0; i < intervals.length; i++) {
            scale[i] = (rootNote + intervals[i]) % 12;
        }
        
        return scale;
    }
    
    /**
     * 生成和弦 / Generate chord
     * <p>
     * 根据根音和和弦类型生成和弦的所有音符。
     * Generate all notes in a chord based on root note and chord type.
     * </p>
     *
     * @param rootNote 根音 (0-11) / Root note (0-11)
     * @param chordType 和弦类型 / Chord type
     * @return 和弦音符数组 / Chord notes array
     */
    public static int[] generateChord(int rootNote, ChordType chordType) {
        Chord chord = new Chord(rootNote, chordType);
        return chord.getNotes();
    }
    
    /**
     * 生成音阶的音频信号 / Generate audio signal for scale
     * <p>
     * 生成指定音阶的音频信号，每个音符持续指定时间。
     * Generate audio signal for specified scale, each note lasts for specified duration.
     * </p>
     *
     * @param rootNote 根音 (0-11) / Root note (0-11)
     * @param scaleType 音阶类型 / Scale type
     * @param octave 八度 / Octave
     * @param noteDuration 每个音符的持续时间 (秒) / Duration of each note (seconds)
     * @param sampleRate 采样率 / Sample rate
     * @param amplitude 幅度 / Amplitude
     * @return 音阶音频数据 / Scale audio data
     */
    public static AudioData generateScaleAudio(int rootNote, ScaleType scaleType, int octave, 
                                             double noteDuration, double sampleRate, double amplitude) {
        int[] scale = generateScale(rootNote, scaleType);
        int samplesPerNote = (int) (noteDuration * sampleRate);
        int totalSamples = scale.length * samplesPerNote;
        
        IVector<Double> samples = Linalg.zeros(totalSamples);
        
        for (int i = 0; i < scale.length; i++) {
            double frequency = noteToFrequency(NOTE_NAMES[scale[i]], octave);
            int startSample = i * samplesPerNote;
            int endSample = Math.min(startSample + samplesPerNote, totalSamples);
            
            // 生成正弦波 / Generate sine wave
            IVector<Double> noteSamples = Signals.sineWave(
                endSample - startSample, frequency, sampleRate, amplitude, 0.0);
            
            // 添加淡入淡出效果 / Add fade in/out effect
            noteSamples = applyFadeInOut(noteSamples, (int)(sampleRate * 0.1));
            
            // 复制到总信号中 / Copy to total signal
            for (int j = 0; j < noteSamples.length(); j++) {
                samples.set(startSample + j, noteSamples.get(j));
            }
        }
        
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 生成和弦的音频信号 / Generate audio signal for chord
     * <p>
     * 生成指定和弦的音频信号，所有音符同时播放。
     * Generate audio signal for specified chord, all notes play simultaneously.
     * </p>
     *
     * @param rootNote 根音 (0-11) / Root note (0-11)
     * @param chordType 和弦类型 / Chord type
     * @param octave 八度 / Octave
     * @param duration 持续时间 (秒) / Duration (seconds)
     * @param sampleRate 采样率 / Sample rate
     * @param amplitude 幅度 / Amplitude
     * @return 和弦音频数据 / Chord audio data
     */
    public static AudioData generateChordAudio(int rootNote, ChordType chordType, int octave,
                                             double duration, double sampleRate, double amplitude) {
        int[] chord = generateChord(rootNote, chordType);
        int totalSamples = (int) (duration * sampleRate);
        
        IVector<Double> samples = Linalg.zeros(totalSamples);
        
        // 为每个音符生成正弦波并叠加 / Generate sine wave for each note and superimpose
        for (int note : chord) {
            double frequency = noteToFrequency(NOTE_NAMES[note], octave);
            IVector<Double> noteSamples = Signals.sineWave(
                totalSamples, frequency, sampleRate, amplitude / chord.length, 0.0);
            
            samples = samples.add(noteSamples);
        }
        
        // 添加淡入淡出效果 / Add fade in/out effect
        samples = applyFadeInOut(samples, (int)(sampleRate * 0.1));
        
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 检测调性 / Detect key
     * <p>
     * 根据音频的频谱特征检测最可能的调性。
     * Detect most likely key based on spectral characteristics of audio.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @return 检测到的调性 / Detected key
     */
    public static Key detectKey(AudioData audioData) {
        // 计算频谱 / Calculate spectrum
        var spectrumResult = AudioAnalyzer.calculateSpectrum(audioData);
        IVector<Double> frequencies = spectrumResult._1;
        IVector<Double> magnitudes = spectrumResult._2;
        
        // 寻找主要频率峰值 / Find main frequency peaks
        double[] peakFrequencies = findPeakFrequencies(frequencies, magnitudes, 10);
        
        // 将频率转换为音符 / Convert frequencies to notes
        int[] notes = new int[peakFrequencies.length];
        for (int i = 0; i < peakFrequencies.length; i++) {
            String[] noteInfo = frequencyToNote(peakFrequencies[i]);
            notes[i] = getNoteIndex(noteInfo[0]);
        }
        
        // 分析调性 / Analyze key
        return analyzeKey(notes);
    }
    
    /**
     * 检测和弦 / Detect chord
     * <p>
     * 根据音频的频谱特征检测最可能的和弦。
     * Detect most likely chord based on spectral characteristics of audio.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @return 检测到的和弦 / Detected chord
     */
    public static Chord detectChord(AudioData audioData) {
        // 计算频谱 / Calculate spectrum
        var spectrumResult = AudioAnalyzer.calculateSpectrum(audioData);
        IVector<Double> frequencies = spectrumResult._1;
        IVector<Double> magnitudes = spectrumResult._2;
        
        // 寻找主要频率峰值 / Find main frequency peaks
        double[] peakFrequencies = findPeakFrequencies(frequencies, magnitudes, 8);
        
        // 将频率转换为音符 / Convert frequencies to notes
        int[] notes = new int[peakFrequencies.length];
        for (int i = 0; i < peakFrequencies.length; i++) {
            String[] noteInfo = frequencyToNote(peakFrequencies[i]);
            notes[i] = getNoteIndex(noteInfo[0]);
        }
        
        // 分析和弦 / Analyze chord
        return analyzeChord(notes);
    }
    
    /**
     * 计算音程 / Calculate interval
     * <p>
     * 计算两个音符之间的音程。
     * Calculate interval between two notes.
     * </p>
     *
     * @param note1 第一个音符 (0-11) / First note (0-11)
     * @param note2 第二个音符 (0-11) / Second note (0-11)
     * @return 音程 (半音数) / Interval (semitones)
     */
    public static int calculateInterval(int note1, int note2) {
        return (note2 - note1 + 12) % 12;
    }
    
    /**
     * 获取音程名称 / Get interval name
     *
     * @param semitones 半音数 / Semitones
     * @return 音程名称 / Interval name
     */
    public static String getIntervalName(int semitones) {
        String[] intervalNames = {
            "纯一度", "小二度", "大二度", "小三度", "大三度", "纯四度",
            "增四度", "纯五度", "小六度", "大六度", "小七度", "大七度"
        };
        return intervalNames[semitones % 12];
    }
    
    // 私有辅助方法 / Private helper methods
    
    /**
     * 获取音名索引 / Get note name index
     */
    private static int getNoteIndex(String noteName) {
        for (int i = 0; i < NOTE_NAMES.length; i++) {
            if (NOTE_NAMES[i].equals(noteName)) {
                return i;
            }
        }
        throw new IllegalArgumentException("Invalid note name: " + noteName);
    }
    
    /**
     * 获取音阶间隔 / Get scale intervals
     */
    private static int[] getScaleIntervals(ScaleType scaleType) {
        switch (scaleType) {
            case MAJOR: return MAJOR_SCALE_INTERVALS;
            case MINOR:
            case NATURAL_MINOR: return NATURAL_MINOR_SCALE_INTERVALS;
            case HARMONIC_MINOR: return HARMONIC_MINOR_SCALE_INTERVALS;
            case MELODIC_MINOR: return MELODIC_MINOR_SCALE_INTERVALS;
            case PENTATONIC_MAJOR: return new int[]{0, 2, 4, 7, 9};
            case PENTATONIC_MINOR: return new int[]{0, 3, 5, 7, 10};
            case BLUES: return new int[]{0, 3, 5, 6, 7, 10};
            case DORIAN: return new int[]{0, 2, 3, 5, 7, 9, 10};
            case PHRYGIAN: return new int[]{0, 1, 3, 5, 7, 8, 10};
            case LYDIAN: return new int[]{0, 2, 4, 6, 7, 9, 11};
            case MIXOLYDIAN: return new int[]{0, 2, 4, 5, 7, 9, 10};
            case LOCRIAN: return new int[]{0, 1, 3, 5, 6, 8, 10};
            default: return MAJOR_SCALE_INTERVALS;
        }
    }
    
    /**
     * 应用淡入淡出效果 / Apply fade in/out effect
     */
    private static IVector<Double> applyFadeInOut(IVector<Double> samples, int fadeSamples) {
        if (samples.length() <= 2 * fadeSamples) {
            return samples;
        }
        
        IVector<Double> fadedSamples = samples.copy();
        
        // 淡入 / Fade in
        for (int i = 0; i < fadeSamples; i++) {
            double factor = (double) i / fadeSamples;
            fadedSamples.set(i, fadedSamples.get(i) * factor);
        }
        
        // 淡出 / Fade out
        for (int i = 0; i < fadeSamples; i++) {
            double factor = (double) i / fadeSamples;
            int index = samples.length() - 1 - i;
            fadedSamples.set(index, fadedSamples.get(index) * factor);
        }
        
        return fadedSamples;
    }
    
    /**
     * 寻找频率峰值 / Find frequency peaks
     */
    private static double[] findPeakFrequencies(IVector<Double> frequencies, IVector<Double> magnitudes, int maxPeaks) {
        // 简化的峰值检测 / Simplified peak detection
        double[] peakFreqs = new double[maxPeaks];
        double[] peakMags = new double[maxPeaks];
        
        for (int i = 0; i < frequencies.length(); i++) {
            double freq = frequencies.get(i);
            double mag = magnitudes.get(i);
            
            // 寻找最小峰值位置 / Find minimum peak position
            int minIndex = 0;
            for (int j = 1; j < maxPeaks; j++) {
                if (peakMags[j] < peakMags[minIndex]) {
                    minIndex = j;
                }
            }
            
            // 如果当前幅度大于最小峰值，替换 / If current magnitude is greater than minimum peak, replace
            if (mag > peakMags[minIndex]) {
                peakFreqs[minIndex] = freq;
                peakMags[minIndex] = mag;
            }
        }
        
        return peakFreqs;
    }
    
    /**
     * 分析调性 / Analyze key
     */
    private static Key analyzeKey(int[] notes) {
        // 简化的调性分析 / Simplified key analysis
        // 统计每个根音的可能性 / Count possibility for each root note
        int[] keyScores = new int[12];
        
        for (int rootNote = 0; rootNote < 12; rootNote++) {
            int[] majorScale = generateScale(rootNote, ScaleType.MAJOR);
            int[] minorScale = generateScale(rootNote, ScaleType.MINOR);
            
            for (int note : notes) {
                for (int scaleNote : majorScale) {
                    if (note == scaleNote) {
                        keyScores[rootNote] += 2; // 大调权重更高 / Major scale has higher weight
                    }
                }
                for (int scaleNote : minorScale) {
                    if (note == scaleNote) {
                        keyScores[rootNote] += 1;
                    }
                }
            }
        }
        
        // 找到得分最高的根音 / Find root note with highest score
        int bestRootNote = 0;
        for (int i = 1; i < 12; i++) {
            if (keyScores[i] > keyScores[bestRootNote]) {
                bestRootNote = i;
            }
        }
        
        // 简化为大调或小调 / Simplify to major or minor
        return new Key(bestRootNote, ScaleType.MAJOR);
    }
    
    /**
     * 分析和弦 / Analyze chord
     */
    private static Chord analyzeChord(int[] notes) {
        // 简化的和弦分析 / Simplified chord analysis
        ChordType[] chordTypes = ChordType.values();
        int bestRootNote = 0;
        ChordType bestChordType = ChordType.MAJOR;
        int bestScore = 0;
        
        for (int rootNote = 0; rootNote < 12; rootNote++) {
            for (ChordType chordType : chordTypes) {
                int[] chord = generateChord(rootNote, chordType);
                int score = 0;
                
                for (int note : notes) {
                    for (int chordNote : chord) {
                        if (note == chordNote) {
                            score++;
                        }
                    }
                }
                
                if (score > bestScore) {
                    bestScore = score;
                    bestRootNote = rootNote;
                    bestChordType = chordType;
                }
            }
        }
        
        return new Chord(bestRootNote, bestChordType);
    }
}
