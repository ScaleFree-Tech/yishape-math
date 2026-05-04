package com.yishape.lab.music.theory;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.Signals;

/**
 * 音阶理论 / Scale Theory Class
 * <p>
 * 提供音阶相关的音乐理论功能
 * Provides music theory functionality related to scales.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class ScaleTheory {

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
     * 生成音阶 / Generate scale
     * <p>
     * 根据根音和音阶类型生成音阶的所有音符
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
     * 生成音阶的音频信号 / Generate audio signal for scale
     * <p>
     * 生成指定音阶的音频信号，每个音符持续指定时间
     * Generate audio signal for specified scale, each note lasts for specified duration.
     * </p>
     *
     * @param rootNote 根音 (0-11) / Root note (0-11)
     * @param scaleType 音阶类型 / Scale type
     * @param octave 八度 / Octave
     * @param noteDuration 每个音符的持续时间(秒) / Duration of each note (seconds)
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

        return new AudioData(Linalg.vector(samples.toDoubleArray()), sampleRate, 1, 16, AudioFormat.WAV);
    }

    // 私有辅助方法 / Private helper methods

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
     * 根据音名和八度计算频率 / Calculate frequency from note name and octave
     *
     * @param noteName 音名 (C, C#, D, etc.) / Note name (C, C#, D, etc.)
     * @param octave 八度 / Octave
     * @return 频率 (Hz) / Frequency (Hz)
     */
    public static double noteToFrequency(String noteName, int octave) {
        int noteIndex = getNoteIndex(noteName);
        // Assuming A4_FREQUENCY = 440.0 Hz as standard
        int semitones = (octave - 4) * 12 + noteIndex;
        return 440.0 * Math.pow(Math.pow(2, 1.0/12), semitones);
    }

    /**
     * 获取音名索引 / Get note name index
     */
    public static int getNoteIndex(String noteName) {
        for (int i = 0; i < NOTE_NAMES.length; i++) {
            if (NOTE_NAMES[i].equals(noteName)) {
                return i;
            }
        }
        throw new IllegalArgumentException("Invalid note name: " + noteName);
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
}