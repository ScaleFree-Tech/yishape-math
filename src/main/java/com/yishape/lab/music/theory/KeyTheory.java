package com.yishape.lab.music.theory;

/**
 * 调性理论类 / Key Theory Class
 * <p>
 * 提供调性相关的音乐理论功能。
 * Provides music theory functionality related to keys.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class KeyTheory {
    
    /** 音名数组 / Note names array */
    public static final String[] NOTE_NAMES = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    
    /**
     * 调性类 / Key Class
     */
    public static class Key {
        private final int rootNote; // 根音 (0-11) / Root note (0-11)
        private final ScaleTheory.ScaleType scaleType; // 音阶类型 / Scale type
        
        public Key(int rootNote, ScaleTheory.ScaleType scaleType) {
            this.rootNote = rootNote % 12;
            this.scaleType = scaleType;
        }
        
        public int getRootNote() { return rootNote; }
        public ScaleTheory.ScaleType getScaleType() { return scaleType; }
        
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
        return (int) Math.round(12 * Math.log(frequency / 440.0) / Math.log(2));
    }
}