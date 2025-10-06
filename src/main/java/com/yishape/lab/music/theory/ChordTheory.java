package com.yishape.lab.music.theory;

/**
 * 和弦理论 / Chord Theory Class
 * <p>
 * 提供和弦相关的音乐理论功能
 * Provides music theory functionality related to chords.
 * </p>
 */
public class ChordTheory {

    /** 音名数组 / Note names array */
    public static final String[] NOTE_NAMES = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

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
     * 生成和弦 / Generate chord
     * <p>
     * 根据根音和和弦类型生成和弦的所有音符
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
}