package com.reremouse.lab.music;

import com.reremouse.lab.music.analysis.basic.BeatDetectionResult;
import com.reremouse.lab.music.analysis.basic.ChordDetectionResult;
import com.reremouse.lab.music.analysis.basic.KeyDetectionResult;

/**
 * 测试音乐检测结果类 / Test Music Detection Result Classes
 */
public class MusicDetectionResultTest {

    public static void main(String[] args) {
        testBeatDetectionResult();
        testKeyDetectionResult();
        testChordDetectionResult();
        testInheritance();
        System.out.println("All tests passed!");
    }

    public static void testBeatDetectionResult() {
        // Test default constructor
        BeatDetectionResult beatResult = new BeatDetectionResult();
        assertEqual(0.0, beatResult.getConfidence(), 0.001);
        assertEqual(0, beatResult.getBeatTimes().length);
        assertEqual(0.0, beatResult.getBpm(), 0.001);

        // Test parameterized constructor
        double[] beats = {1.0, 2.0, 3.0};
        BeatDetectionResult beatResult2 = new BeatDetectionResult(beats, 120.0);
        assertArrayEqual(beats, beatResult2.getBeatTimes(), 0.001);
        assertEqual(120.0, beatResult2.getBpm(), 0.001);

        // Test with confidence and algorithm
        BeatDetectionResult beatResult3 = new BeatDetectionResult(beats, 120.0, 0.85, "fft");
        assertEqual(0.85, beatResult3.getConfidence(), 0.001);
        assertEqual("fft", beatResult3.getAlgorithm());

        System.out.println("BeatDetectionResult tests passed!");
    }

    public static void testKeyDetectionResult() {
        // Test default constructor
        KeyDetectionResult keyResult = new KeyDetectionResult();
        assertEqual(0.0, keyResult.getConfidence(), 0.001);
        assertEqual("", keyResult.getKeyName());
        assertEqual("", keyResult.getScaleType());

        // Test parameterized constructor
        double[] profile = {0.1, 0.2, 0.3};
        KeyDetectionResult keyResult2 = new KeyDetectionResult("C", "MAJOR", 0.9, profile);
        assertEqual("C", keyResult2.getKeyName());
        assertEqual("MAJOR", keyResult2.getScaleType());
        assertEqual(0.9, keyResult2.getConfidence(), 0.001);
        assertArrayEqual(profile, keyResult2.getPitchClassProfile(), 0.001);

        System.out.println("KeyDetectionResult tests passed!");
    }

    public static void testChordDetectionResult() {
        // Test default constructor
        ChordDetectionResult chordResult = new ChordDetectionResult();
        assertEqual(0.0, chordResult.getConfidence(), 0.001);
        assertEqual(0.0, chordResult.getStartTime(), 0.001);
        assertEqual(0.0, chordResult.getEndTime(), 0.001);
        assertEqual("", chordResult.getChordName());

        // Test parameterized constructor
        ChordDetectionResult chordResult2 = new ChordDetectionResult(1.0, 2.0, "Cmaj", 0.75);
        assertEqual(1.0, chordResult2.getStartTime(), 0.001);
        assertEqual(2.0, chordResult2.getEndTime(), 0.001);
        assertEqual("Cmaj", chordResult2.getChordName());
        assertEqual(0.75, chordResult2.getConfidence(), 0.001);

        System.out.println("ChordDetectionResult tests passed!");
    }

    public static void testInheritance() {
        BeatDetectionResult beatResult = new BeatDetectionResult();
        KeyDetectionResult keyResult = new KeyDetectionResult();
        ChordDetectionResult chordResult = new ChordDetectionResult();

        // All should be instances of MusicDetectionResult
        // Test setting common properties
        beatResult.setConfidence(0.8);
        keyResult.setConfidence(0.9);
        chordResult.setConfidence(0.7);

        assertEqual(0.8, beatResult.getConfidence(), 0.001);
        assertEqual(0.9, keyResult.getConfidence(), 0.001);
        assertEqual(0.7, chordResult.getConfidence(), 0.001);

        System.out.println("Inheritance tests passed!");
    }

    private static void assertEqual(double expected, double actual, double delta) {
        if (Math.abs(expected - actual) > delta) {
            throw new AssertionError("Expected: " + expected + ", Actual: " + actual);
        }
    }

    private static void assertEqual(int expected, int actual) {
        if (expected != actual) {
            throw new AssertionError("Expected: " + expected + ", Actual: " + actual);
        }
    }

    private static void assertEqual(String expected, String actual) {
        if (!expected.equals(actual)) {
            throw new AssertionError("Expected: " + expected + ", Actual: " + actual);
        }
    }

    private static void assertArrayEqual(double[] expected, double[] actual, double delta) {
        if (expected.length != actual.length) {
            throw new AssertionError("Array lengths differ: expected " + expected.length + ", actual " + actual.length);
        }
        for (int i = 0; i < expected.length; i++) {
            if (Math.abs(expected[i] - actual[i]) > delta) {
                throw new AssertionError("Array elements differ at index " + i + ": expected " + expected[i] + ", actual " + actual[i]);
            }
        }
    }
}