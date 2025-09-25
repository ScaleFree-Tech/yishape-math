package com.reremouse.lab.audio.music.analysis;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioFormat;
import com.reremouse.lab.music.analysis.MusicDetectionResult;
import com.reremouse.lab.music.analysis.basic.BeatDetectionResult;
import com.reremouse.lab.music.analysis.basic.KeyDetectionResult;
import com.reremouse.lab.music.analysis.basic.ChordDetectionResult;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.music.analysis.BasicMusicAnalyzer;

/**
 * BasicMusicAnalyzer测试类 / BasicMusicAnalyzer Test Class
 */
public class BasicMusicAnalyzerTest {
    
    public static void main(String[] args) {
        try {
            // Create a simple test audio data
            double[] testData = new double[44100]; // 1 second of audio at 44.1kHz
            for (int i = 0; i < testData.length; i++) {
                // Generate a simple sine wave
                testData[i] = Math.sin(2 * Math.PI * 440 * i / 44100);
            }
            
            IVector<Double> samples = Linalg.vector(testData);
            AudioData audioData = new AudioData(samples, 44100, 1, 16, AudioFormat.WAV);
            
            // Create analyzer
            BasicMusicAnalyzer analyzer = new BasicMusicAnalyzer();
            
            // Test beat analysis
            System.out.println("Testing beat analysis...");
            BeatDetectionResult beatResult = analyzer.analyzeBeat(audioData);
            System.out.println("Beat result: " + beatResult);
            
            // Test key analysis
            System.out.println("Testing key analysis...");
            KeyDetectionResult keyResult = analyzer.analyzeKey(audioData);
            System.out.println("Key result: " + keyResult);
            
            // Test chord analysis
            System.out.println("Testing chord analysis...");
            ChordDetectionResult chordResult = analyzer.analyzeChord(audioData);
            System.out.println("Chord result: " + chordResult);
            
            // Test full music analysis
            System.out.println("Testing full music analysis...");
            MusicDetectionResult musicResult = analyzer.analyzeMusic(audioData);
            System.out.println("Music result: " + musicResult);
            
            System.out.println("All tests passed!");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}