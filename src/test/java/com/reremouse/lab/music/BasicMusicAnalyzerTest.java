package com.reremouse.lab.music;

import com.reremouse.lab.music.analysis.BasicMusicAnalyzer;
import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioFormat;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.music.analysis.MusicDetectionResult;

public class BasicMusicAnalyzerTest {
    public static void main(String[] args) {
        System.out.println("Testing BasicMusicAnalyzer to check if hanging issue is resolved...");
        
        try {
            // Create a simple audio data for testing
            double[] samples = new double[44100]; // 1 second of audio at 44.1kHz
            for (int i = 0; i < samples.length; i++) {
                samples[i] = Math.sin(2 * Math.PI * 440 * i / 44100); // 440Hz sine wave
            }
            
            IVector<Double> sampleVector = Linalg.vector(samples);
            AudioData audioData = new AudioData(sampleVector, 44100.0, 1, 16, AudioFormat.WAV);
            
            System.out.println("Created test audio data with " + samples.length + " samples");
            
            // Create BasicMusicAnalyzer
            BasicMusicAnalyzer analyzer = new BasicMusicAnalyzer();
            System.out.println("Created BasicMusicAnalyzer instance");
            
            // Test the analysis - this was hanging before our fix
            System.out.println("Starting basic analysis...");
            long startTime = System.currentTimeMillis();
            
            MusicDetectionResult result = analyzer.analyzeMusic(audioData);
            
            long endTime = System.currentTimeMillis();
            System.out.println("Analysis completed in " + (endTime - startTime) + " ms");
            
            if (result != null) {
                System.out.println("Analysis successful!");
                System.out.println("Result class: " + result.getClass().getName());
            } else {
                System.out.println("Analysis returned null result");
            }
            
        } catch (Exception e) {
            System.err.println("Error during test: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("Test completed.");
    }
}