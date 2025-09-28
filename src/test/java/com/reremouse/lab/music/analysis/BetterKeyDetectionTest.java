package com.reremouse.lab.music.analysis;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.music.analysis.basic.KeyAnalyzerImpl;
import com.reremouse.lab.music.analysis.basic.KeyDetectionResult;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IVector;

import java.util.HashMap;
import java.util.Map;

/**
 * Test to verify key detection improvements with better signal
 */
public class BetterKeyDetectionTest {

    public static void main(String[] args) {
        try {
            System.out.println("Testing Key Detection with Better Signal...");
            
            // Create a test signal with clear musical content
            double sampleRate = 44100.0;
            int durationSeconds = 3;
            int numSamples = (int) (sampleRate * durationSeconds);
            double[] samples = new double[numSamples];
            
            // Generate a more complex musical signal with multiple notes
            // Simulate a C major chord: C (261.63Hz), E (329.63Hz), G (392.00Hz)
            for (int i = 0; i < numSamples; i++) {
                double t = i / sampleRate;
                // Fundamental frequency (C4 = 261.63Hz) and its harmonics
                samples[i] = 0.3 * Math.sin(2 * Math.PI * 261.63 * t) + 
                            0.25 * Math.sin(2 * Math.PI * 329.63 * t) + 
                            0.2 * Math.sin(2 * Math.PI * 392.00 * t) +
                            0.15 * Math.sin(2 * Math.PI * 523.25 * t) + // C5
                            0.1 * Math.sin(2 * Math.PI * 659.25 * t);   // E5
            }
            
            // Convert to AudioData
            IVector<Double> sampleVector = Linalg.vector(samples);
            AudioData audioData = new AudioData(sampleVector, sampleRate, 1, 16, 
                com.reremouse.lab.audio.core.AudioFormat.WAV);
            
            // Test key detection with improved parameters
            KeyAnalyzerImpl keyAnalyzer = new KeyAnalyzerImpl();
            
            Map<String, Object> parameters = new HashMap<>();
            parameters.put("windowSize", 8192);
            parameters.put("hopSize", 2048);
            
            System.out.println("Detecting key with improved algorithm...");
            KeyDetectionResult result = keyAnalyzer.detectKey(audioData, parameters);
            
            System.out.println("Key Detection Result:");
            System.out.println("Key: " + result.getKeyName());
            System.out.println("Scale: " + result.getScaleType());
            System.out.println("Confidence: " + String.format("%.4f", result.getConfidence()));
            System.out.println("Full result: " + result);
            
            // Also test chroma features directly
            System.out.println("\nAnalyzing chroma features...");
            double[] chromaFeatures = keyAnalyzer.analyzeChromaFeatures(audioData, parameters);
            
            System.out.println("Chroma Features:");
            String[] noteNames = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
            double sum = 0.0;
            for (int i = 0; i < 12; i++) {
                System.out.println(noteNames[i] + ": " + String.format("%.6f", chromaFeatures[i]));
                sum += chromaFeatures[i];
            }
            System.out.println("Sum of chroma features: " + String.format("%.6f", sum));
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}