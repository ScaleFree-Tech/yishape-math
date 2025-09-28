package com.reremouse.lab.music.analysis;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.music.analysis.basic.KeyAnalyzerImpl;
import com.reremouse.lab.music.analysis.basic.KeyDetectionResult;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IVector;

import java.util.HashMap;
import java.util.Map;

/**
 * Simple test to verify key detection improvements
 */
public class KeyDetectionImprovementTest {

    public static void main(String[] args) {
        try {
            System.out.println("Testing Key Detection Improvements...");
            
            // Create a test signal with a clear pitch (A4 = 440Hz)
            double sampleRate = 44100.0;
            int durationSeconds = 2;
            int numSamples = (int) (sampleRate * durationSeconds);
            double[] samples = new double[numSamples];
            
            // Generate a combination of sine waves to simulate a musical key
            // Fundamental frequency (A4 = 440Hz) and its harmonics
            for (int i = 0; i < numSamples; i++) {
                double t = i / sampleRate;
                // A major chord: A (440Hz), C# (554.37Hz), E (659.25Hz)
                samples[i] = Math.sin(2 * Math.PI * 440 * t) + 
                            0.8 * Math.sin(2 * Math.PI * 554.37 * t) + 
                            0.6 * Math.sin(2 * Math.PI * 659.25 * t);
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
            for (int i = 0; i < 12; i++) {
                System.out.println(noteNames[i] + ": " + String.format("%.6f", chromaFeatures[i]));
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}