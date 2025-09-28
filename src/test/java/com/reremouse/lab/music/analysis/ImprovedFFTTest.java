package com.reremouse.lab.music.analysis;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.music.analysis.basic.KeyAnalyzerImpl;
import com.reremouse.lab.music.analysis.basic.KeyDetectionResult;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IVector;

import java.util.HashMap;
import java.util.Map;

/**
 * Improved test to verify FFT and chroma feature improvements
 */
public class ImprovedFFTTest {

    public static void main(String[] args) {
        try {
            System.out.println("Testing Improved FFT and Chroma Feature Improvements...");
            
            // Create a test signal with multiple pitches (C major chord: C4, E4, G4)
            double sampleRate = 44100.0;
            int durationSeconds = 3;
            int numSamples = (int) (sampleRate * durationSeconds);
            double[] samples = new double[numSamples];
            
            // Generate a C major chord (C4=261.63Hz, E4=329.63Hz, G4=392.00Hz)
            for (int i = 0; i < numSamples; i++) {
                double t = i / sampleRate;
                // Combine multiple sine waves
                samples[i] = 0.3 * Math.sin(2 * Math.PI * 261.63 * t) +  // C4
                             0.3 * Math.sin(2 * Math.PI * 329.63 * t) +  // E4
                             0.3 * Math.sin(2 * Math.PI * 392.00 * t);   // G4
            }
            
            // Convert to AudioData
            IVector<Double> sampleVector = Linalg.vector(samples);
            AudioData audioData = new AudioData(sampleVector, sampleRate, 1, 16, 
                com.reremouse.lab.audio.core.AudioFormat.WAV);
            
            // Test key detection with improved parameters
            KeyAnalyzerImpl keyAnalyzer = new KeyAnalyzerImpl();
            
            Map<String, Object> parameters = new HashMap<>();
            parameters.put("windowSize", 8192); // Use a power of 2
            parameters.put("hopSize", 2048);
            
            System.out.println("Analyzing chroma features...");
            double[] chromaFeatures = keyAnalyzer.analyzeChromaFeatures(audioData, parameters);
            
            System.out.println("Chroma Features:");
            String[] noteNames = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
            double sum = 0.0;
            for (int i = 0; i < 12; i++) {
                System.out.println(noteNames[i] + ": " + String.format("%.6f", chromaFeatures[i]));
                sum += chromaFeatures[i];
            }
            System.out.println("Sum of chroma features: " + String.format("%.6f", sum));
            
            // Check if chroma features are non-uniform
            double avg = sum / 12;
            double variance = 0.0;
            for (int i = 0; i < 12; i++) {
                variance += (chromaFeatures[i] - avg) * (chromaFeatures[i] - avg);
            }
            variance /= 12;
            System.out.println("Variance of chroma features: " + String.format("%.6f", variance));
            
            // Test key detection
            System.out.println("\nDetecting key...");
            KeyDetectionResult result = keyAnalyzer.detectKey(audioData, parameters);
            
            System.out.println("Key Detection Result:");
            System.out.println("Key: " + result.getKeyName());
            System.out.println("Scale: " + result.getScaleType());
            System.out.println("Confidence: " + String.format("%.4f", result.getConfidence()));
            
            // Print chroma features from result
            double[] resultChroma = result.getChromaFeatures();
            if (resultChroma != null) {
                System.out.println("Chroma features from result:");
                for (int i = 0; i < 12; i++) {
                    System.out.println(noteNames[i] + ": " + String.format("%.6f", resultChroma[i]));
                }
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}