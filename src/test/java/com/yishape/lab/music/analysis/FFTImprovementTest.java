package com.yishape.lab.music.analysis;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.music.analysis.basic.KeyAnalyzerImpl;
import com.yishape.lab.music.analysis.basic.KeyDetectionResult;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;

import java.util.HashMap;
import java.util.Map;

/**
 * Simple test to verify FFT and chroma feature improvements
 */
public class FFTImprovementTest {

    public static void main(String[] args) {
        try {
            System.out.println("Testing FFT and Chroma Feature Improvements...");
            
            // Create a test signal with a clear pitch (A4 = 440Hz)
            double sampleRate = 44100.0;
            int durationSeconds = 2;
            int numSamples = (int) (sampleRate * durationSeconds);
            double[] samples = new double[numSamples];
            
            // Generate a pure sine wave at 440Hz
            for (int i = 0; i < numSamples; i++) {
                double t = i / sampleRate;
                samples[i] = Math.sin(2 * Math.PI * 440 * t);
            }
            
            // Convert to AudioData
            IVector<Double> sampleVector = Linalg.vector(samples);
            AudioData audioData = new AudioData(sampleVector, sampleRate, 1, 16, 
                AudioFormat.WAV);
            
            // Test key detection with improved parameters
            KeyAnalyzerImpl keyAnalyzer = new KeyAnalyzerImpl();
            
            Map<String, Object> parameters = new HashMap<>();
            parameters.put("windowSize", 4096); // Use a power of 2
            parameters.put("hopSize", 1024);
            
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
            
            // Test key detection
            System.out.println("\nDetecting key...");
            KeyDetectionResult result = keyAnalyzer.detectKey(audioData, parameters);
            
            System.out.println("Key Detection Result:");
            System.out.println("Key: " + result.getKeyName());
            System.out.println("Scale: " + result.getScaleType());
            System.out.println("Confidence: " + String.format("%.4f", result.getConfidence()));
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}