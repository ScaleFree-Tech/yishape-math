package com.reremouse.lab.music.analysis;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioIO;
import com.reremouse.lab.music.analysis.basic.KeyAnalyzerImpl;
import com.reremouse.lab.music.analysis.basic.KeyDetectionResult;
import com.reremouse.lab.audio.exception.AudioProcessingException;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

/**
 * Test class for KeyAnalyzerImpl
 */
public class KeyAnalyzerTest {

    public static void main(String[] args) {
        try {
            // Create a simple test audio signal (sine wave at 440Hz for 1 second)
            double sampleRate = 44100.0;
            int durationSeconds = 1;
            int numSamples = (int) (sampleRate * durationSeconds);
            double[] samples = new double[numSamples];
            
            // Generate a sine wave at 440Hz (A4)
            for (int i = 0; i < numSamples; i++) {
                samples[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate);
            }
            
            // Convert to AudioData
            com.reremouse.lab.math.linalg.IVector<Double> sampleVector = 
                com.reremouse.lab.math.linalg.Linalg.vector(samples);
            AudioData audioData = new AudioData(sampleVector, sampleRate, 1, 16, 
                com.reremouse.lab.audio.core.AudioFormat.WAV);
            
            // Test key detection
            KeyAnalyzerImpl keyAnalyzer = new KeyAnalyzerImpl();
            
            // Set parameters for better analysis
            Map<String, Object> parameters = new HashMap<>();
            parameters.put("windowSize", 8192);
            parameters.put("hopSize", 2048);
            
            KeyDetectionResult result = keyAnalyzer.detectKey(audioData, parameters);
            
            System.out.println("Key Detection Result:");
            System.out.println("Key: " + result.getKeyName());
            System.out.println("Scale: " + result.getScaleType());
            System.out.println("Confidence: " + String.format("%.2f", result.getConfidence()));
            System.out.println("Chroma Features: " + result);
            
            // Test chroma feature extraction directly
            double[] chromaFeatures = keyAnalyzer.analyzeChromaFeatures(audioData, parameters);
            System.out.println("\nChroma Features:");
            String[] noteNames = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
            for (int i = 0; i < 12; i++) {
                System.out.println(noteNames[i] + ": " + String.format("%.4f", chromaFeatures[i]));
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}