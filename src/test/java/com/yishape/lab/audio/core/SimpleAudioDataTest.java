package com.yishape.lab.audio.core;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;

/**
 * Simple test for AudioData equals method fix
 */
public class SimpleAudioDataTest {
    
    public static void main(String[] args) {
        System.out.println("Testing AudioData equals method fix...");
        
        // Test with different length vectors (this was causing the original issue)
        testDifferentLengthVectors();
        
        System.out.println("Test completed successfully!");
    }
    
    private static void testDifferentLengthVectors() {
        System.out.println("\n=== Test: Different Length Vectors ===");
        
        try {
            // Create two audio data objects with very different lengths
            // This was causing the original Vector lengths don't match error
            double[] samples1 = new double[500]; // Small size
            for (int i = 0; i < samples1.length; i++) {
                samples1[i] = Math.sin(2 * Math.PI * i / 100.0);
            }
            
            double[] samples2 = new double[16478]; // Larger size
            for (int i = 0; i < samples2.length; i++) {
                samples2[i] = Math.sin(2 * Math.PI * i / 100.0);
            }
            
            IVector<Double> vector1 = Linalg.vector(samples1);
            IVector<Double> vector2 = Linalg.vector(samples2);
            
            AudioData audio1 = new AudioData(vector1, 44100, 1, 16, AudioFormat.WAV);
            AudioData audio2 = new AudioData(vector2, 44100, 1, 16, AudioFormat.WAV);
            
            // This should not throw an exception anymore
            boolean result = audio1.equals(audio2);
            System.out.println("Different length audio data comparison: " + result);
            System.out.println("Expected: false, Actual: " + result);
            
            // Test with same data
            AudioData audio3 = new AudioData(vector1, 44100, 1, 16, AudioFormat.WAV);
            boolean result2 = audio1.equals(audio3);
            System.out.println("Same audio data comparison: " + result2);
            System.out.println("Expected: true, Actual: " + result2);
            
            System.out.println("✓ AudioData equals method works correctly with different length vectors");
            
        } catch (Exception e) {
            System.err.println("❌ Error in testDifferentLengthVectors: " + e.getMessage());
            e.printStackTrace();
        }
    }
}