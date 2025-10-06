package com.yishape.lab.audio.core;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;

/**
 * Test for AudioData equals method fix
 */
public class AudioDataEqualsTest {
    
    public static void main(String[] args) {
        System.out.println("Testing AudioData equals method fix...");
        
        // Test 1: Same audio data
        testSameAudioData();
        
        // Test 2: Different length audio data (this was causing the issue)
        testDifferentLengthAudioData();
        
        // Test 3: Same length but different content
        testSameLengthDifferentContent();
        
        System.out.println("All tests completed.");
    }
    
    private static void testSameAudioData() {
        System.out.println("\n=== Test 1: Same Audio Data ===");
        
        try {
            // Create two identical audio data objects
            double[] samples1 = new double[1000];
            for (int i = 0; i < samples1.length; i++) {
                samples1[i] = Math.sin(2 * Math.PI * i / 100.0);
            }
            
            IVector<Double> vector1 = Linalg.vector(samples1);
            AudioData audio1 = new AudioData(vector1, 44100, 1, 16, AudioFormat.WAV);
            AudioData audio2 = new AudioData(vector1, 44100, 1, 16, AudioFormat.WAV);
            
            boolean result = audio1.equals(audio2);
            System.out.println("Same audio data comparison: " + result);
            System.out.println("Expected: true, Actual: " + result);
            
        } catch (Exception e) {
            System.err.println("Error in testSameAudioData: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testDifferentLengthAudioData() {
        System.out.println("\n=== Test 2: Different Length Audio Data ===");
        
        try {
            // Create two audio data objects with different lengths
            // This was causing the original issue
            double[] samples1 = new double[500000]; // Cache size
            for (int i = 0; i < samples1.length; i++) {
                samples1[i] = Math.sin(2 * Math.PI * i / 1000.0);
            }
            
            double[] samples2 = new double[16478208]; // New size
            for (int i = 0; i < samples2.length; i++) {
                samples2[i] = Math.sin(2 * Math.PI * i / 1000.0);
            }
            
            IVector<Double> vector1 = Linalg.vector(samples1);
            IVector<Double> vector2 = Linalg.vector(samples2);
            
            AudioData audio1 = new AudioData(vector1, 44100, 1, 16, AudioFormat.WAV);
            AudioData audio2 = new AudioData(vector2, 44100, 1, 16, AudioFormat.WAV);
            
            // This should not throw an exception anymore
            boolean result = audio1.equals(audio2);
            System.out.println("Different length audio data comparison: " + result);
            System.out.println("Expected: false, Actual: " + result);
            
        } catch (Exception e) {
            System.err.println("Error in testDifferentLengthAudioData: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testSameLengthDifferentContent() {
        System.out.println("\n=== Test 3: Same Length, Different Content ===");
        
        try {
            // Create two audio data objects with same length but different content
            double[] samples1 = new double[1000];
            double[] samples2 = new double[1000];
            
            for (int i = 0; i < samples1.length; i++) {
                samples1[i] = Math.sin(2 * Math.PI * i / 100.0);
                samples2[i] = Math.cos(2 * Math.PI * i / 100.0);
            }
            
            IVector<Double> vector1 = Linalg.vector(samples1);
            IVector<Double> vector2 = Linalg.vector(samples2);
            
            AudioData audio1 = new AudioData(vector1, 44100, 1, 16, AudioFormat.WAV);
            AudioData audio2 = new AudioData(vector2, 44100, 1, 16, AudioFormat.WAV);
            
            boolean result = audio1.equals(audio2);
            System.out.println("Same length, different content comparison: " + result);
            System.out.println("Expected: false, Actual: " + result);
            
        } catch (Exception e) {
            System.err.println("Error in testSameLengthDifferentContent: " + e.getMessage());
            e.printStackTrace();
        }
    }
}