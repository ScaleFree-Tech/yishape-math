package com.yishape.lab.music.analysis.feature;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;

/**
 * Test for FeatureExtractorImpl cache mechanism improvements
 */
public class FeatureExtractorCacheTest {
    
    public static void main(String[] args) {
        System.out.println("Testing FeatureExtractorImpl cache mechanism improvements...");
        
        // Test 1: Cache with same audio data
        testCacheWithSameAudioData();
        
        // Test 2: Cache with different audio data
        testCacheWithDifferentAudioData();
        
        // Test 3: Cache with large audio data
        testCacheWithLargeAudioData();
        
        // Test 4: Cache clearing
        testCacheClearing();
        
        System.out.println("All cache tests completed.");
    }
    
    private static void testCacheWithSameAudioData() {
        System.out.println("\n=== Test 1: Cache with Same Audio Data ===");
        
        try {
            FeatureExtractorImpl extractor = new FeatureExtractorImpl();
            
            // Create audio data
            double[] samples = new double[1000];
            for (int i = 0; i < samples.length; i++) {
                samples[i] = Math.sin(2 * Math.PI * i / 100.0);
            }
            
            IVector<Double> vector = Linalg.vector(samples);
            AudioData audioData1 = new AudioData(vector, 44100, 1, 16, AudioFormat.WAV);
            AudioData audioData2 = new AudioData(vector, 44100, 1, 16, AudioFormat.WAV);
            
            // Extract features twice
            long startTime1 = System.currentTimeMillis();
            var result1 = extractor.extractRhythmFeatures(audioData1);
            long endTime1 = System.currentTimeMillis();
            
            long startTime2 = System.currentTimeMillis();
            var result2 = extractor.extractRhythmFeatures(audioData2);
            long endTime2 = System.currentTimeMillis();
            
            System.out.println("First extraction time: " + (endTime1 - startTime1) + " ms");
            System.out.println("Second extraction time: " + (endTime2 - startTime2) + " ms");
            System.out.println("Cache hit: " + (endTime2 - startTime2 < endTime1 - startTime1));
            
        } catch (Exception e) {
            System.err.println("Error in testCacheWithSameAudioData: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testCacheWithDifferentAudioData() {
        System.out.println("\n=== Test 2: Cache with Different Audio Data ===");
        
        try {
            FeatureExtractorImpl extractor = new FeatureExtractorImpl();
            
            // Create two different audio data objects
            double[] samples1 = new double[1000];
            double[] samples2 = new double[1000];
            
            for (int i = 0; i < samples1.length; i++) {
                samples1[i] = Math.sin(2 * Math.PI * i / 100.0);
                samples2[i] = Math.cos(2 * Math.PI * i / 100.0);
            }
            
            IVector<Double> vector1 = Linalg.vector(samples1);
            IVector<Double> vector2 = Linalg.vector(samples2);
            
            AudioData audioData1 = new AudioData(vector1, 44100, 1, 16, AudioFormat.WAV);
            AudioData audioData2 = new AudioData(vector2, 44100, 1, 16, AudioFormat.WAV);
            
            // Extract features for both
            long startTime1 = System.currentTimeMillis();
            var result1 = extractor.extractRhythmFeatures(audioData1);
            long endTime1 = System.currentTimeMillis();
            
            long startTime2 = System.currentTimeMillis();
            var result2 = extractor.extractRhythmFeatures(audioData2);
            long endTime2 = System.currentTimeMillis();
            
            System.out.println("First extraction time: " + (endTime1 - startTime1) + " ms");
            System.out.println("Second extraction time: " + (endTime2 - startTime2) + " ms");
            System.out.println("Different audio data should not use cache");
            
        } catch (Exception e) {
            System.err.println("Error in testCacheWithDifferentAudioData: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testCacheWithLargeAudioData() {
        System.out.println("\n=== Test 3: Cache with Large Audio Data ===");
        
        try {
            FeatureExtractorImpl extractor = new FeatureExtractorImpl();
            
            // Create large audio data (this was causing issues before)
            double[] samples1 = new double[500000]; // Cache size
            double[] samples2 = new double[500000]; // Same size
            
            for (int i = 0; i < samples1.length; i++) {
                samples1[i] = Math.sin(2 * Math.PI * i / 1000.0);
                samples2[i] = Math.sin(2 * Math.PI * i / 1000.0);
            }
            
            IVector<Double> vector1 = Linalg.vector(samples1);
            IVector<Double> vector2 = Linalg.vector(samples2);
            
            AudioData audioData1 = new AudioData(vector1, 44100, 1, 16, AudioFormat.WAV);
            AudioData audioData2 = new AudioData(vector2, 44100, 1, 16, AudioFormat.WAV);
            
            // This should work without throwing vector length mismatch exceptions
            long startTime1 = System.currentTimeMillis();
            var result1 = extractor.extractRhythmFeatures(audioData1);
            long endTime1 = System.currentTimeMillis();
            
            long startTime2 = System.currentTimeMillis();
            var result2 = extractor.extractRhythmFeatures(audioData2);
            long endTime2 = System.currentTimeMillis();
            
            System.out.println("Large audio first extraction: " + (endTime1 - startTime1) + " ms");
            System.out.println("Large audio second extraction: " + (endTime2 - startTime2) + " ms");
            System.out.println("Cache should work with large audio data now");
            
        } catch (Exception e) {
            System.err.println("Error in testCacheWithLargeAudioData: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testCacheClearing() {
        System.out.println("\n=== Test 4: Cache Clearing ===");
        
        try {
            FeatureExtractorImpl extractor = new FeatureExtractorImpl();
            
            // Create audio data
            double[] samples = new double[1000];
            for (int i = 0; i < samples.length; i++) {
                samples[i] = Math.sin(2 * Math.PI * i / 100.0);
            }
            
            IVector<Double> vector = Linalg.vector(samples);
            AudioData audioData = new AudioData(vector, 44100, 1, 16, AudioFormat.WAV);
            
            // Extract features
            var result1 = extractor.extractRhythmFeatures(audioData);
            
            // Clear cache
            extractor.clearCache();
            
            // Extract features again - should not use cache
            long startTime = System.currentTimeMillis();
            var result2 = extractor.extractRhythmFeatures(audioData);
            long endTime = System.currentTimeMillis();
            
            System.out.println("Extraction after cache clear: " + (endTime - startTime) + " ms");
            System.out.println("Cache should be cleared and not used");
            
        } catch (Exception e) {
            System.err.println("Error in testCacheClearing: " + e.getMessage());
            e.printStackTrace();
        }
    }
}