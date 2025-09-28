package com.reremouse.lab.music.analysis.features;

import com.reremouse.lab.music.analysis.feature.FeatureExtractorImpl;
import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioFormat;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.music.Musics;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;

/**
 * MFCC调试测试类
 * MFCC debug test class
 */
public class MFCCDebugTest {
    
    /**
     * 调试MFCC特征提取功能
     * Debug MFCC feature extraction functionality
     */
    @Test
    public void debugMFCCFeatureExtraction() {
        try {
            System.out.println("Starting MFCC debug test...");
            
            // 创建测试音频数据 / Create test audio data
            int sampleRate = 22050;
            int duration = 1; // 1秒 / 1 second
            int sampleCount = sampleRate * duration;
            
            // 创建一个简单的正弦波信号 / Create a simple sine wave signal
            Double[] samples = new Double[sampleCount];
            double frequency = 440.0; // A4音符 / A4 note
            for (int i = 0; i < sampleCount; i++) {
                samples[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate);
            }
            
            AudioData audioData = new AudioData(
                Linalg.vector(samples),
                sampleRate,
                1,  // 单声道 / Mono
                16, // 16位 / 16 bit
                AudioFormat.WAV
            );
            
            System.out.println("Audio data created successfully");
            
            // 直接测试FeatureExtractorImpl
            FeatureExtractorImpl extractor = new FeatureExtractorImpl();
            System.out.println("FeatureExtractorImpl created successfully");
            
            // 测试基本的音乐特征提取
            try {
                var result = extractor.extractMusicFeatures(audioData);
                System.out.println("Basic music feature extraction successful");
                System.out.println("Expressiveness features: " + (result.getExpressivenessFeatures() != null));
                if (result.getExpressivenessFeatures() != null) {
                    System.out.println("Timbre variation: " + result.getExpressivenessFeatures().getTimbreVariation());
                }
            } catch (Exception e) {
                System.err.println("Error in basic music feature extraction: " + e.getMessage());
                e.printStackTrace();
            }
            
            // 测试表现力特征提取
            try {
                var expressivenessResult = extractor.extractExpressivenessFeatures(audioData);
                System.out.println("Expressiveness feature extraction successful");
                System.out.println("Timbre variation: " + expressivenessResult.getTimbreVariation());
            } catch (Exception e) {
                System.err.println("Error in expressiveness feature extraction: " + e.getMessage());
                e.printStackTrace();
            }
            
            // 测试Musics类的MFCC提取
            try {
                Map<String, Object> mfccFeatures = Musics.extractMFCCFeatures(audioData);
                System.out.println("MFCC feature extraction via Musics class successful");
                System.out.println("MFCC features map size: " + mfccFeatures.size());
                System.out.println("MFCC features keys: " + mfccFeatures.keySet());
            } catch (Exception e) {
                System.err.println("Error in MFCC feature extraction via Musics class: " + e.getMessage());
                e.printStackTrace();
            }
            
        } catch (Exception e) {
            System.err.println("Unexpected error in debug test: " + e.getMessage());
            e.printStackTrace();
        }
    }
}