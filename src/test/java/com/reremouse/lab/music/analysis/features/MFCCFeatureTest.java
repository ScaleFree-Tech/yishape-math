package com.reremouse.lab.music.analysis.features;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioFormat;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.music.Musics;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;

/**
 * MFCC特征提取测试类
 * MFCC feature extraction test class
 */
public class MFCCFeatureTest {
    
    /**
     * 测试MFCC特征提取功能
     * Test MFCC feature extraction functionality
     */
    @Test
    public void testMFCCFeatureExtraction() {
        try {
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
            
            // 提取MFCC特征 / Extract MFCC features
            Map<String, Object> mfccFeatures = Musics.extractMFCCFeatures(audioData);
            
            // 验证输出 / Verify output
            assertNotNull(mfccFeatures, "MFCC features should not be null");
            assertTrue(mfccFeatures.containsKey("mfcc"), "Should contain MFCC features");
            
            System.out.println("MFCC feature extraction successful");
            System.out.println("MFCC features map size: " + mfccFeatures.size());
            
        } catch (Exception e) {
            fail("Exception occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试音乐特征提取中的MFCC功能
     * Test MFCC functionality in music feature extraction
     */
    @Test
    public void testMusicFeatureExtractionWithMFCC() {
        try {
            // 创建测试音频数据 / Create test audio data
            int sampleRate = 22050;
            int duration = 2; // 2秒 / 2 seconds
            int sampleCount = sampleRate * duration;
            
            // 创建一个扫频信号 / Create a sweep signal
            Double[] samples = new Double[sampleCount];
            for (int i = 0; i < sampleCount; i++) {
                double t = (double) i / sampleRate;
                double frequency = 100 + (1000 - 100) * t / duration; // 100Hz到1000Hz的扫频 / Sweep from 100Hz to 1000Hz
                samples[i] = Math.sin(2 * Math.PI * frequency * t);
            }
            
            AudioData audioData = new AudioData(
                Linalg.vector(samples),
                sampleRate,
                1,  // 单声道 / Mono
                16, // 16位 / 16 bit
                AudioFormat.WAV
            );
            
            // 提取所有音乐特征 / Extract all music features
            Map<String, Object> allFeatures = Musics.extractMusicFeatures(audioData);
            
            // 验证输出 / Verify output
            assertNotNull(allFeatures, "All features should not be null");
            assertTrue(allFeatures.containsKey("expressiveness"), "Should contain expressiveness features");
            
            // 检查表现力特征中的MFCC相关特征 / Check MFCC-related features in expressiveness features
            @SuppressWarnings("unchecked")
            Map<String, Object> expressivenessFeatures = (Map<String, Object>) allFeatures.get("expressiveness");
            assertNotNull(expressivenessFeatures, "Expressiveness features should not be null");
            
            System.out.println("Music feature extraction with MFCC successful");
            System.out.println("Total features extracted: " + allFeatures.size());
            System.out.println("Expressiveness features: " + expressivenessFeatures.size());
            
        } catch (Exception e) {
            fail("Exception occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }
}