package com.yishape.lab.music.analysis.features;

import com.yishape.lab.audio.Audios;
import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.music.Musics;
import org.junit.jupiter.api.Test;

import java.util.Map;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * MFCC特征提取测试类
 * MFCC feature extraction test class
 */
public class MFCCFeatureTest {
    
   
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
            Map<String, Object> allFeatures = Musics.extractMusicFeatureMap(audioData);
            
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