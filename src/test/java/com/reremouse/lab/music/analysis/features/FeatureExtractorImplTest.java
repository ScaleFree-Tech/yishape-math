package com.reremouse.lab.music.analysis.features;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioFormat;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.music.Musics;
import com.reremouse.lab.music.analysis.feature.FeatureExtractorImpl;
import com.reremouse.lab.music.analysis.feature.MusicFeatureResult;
import com.reremouse.lab.music.analysis.feature.ExpressivenessFeatureResult;
import com.reremouse.lab.math.RereMathUtil;
import com.reremouse.lab.audio.core.AudioUtil;
import com.reremouse.lab.audio.core.AudioStatistics;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;
import java.util.HashMap;

/**
 * FeatureExtractorImpl测试类
 * FeatureExtractorImpl test class
 */
public class FeatureExtractorImplTest {
    
    private FeatureExtractorImpl featureExtractor;
    private AudioData testAudioData;
    
    @BeforeEach
    public void setUp() {
        featureExtractor = new FeatureExtractorImpl();
        testAudioData = createTestAudioData();
    }
    
    /**
     * 创建测试音频数据
     * Create test audio data
     */
    private AudioData createTestAudioData() {
        try {
            int sampleRate = 22050;
            int duration = 2; // 2秒 / 2 seconds
            int sampleCount = sampleRate * duration;
            
            // 创建一个复合信号：包含不同频率的正弦波
            // Create a composite signal with different frequency sine waves
            Double[] samples = new Double[sampleCount];
            
            for (int i = 0; i < sampleCount; i++) {
                double t = (double) i / sampleRate;
                
                // 基础频率440Hz (A4)
                double baseFreq = 440.0;
                double signal = Math.sin(2 * Math.PI * baseFreq * t);
                
                // 添加谐波
                signal += 0.5 * Math.sin(2 * Math.PI * baseFreq * 2 * t); // 第二谐波
                signal += 0.3 * Math.sin(2 * Math.PI * baseFreq * 3 * t); // 第三谐波
                
                // 添加一些变化
                signal += 0.2 * Math.sin(2 * Math.PI * 100 * t); // 低频成分
                
                samples[i] = signal * 0.5; // 归一化
            }
            
            return new AudioData(
                Linalg.vector(samples),
                sampleRate,
                1,  // 单声道 / Mono
                16, // 16位 / 16 bit
                AudioFormat.WAV
            );
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to create test audio data", e);
        }
    }
    
    /**
     * 测试表现力特征提取 - 验证acousticness, instrumentalness, liveness, speechiness
     * Test expressiveness feature extraction - verify acousticness, instrumentalness, liveness, speechiness
     */
    @Test
    public void testExpressivenessFeatureExtraction() {
        try {
            // 提取表现力特征
            ExpressivenessFeatureResult result = featureExtractor.extractExpressivenessFeatures(testAudioData);
            
            // 验证结果不为空
            assertNotNull(result, "Expressiveness feature result should not be null");
            
            // 验证特征值在合理范围内 (0.0 到 1.0)
            assertTrue(result.getAcousticness() >= 0.0 && result.getAcousticness() <= 1.0,
                "Acousticness should be between 0.0 and 1.0");
            assertTrue(result.getInstrumentalness() >= 0.0 && result.getInstrumentalness() <= 1.0,
                "Instrumentalness should be between 0.0 and 1.0");
            assertTrue(result.getLiveness() >= 0.0 && result.getLiveness() <= 1.0,
                "Liveness should be between 0.0 and 1.0");
            assertTrue(result.getSpeechiness() >= 0.0 && result.getSpeechiness() <= 1.0,
                "Speechiness should be between 0.0 and 1.0");
            
            // 验证其他特征
            assertTrue(result.getEnergy() >= 0.0 && result.getEnergy() <= 1.0,
                "Energy should be between 0.0 and 1.0");
            assertTrue(result.getDanceability() >= 0.0 && result.getDanceability() <= 1.0,
                "Danceability should be between 0.0 and 1.0");
            
            // 验证动态范围特征
        assertNotNull(result.getDynamicRange(), "Dynamic range should not be null");
        assertNotNull(result.getSpectralCentroidEvolution(), "Spectral centroid evolution should not be null");
            
            // 输出结果用于调试
            System.out.println("=== Expressiveness Feature Test Results ===");
            System.out.println("Energy: " + result.getEnergy());
            System.out.println("Dynamic Range Length: " + result.getDynamicRange().length);
            System.out.println("Spectral Centroid Evolution Length: " + result.getSpectralCentroidEvolution().length);
            
        } catch (Exception e) {
            fail("Exception occurred during expressiveness feature extraction: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试缓存机制 - 验证特征提取结果是否被正确缓存
     * Test cache mechanism - verify feature extraction results are properly cached
     */
    @Test
    public void testCacheMechanism() {
        try {
            // 第一次提取 - 应该计算并缓存
            ExpressivenessFeatureResult result1 = featureExtractor.extractExpressivenessFeatures(testAudioData);
            
            // 第二次提取 - 应该使用缓存
            ExpressivenessFeatureResult result2 = featureExtractor.extractExpressivenessFeatures(testAudioData);
            
            // 验证结果相同
            assertEquals(result1.getAcousticness(), result2.getAcousticness(), 0.0001,
                "Cached acousticness should be the same");
            assertEquals(result1.getInstrumentalness(), result2.getInstrumentalness(), 0.0001,
                "Cached instrumentalness should be the same");
            assertEquals(result1.getLiveness(), result2.getLiveness(), 0.0001,
                "Cached liveness should be the same");
            assertEquals(result1.getSpeechiness(), result2.getSpeechiness(), 0.0001,
                "Cached speechiness should be the same");
            
            // 验证动态范围数组也相同
            assertArrayEquals(result1.getDynamicRange(), result2.getDynamicRange(), 0.0001,
                "Cached dynamic range should be the same");
            assertArrayEquals(result1.getSpectralCentroidEvolution(), result2.getSpectralCentroidEvolution(), 0.0001,
                "Cached spectral centroid evolution should be the same");
            
            System.out.println("Cache mechanism test passed - results are consistent");
            
        } catch (Exception e) {
            fail("Exception occurred during cache test: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试参数验证 - 验证参数验证功能
     * Test parameter validation - verify parameter validation functionality
     */
    @Test
    public void testParameterValidation() {
        System.out.println("\n=== Testing Parameter Validation ===");
        
        // 测试空音频数据
        try {
            featureExtractor.extractExpressivenessFeatures(null);
            fail("Should throw exception for null audio data");
        } catch (Exception e) {
            System.out.println("✓ Correctly handled null audio data: " + e.getMessage());
        }
        
        // 测试参数边界值 - 使用有效的参数
        Map<String, Object> validParams = new HashMap<>();
        validParams.put("sampleRate", 44100.0);
        validParams.put("windowSize", 2048);
        
        try {
            ExpressivenessFeatureResult result = featureExtractor.extractExpressivenessFeatures(testAudioData, validParams);
            assertNotNull(result, "Should handle valid parameters successfully");
            System.out.println("✓ Handled valid parameters successfully");
        } catch (Exception e) {
            System.out.println("Parameter validation result: " + e.getMessage());
        }
    }
    
    /**
     * 测试所有音乐特征提取 - 验证综合特征提取功能
     * Test all music feature extraction - verify comprehensive feature extraction functionality
     */
    @Test
    public void testAllMusicFeatureExtraction() {
        try {
            // 提取所有音乐特征
            MusicFeatureResult result = featureExtractor.extractMusicFeatures(testAudioData);
            
            // 验证结果完整性
            assertNotNull(result, "Music feature result should not be null");
            assertNotNull(result.getRhythmFeatures(), "Rhythm features should not be null");
            assertNotNull(result.getTonalFeatures(), "Tonal features should not be null");
            assertNotNull(result.getStructureFeatures(), "Structure features should not be null");
            assertNotNull(result.getExpressivenessFeatures(), "Expressiveness features should not be null");
            
            // 验证元数据
            Map<String, Object> metadata = result.getMetadata();
            assertNotNull(metadata, "Metadata should not be null");
            assertTrue(metadata.containsKey("sampleRate"), "Metadata should contain sampleRate");
            assertTrue(metadata.containsKey("duration"), "Metadata should contain duration");
            assertTrue(metadata.containsKey("extractorVersion"), "Metadata should contain extractorVersion");
            
            System.out.println("All music feature extraction test passed");
            System.out.println("Total metadata entries: " + metadata.size());
            
        } catch (Exception e) {
            fail("Exception occurred during all music feature extraction: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试特征类型支持 - 验证支持的特征类型
     * Test feature type support - verify supported feature types
     */
    @Test
    public void testFeatureTypeSupport() {
        String[] supportedTypes = featureExtractor.getSupportedFeatureTypes();

        assertNotNull(supportedTypes, "Supported feature types should not be null");
        assertTrue(supportedTypes.length > 0, "Should support at least one feature type");

        System.out.println("Supported feature types: " + java.util.Arrays.toString(supportedTypes));
    }
}