package com.yishape.lab.audio.factory;

import com.yishape.lab.audio.feature.IAudioFeatureExtractor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 音频组件工厂测试类 / Audio Component Factory Test Class
 * <p>
 * 测试音频组件工厂的功能，包括新添加的特征提取器功能。
 * Tests the functionality of the audio component factory, including the newly added feature extractor functionality.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class AudioComponentFactoryTest {

    /**
     * 测试创建音频特征提取器 / Test creating audio feature extractor
     */
    @Test
    public void testCreateFeatureExtractor() {
        AudioComponentFactory factory = AudioComponentFactory.getInstance();
        
        // 测试创建默认特征提取器 / Test creating default feature extractor
        IAudioFeatureExtractor extractor = factory.createFeatureExtractor("default");
        assertNotNull(extractor, "Extractor should not be null");
        assertEquals("AudioFeatureExtractorImpl", extractor.getExtractorName(), "Extractor name should match");
        assertEquals("2.0", extractor.getVersion(), "Version should match");
        
        // 测试创建标准特征提取器 / Test creating standard feature extractor
        IAudioFeatureExtractor standardExtractor = factory.createFeatureExtractor("standard");
        assertNotNull(standardExtractor, "Standard extractor should not be null");
        assertEquals("AudioFeatureExtractorImpl", standardExtractor.getExtractorName(), "Standard extractor name should match");
    }

    /**
     * 测试创建不存在的音频特征提取器 / Test creating non-existent audio feature extractor
     */
    @Test
    public void testCreateNonExistentFeatureExtractor() {
        AudioComponentFactory factory = AudioComponentFactory.getInstance();
        
        // 测试创建不存在的特征提取器 / Test creating non-existent feature extractor
        assertThrows(IllegalArgumentException.class, () -> {
            factory.createFeatureExtractor("nonexistent");
        }, "Should throw IllegalArgumentException for non-existent extractor");
    }

    /**
     * 测试组件类型枚举 / Test component type enum
     */
    @Test
    public void testComponentTypeEnum() {
        AudioComponentFactory.ComponentType[] types = AudioComponentFactory.ComponentType.values();
        assertNotNull(types, "Component types should not be null");
        assertTrue(types.length > 0, "Should have at least one component type");
        
        // 检查是否包含FEATURE_EXTRACTOR类型 / Check if FEATURE_EXTRACTOR type is included
        boolean hasFeatureExtractorType = false;
        for (AudioComponentFactory.ComponentType type : types) {
            if (type == AudioComponentFactory.ComponentType.FEATURE_EXTRACTOR) {
                hasFeatureExtractorType = true;
                break;
            }
        }
        assertTrue(hasFeatureExtractorType, "Should include FEATURE_EXTRACTOR type");
    }
}