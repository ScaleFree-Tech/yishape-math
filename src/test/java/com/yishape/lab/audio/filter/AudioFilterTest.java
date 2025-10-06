package com.yishape.lab.audio.filter;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 音频滤波器测试类 / Audio Filter Test Class
 * <p>
 * 测试音频滤波器接口及其实现类的功能。
 * Tests the functionality of audio filter interfaces and their implementations.
 * </p>
 */
public class AudioFilterTest {
    
    /**
     * 测试低通滤波器基本功能 / Test low-pass filter basic functionality
     */
    @Test
    public void testLowPassFilter() throws AudioProcessingException {
        // 创建测试音频数据 / Create test audio data
        IVector<Double> samples = Linalg.linspace(0.0, 1000.0, 100);
        AudioData audioData = new AudioData(samples, 44100.0, 1, 16, AudioFormat.PCM);
        
        // 创建低通滤波器 / Create low-pass filter
        LowPassFilter filter = new LowPassFilter();
        filter.setCutoffFrequency(1000);
        filter.setOrder(4);
        
        // 应用滤波器 / Apply filter
        AudioData filteredAudio = filter.filter(audioData);
        
        // 验证结果 / Verify results
        assertNotNull(filteredAudio, "Filtered audio should not be null");
        assertEquals(audioData.getSampleRate(), filteredAudio.getSampleRate(), 0.001, "Sample rate should be preserved");
        assertEquals(audioData.getChannels(), filteredAudio.getChannels(), "Channels should be preserved");
    }
    
    /**
     * 测试高级低通滤波器功能 / Test advanced low-pass filter functionality
     */
    @Test
    public void testAdvancedLowPassFilter() throws AudioProcessingException {
        // 创建测试音频数据 / Create test audio data
        IVector<Double> samples = Linalg.linspace(0.0, 1000.0, 100);
        AudioData audioData = new AudioData(samples, 44100.0, 1, 16, AudioFormat.PCM);
        
        // 创建高级低通滤波器 / Create advanced low-pass filter
        AdvancedLowPassFilter filter = new AdvancedLowPassFilter();
        
        // 使用参数映射来设置滤波器参数 / Use parameter map to set filter parameters
        java.util.Map<String, Object> params = new java.util.HashMap<>();
        params.put("cutoffFrequency", 1000.0);
        params.put("order", 4);
        params.put("qualityFactor", 1.5);
        params.put("bandwidth", 2000.0);
        
        // 应用滤波器 / Apply filter
        AudioData filteredAudio = filter.filter(audioData, params);
        
        // 验证结果 / Verify results
        assertNotNull(filteredAudio, "Filtered audio should not be null");
        assertEquals(audioData.getSampleRate(), filteredAudio.getSampleRate(), 0.001, "Sample rate should be preserved");
        assertEquals(audioData.getChannels(), filteredAudio.getChannels(), "Channels should be preserved");
        
        // 验证参数是否正确设置 / Verify parameters are set correctly
        assertEquals(1.5, filter.getQualityFactor(), 0.001, "Quality factor should be set");
        assertEquals(2000, filter.getBandwidth(), 0.001, "Bandwidth should be set");
    }
    
    /**
     * 测试滤波器参数管理 / Test filter parameter management
     */
    @Test
    public void testFilterParameterManagement() {
        LowPassFilter filter = new LowPassFilter();
        
        // 测试设置和获取参数 / Test setting and getting parameters
        filter.setCutoffFrequency(2000);
        filter.setOrder(6);
        filter.setFilterType(IBaseAudioFilter.FilterType.HIGH_PASS);
        
        assertEquals(2000, filter.getCutoffFrequency(), 0.001, "Cutoff frequency should be set");
        assertEquals(6, filter.getOrder(), "Order should be set");
        assertEquals(IBaseAudioFilter.FilterType.HIGH_PASS, filter.getFilterType(), "Filter type should be set");
        
        // 测试通过IAudioProcessor接口设置参数 / Test setting parameters through IAudioProcessor interface
        filter.setParameter("cutoffFrequency", 3000.0);
        filter.setParameter("order", 8);
        
        assertEquals(3000, filter.getCutoffFrequency(), 0.001, "Cutoff frequency should be updated");
        assertEquals(8, filter.getOrder(), "Order should be updated");
    }
    
    /**
     * 测试滤波器克隆功能 / Test filter cloning functionality
     */
    @Test
    public void testFilterCloning() {
        LowPassFilter filter = new LowPassFilter();
        filter.setCutoffFrequency(1500);
        filter.setOrder(5);
        
        // 克隆滤波器 / Clone filter
        IBaseAudioFilter clonedFilter = filter.clone();
        
        // 验证克隆 / Verify clone
        assertNotNull(clonedFilter, "Cloned filter should not be null");
        assertNotSame(filter, clonedFilter, "Cloned filter should be a different object");
        assertEquals(1500, clonedFilter.getCutoffFrequency(), 0.001, "Cutoff frequency should be preserved");
        assertEquals(5, clonedFilter.getOrder(), "Order should be preserved");
    }
    
    /**
     * 测试滤波器验证功能 / Test filter validation functionality
     */
    @Test
    public void testFilterValidation() {
        LowPassFilter filter = new LowPassFilter();
        
        // 创建有效的测试音频数据 / Create valid test audio data
        IVector<Double> validSamples = Linalg.linspace(0.0, 100.0, 10);
        AudioData validAudio = new AudioData(validSamples, 44100.0, 1, 16, AudioFormat.PCM);
        
        // 验证输入 / Validate input
        assertTrue(filter.validateInput(validAudio), "Valid audio should pass validation");
        
        // 验证参数 / Validate parameters
        assertTrue(filter.validateParameters(null), "Null parameters should pass validation");
    }
}