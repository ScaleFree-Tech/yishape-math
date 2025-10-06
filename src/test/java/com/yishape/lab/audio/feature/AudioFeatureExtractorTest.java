package com.yishape.lab.audio.feature;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.audio.factory.AudioComponentFactory;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.HashMap;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 音频特征提取器测试类 / Audio Feature Extractor Test Class
 * <p>
 * 测试音频特征提取器的功能和性能。
 * Tests the functionality and performance of the audio feature extractor.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public class AudioFeatureExtractorTest {

    /**
     * 测试创建音频特征提取器 / Test creating audio feature extractor
     */
    @Test
    public void testCreateExtractor() {
        IAudioFeatureExtractor extractor = AudioComponentFactory.getInstance().createFeatureExtractor("default");
        assertNotNull(extractor, "Extractor should not be null");
        assertEquals(extractor.getExtractorName(), "AudioFeatureExtractorImpl", "Extractor name should match");
        assertEquals(extractor.getVersion(), "2.0", "Version should match");
    }

    /**
     * 测试提取音频特征 / Test extracting audio features
     */
    @Test
    public void testExtractAudioFeatures() throws AudioProcessingException {
        IAudioFeatureExtractor extractor = AudioComponentFactory.getInstance().createFeatureExtractor("default");
        
        // 创建测试音频数据 / Create test audio data
        AudioData audioData = createTestAudioData();
        
        // 提取特征 / Extract features
        AudioFeatureResult features = extractor.extractAudioFeatures(audioData);
        
        assertNotNull(features, "Features should not be null");
        assertNotNull(features.getMetadata(), "Metadata should not be null");
        assertTrue(features.getMetadata().containsKey("sampleRate"), "Metadata should contain sampleRate");
    }

    /**
     * 测试提取时域特征 / Test extracting time-domain features
     */
    @Test
    public void testExtractTimeDomainFeatures() throws AudioProcessingException {
        IAudioFeatureExtractor extractor = AudioComponentFactory.getInstance().createFeatureExtractor("default");
        
        // 创建测试音频数据 / Create test audio data
        AudioData audioData = createTestAudioData();
        
        // 提取时域特征 / Extract time-domain features
        TimeDomainFeatureResult features = extractor.extractTimeDomainFeatures(audioData);
        
        assertNotNull(features, "Time-domain features should not be null");
        assertTrue(features.getRmsEnergy() >= 0.0, "RMS energy should be non-negative");
        assertTrue(features.getZeroCrossingRate() >= 0.0 && features.getZeroCrossingRate() <= 1.0, 
                  "Zero crossing rate should be between 0 and 1");
        assertTrue(features.getConfidence() >= 0.0 && features.getConfidence() <= 1.0, 
                  "Confidence should be between 0 and 1");
    }

    /**
     * 测试提取频域特征 / Test extracting frequency-domain features
     */
    @Test
    public void testExtractFrequencyDomainFeatures() throws AudioProcessingException {
        IAudioFeatureExtractor extractor = AudioComponentFactory.getInstance().createFeatureExtractor("default");
        
        // 创建测试音频数据 / Create test audio data
        AudioData audioData = createTestAudioData();
        
        // 提取频域特征 / Extract frequency-domain features
        FrequencyDomainFeatureResult features = extractor.extractFrequencyDomainFeatures(audioData);
        
        assertNotNull(features, "Frequency-domain features should not be null");
        assertTrue(features.getSpectralCentroid() >= 0.0, "Spectral centroid should be non-negative");
        assertTrue(features.getSpectralBandwidth() >= 0.0, "Spectral bandwidth should be non-negative");
        assertTrue(features.getSpectralRolloff() >= 0.0, "Spectral rolloff should be non-negative");
        assertTrue(features.getConfidence() >= 0.0 && features.getConfidence() <= 1.0, 
                  "Confidence should be between 0 and 1");
    }

    /**
     * 测试提取谱特征 / Test extracting spectral features
     */
    @Test
    public void testExtractSpectralFeatures() throws AudioProcessingException {
        IAudioFeatureExtractor extractor = AudioComponentFactory.getInstance().createFeatureExtractor("default");
        
        // 创建测试音频数据 / Create test audio data
        AudioData audioData = createTestAudioData();
        
        // 提取谱特征 / Extract spectral features
        SpectralFeatureResult features = extractor.extractSpectralFeatures(audioData);
        
        assertNotNull(features, "Spectral features should not be null");
        assertNotNull(features.getMfcc(), "MFCC should not be null");
        assertNotNull(features.getChroma(), "Chroma should not be null");
        assertTrue(features.getMfccCount() >= 0, "MFCC count should be non-negative");
        assertTrue(features.getConfidence() >= 0.0 && features.getConfidence() <= 1.0, 
                  "Confidence should be between 0 and 1");
    }

    /**
     * 测试参数验证 / Test parameter validation
     */
    @Test
    public void testParameterValidation() {
        IAudioFeatureExtractor extractor = AudioComponentFactory.getInstance().createFeatureExtractor("default");
        
        // 测试空参数 / Test null parameters
        assertFalse(extractor.validateParameters(null), "Null parameters should be invalid");
        
        // 测试有效参数 / Test valid parameters
        Map<String, Object> validParams = new HashMap<>();
        validParams.put("windowSize", 4096);
        validParams.put("hopSize", 2048);
        assertTrue(extractor.validateParameters(validParams), "Valid parameters should be accepted");
        
        // 测试无效窗口大小 / Test invalid window size
        Map<String, Object> invalidParams1 = new HashMap<>();
        invalidParams1.put("windowSize", -1);
        assertFalse(extractor.validateParameters(invalidParams1), "Negative window size should be invalid");
        
        // 测试过大窗口大小 / Test too large window size
        Map<String, Object> invalidParams2 = new HashMap<>();
        invalidParams2.put("windowSize", 20000);
        assertFalse(extractor.validateParameters(invalidParams2), "Too large window size should be invalid");
    }

    /**
     * 测试支持的特征类型 / Test supported feature types
     */
    @Test
    public void testSupportedFeatureTypes() {
        IAudioFeatureExtractor extractor = AudioComponentFactory.getInstance().createFeatureExtractor("default");
        
        String[] supportedTypes = extractor.getSupportedFeatureTypes();
        assertNotNull(supportedTypes, "Supported types should not be null");
        assertTrue(supportedTypes.length > 0, "Should support at least one feature type");
        
        // 检查一些关键特征类型 / Check some key feature types
        assertTrue(extractor.isFeatureTypeSupported("time_domain"), "Should support time_domain");
        assertTrue(extractor.isFeatureTypeSupported("spectral"), "Should support spectral");
        assertTrue(extractor.isFeatureTypeSupported("rms_energy"), "Should support rms_energy");
    }

    /**
     * 创建测试音频数据 / Create test audio data
     */
    private AudioData createTestAudioData() {
        // 创建简单的正弦波信号 / Create a simple sine wave signal
        int sampleRate = 44100;
        int durationSeconds = 1;
        int numSamples = sampleRate * durationSeconds;
        
        IVector<Double> samples = Linalg.zeros(numSamples);
        double frequency = 440.0; // A4 note
        
        for (int i = 0; i < numSamples; i++) {
            double time = (double) i / sampleRate;
            samples.set(i, Math.sin(2 * Math.PI * frequency * time));
        }
        
        return new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);
    }
}