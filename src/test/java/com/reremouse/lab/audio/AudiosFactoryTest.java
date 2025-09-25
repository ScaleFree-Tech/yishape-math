package com.reremouse.lab.audio;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioFormat;
import com.reremouse.lab.audio.processing.IAdvancedAudioProcessor;
import com.reremouse.lab.audio.filter.IBaseAudioFilter;
import com.reremouse.lab.audio.effect.IAudioEffect;
import com.reremouse.lab.audio.analysis.IAudioAnalyzer;
import com.reremouse.lab.audio.feature.IAudioFeatureExtractor;
import com.reremouse.lab.audio.core.IAudioCodec;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.util.Tuple2;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 测试Audios工厂方法 / Test Audios factory methods
 */
public class AudiosFactoryTest {
    
    private AudioData testAudioData;
    
    @BeforeEach
    public void setUp() {
        // 创建测试音频数据 / Create test audio data
        double[] samples = new double[44100]; // 1秒的音频数据 / 1 second of audio data
        for (int i = 0; i < samples.length; i++) {
            samples[i] = Math.sin(2 * Math.PI * 440 * i / 44100); // 440Hz 正弦波 / 440Hz sine wave
        }
        IVector<Double> sampleVector = Linalg.vector(samples);
        testAudioData = new AudioData(sampleVector, 44100, 1, 16, AudioFormat.WAV);
    }
    
    @Test
    public void testCreateProcessor() {
        IAdvancedAudioProcessor processor = Audios.createProcessor("volume");
        assertNotNull(processor, "Volume processor should be created");
    }
    
    @Test
    public void testCreateVolumeProcessor() {
        IAdvancedAudioProcessor processor = Audios.createVolumeProcessor();
        assertNotNull(processor, "Volume processor should be created");
    }
    
    @Test
    public void testCreateNormalizeProcessor() {
        IAdvancedAudioProcessor processor = Audios.createNormalizeProcessor();
        assertNotNull(processor, "Normalize processor should be created");
    }
    
    @Test
    public void testCreateChannelProcessor() {
        IAdvancedAudioProcessor processor = Audios.createChannelProcessor();
        assertNotNull(processor, "Channel processor should be created");
    }
    
    @Test
    public void testCreateAnalyzer() {
        IAudioAnalyzer analyzer = Audios.createAnalyzer("spectrum");
        assertNotNull(analyzer, "Spectrum analyzer should be created");
    }
    
    @Test
    public void testCreateSpectrumAnalyzer() {
        IAudioAnalyzer analyzer = Audios.createSpectrumAnalyzer();
        assertNotNull(analyzer, "Spectrum analyzer should be created");
    }
    
    @Test
    public void testCreatePitchDetector() {
        IAudioAnalyzer analyzer = Audios.createPitchDetector();
        assertNotNull(analyzer, "Pitch detector should be created");
    }
    
    @Test
    public void testCreateSTFTAnalyzer() {
        IAudioAnalyzer analyzer = Audios.createSTFTAnalyzer();
        assertNotNull(analyzer, "STFT analyzer should be created");
    }
    
    @Test
    public void testCreateFilter() {
        IBaseAudioFilter filter = Audios.createFilter("lowpass");
        assertNotNull(filter, "Low-pass filter should be created");
    }
    
    @Test
    public void testCreateLowPassFilter() {
        IBaseAudioFilter filter = Audios.createLowPassFilter();
        assertNotNull(filter, "Low-pass filter should be created");
    }
    
    @Test
    public void testCreateAdvancedLowPassFilter() {
        IBaseAudioFilter filter = Audios.createAdvancedLowPassFilter();
        assertNotNull(filter, "Advanced low-pass filter should be created");
    }
    
    @Test
    public void testCreateEffect() {
        IAudioEffect effect = Audios.createEffect("reverb");
        assertNotNull(effect, "Reverb effect should be created");
    }
    
    @Test
    public void testCreateReverbEffect() {
        IAudioEffect effect = Audios.createReverbEffect();
        assertNotNull(effect, "Reverb effect should be created");
    }
    
    @Test
    public void testCreateFeatureExtractor() {
        IAudioFeatureExtractor extractor = Audios.createFeatureExtractor("standard");
        assertNotNull(extractor, "Standard feature extractor should be created");
    }
    
    @Test
    public void testCreateStandardFeatureExtractor() {
        IAudioFeatureExtractor extractor = Audios.createStandardFeatureExtractor();
        assertNotNull(extractor, "Standard feature extractor should be created");
    }
    
    @Test
    public void testCreateCodec() {
        // Note: There are no codec implementations registered in the factory by default
        assertThrows(RuntimeException.class, () -> {
            Audios.createCodec("wav");
        }, "Should throw exception for unknown codec");
    }
    
    @Test
    public void testAdjustVolume() {
        AudioData result = Audios.adjustVolume(testAudioData, 1.5);
        assertNotNull(result, "Adjusted audio should be returned");
        assertNotSame(testAudioData, result, "Should return a new AudioData instance");
    }
    
    @Test
    public void testNormalize() {
        AudioData result = Audios.normalize(testAudioData);
        assertNotNull(result, "Normalized audio should be returned");
        assertNotSame(testAudioData, result, "Should return a new AudioData instance");
    }
    
    @Test
    public void testConvertChannels() {
        AudioData result = Audios.convertChannels(testAudioData, 2);
        assertNotNull(result, "Converted audio should be returned");
        assertNotSame(testAudioData, result, "Should return a new AudioData instance");
    }
    
    @Test
    public void testToMono() {
        AudioData result = Audios.toMono(testAudioData);
        assertNotNull(result, "Mono audio should be returned");
        assertNotSame(testAudioData, result, "Should return a new AudioData instance");
    }
    
    @Test
    public void testToStereo() {
        AudioData result = Audios.toStereo(testAudioData);
        assertNotNull(result, "Stereo audio should be returned");
        assertNotSame(testAudioData, result, "Should return a new AudioData instance");
    }
    
    @Test
    public void testSpectrum() {
        Tuple2<IVector<Double>, IVector<Double>> result = Audios.spectrum(testAudioData);
        assertNotNull(result, "Spectrum result should be returned");
        assertNotNull(result._1, "Frequency values should be returned");
        assertNotNull(result._2, "Magnitude values should be returned");
    }
    
    @Test
    public void testSpectrumWithParameters() {
        Tuple2<IVector<Double>, IVector<Double>> result = Audios.spectrum(testAudioData, 1024, 0.5);
        assertNotNull(result, "Spectrum result should be returned");
        assertNotNull(result._1, "Frequency values should be returned");
        assertNotNull(result._2, "Magnitude values should be returned");
    }
    
    @Test
    public void testExtractFeatures() {
        IVector<Double> result = Audios.extractFeatures(testAudioData);
        assertNotNull(result, "Features should be returned");
    }
    
    @Test
    public void testStft() {
        Tuple2<IVector<Double>, IVector<Double>> result = Audios.stft(testAudioData);
        assertNotNull(result, "STFT result should be returned");
        assertNotNull(result._1, "Frequency values should be returned");
        assertNotNull(result._2, "Magnitude values should be returned");
    }
    
    @Test
    public void testDetectPitch() {
        double pitch = Audios.detectPitch(testAudioData);
        assertTrue(pitch >= 0, "Pitch should be non-negative");
    }
    
    @Test
    public void testLowPassFilter() {
        AudioData result = Audios.lowPassFilter(testAudioData, 1000);
        assertNotNull(result, "Filtered audio should be returned");
        assertNotSame(testAudioData, result, "Should return a new AudioData instance");
    }
    
    @Test
    public void testReverb() {
        AudioData result = Audios.reverb(testAudioData, 0.5, 0.3);
        assertNotNull(result, "Reverberated audio should be returned");
        assertNotSame(testAudioData, result, "Should return a new AudioData instance");
    }
    
    @Test
    public void testReduceNoise() {
        AudioData result = Audios.reduceNoise(testAudioData, 0.1);
        assertNotNull(result, "Noise reduced audio should be returned");
        // Note: Currently returns the same audio data as enhancement is not implemented
        assertSame(testAudioData, result, "Should return the same AudioData instance");
    }
    
    @Test
    public void testEqualize() {
        AudioData result = Audios.equalize(testAudioData, null);
        assertNotNull(result, "Equalized audio should be returned");
        // Note: Currently returns the same audio data as enhancement is not implemented
        assertSame(testAudioData, result, "Should return the same AudioData instance");
    }
    
    @Test
    public void testCompress() {
        AudioData result = Audios.compress(testAudioData, 0.5, 2.0);
        assertNotNull(result, "Compressed audio should be returned");
        // Note: Currently returns the same audio data as enhancement is not implemented
        assertSame(testAudioData, result, "Should return the same AudioData instance");
    }
}