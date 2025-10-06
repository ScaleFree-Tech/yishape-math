package com.yishape.lab.audio;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.audio.enhancement.IAudioEnhancer;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.HashMap;
import java.util.Map;

public class AudioEnhancerTest {

    @Test
    public void testCreateEnhancer() {
        // Test creating different types of enhancers
        IAudioEnhancer noiseReduction = Audios.createNoiseReductionEnhancer();
        assertNotNull(noiseReduction, "Noise reduction enhancer should be created");
        assertEquals("noise_reduction", noiseReduction.getName());
        
        IAudioEnhancer equalizer = Audios.createEqualizerEnhancer();
        assertNotNull(equalizer, "Equalizer enhancer should be created");
        assertEquals("equalizer", equalizer.getName());
        
        IAudioEnhancer compressor = Audios.createCompressorEnhancer();
        assertNotNull(compressor, "Compressor enhancer should be created");
        assertEquals("compressor", compressor.getName());
    }
    
    @Test
    public void testNoiseReductionEnhancer() {
        // Create test audio data with a size compatible with the default frame size (1024)
        IVector<Double> samples = Linalg.zeros(2048); // Use a multiple of frame size
        // Add some noise-like values
        for (int i = 0; i < samples.length(); i++) {
            samples.set(i, (Math.random() - 0.5) * 0.1); // Low amplitude noise
        }
        
        AudioData input = new AudioData(samples, 44100, 1, 16, AudioFormat.WAV);
        
        // Apply noise reduction with specific parameters to avoid frame size issues
        Map<String, Object> params = new HashMap<>();
        params.put("noiseThreshold", 0.05);
        params.put("attenuationFactor", 0.1);
        params.put("frameSize", 512); // Use a smaller frame size
        params.put("hopSize", 256);
        
        try {
            IAudioEnhancer enhancer = Audios.createNoiseReductionEnhancer();
            AudioData output = enhancer.enhance(input, params);
            
            assertNotNull(output, "Output should not be null");
            assertNotNull(output.getSamples(), "Output samples should not be null");
            assertEquals(input.getSamples().length(), output.getSamples().length(), 
                       "Sample count should be preserved");
        } catch (AudioProcessingException e) {
            fail("Audio processing failed: " + e.getMessage());
        }
    }
    
    @Test
    public void testEqualizerEnhancer() {
        // Create test audio data
        IVector<Double> samples = Linalg.zeros(1000);
        // Add a simple sine wave
        for (int i = 0; i < samples.length(); i++) {
            samples.set(i, Math.sin(2 * Math.PI * 440 * i / 44100)); // 440Hz tone
        }
        
        AudioData input = new AudioData(samples, 44100, 1, 16, AudioFormat.WAV);
        
        // Apply equalizer with some gains
        Map<String, Double> bandGains = new HashMap<>();
        bandGains.put("lowGain", 2.0);
        bandGains.put("midGain", -1.0);
        bandGains.put("highGain", 1.5);
        
        AudioData output = Audios.equalize(input, bandGains);
        
        assertNotNull(output, "Output should not be null");
        assertNotNull(output.getSamples(), "Output samples should not be null");
        assertEquals(input.getSamples().length(), output.getSamples().length(), 
                   "Sample count should be preserved");
    }
    
    @Test
    public void testCompressorEnhancer() {
        // Create test audio data with varying amplitudes
        IVector<Double> samples = Linalg.zeros(1000);
        // Add a signal that exceeds the threshold
        for (int i = 0; i < samples.length(); i++) {
            double amplitude = (i < 500) ? 0.3 : 0.8; // Increase amplitude in second half
            samples.set(i, amplitude * Math.sin(2 * Math.PI * 440 * i / 44100));
        }
        
        AudioData input = new AudioData(samples, 44100, 1, 16, AudioFormat.WAV);
        
        // Apply compression
        AudioData output = Audios.compress(input, -20.0, 4.0);
        
        assertNotNull(output, "Output should not be null");
        assertNotNull(output.getSamples(), "Output samples should not be null");
        assertEquals(input.getSamples().length(), output.getSamples().length(), 
                   "Sample count should be preserved");
    }
}