package com.yishape.lab.audio.effect;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;
import java.util.HashMap;

public class ReverbEffectTest {
    
    @Test
    public void testReverbEffectCreation() {
        ReverbEffect reverb = new ReverbEffect();
        assertNotNull(reverb);
        assertEquals("reverb", reverb.getName());
        assertEquals("Reverb effect", reverb.getDescription());
        assertEquals(IAudioEffect.EffectType.REVERB, reverb.getEffectType());
    }
    
    @Test
    public void testReverbEffectApply() throws AudioProcessingException {
        ReverbEffect reverb = new ReverbEffect();
        
        // Create a simple test audio data
        double[] samples = {0.1, 0.2, 0.3, 0.4, 0.5};
        IVector<Double> vector = Linalg.vector(samples);
        AudioData input = new AudioData(vector, 44100, 1, 16, AudioFormat.PCM);
        
        // Apply reverb effect
        AudioData output = reverb.applyEffect(input);
        
        // Verify output
        assertNotNull(output);
        assertEquals(input.getSampleRate(), output.getSampleRate(), 0.0);
        assertEquals(input.getChannels(), output.getChannels());
        assertEquals(input.getBitDepth(), output.getBitDepth());
    }
    
    @Test
    public void testReverbEffectParameters() {
        ReverbEffect reverb = new ReverbEffect();
        
        // Test supported parameters
        String[] supportedParams = reverb.getSupportedParameters();
        assertNotNull(supportedParams);
        assertTrue(supportedParams.length > 0);
        
        // Test setting and getting parameters
        reverb.setDryWetMix(0.5);
        assertEquals(0.5, reverb.getDryWetMix(), 0.001);
        
        reverb.setIntensity(0.8);
        assertEquals(0.8, reverb.getIntensity(), 0.001);
        
        reverb.setEnabled(false);
        assertFalse(reverb.isEnabled());
    }
    
    @Test
    public void testReverbEffectPresets() {
        ReverbEffect reverb = new ReverbEffect();
        
        // Test presets
        String[] presets = reverb.getPresets();
        assertNotNull(presets);
        assertEquals(4, presets.length);
        
        // Test loading a preset
        assertDoesNotThrow(() -> {
            reverb.loadPreset("hall");
        });
    }
    
    @Test
    public void testReverbEffectProcessorMethods() throws AudioProcessingException {
        ReverbEffect reverb = new ReverbEffect();
        
        // Test IAudioProcessor methods
        assertEquals("1.0", reverb.getVersion());
        assertNotNull(reverb.clone());
        
        // Test setting parameter
        reverb.setParameter("testParam", 0.5);
        
        // Test getting parameter
        Object param = reverb.getParameter("testParam");
        assertNotNull(param);
        
        // Test reset
        reverb.reset();
        
        // Test format support
        double[] samples = {0.1, 0.2, 0.3};
        IVector<Double> vector = Linalg.vector(samples);
        AudioData input = new AudioData(vector, 44100, 1, 16, AudioFormat.PCM);
        assertTrue(reverb.supportsFormat(input));
        
        // Test latency
        assertTrue(reverb.getLatency() >= 0);
    }
}