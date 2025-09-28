package com.reremouse.lab.music;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioFormat;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.viz.IPlot;

public class MusicPlotsTest {
    
    public static void main(String[] args) {
        // Create a simple sine wave audio data for testing
        double sampleRate = 44100;
        double duration = 1.0; // 1 second
        int numSamples = (int) (sampleRate * duration);
        double[] samples = new double[numSamples];
        
        // Generate a simple sine wave at 440Hz
        for (int i = 0; i < numSamples; i++) {
            samples[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate);
        }
        
        // Create AudioData object
        IVector<Double> sampleVector = Linalg.vector(samples);
        AudioData audioData = new AudioData(sampleVector, sampleRate, 1, 16, AudioFormat.WAV);
        
        // Test the radar chart plotting
        IPlot radarPlot = MusicPlots.plotMusicFeaturesRadar(audioData, "Test Music Features Radar Chart");
        
        // Verify that the plot was created
        if (radarPlot != null) {
            System.out.println("Radar chart created successfully");
        } else {
            System.out.println("Failed to create radar chart");
        }
    }
}