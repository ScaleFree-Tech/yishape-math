package com.yishape.lab.music;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.audio.Audios;
import java.io.IOException;
import com.yishape.lab.audio.core.UnsupportedAudioFormatException;

/**
 * Test for MusicPlots radar chart functionality
 */
public class MusicPlotsRadarTest {
    
    public static void main(String[] args) {
        // Test with generated audio data
        testWithGeneratedAudio();
        
        // If you have a real audio file to test with, uncomment and update the path below
        // testWithRealAudioFile("path/to/your/audio/file.wav");
    }
    
    /**
     * Test the radar chart with generated audio data
     */
    public static void testWithGeneratedAudio() {
        System.out.println("Testing radar chart with generated audio data...");
        
        try {
            // Create a test audio signal with multiple frequencies
            double sampleRate = 22050;
            double duration = 15.0; // 15 seconds for better feature extraction
            int numSamples = (int)(sampleRate * duration);
            double[] samples = new double[numSamples];
            
            // Generate a complex waveform
            for (int i = 0; i < numSamples; i++) {
                double t = (double)i / sampleRate;
                // Combine multiple sine waves at different frequencies
                samples[i] = 0.2 * Math.sin(2 * Math.PI * 220 * t) +   // A3
                            0.3 * Math.sin(2 * Math.PI * 440 * t) +   // A4
                            0.15 * Math.sin(2 * Math.PI * 660 * t) +  // E5
                            0.1 * Math.sin(2 * Math.PI * 880 * t) +   // A5
                            0.05 * Math.sin(2 * Math.PI * 1100 * t);  // C#6
            }
            
            // Create AudioData object
            IVector<Double> sampleVector = Linalg.vector(samples);
            AudioData audioData = new AudioData(sampleVector, sampleRate, 1, 16, AudioFormat.WAV);
            
            // Test the radar chart
            IPlot radarPlot = MusicPlots.plotMusicFeaturesRadar(audioData, "Generated Audio Features");
            
            System.out.println("Radar chart created successfully with generated audio data!");
            System.out.println("Chart title: " + "Generated Audio Features");
            
        } catch (Exception e) {
            System.err.println("Error testing with generated audio: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Test the radar chart with a real audio file
     * @param filePath Path to the audio file
     */
    public static void testWithRealAudioFile(String filePath) {
        System.out.println("Testing radar chart with real audio file: " + filePath);
        
        try {
            // Load the audio file
            AudioData audioData = Audios.readAudio(filePath);
            
            System.out.println("Loaded audio file:");
            System.out.println("  Duration: " + audioData.getDuration() + " seconds");
            System.out.println("  Sample rate: " + audioData.getSampleRate() + " Hz");
            System.out.println("  Samples: " + audioData.getSamples().length());
            
            // Test the radar chart
            IPlot radarPlot = MusicPlots.plotMusicFeaturesRadar(audioData, "Real Audio Features");
            
            System.out.println("Radar chart created successfully with real audio file!");
            System.out.println("Chart title: " + "Real Audio Features");
            
            // In a GUI environment, you could display the chart with:
            // radarPlot.show();
            
        } catch (IOException e) {
            System.err.println("IO Error loading audio file: " + e.getMessage());
        } catch (UnsupportedAudioFormatException e) {
            System.err.println("Unsupported audio format: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Error testing with real audio file: " + e.getMessage());
            e.printStackTrace();
        }
    }
}