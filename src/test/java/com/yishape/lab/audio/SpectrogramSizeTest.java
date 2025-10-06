package com.yishape.lab.audio;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioIO;
import com.yishape.lab.math.viz.IPlot;

/**
 * Test to verify that spectrogram file size issue is fixed
 */
public class SpectrogramSizeTest {

    public static void main(String args[]) {
        // 音频可视化 / Audio visualization
        String path = "F:\\music\\test\\";
        String f1 = path + "shadow of love.mp3";

        try {
            System.out.println("Loading audio file...");
            long startTime = System.currentTimeMillis();
            AudioData data = AudioIO.readAudio(f1);
            long endTime = System.currentTimeMillis();
            System.out.println("Audio loaded in " + (endTime - startTime) + " ms");
            System.out.println("Audio data length: " + data.getSamples().length() + " samples");
            
            System.out.println("Generating spectrogram plot with size limitation...");
            startTime = System.currentTimeMillis();
            IPlot spectrogram = AudioPlots.plotSpectrogram(data, "音频频谱图 / Audio Spectrogram", 1024, 256);
            endTime = System.currentTimeMillis();
            System.out.println("Spectrogram plot generated in " + (endTime - startTime) + " ms");
            
            // Show the plot
            spectrogram.show();
            
            System.out.println("Test completed successfully!");
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}