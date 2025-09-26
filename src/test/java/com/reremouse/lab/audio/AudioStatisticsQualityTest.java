package com.reremouse.lab.audio;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioIO;
import com.reremouse.lab.math.viz.IPlot;

/**
 * Test to verify that audio statistics and quality plots show correct x-axis labels
 */
public class AudioStatisticsQualityTest {

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
            
            System.out.println("Generating audio statistics plot with proper x-axis labels...");
            startTime = System.currentTimeMillis();
            IPlot statsPlot = AudioPlots.plotAudioStatistics(data, "音频统计信息 / Audio Statistics");
            endTime = System.currentTimeMillis();
            System.out.println("Statistics plot generated in " + (endTime - startTime) + " ms");
            statsPlot.show();
            
            System.out.println("Generating audio quality plot with proper x-axis labels...");
            startTime = System.currentTimeMillis();
            IPlot qualityPlot = AudioPlots.plotAudioQuality(data, "音频质量评估 / Audio Quality Assessment");
            endTime = System.currentTimeMillis();
            System.out.println("Quality plot generated in " + (endTime - startTime) + " ms");
            qualityPlot.show();
            
            System.out.println("Test completed successfully!");
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}