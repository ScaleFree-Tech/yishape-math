package com.yishape.lab.audio;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioIO;

/**
 * Extended version of VizTest to address the issue of only displaying 1 second of information
 * for long audio files
 */
public class VizTestExtended {

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
            System.out.println("Audio duration: " + data.getDuration() + " seconds");
            System.out.println("Audio data length: " + data.getSamples().length() + " samples");
            
            System.out.println("Generating spectrogram plot with extended time range...");
            startTime = System.currentTimeMillis();
            // 使用更大的时间帧数来显示更长的音频时段
            AudioPlots.plotSpectrogram(data, "音频频谱图 (扩展) / Audio Spectrogram (Extended)", 1024, 256).show();
            endTime = System.currentTimeMillis();
            System.out.println("Extended spectrogram plot generated in " + (endTime - startTime) + " ms");
            
            System.out.println("Generating MFCC plot with extended time range...");
            startTime = System.currentTimeMillis();
            // 使用更大的时间帧数来显示更长的音频时段
            AudioPlots.plotMFCC(data, "MFCC特征 (扩展) / MFCC Features (Extended)", 13, 1024, 256).show();
            endTime = System.currentTimeMillis();
            System.out.println("Extended MFCC plot generated in " + (endTime - startTime) + " ms");
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}