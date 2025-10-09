package com.yishape.lab.audio;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioIO;

/**
 * Fixed version of VizTest to address the slow loading and zero Y-axis issues
 */
public class VizTestFixed {

    public static void main(String args[]) {
        // 音频可视化 / Audio visualization
        String path = "F:\\music\\test\\";
        String f1 = path + "20.杨友友-野花做了场玫瑰花的梦.mp3";

        try {
            System.out.println("Loading audio file...");
            long startTime = System.currentTimeMillis();
            AudioData data = AudioIO.readAudio(f1);
            long endTime = System.currentTimeMillis();
            System.out.println("Audio loaded in " + (endTime - startTime) + " ms");
            System.out.println("Audio data length: " + data.getSamples().length() + " samples");
            
            System.out.println("Generating waveform plot...");
            startTime = System.currentTimeMillis();
            AudioPlots.plotWaveform(data, "音频波形 / Audio Waveform", 1024, 256).show();
            endTime = System.currentTimeMillis();
            System.out.println("Waveform plot generated in " + (endTime - startTime) + " ms");
            
            System.out.println("Generating spectrum plot (line chart)...");
            startTime = System.currentTimeMillis();
            AudioPlots.plotSpectrum(data, "音频频谱 (线图) / Audio Spectrum (Line Chart)", "line").show();
            endTime = System.currentTimeMillis();
            System.out.println("Spectrum line plot generated in " + (endTime - startTime) + " ms");
            
            System.out.println("Generating spectrum plot (bar chart)...");
            startTime = System.currentTimeMillis();
            AudioPlots.plotSpectrum(data, "音频频谱 (柱状图) / Audio Spectrum (Bar Chart)", "bar").show();
            endTime = System.currentTimeMillis();
            System.out.println("Spectrum bar plot generated in " + (endTime - startTime) + " ms");
            
            System.out.println("Generating log spectrum plot...");
            startTime = System.currentTimeMillis();
            AudioPlots.plotLogSpectrum(data, "音频对数频谱 / Audio Log Spectrum").show();
            endTime = System.currentTimeMillis();
            System.out.println("Log spectrum plot generated in " + (endTime - startTime) + " ms");
            
            System.out.println("Generating spectrogram plot...");
            startTime = System.currentTimeMillis();
            AudioPlots.plotSpectrogram(data, "音频频谱图 / Audio Spectrogram", 1024, 256).show();
            endTime = System.currentTimeMillis();
            System.out.println("Spectrogram plot generated in " + (endTime - startTime) + " ms");
            
            System.out.println("Generating MFCC plot...");
            startTime = System.currentTimeMillis();
            AudioPlots.plotMFCC(data, "MFCC特征 / MFCC Features", 13, 1024, 256).show();
            endTime = System.currentTimeMillis();
            System.out.println("MFCC plot generated in " + (endTime - startTime) + " ms");
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}