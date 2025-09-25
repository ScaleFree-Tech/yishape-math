package com.reremouse.lab.audio;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioIO;

/**
 *
 * @author lteb2
 */
public class VizTest {

    public static void main(String args[]) {

        // 音频可视化 / Audio visualization
        String path = "F:\\music\\test\\";
        String f1 = path + "shadow of love.mp3";

        try {
            AudioData data = AudioIO.readAudio(f1);
//        AudioVisualizer.plotWaveform(data, "音频波形").show();
//        AudioVisualizer.plotSpectrum(data, "频谱分析").show();
//        AudioVisualizer.plotSpectrogram(data, "频谱图").show();
        AudioVisualizer.plotAudioStatistics(data, "统计信息").show();
        AudioVisualizer.plotAudioQuality(data, "质量分析").show();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
