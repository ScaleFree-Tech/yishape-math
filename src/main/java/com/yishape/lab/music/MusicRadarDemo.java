package com.yishape.lab.music;

import com.yishape.lab.audio.Audios;
import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.math.viz.IPlot;

/**
 * Music Radar Chart Demo This demo shows how to use the plotMusicFeaturesRadar
 * method
 */
public class MusicRadarDemo {

    public static void main(String[] args) {

        try {
            String path = "F:\\music\\test\\";
            String f1 = path + "shadow of love.mp3";

            AudioData audioData = Audios.readAudio(f1);

            // Create the radar chart
            IPlot radarPlot = MusicPlots.plotMusicFeaturesRadar(audioData, "Music Features Radar Chart");
            radarPlot.show();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}
