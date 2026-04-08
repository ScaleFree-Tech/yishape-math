package com.yishape.lab.music;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.audio.Audios;
import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.math.viz.IPlot;

/**
 * Music Radar Chart Demo This demo shows how to use the plotMusicFeaturesRadar
 * method
 */
public class MusicRadarDemo {

    private static final Logger log = LoggerFactory.getLogger(MusicRadarDemo.class);


    public static void main(String[] args) {

        try {
            String path = "F:\\music\\test\\";
//            String f1 = path + "20.杨友友-野花做了场玫瑰花的梦.mp3";
            String f1 = path + "shadow of love.mp3";

            AudioData audioData = Audios.readAudio(f1);

            // Create the radar chart
            IPlot radarPlot = MusicPlots.plotMusicFeaturesRadar(audioData, "Music Features Radar Chart");
            log.debug(radarPlot.toJson());
            radarPlot.show();
        } catch (Exception e) {
            log.error("exception", e);
        }

    }
}
