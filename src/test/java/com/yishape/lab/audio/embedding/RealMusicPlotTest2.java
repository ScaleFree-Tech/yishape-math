package com.yishape.lab.audio.embedding;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioIO;
import com.yishape.lab.music.MusicPlots;


/**
 *
 * @author lteb2
 */
public class RealMusicPlotTest2 {

    public static void main(String args[]) {

        String path = "F:\\music\\test\\";
        String f1 = path + "shadow of love.mp3";

        try {
            AudioData ad = AudioIO.readAudio(f1);

            MusicPlots.plotMusicFeaturesRadar(ad, "").show();


        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
