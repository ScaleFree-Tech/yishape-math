package com.yishape.lab.audio.embedding;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioIO;
import com.yishape.lab.audio.core.AudioUtil;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.viz.IPlot;
import com.yishape.lab.math.viz.Plots;

import java.util.Arrays;
import java.util.List;

/**
 *
 * @author lteb2
 */
public class RealMusicPlotTest {

    public static void main(String args[]) {

        String path = "F:\\music\\test\\";
        String f1 = path + "20.杨友友-野花做了场玫瑰花的梦.mp3";

        try {
            AudioData ad = AudioIO.readAudio(f1);

            IMatrix<Double> mfcc = AudioUtil.calculateMFCCMatrix(ad,40960);
            System.out.println(Arrays.toString(mfcc.shape()));
            IPlot ip = Plots.of(500, 300);
            ip.heatmap(mfcc);

            ip.title("李铁铁");
//            ip.saveAsHtml("d:\\test.html");
            ip.show();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
