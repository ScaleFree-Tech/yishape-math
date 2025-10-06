package com.yishape.lab.audio.embedding;

import com.yishape.lab.audio.core.AudioIO;
import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.UnsupportedAudioFormatException;
import com.yishape.lab.music.Musics;
import com.yishape.lab.music.analysis.feature.FeatureExtractorImpl;
import com.yishape.lab.music.analysis.feature.IFeatureExtractor;

/**
 * Test class for audio functionality
 *
 * @author lteb2
 */
public class AudioTest {

    public static void main(String args[]) {

        String path = "F:\\music\\test\\";
        String f1 = path + "shadow of love.mp3";
        try {
            AudioData data = AudioIO.readAudio(f1);
            System.out.println("Successfully read MP3 file!");
            System.out.println("Sample rate: " + data.getSampleRate());
            System.out.println("Channels: " + data.getChannels());
            System.out.println("Bit depth: " + data.getBitDepth());
            System.out.println("Samples: " + data.getSamples().length());
            long start = System.currentTimeMillis();
            var result = Musics.advancedAnalysis(data);
//            var result = Musics.basicAnalysis(data);
//            System.out.println(result);
//
            IFeatureExtractor ex = new FeatureExtractorImpl();

            var fs1 = ex.extractExpressivenessFeatures(data);
            
            long end = System.currentTimeMillis();
            long interval = (end - start);
            System.out.println("FS1 completed in " + interval + " ms");
            System.out.println(fs1);
            start = System.currentTimeMillis();
            
            var fs2 = ex.extractRhythmFeatures(data);
            
            end = System.currentTimeMillis();
            interval = (end - start);
            System.out.println("FS2 completed in " + interval + " ms");
            System.out.println(fs2);
            start = System.currentTimeMillis();
            
            var fs3 = ex.extractStructureFeatures(data);
            
            end = System.currentTimeMillis();
            interval = (end - start);
            System.out.println("FS3 completed in " + interval + " ms");
            System.out.println(fs3);
            start = System.currentTimeMillis();
            
            var fs4 = ex.extractTonalFeatures(data);

            end = System.currentTimeMillis();
            interval = (end - start);
            System.out.println("FS4 completed in " + interval + " ms");
            System.out.println(fs4);

        } catch (UnsupportedAudioFormatException e) {
            System.err.println("Audio format not supported: " + e.getMessage());
            if (e.getMessage().contains("MP3 parsing not yet implemented")) {
                System.err.println("MP3 support is available but requires proper MP3 file data.");
                System.err.println("Please make sure the file is a valid MP3 file.");
            } else {
                System.err.println("Please convert the MP3 file to WAV format or implement MP3 decoding.");
            }
        } catch (java.io.FileNotFoundException e) {
            System.err.println("Audio file not found: " + f1);
            System.err.println("Please check the file path and make sure the file exists.");
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
            e.printStackTrace();
        }
    }

}
