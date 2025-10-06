package com.yishape.lab.audio.embedding;

import com.yishape.lab.audio.core.AudioIO;
import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.UnsupportedAudioFormatException;
import com.yishape.lab.music.Musics;
import com.yishape.lab.music.analysis.UnifiedMusicAnalysisResult;

/**
 * Test class for advanced music analysis functionality
 */
public class AdvancedAnalysisTest {

    public static void main(String args[]) {

        String path = "F:\\music\\test\\";
        String f1 = path + "20.杨友友-野花做了场玫瑰花的梦.mp3";
        try {
            AudioData data = AudioIO.readAudio(f1);
            System.out.println("Successfully read MP3 file!");
            System.out.println("Sample rate: " + data.getSampleRate());
            System.out.println("Channels: " + data.getChannels());
            System.out.println("Bit depth: " + data.getBitDepth());
            System.out.println("Samples: " + data.getSamples().length());
            System.out.println("Duration: " + (data.getSamples().length() / data.getSampleRate()) + " seconds");

            long start = System.currentTimeMillis();
            var result = Musics.advancedAnalysis(data);
            long end = System.currentTimeMillis();
            long interval = (end-start);
            System.out.println("Analysis took: " + interval + " ms");

            // Display detailed results
            if (result instanceof UnifiedMusicAnalysisResult) {
                UnifiedMusicAnalysisResult unifiedResult = (UnifiedMusicAnalysisResult) result;
                System.out.println("Overall confidence: " + String.format("%.2f", unifiedResult.getConfidence()));
                System.out.println("Algorithm: " + unifiedResult.getAlgorithm());

                // Display emotion analysis if available
                if (unifiedResult.getEmotionAnalysis() != null && !unifiedResult.getEmotionAnalysis().isEmpty()) {
                    System.out.println("Emotion Analysis:");
                    for (java.util.Map.Entry<String, Object> entry : unifiedResult.getEmotionAnalysis().entrySet()) {
                        System.out.println("  " + entry.getKey() + ": " + entry.getValue());
                    }
                }

                // Display genre analysis if available
                if (unifiedResult.getGenreAnalysis() != null && !unifiedResult.getGenreAnalysis().isEmpty()) {
                    System.out.println("Genre Analysis:");
                    for (java.util.Map.Entry<String, Object> entry : unifiedResult.getGenreAnalysis().entrySet()) {
                        System.out.println("  " + entry.getKey() + ": " + entry.getValue());
                    }
                }

                // Display complexity analysis if available
                if (unifiedResult.getComplexityAnalysis() != null && !unifiedResult.getComplexityAnalysis().isEmpty()) {
                    System.out.println("Complexity Analysis:");
                    for (java.util.Map.Entry<String, Object> entry : unifiedResult.getComplexityAnalysis().entrySet()) {
                        System.out.println("  " + entry.getKey() + ": " + entry.getValue());
                    }
                }
            }

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