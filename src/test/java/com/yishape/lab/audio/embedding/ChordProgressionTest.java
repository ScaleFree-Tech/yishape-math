package com.yishape.lab.audio.embedding;

import com.yishape.lab.audio.core.AudioIO;
import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.music.analysis.BasicMusicAnalyzer;
import com.yishape.lab.music.analysis.MusicDetectionResult;
import com.yishape.lab.music.analysis.UnifiedMusicAnalysisResult;
import com.yishape.lab.music.analysis.basic.ChordDetectionResult;

/**
 * 和弦进行分析测试类 / Test class for chord progression analysis
 * <p>
 * 测试和弦进行分析功能，包括和弦检测、时间窗口分析和结果显示。
 * Tests chord progression analysis functionality including chord detection,
 * time window analysis and result display.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ChordProgressionTest {
    
    public static void main(String args[]) {
        String path = "F:\\music\\test\\";
        String f1 = path + "20.杨友-野花做了场玫瑰花的梦.mp3";
        
        try {
            AudioData data = AudioIO.readAudio(f1);
            System.out.println("成功读取MP3文件！/ Successfully read MP3 file!");
            System.out.println("采样率 / Sample rate: " + data.getSampleRate());
            System.out.println("声道数 / Channels: " + data.getChannels());
            System.out.println("位深度 / Bit depth: " + data.getBitDepth());
            System.out.println("样本数 / Samples: " + data.getSamples().length());
            
            // 创建分析器 / Create analyzer
            BasicMusicAnalyzer analyzer = new BasicMusicAnalyzer();
            
            // 分析和弦进行 / Analyze chord progression
            System.out.println("分析和弦进行... / Analyzing chord progression...");
            long start = System.currentTimeMillis();
            MusicDetectionResult[] chordProgression = analyzer.analyzeStream(data, 30.0, 15.0); // 30秒窗口，15秒跳跃
            long end = System.currentTimeMillis();
            System.out.println("和弦进行分析耗时 " + (end-start) + " 毫秒 / Chord progression analysis took " + 
                              (end-start) + " ms");
            
            // 显示结果 / Display results
            System.out.println("和弦进行结果 / Chord Progression Results:");
            for (int i = 0; i < Math.min(chordProgression.length, 10); i++) { // 显示前10个和弦
                MusicDetectionResult result = chordProgression[i];
                if (result instanceof UnifiedMusicAnalysisResult) {
                    UnifiedMusicAnalysisResult unifiedResult = (UnifiedMusicAnalysisResult) result;
                    ChordDetectionResult chordResult = unifiedResult.getChordDetectionResult();
                    if (chordResult != null && chordResult.getChordName() != null && !chordResult.getChordName().equals("N/A")) {
                        System.out.println(String.format("  %.2f-%.2fs: %s (置信度 / confidence: %.2f)",
                             chordResult.getStartTime(), chordResult.getEndTime(), chordResult.getChordName(), chordResult.getConfidence()));
                    }
                }
            }
            
            if (chordProgression.length > 10) {
                System.out.println("  ... 还有 " + (chordProgression.length - 10) + " 个片段 / ... and " + 
                                  (chordProgression.length - 10) + " more segments");
            }
            
        } catch (AudioProcessingException e) {
            System.err.println("音频处理错误 / Audio processing error: " + e.getMessage());
            if (e.getMessage().contains("MP3 parsing not yet implemented")) {
                System.err.println("MP3支持可用但需要正确的MP3文件数据。/ MP3 support is available but requires proper MP3 file data.");
                System.err.println("请确保文件是有效的MP3文件。/ Please make sure the file is a valid MP3 file.");
            } else {
                System.err.println("请将MP3文件转换为WAV格式或实现MP3解码。/ Please convert the MP3 file to WAV format or implement MP3 decoding.");
            }
        } catch (java.io.FileNotFoundException e) {
            System.err.println("音频文件未找到 / Audio file not found: " + f1);
            System.err.println("请检查文件路径并确保文件存在。/ Please check the file path and make sure the file exists.");
        } catch (Exception e) {
            System.err.println("意外错误 / Unexpected error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试和弦检测功能 / Test chord detection functionality
     * 
     * @param audioData 音频数据 / Audio data
     * @param analyzer 音乐分析器 / Music analyzer
     */
    public static void testChordDetection(AudioData audioData, BasicMusicAnalyzer analyzer) {
        try {
            System.out.println("\n=== 和弦检测测试 / Chord Detection Test ===");
            
            // 测试单个时间窗口的和弦检测 / Test chord detection for a single time window
            double startTime = 30.0; // 从30秒开始
            double windowSize = 5.0;  // 5秒窗口
            
            // 提取音频片段
            AudioData segment = audioData.extractSegment(startTime, startTime + windowSize);
            ChordDetectionResult result = analyzer.analyzeChord(segment);
            
            if (result != null) {
                System.out.println("检测到的和弦 / Detected chord: " + result.getChordName());
                System.out.println("置信度 / Confidence: " + String.format("%.2f", result.getConfidence()));
                System.out.println("时间范围 / Time range: " + String.format("%.2f-%.2fs", 
                                  startTime, startTime + windowSize));
            } else {
                System.out.println("未检测到和弦 / No chord detected");
            }
            
        } catch (Exception e) {
            System.err.println("和弦检测测试失败 / Chord detection test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试和弦进行分析的不同参数 / Test chord progression analysis with different parameters
     * 
     * @param audioData 音频数据 / Audio data
     * @param analyzer 音乐分析器 / Music analyzer
     */
    public static void testDifferentParameters(AudioData audioData, BasicMusicAnalyzer analyzer) {
        try {
            System.out.println("\n=== 不同参数测试 / Different Parameters Test ===");
            
            // 测试不同的窗口大小 / Test different window sizes
            double[] windowSizes = {10.0, 20.0, 30.0}; // 秒
            double[] hopSizes = {5.0, 10.0, 15.0};     // 秒
            
            for (int i = 0; i < windowSizes.length; i++) {
                double windowSize = windowSizes[i];
                double hopSize = hopSizes[i];
                
                System.out.println("\n窗口大小 / Window size: " + windowSize + "s, 跳跃大小 / Hop size: " + hopSize + "s");
                
                long start = System.currentTimeMillis();
                MusicDetectionResult[] results = analyzer.analyzeStream(audioData, windowSize, hopSize);
                long end = System.currentTimeMillis();
                
                System.out.println("分析时间 / Analysis time: " + (end - start) + "ms");
                System.out.println("检测到的和弦段数 / Number of chord segments: " + results.length);
                
                // 显示前3个结果 / Show first 3 results
                int count = Math.min(3, results.length);
                for (int j = 0; j < count; j++) {
                    MusicDetectionResult result = results[j];
                    if (result instanceof UnifiedMusicAnalysisResult) {
                        UnifiedMusicAnalysisResult unifiedResult = (UnifiedMusicAnalysisResult) result;
                        ChordDetectionResult chordResult = unifiedResult.getChordDetectionResult();
                        if (chordResult != null && chordResult.getChordName() != null) {
                            System.out.println("  " + String.format("%.1f-%.1fs: %s (%.2f)", 
                                              chordResult.getStartTime(), chordResult.getEndTime(), 
                                              chordResult.getChordName(), chordResult.getConfidence()));
                        }
                    }
                }
            }
            
        } catch (Exception e) {
            System.err.println("参数测试失败 / Parameter test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 分析和弦进行的统计信息 / Analyze chord progression statistics
     * 
     * @param chordProgression 和弦进行结果 / Chord progression results
     */
    public static void analyzeChordStatistics(MusicDetectionResult[] chordProgression) {
        System.out.println("\n=== 和弦统计分析 / Chord Statistics Analysis ===");
        
        if (chordProgression == null || chordProgression.length == 0) {
            System.out.println("没有和弦数据可分析 / No chord data to analyze");
            return;
        }
        
        // 统计不同和弦的出现次数 / Count occurrences of different chords
        java.util.Map<String, Integer> chordCounts = new java.util.HashMap<>();
        double totalConfidence = 0.0;
        int validChords = 0;
        
        for (MusicDetectionResult result : chordProgression) {
            if (result instanceof UnifiedMusicAnalysisResult) {
                UnifiedMusicAnalysisResult unifiedResult = (UnifiedMusicAnalysisResult) result;
                ChordDetectionResult chordResult = unifiedResult.getChordDetectionResult();
                if (chordResult != null && chordResult.getChordName() != null && !chordResult.getChordName().equals("N/A")) {
                    String chordName = chordResult.getChordName();
                    chordCounts.put(chordName, chordCounts.getOrDefault(chordName, 0) + 1);
                    totalConfidence += chordResult.getConfidence();
                    validChords++;
                }
            }
        }
        
        System.out.println("总和弦段数 / Total chord segments: " + chordProgression.length);
        System.out.println("有效和弦段数 / Valid chord segments: " + validChords);
        System.out.println("平均置信度 / Average confidence: " + 
                          String.format("%.2f", validChords > 0 ? totalConfidence / validChords : 0.0));
        
        // 显示最常见的和弦 / Show most common chords
        System.out.println("\n最常见的和弦 / Most common chords:");
        chordCounts.entrySet().stream()
            .sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue()))
            .limit(5)
            .forEach(entry -> System.out.println("  " + entry.getKey() + ": " + entry.getValue() + " 次"));
    }
}