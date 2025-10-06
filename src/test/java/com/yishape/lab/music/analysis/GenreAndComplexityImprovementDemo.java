package com.yishape.lab.music.analysis;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.music.analysis.advanced.GenreAnalyzer;
import com.yishape.lab.music.analysis.advanced.ComplexityAnalyzer;

import java.util.Map;

/**
 * 流派分析和复杂度分析改进演示类 / Demo class to showcase the improvements in genre analysis confidence and complexity analysis performance
 * <p>
 * 展示流派分析置信度和复杂度分析性能的改进
 * Demonstrates improvements in genre analysis confidence and complexity analysis performance
 * </p>
 */
public class GenreAndComplexityImprovementDemo {
    
    public static void main(String[] args) {
        try {
            System.out.println("=== 流派分析和复杂度分析改进演示 / Genre Analysis and Complexity Analysis Improvements Demo ===\n");
            
            // 创建演示分析器 / Create demo analyzers
            GenreAnalyzer genreAnalyzer = new GenreAnalyzer();
            ComplexityAnalyzer complexityAnalyzer = new ComplexityAnalyzer();
            
            // 创建具有明确特征的测试音频信号 / Create a test audio signal with clear characteristics
            System.out.println("创建具有明确音乐特征的测试音频信号... / Creating test audio signal with clear musical characteristics...");
            IVector<Double> samples = createMusicLikeSignal(44100, 5.0); // 5秒，44.1kHz
            AudioData audioData = new AudioData(samples, 44100, 1, samples.length(), null);
            
            System.out.println("音频数据已创建 / Audio data created: " + samples.length() + " 个样本，采样率 " +
                              audioData.getSampleRate() + " Hz / samples at " + audioData.getSampleRate() + " Hz\n");
            
            // 测试流派分析改进 / Test Genre Analysis Improvements
            System.out.println("=== 流派分析改进 / Genre Analysis Improvements ===");
            testGenreAnalysis(genreAnalyzer, audioData);
            
            System.out.println("\n" + "=".repeat(50) + "\n");
            
            // 测试复杂度分析性能改进 / Test Complexity Analysis Performance Improvements
            System.out.println("=== 复杂度分析性能改进 / Complexity Analysis Performance Improvements ===");
            testComplexityAnalysis(complexityAnalyzer, audioData);
            
            System.out.println("\n=== 演示完成 / Demo Completed ===");
            
        } catch (Exception e) {
            System.err.println("演示中出现错误 / Error in demo: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testGenreAnalysis(GenreAnalyzer genreAnalyzer, AudioData audioData) throws AudioProcessingException {
        System.out.println("执行流派分析... / Performing genre analysis...");
        long startTime = System.currentTimeMillis();
        Map<String, Object> results = genreAnalyzer.analyze(audioData);
        long endTime = System.currentTimeMillis();
        
        System.out.println("流派分析完成，耗时 " + (endTime - startTime) + " 毫秒 / Genre analysis completed in " + 
                          (endTime - startTime) + " ms");
        
        // 显示结果 / Display results
        String predictedGenre = (String) results.get("predictedGenre");
        double confidence = (Double) results.get("confidence");
        Map<String, Double> probabilities = (Map<String, Double>) results.get("genreProbabilities");
        
        System.out.println("\n--- 结果 / Results ---");
        System.out.println("预测流派 / Predicted Genre: " + predictedGenre);
        System.out.println("置信度 / Confidence: " + String.format("%.2f", confidence));
        
        // 显示概率不全相等（表明更好的区分能力）/ Show that probabilities are not all equal (indicating better discrimination)
        System.out.println("\n前5个流派及其概率 / Top 5 Genres with Probabilities:");
        probabilities.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(5)
            .forEach(entry -> 
                System.out.println("  " + entry.getKey() + ": " + String.format("%.3f", entry.getValue())));
        
        // 计算概率方差以显示改进 / Calculate variance in probabilities to show improvement
        double mean = probabilities.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double variance = probabilities.values().stream()
            .mapToDouble(v -> Math.pow(v - mean, 2))
            .average().orElse(0.0);
        
        System.out.println("\n概率分布分析 / Probability Distribution Analysis:");
        System.out.println("  平均概率 / Mean Probability: " + String.format("%.3f", mean));
        System.out.println("  方差 / Variance: " + String.format("%.3f", variance));
        System.out.println("  标准差 / Standard Deviation: " + String.format("%.3f", Math.sqrt(variance)));
        
        if (variance > 0.01) {
            System.out.println("  ✓ 良好的区分能力 - 概率差异显著 / Good discrimination - probabilities vary significantly");
        } else {
            System.out.println("  ✗ 区分能力较差 - 概率过于相似 / Poor discrimination - probabilities are too similar");
        }
    }
    
    private static void testComplexityAnalysis(ComplexityAnalyzer complexityAnalyzer, AudioData audioData) throws AudioProcessingException {
        System.out.println("执行复杂度分析... / Performing complexity analysis...");
        long startTime = System.currentTimeMillis();
        Map<String, Object> results = complexityAnalyzer.analyze(audioData);
        long endTime = System.currentTimeMillis();
        
        System.out.println("复杂度分析完成，耗时 " + (endTime - startTime) + " 毫秒 / Complexity analysis completed in " + 
                          (endTime - startTime) + " ms");
        
        // 显示结果 / Display results
        double harmonicComplexity = (Double) results.get("harmonicComplexity");
        double rhythmicComplexity = (Double) results.get("rhythmicComplexity");
        double melodicComplexity = (Double) results.get("melodicComplexity");
        double spectralComplexity = (Double) results.get("spectralComplexity");
        double overallComplexity = (Double) results.get("overallComplexity");
        String complexityLevel = (String) results.get("complexityLevel");
        
        System.out.println("\n--- 结果 / Results ---");
        System.out.println("和声复杂度 / Harmonic Complexity: " + String.format("%.2f", harmonicComplexity));
        System.out.println("节奏复杂度 / Rhythmic Complexity: " + String.format("%.2f", rhythmicComplexity));
        System.out.println("旋律复杂度 / Melodic Complexity: " + String.format("%.2f", melodicComplexity));
        System.out.println("频谱复杂度 / Spectral Complexity: " + String.format("%.2f", spectralComplexity));
        System.out.println("整体复杂度 / Overall Complexity: " + String.format("%.2f", overallComplexity));
        System.out.println("复杂度级别 / Complexity Level: " + complexityLevel);
        
        // 性能分析 / Performance analysis
        long durationMs = endTime - startTime;
        System.out.println("\n性能分析 / Performance Analysis:");
        System.out.println("  分析时间 / Analysis Time: " + durationMs + " ms");
        
        if (durationMs < 1000) {
            System.out.println("  ✓ 性能良好 - 分析快速完成 / Good performance - analysis completed quickly");
        } else if (durationMs < 2000) {
            System.out.println("  ✓ 性能可接受 - 分析时间合理 / Acceptable performance - analysis time is reasonable");
        } else {
            System.out.println("  ✗ 性能较差 - 分析时间过长 / Poor performance - analysis took too long");
        }
    }
    
    /**
     * 创建用于测试的音乐类信号，具有明确特征 / Create a music-like signal for testing with clear characteristics
     */
    private static IVector<Double> createMusicLikeSignal(double sampleRate, double duration) {
        int numSamples = (int) (sampleRate * duration);
        IVector<Double> samples = Linalg.zeros(numSamples);
        
        // 创建更复杂的信号来模拟具有明确特征的音乐 / Create a more complex signal that simulates music with clear characteristics
        for (int i = 0; i < numSamples; i++) {
            double t = i / sampleRate;
            
            // 基频（A4 = 440 Hz）/ Fundamental frequency (A4 = 440 Hz)
            double fundamental = 440.0;
            
            // 创建谐波和调制来模拟音乐 / Create harmonics and modulation to simulate music
            double value = 0.0;
            
            // 添加基频和谐波，振幅变化 / Add fundamental and harmonics with varying amplitudes
            for (int harmonic = 1; harmonic <= 6; harmonic++) {
                // 随时间变化振幅以模拟音乐短语 / Vary amplitude over time to simulate musical phrasing
                double amplitude = 0.1 + 0.15 * Math.sin(2 * Math.PI * 0.3 * t);
                value += amplitude * Math.sin(2 * Math.PI * fundamental * harmonic * t) / harmonic;
            }
            
            // 添加节奏成分（节拍）/ Add rhythmic component (beat)
            double beatFreq = 2.0; // 2 Hz (120 BPM)
            double beatComponent = 0.1 * Math.sin(2 * Math.PI * beatFreq * t);
            value += beatComponent;
            
            // 添加一些噪声使其更真实 / Add some noise to make it more realistic
            value += 0.03 * (Math.random() * 2 - 1);
            
            // 应用包络来模拟音符的起音和衰减 / Apply envelope to simulate note attacks and decays
            double envelope = Math.min(1.0, t * 3); // 更快的起音 / Faster attack
            if (t > duration * 0.7) {
                envelope *= (duration - t) / (duration * 0.3); // 更快的衰减 / Faster decay
            }
            
            samples.set(i, value * envelope);
        }
        
        return samples;
    }
}