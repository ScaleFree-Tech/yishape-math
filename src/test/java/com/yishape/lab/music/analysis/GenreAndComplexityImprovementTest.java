package com.yishape.lab.music.analysis;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.music.analysis.advanced.GenreAnalyzer;
import com.yishape.lab.music.analysis.advanced.ComplexityAnalyzer;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;

/**
 * 流派分析置信度和复杂度分析性能改进测试类 / Test class to verify improvements in genre analysis confidence and complexity analysis performance
 * <p>
 * 验证流派分析置信度和复杂度分析性能的改进
 * Verifies improvements in genre analysis confidence and complexity analysis performance
 * </p>
 */
public class GenreAndComplexityImprovementTest {
    
    private GenreAnalyzer genreAnalyzer;
    private ComplexityAnalyzer complexityAnalyzer;
    
    @BeforeEach
    void setUp() {
        genreAnalyzer = new GenreAnalyzer();
        complexityAnalyzer = new ComplexityAnalyzer();
    }
    
    @Test
    void testImprovedGenreAnalysisConfidence() throws AudioProcessingException {
        // 创建具有明确特征的测试音频信号 / Create a test audio signal with clear characteristics
        IVector<Double> samples = createTestSignalWithClearGenreCharacteristics(44100, 5.0);
        AudioData audioData = new AudioData(samples, 44100, 1, samples.length(), null);
        
        // 执行流派分析 / Perform genre analysis
        Map<String, Object> results = genreAnalyzer.analyze(audioData);
        
        // 验证我们得到了结果 / Verify that we get results
        assertNotNull(results);
        assertTrue(results.containsKey("predictedGenre"));
        assertTrue(results.containsKey("confidence"));
        assertTrue(results.containsKey("genreProbabilities"));
        
        // 检查置信度 - 应该比之前更高 / Check confidence - should be higher than before
        double confidence = (Double) results.get("confidence");
        System.out.println("流派分析置信度 / Genre analysis confidence: " + confidence);
        
        // 置信度应该是合理的（不是极低的值如0.11）/ Confidence should be reasonable (not extremely low like 0.11)
        assertTrue(confidence > 0.2, "置信度应该大于0.2 / Confidence should be greater than 0.2");
        
        // 检查概率不全相等 / Check that probabilities are not all equal
        Map<String, Double> probabilities = (Map<String, Double>) results.get("genreProbabilities");
        assertNotNull(probabilities);
        
        // 找到最大和最小概率 / Find the maximum and minimum probabilities
        double maxProb = probabilities.values().stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        double minProb = probabilities.values().stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        
        System.out.println("最大概率 / Max probability: " + maxProb);
        System.out.println("最小概率 / Min probability: " + minProb);
        
        // 差异应该显著，表明分类器在做区分 / The difference should be significant, indicating the classifier is making distinctions
        double diff = maxProb - minProb;
        assertTrue(diff > 0.1, "最大和最小概率应该有显著差异 / Maximum and minimum probabilities should differ significantly (diff=" + diff + ")");
        
        // 打印前3个流派 / Print top 3 genres
        System.out.println("前3个流派 / Top 3 genres:");
        probabilities.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(3)
            .forEach(entry -> 
                System.out.println("  " + entry.getKey() + ": " + String.format("%.2f", entry.getValue())));
    }
    
    @Test
    void testImprovedComplexityAnalysisPerformance() throws AudioProcessingException {
        // 创建测试音频信号 / Create a test audio signal
        IVector<Double> samples = createTestSignalWithClearGenreCharacteristics(44100, 3.0);
        AudioData audioData = new AudioData(samples, 44100, 1, samples.length(), null);
        
        // 执行复杂度分析并测量时间 / Perform complexity analysis and measure time
        long startTime = System.nanoTime();
        Map<String, Object> results = complexityAnalyzer.analyze(audioData);
        long endTime = System.nanoTime();
        
        long durationMs = (endTime - startTime) / 1_000_000;
        System.out.println("复杂度分析时间 / Complexity analysis time: " + durationMs + " ms");
        
        // 验证我们得到了结果 / Verify that we get results
        assertNotNull(results);
        assertTrue(results.containsKey("harmonicComplexity"));
        assertTrue(results.containsKey("rhythmicComplexity"));
        assertTrue(results.containsKey("melodicComplexity"));
        assertTrue(results.containsKey("spectralComplexity"));
        assertTrue(results.containsKey("overallComplexity"));
        
        // 性能应该是合理的（3秒音频应在2秒内完成分析）/ Performance should be reasonable (less than 2 seconds for a 3-second audio)
        assertTrue(durationMs < 2000, "复杂度分析应在2秒内完成 / Complexity analysis should complete in less than 2 seconds (took " + durationMs + " ms)");
        
        // 值应该在有效范围内 / Values should be in valid ranges
        double overallComplexity = (Double) results.get("overallComplexity");
        assertTrue(overallComplexity >= 0.0 && overallComplexity <= 10.0,
                   "整体复杂度应该在0到10之间 / Overall complexity should be between 0 and 10");
        
        System.out.println("整体复杂度 / Overall complexity: " + overallComplexity);
    }
    
    @Test
    void testGenreAnalysisConsistency() throws AudioProcessingException {
        // 测试分析结果的一致性 / Test consistency of analysis results
        IVector<Double> samples = createTestSignalWithClearGenreCharacteristics(44100, 2.0);
        AudioData audioData = new AudioData(samples, 44100, 1, samples.length(), null);
        
        // 多次执行分析 / Perform analysis multiple times
        String firstGenre = null;
        double totalConfidence = 0.0;
        int iterations = 5;
        
        for (int i = 0; i < iterations; i++) {
            Map<String, Object> results = genreAnalyzer.analyze(audioData);
            String predictedGenre = (String) results.get("predictedGenre");
            double confidence = (Double) results.get("confidence");
            
            if (firstGenre == null) {
                firstGenre = predictedGenre;
            }
            
            totalConfidence += confidence;
            
            // 结果应该一致 / Results should be consistent
            assertEquals(firstGenre, predictedGenre, "流派预测应该一致 / Genre prediction should be consistent");
        }
        
        double avgConfidence = totalConfidence / iterations;
        System.out.println("平均置信度 / Average confidence: " + avgConfidence);
        
        // 平均置信度应该合理 / Average confidence should be reasonable
        assertTrue(avgConfidence > 0.15, "平均置信度应该大于0.15 / Average confidence should be greater than 0.15");
    }
    
    @Test
    void testComplexityAnalysisAccuracy() throws AudioProcessingException {
        // 测试复杂度分析的准确性 / Test accuracy of complexity analysis
        
        // 创建简单信号 / Create simple signal
        IVector<Double> simpleSignal = createSimpleSignal(44100, 2.0);
        AudioData simpleAudio = new AudioData(simpleSignal, 44100, 1, simpleSignal.length(), null);
        
        // 创建复杂信号 / Create complex signal
        IVector<Double> complexSignal = createComplexSignal(44100, 2.0);
        AudioData complexAudio = new AudioData(complexSignal, 44100, 1, complexSignal.length(), null);
        
        // 分析两个信号 / Analyze both signals
        Map<String, Object> simpleResults = complexityAnalyzer.analyze(simpleAudio);
        Map<String, Object> complexResults = complexityAnalyzer.analyze(complexAudio);
        
        double simpleComplexity = (Double) simpleResults.get("overallComplexity");
        double complexComplexity = (Double) complexResults.get("overallComplexity");
        
        System.out.println("简单信号复杂度 / Simple signal complexity: " + simpleComplexity);
        System.out.println("复杂信号复杂度 / Complex signal complexity: " + complexComplexity);
        
        // 复杂信号的复杂度应该更高 / Complex signal should have higher complexity
        assertTrue(complexComplexity > simpleComplexity, 
                   "复杂信号的复杂度应该高于简单信号 / Complex signal should have higher complexity than simple signal");
    }
    
    /**
     * 创建具有明确流派特征的测试信号 / Create a test signal with characteristics that should lead to better genre classification
     */
    private IVector<Double> createTestSignalWithClearGenreCharacteristics(double sampleRate, double duration) {
        int numSamples = (int) (sampleRate * duration);
        IVector<Double> samples = Linalg.zeros(numSamples);
        
        // 创建具有明确节奏模式和谐波内容的信号 / Create a signal with clear rhythmic patterns and harmonic content
        for (int i = 0; i < numSamples; i++) {
            double t = i / sampleRate;
            
            // 基频（A4 = 440 Hz）/ Fundamental frequency (A4 = 440 Hz)
            double fundamental = 440.0;
            
            // 创建具有强节拍模式的信号 / Create a signal with strong beat patterns
            double value = 0.0;
            
            // 添加基频和谐波，变化振幅以创建流派特征 / Add fundamental and harmonics with varying amplitudes to create genre-like characteristics
            for (int harmonic = 1; harmonic <= 8; harmonic++) {
                // 随时间变化振幅以模拟音乐短语 / Vary amplitude over time to simulate musical phrasing
                double amplitude = 0.1 + 0.2 * Math.sin(2 * Math.PI * 0.5 * t); // 慢速振幅调制 / Slow amplitude modulation
                value += amplitude * Math.sin(2 * Math.PI * fundamental * harmonic * t) / harmonic;
            }
            
            // 添加强节拍成分 / Add strong beat component
            double beatFreq = 2.0; // 2 Hz节拍（120 BPM）/ 2 Hz beat (120 BPM)
            double beatComponent = 0.1 * Math.sin(2 * Math.PI * beatFreq * t);
            value += beatComponent;
            
            // 添加一些噪声使其更真实 / Add some noise to make it more realistic
            value += 0.05 * (Math.random() * 2 - 1);
            
            // 应用包络来模拟音符的起音和衰减 / Apply envelope to simulate note attacks and decays
            double envelope = Math.min(1.0, t * 2); // 起音 / Attack
            if (t > duration * 0.8) {
                envelope *= (duration - t) / (duration * 0.2); // 衰减 / Decay
            }
            
            samples.set(i, value * envelope);
        }
        
        return samples;
    }
    
    /**
     * 创建简单信号用于复杂度测试 / Create simple signal for complexity testing
     */
    private IVector<Double> createSimpleSignal(double sampleRate, double duration) {
        int numSamples = (int) (sampleRate * duration);
        IVector<Double> samples = Linalg.zeros(numSamples);
        
        // 简单的正弦波 / Simple sine wave
        for (int i = 0; i < numSamples; i++) {
            double t = i / sampleRate;
            samples.set(i, 0.5 * Math.sin(2 * Math.PI * 440 * t));
        }
        
        return samples;
    }
    
    /**
     * 创建复杂信号用于复杂度测试 / Create complex signal for complexity testing
     */
    private IVector<Double> createComplexSignal(double sampleRate, double duration) {
        int numSamples = (int) (sampleRate * duration);
        IVector<Double> samples = Linalg.zeros(numSamples);
        
        // 复杂的多谐波信号 / Complex multi-harmonic signal
        for (int i = 0; i < numSamples; i++) {
            double t = i / sampleRate;
            double value = 0.0;
            
            // 多个频率成分 / Multiple frequency components
            for (int freq = 1; freq <= 10; freq++) {
                value += 0.1 * Math.sin(2 * Math.PI * 440 * freq * t) / freq;
            }
            
            // 调制 / Modulation
            value *= (1 + 0.5 * Math.sin(2 * Math.PI * 5 * t));
            
            // 噪声 / Noise
            value += 0.1 * (Math.random() * 2 - 1);
            
            samples.set(i, value);
        }
        
        return samples;
    }
}