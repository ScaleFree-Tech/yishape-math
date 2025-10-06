package com.yishape.lab.music.analysis;

import com.yishape.lab.audio.preprocessing.AudioPreprocessingOptions;

import java.util.Map;

/**
 * 高级音乐分析器性能测试 / Advanced Music Analyzer Performance Test
 * <p>
 * 测试性能优化后的高级音乐分析器的性能表现。
 * Test the performance of the optimized advanced music analyzer.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class AdvancedMusicAnalyzerPerformanceTest {
    
    /**
     * 主函数 / Main function
     */
    public static void main(String[] args) {
        System.out.println("=== 高级音乐分析器性能测试 / Advanced Music Analyzer Performance Test ===\n");
        
        try {
            // 创建分析器 / Create analyzer
            AdvancedMusicAnalyzer analyzer = new AdvancedMusicAnalyzer();
            
            // 测试不同大小的音频数据 / Test different audio data sizes
            testWithDifferentSizes(analyzer);
            
            // 测试缓存效果 / Test cache effectiveness
            testCacheEffectiveness(analyzer);
            
            // 测试预处理选项 / Test preprocessing options
            testPreprocessingOptions(analyzer);
            
            // 打印最终性能统计 / Print final performance statistics
            printFinalStatistics(analyzer);
            
        } catch (Exception e) {
            System.err.println("测试失败 / Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试不同大小的音频数据 / Test different audio data sizes
     */
    private static void testWithDifferentSizes(AdvancedMusicAnalyzer analyzer) {
        System.out.println("1. 测试不同大小的音频数据 / Testing different audio data sizes:");
        
        int[] sizes = {10000, 50000, 100000, 200000};
        double sampleRate = 44100.0;
        
        for (int size : sizes) {
            // 生成测试音频数据 / Generate test audio data
            double[] audioData = generateTestAudioData(size);
            
            System.out.printf("  测试数据大小 / Testing data size: %d samples\n", size);
            
            long startTime = System.currentTimeMillis();
            
            try {
                // 执行分析 / Execute analysis
                UnifiedMusicAnalysisResult result = analyzer.analyze(audioData, sampleRate);
                
                long endTime = System.currentTimeMillis();
                long duration = endTime - startTime;
                
                System.out.printf("  分析完成 / Analysis completed in %d ms\n", duration);
                System.out.printf("  结果置信度 / Result confidence: %.2f%%\n", result.getConfidence() * 100);
                
            } catch (Exception e) {
                System.err.println("  分析失败 / Analysis failed: " + e.getMessage());
            }
            
            System.out.println();
        }
    }
    
    /**
     * 测试缓存效果 / Test cache effectiveness
     */
    private static void testCacheEffectiveness(AdvancedMusicAnalyzer analyzer) {
        System.out.println("2. 测试缓存效果 / Testing cache effectiveness:");
        
        // 生成测试音频数据 / Generate test audio data
        double[] audioData = generateTestAudioData(100000);
        double sampleRate = 44100.0;
        
        System.out.println("  第一次分析（无缓存）/ First analysis (no cache):");
        long startTime = System.currentTimeMillis();
        
        try {
            analyzer.analyze(audioData, sampleRate);
            long firstDuration = System.currentTimeMillis() - startTime;
            System.out.printf("  用时 / Duration: %d ms\n", firstDuration);
            
            System.out.println("  第二次分析（有缓存）/ Second analysis (with cache):");
            startTime = System.currentTimeMillis();
            analyzer.analyze(audioData, sampleRate);
            long secondDuration = System.currentTimeMillis() - startTime;
            System.out.printf("  用时 / Duration: %d ms\n", secondDuration);
            
            double speedup = (double) firstDuration / secondDuration;
            System.out.printf("  缓存加速比 / Cache speedup: %.2fx\n", speedup);
            
        } catch (Exception e) {
            System.err.println("  测试失败 / Test failed: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    /**
     * 测试预处理选项 / Test preprocessing options
     */
    private static void testPreprocessingOptions(AdvancedMusicAnalyzer analyzer) {
        System.out.println("3. 测试预处理选项 / Testing preprocessing options:");
        
        // 生成测试音频数据 / Generate test audio data
        double[] audioData = generateTestAudioData(100000);
        double sampleRate = 44100.0;
        
        // 测试默认选项 / Test default options
        System.out.println("  默认预处理选项 / Default preprocessing options:");
        testWithOptions(analyzer, AudioPreprocessingOptions.getDefault(), audioData, sampleRate);
        
        // 测试高质量选项 / Test high quality options
        System.out.println("  高质量预处理选项 / High quality preprocessing options:");
        testWithOptions(analyzer, AudioPreprocessingOptions.getHighQuality(), audioData, sampleRate);
        
        // 测试快速选项 / Test fast options
        System.out.println("  快速预处理选项 / Fast preprocessing options:");
        testWithOptions(analyzer, AudioPreprocessingOptions.getFast(), audioData, sampleRate);
        
        System.out.println();
    }
    
    /**
     * 使用指定选项进行测试 / Test with specified options
     */
    private static void testWithOptions(AdvancedMusicAnalyzer analyzer, AudioPreprocessingOptions options,
                                      double[] audioData, double sampleRate) {
        analyzer.setPreprocessingOptions(options);
        
        long startTime = System.currentTimeMillis();
        
        try {
            UnifiedMusicAnalysisResult result = analyzer.analyze(audioData, sampleRate);
            long duration = System.currentTimeMillis() - startTime;
            
            System.out.printf("  分析完成 / Analysis completed in %d ms\n", duration);
            System.out.printf("  结果置信度 / Result confidence: %.2f%%\n", result.getConfidence() * 100);
            
        } catch (Exception e) {
            System.err.println("  分析失败 / Analysis failed: " + e.getMessage());
        }
    }
    
    /**
     * 生成测试音频数据 / Generate test audio data
     */
    private static double[] generateTestAudioData(int size) {
        double[] data = new double[size];
        double sampleRate = 44100.0;
        
        // 生成复合音频信号 / Generate composite audio signal
        for (int i = 0; i < size; i++) {
            double t = i / sampleRate;
            
            // 基础频率 / Base frequency
            double signal = Math.sin(2 * Math.PI * 440 * t); // A4音符 / A4 note
            
            // 添加谐波 / Add harmonics
            signal += 0.5 * Math.sin(2 * Math.PI * 880 * t); // A5音符 / A5 note
            signal += 0.25 * Math.sin(2 * Math.PI * 1320 * t); // E6音符 / E6 note
            
            // 添加一些噪声 / Add some noise
            signal += 0.1 * (Math.random() - 0.5);
            
            // 应用衰减 / Apply decay
            signal *= Math.exp(-t * 2.0);
            
            data[i] = signal;
        }
        
        return data;
    }
    
    /**
     * 打印最终性能统计 / Print final performance statistics
     */
    private static void printFinalStatistics(AdvancedMusicAnalyzer analyzer) {
        System.out.println("=== 最终性能统计 / Final Performance Statistics ===");
        
        PerformanceMonitor.PerformanceStatistics stats = analyzer.getPerformanceStatistics();
        System.out.println(stats.toString());
        
        // 打印算法特定的统计 / Print algorithm-specific statistics
        java.util.Map<String, PerformanceMonitor.AlgorithmStatistics> algoStats = stats.getAlgorithmStatistics();
        if (!algoStats.isEmpty()) {
            System.out.println("\n=== 算法性能统计 / Algorithm Performance Statistics ===");
            for (java.util.Map.Entry<String, PerformanceMonitor.AlgorithmStatistics> entry : algoStats.entrySet()) {
                System.out.printf("算法 / Algorithm: %s\n", entry.getKey());
                System.out.printf("  %s\n", entry.getValue().toString());
            }
        }
        
        System.out.printf("\n缓存命中率 / Cache hit rate: %.2f%%\n", getCacheHitRate(analyzer));
    }
    
    /**
     * 获取缓存命中率 / Get cache hit rate
     */
    private static double getCacheHitRate(AdvancedMusicAnalyzer analyzer) {
        // 这是一个简化的缓存命中率计算 / This is a simplified cache hit rate calculation
        // 实际实现可能需要更复杂的逻辑 / Actual implementation may need more complex logic
        return 0.0; // 占位符 / Placeholder
    }
}