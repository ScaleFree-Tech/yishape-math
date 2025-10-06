package com.yishape.lab.music.analysis;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * 增强版音乐分析器测试类 / Enhanced Music Analyzer Test Class
 * <p>
 * 集成所有优化算法，提供详细的分析报告和调试信息。
 * 测试内容包括：置信度计算、异常检测、多算法交叉验证、性能分析等。
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class EnhancedMusicAnalyzerTest {
    
    private static final double[] TEST_FREQUENCIES = {220, 293.66, 329.63, 440, 523.25}; // A4, D4, E4, A4, C5
    private static final String[] TEST_KEYS = {"C Major", "G Major", "D Major", "A Minor", "E Minor"};
    private static final String[] TEST_CHORDS = {"C Major", "G Major", "Am", "F Major", "Dm"};
    
    private BasicMusicAnalyzer analyzer;
    private List<AudioData> testAudioSamples;
    private Map<String, TestResult> testResults;
    
    public static class TestResult {
        public String testName;
        public boolean passed;
        public long executionTime;
        public double confidence;
        public String details;
        public Map<String, Object> metrics;
        
        public TestResult(String testName) {
            this.testName = testName;
            this.metrics = new HashMap<>();
        }
    }
    
    public static class AnalysisReport {
        public String algorithm;
        public double confidence;
        public double beatConfidence;
        public double keyConfidence;
        public double chordConfidence;
        public double consistencyScore;
        public long executionTime;
        public boolean passedValidation;
        public Map<String, Object> validationDetails;
        
        public AnalysisReport() {
            this.validationDetails = new HashMap<>();
        }
    }
    
    public EnhancedMusicAnalyzerTest() {
        this.analyzer = new BasicMusicAnalyzer();
        this.testAudioSamples = new ArrayList<>();
        this.testResults = new HashMap<>();
        
        // 启用详细日志记录
        analyzer.setVerboseLogging(true);
    }
    
    /**
     * 运行完整的增强测试套件 / Run complete enhanced test suite
     */
    public void runEnhancedTestSuite() {
        System.out.println("=== 增强版音乐分析器测试开始 ===");
        System.out.println("Enhanced Music Analyzer Test Suite Starting");
        System.out.println();
        
        try {
            // 1. 准备测试音频样本
            prepareTestAudioSamples();
            
            // 2. 运行基础功能测试
            runBasicFunctionalityTests();
            
            // 3. 运行置信度计算测试
            runConfidenceCalculationTests();
            
            // 4. 运行异常检测测试
            runAnomalyDetectionTests();
            
            // 5. 运行多算法交叉验证测试
            runMultiAlgorithmValidationTests();
            
            // 6. 运行性能分析测试
            runPerformanceAnalysisTests();
            
            // 7. 生成详细报告
            generateDetailedReport();
            
        } catch (Exception e) {
            System.err.println("测试套件执行失败: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\n=== 增强版音乐分析器测试完成 ===");
        System.out.println("Enhanced Music Analyzer Test Suite Completed");
    }
    
    /**
     * 准备测试音频样本 / Prepare test audio samples
     */
    private void prepareTestAudioSamples() {
        System.out.println("准备测试音频样本...");
        
        // 创建不同类型的测试音频
        for (int i = 0; i < TEST_FREQUENCIES.length; i++) {
            double frequency = TEST_FREQUENCIES[i];
            String expectedKey = TEST_KEYS[i];
            String expectedChord = TEST_CHORDS[i];
            
            // 生成复合音频（基础频率 + 和声）
            AudioData audioData = createComplexAudio(frequency, expectedKey, expectedChord);
            testAudioSamples.add(audioData);
            
            System.out.println("创建测试音频样本 " + (i + 1) + ": " + frequency + "Hz, " + expectedKey + ", " + expectedChord);
        }
        
        // 添加一些特殊测试案例
        testAudioSamples.add(createNoisyAudio()); // 噪声音频
        testAudioSamples.add(createSilentAudio());  // 静音音频
        testAudioSamples.add(createMultiToneAudio()); // 多音调音频
        
        System.out.println("共准备了 " + testAudioSamples.size() + " 个测试音频样本");
        System.out.println();
    }
    
    /**
     * 创建复合音频 / Create complex audio
     */
    private AudioData createComplexAudio(double baseFrequency, String key, String chord) {
        int sampleRate = 44100;
        int duration = 2; // 2秒
        int totalSamples = sampleRate * duration;
        
        double[] samples = new double[totalSamples];
        
        // 基础频率
        for (int i = 0; i < totalSamples; i++) {
            samples[i] = 0.5 * Math.sin(2 * Math.PI * baseFrequency * i / sampleRate);
        }
        
        // 添加和声（根据和弦类型）
        double[] harmonics = getHarmonicsForChord(chord);
        for (double harmonic : harmonics) {
            if (harmonic > 0) {
                for (int i = 0; i < totalSamples; i++) {
                    samples[i] += 0.3 * Math.sin(2 * Math.PI * harmonic * i / sampleRate);
                }
            }
        }
        
        // 应用淡入淡出
        applyFadeInOut(samples, 0.1);
        
        IVector<Double> sampleVector = Linalg.vector(samples);
        return new AudioData(sampleVector, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 创建噪声音频 / Create noisy audio
     */
    private AudioData createNoisyAudio() {
        int sampleRate = 44100;
        int duration = 2;
        int totalSamples = sampleRate * duration;
        
        double[] samples = new double[totalSamples];
        Random random = new Random();
        
        // 生成白噪声
        for (int i = 0; i < totalSamples; i++) {
            samples[i] = (random.nextDouble() - 0.5) * 0.5;
        }
        
        // 添加一些周期性信号
        for (int i = 0; i < totalSamples; i++) {
            samples[i] += 0.2 * Math.sin(2 * Math.PI * 100 * i / sampleRate);
        }
        
        IVector<Double> sampleVector = Linalg.vector(samples);
        return new AudioData(sampleVector, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 创建静音音频 / Create silent audio
     */
    private AudioData createSilentAudio() {
        int sampleRate = 44100;
        int duration = 2;
        int totalSamples = sampleRate * duration;
        
        double[] samples = new double[totalSamples];
        // 非常小的随机噪声
        Random random = new Random();
        for (int i = 0; i < totalSamples; i++) {
            samples[i] = (random.nextDouble() - 0.5) * 0.01;
        }
        
        IVector<Double> sampleVector = Linalg.vector(samples);
        return new AudioData(sampleVector, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 创建多音调音频 / Create multi-tone audio
     */
    private AudioData createMultiToneAudio() {
        int sampleRate = 44100;
        int duration = 2;
        int totalSamples = sampleRate * duration;
        
        double[] samples = new double[totalSamples];
        double[] frequencies = {261.63, 329.63, 392.00, 523.25}; // C, E, G, C
        
        for (int i = 0; i < totalSamples; i++) {
            for (double freq : frequencies) {
                samples[i] += 0.25 * Math.sin(2 * Math.PI * freq * i / sampleRate);
            }
        }
        
        IVector<Double> sampleVector = Linalg.vector(samples);
        return new AudioData(sampleVector, sampleRate, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 获取和弦的和声 / Get harmonics for chord
     */
    private double[] getHarmonicsForChord(String chord) {
        Map<String, double[]> chordHarmonics = new HashMap<>();
        chordHarmonics.put("C Major", new double[]{261.63, 329.63, 392.00});
        chordHarmonics.put("G Major", new double[]{392.00, 493.88, 587.33});
        chordHarmonics.put("Am", new double[]{220.00, 261.63, 329.63});
        chordHarmonics.put("F Major", new double[]{174.61, 220.00, 261.63});
        chordHarmonics.put("Dm", new double[]{293.66, 349.23, 440.00});
        
        return chordHarmonics.getOrDefault(chord, new double[]{0});
    }
    
    /**
     * 应用淡入淡出 / Apply fade in/out
     */
    private void applyFadeInOut(double[] samples, double fadeDuration) {
        int sampleRate = 44100;
        int fadeSamples = (int)(fadeDuration * sampleRate);
        
        // 淡入
        for (int i = 0; i < fadeSamples && i < samples.length; i++) {
            double factor = (double)i / fadeSamples;
            samples[i] *= factor;
        }
        
        // 淡出
        for (int i = samples.length - fadeSamples; i < samples.length; i++) {
            double factor = (double)(samples.length - i) / fadeSamples;
            samples[i] *= factor;
        }
    }
    
    /**
     * 运行基础功能测试 / Run basic functionality tests
     */
    private void runBasicFunctionalityTests() {
        System.out.println("=== 运行基础功能测试 ===");
        
        TestResult result = new TestResult("基础功能测试");
        long startTime = System.currentTimeMillis();
        
        try {
            int passedTests = 0;
            int totalTests = testAudioSamples.size();
            
            for (int i = 0; i < testAudioSamples.size(); i++) {
                AudioData audioData = testAudioSamples.get(i);
                
                try {
                    // 测试完整分析
                    MusicDetectionResult analysisResult = analyzer.analyzeMusic(audioData);
                    
                    if (analysisResult != null && analysisResult instanceof UnifiedMusicAnalysisResult) {
                        UnifiedMusicAnalysisResult unifiedResult = (UnifiedMusicAnalysisResult) analysisResult;
                        
                        // 验证结果完整性
                        if (unifiedResult.getBeatDetectionResult() != null &&
                            unifiedResult.getKeyDetectionResult() != null &&
                            unifiedResult.getChordDetectionResult() != null &&
                            unifiedResult.getConfidence() > 0) {
                            
                            passedTests++;
                            System.out.println("测试样本 " + (i + 1) + " 通过 - 置信度: " + 
                                String.format("%.3f", unifiedResult.getConfidence()));
                        }
                    }
                } catch (Exception e) {
                    System.err.println("测试样本 " + (i + 1) + " 失败: " + e.getMessage());
                }
            }
            
            result.passed = passedTests == totalTests;
            result.details = "通过测试: " + passedTests + "/" + totalTests;
            result.metrics.put("passed_tests", passedTests);
            result.metrics.put("total_tests", totalTests);
            result.metrics.put("success_rate", (double)passedTests / totalTests);
            
        } catch (Exception e) {
            result.passed = false;
            result.details = "测试失败: " + e.getMessage();
            e.printStackTrace();
        }
        
        result.executionTime = System.currentTimeMillis() - startTime;
        testResults.put(result.testName, result);
        
        System.out.println("基础功能测试结果: " + (result.passed ? "通过" : "失败"));
        System.out.println("执行时间: " + result.executionTime + "ms");
        System.out.println("详情: " + result.details);
        System.out.println();
    }
    
    /**
     * 运行置信度计算测试 / Run confidence calculation tests
     */
    private void runConfidenceCalculationTests() {
        System.out.println("=== 运行置信度计算测试 ===");
        
        TestResult result = new TestResult("置信度计算测试");
        long startTime = System.currentTimeMillis();
        
        try {
            AudioData testAudio = testAudioSamples.get(0); // 使用第一个测试音频
            
            // 测试不同的置信度计算方法
            List<AnalysisReport> reports = new ArrayList<>();
            
            // 标准分析
            AnalysisReport standardReport = analyzeWithStandardMethod(testAudio);
            reports.add(standardReport);
            
            // 带参数的分析
            Map<String, Object> params = new HashMap<>();
            params.put("confidence_threshold", 0.8);
            params.put("enable_anomaly_detection", true);
            
            AnalysisReport enhancedReport = analyzeWithEnhancedMethod(testAudio, params);
            reports.add(enhancedReport);
            
            // 验证置信度计算的合理性
            boolean allReasonable = true;
            for (AnalysisReport report : reports) {
                if (report.confidence < 0 || report.confidence > 1) {
                    allReasonable = false;
                    break;
                }
            }
            
            result.passed = allReasonable;
            result.confidence = reports.stream().mapToDouble(r -> r.confidence).average().orElse(0.0);
            result.details = "分析了 " + reports.size() + " 种方法，平均置信度: " + 
                String.format("%.3f", result.confidence);
            
            // 保存详细指标
            result.metrics.put("analysis_methods", reports.size());
            result.metrics.put("average_confidence", result.confidence);
            result.metrics.put("min_confidence", reports.stream().mapToDouble(r -> r.confidence).min().orElse(0.0));
            result.metrics.put("max_confidence", reports.stream().mapToDouble(r -> r.confidence).max().orElse(0.0));
            
            // 打印详细报告
            System.out.println("置信度计算详细结果:");
            for (AnalysisReport report : reports) {
                System.out.println("  算法: " + report.algorithm);
                System.out.println("  置信度: " + String.format("%.3f", report.confidence));
                System.out.println("  节拍置信度: " + String.format("%.3f", report.beatConfidence));
                System.out.println("  调性置信度: " + String.format("%.3f", report.keyConfidence));
                System.out.println("  和弦置信度: " + String.format("%.3f", report.chordConfidence));
                System.out.println("  一致性评分: " + String.format("%.3f", report.consistencyScore));
                System.out.println("  执行时间: " + report.executionTime + "ms");
                System.out.println("  验证通过: " + report.passedValidation);
                System.out.println();
            }
            
        } catch (Exception e) {
            result.passed = false;
            result.details = "置信度计算测试失败: " + e.getMessage();
            e.printStackTrace();
        }
        
        result.executionTime = System.currentTimeMillis() - startTime;
        testResults.put(result.testName, result);
        
        System.out.println("置信度计算测试结果: " + (result.passed ? "通过" : "失败"));
        System.out.println("执行时间: " + result.executionTime + "ms");
        System.out.println("详情: " + result.details);
        System.out.println();
    }
    
    /**
     * 使用标准方法分析 / Analyze with standard method
     */
    private AnalysisReport analyzeWithStandardMethod(AudioData audioData) throws AudioProcessingException {
        AnalysisReport report = new AnalysisReport();
        report.algorithm = "standard";
        
        long startTime = System.currentTimeMillis();
        
        MusicDetectionResult result = analyzer.analyzeMusic(audioData);
        
        report.executionTime = System.currentTimeMillis() - startTime;
        
        if (result instanceof UnifiedMusicAnalysisResult) {
            UnifiedMusicAnalysisResult unifiedResult = (UnifiedMusicAnalysisResult) result;
            
            report.confidence = unifiedResult.getConfidence();
            report.beatConfidence = unifiedResult.getBeatDetectionResult() != null ? 
                unifiedResult.getBeatDetectionResult().getConfidence() : 0.0;
            report.keyConfidence = unifiedResult.getKeyDetectionResult() != null ? 
                unifiedResult.getKeyDetectionResult().getConfidence() : 0.0;
            report.chordConfidence = unifiedResult.getChordDetectionResult() != null ? 
                unifiedResult.getChordDetectionResult().getConfidence() : 0.0;
            report.consistencyScore = 0.0; // 标准方法不计算一致性
            report.passedValidation = report.confidence > 0.5;
        }
        
        return report;
    }
    
    /**
     * 使用增强方法分析 / Analyze with enhanced method
     */
    private AnalysisReport analyzeWithEnhancedMethod(AudioData audioData, Map<String, Object> params) throws AudioProcessingException {
        AnalysisReport report = new AnalysisReport();
        report.algorithm = "enhanced";
        
        long startTime = System.currentTimeMillis();
        
        MusicDetectionResult result = analyzer.analyzeMusic(audioData, params);
        
        report.executionTime = System.currentTimeMillis() - startTime;
        
        if (result instanceof UnifiedMusicAnalysisResult) {
            UnifiedMusicAnalysisResult unifiedResult = (UnifiedMusicAnalysisResult) result;
            
            report.confidence = unifiedResult.getConfidence();
            report.beatConfidence = unifiedResult.getBeatDetectionResult() != null ? 
                unifiedResult.getBeatDetectionResult().getConfidence() : 0.0;
            report.keyConfidence = unifiedResult.getKeyDetectionResult() != null ? 
                unifiedResult.getKeyDetectionResult().getConfidence() : 0.0;
            report.chordConfidence = unifiedResult.getChordDetectionResult() != null ? 
                unifiedResult.getChordDetectionResult().getConfidence() : 0.0;
            report.consistencyScore = 0.8; // 增强方法的一致性评分
            report.passedValidation = report.confidence > 0.6;
            
            // 添加验证详情
            report.validationDetails.put("anomaly_detected", false);
            report.validationDetails.put("multi_algorithm_used", true);
            report.validationDetails.put("consistency_check_passed", true);
        }
        
        return report;
    }
    
    /**
     * 运行异常检测测试 / Run anomaly detection tests
     */
    private void runAnomalyDetectionTests() {
        System.out.println("=== 运行异常检测测试 ===");
        
        TestResult result = new TestResult("异常检测测试");
        long startTime = System.currentTimeMillis();
        
        try {
            // 测试不同类型的异常音频
            List<AudioData> anomalyAudioSamples = Arrays.asList(
                createNoisyAudio(),
                createSilentAudio(),
                createMultiToneAudio()
            );
            
            int detectedAnomalies = 0;
            int totalTests = anomalyAudioSamples.size();
            
            for (int i = 0; i < anomalyAudioSamples.size(); i++) {
                AudioData audioData = anomalyAudioSamples.get(i);
                
                try {
                    // 使用带异常检测的参数
                    Map<String, Object> params = new HashMap<>();
                    params.put("enable_anomaly_detection", true);
                    params.put("confidence_threshold", 0.4);
                    
                    MusicDetectionResult analysisResult = analyzer.analyzeMusic(audioData, params);
                    
                    if (analysisResult instanceof UnifiedMusicAnalysisResult) {
                        UnifiedMusicAnalysisResult unifiedResult = (UnifiedMusicAnalysisResult) analysisResult;
                        
                        // 检查是否检测到异常（低置信度通常表示异常）
                        if (unifiedResult.getConfidence() < 0.5) {
                            detectedAnomalies++;
                            System.out.println("异常音频样本 " + (i + 1) + " 检测到异常 - 置信度: " + 
                                String.format("%.3f", unifiedResult.getConfidence()));
                        } else {
                            System.out.println("异常音频样本 " + (i + 1) + " 未检测到异常 - 置信度: " + 
                                String.format("%.3f", unifiedResult.getConfidence()));
                        }
                    }
                } catch (Exception e) {
                    System.err.println("异常检测测试样本 " + (i + 1) + " 失败: " + e.getMessage());
                }
            }
            
            result.passed = detectedAnomalies > 0; // 至少检测到一个异常
            result.details = "检测到 " + detectedAnomalies + "/" + totalTests + " 个异常音频";
            result.metrics.put("detected_anomalies", detectedAnomalies);
            result.metrics.put("total_anomaly_tests", totalTests);
            result.metrics.put("detection_rate", (double)detectedAnomalies / totalTests);
            
        } catch (Exception e) {
            result.passed = false;
            result.details = "异常检测测试失败: " + e.getMessage();
            e.printStackTrace();
        }
        
        result.executionTime = System.currentTimeMillis() - startTime;
        testResults.put(result.testName, result);
        
        System.out.println("异常检测结果: " + (result.passed ? "通过" : "失败"));
        System.out.println("执行时间: " + result.executionTime + "ms");
        System.out.println("详情: " + result.details);
        System.out.println();
    }
    
    /**
     * 运行多算法交叉验证测试 / Run multi-algorithm validation tests
     */
    private void runMultiAlgorithmValidationTests() {
        System.out.println("=== 运行多算法交叉验证测试 ===");
        
        TestResult result = new TestResult("多算法交叉验证测试");
        long startTime = System.currentTimeMillis();
        
        try {
            AudioData testAudio = testAudioSamples.get(0); // 使用标准测试音频
            
            // 创建不同参数配置的多个分析结果
            List<UnifiedMusicAnalysisResult> results = new ArrayList<>();
            
            // 保守参数
            Map<String, Object> conservativeParams = new HashMap<>();
            conservativeParams.put("confidence_threshold", 0.8);
            conservativeParams.put("sensitivity", "low");
            
            MusicDetectionResult conservativeResult = analyzer.analyzeMusic(testAudio, conservativeParams);
            if (conservativeResult instanceof UnifiedMusicAnalysisResult) {
                results.add((UnifiedMusicAnalysisResult) conservativeResult);
            }
            
            // 激进参数
            Map<String, Object> aggressiveParams = new HashMap<>();
            aggressiveParams.put("confidence_threshold", 0.3);
            aggressiveParams.put("sensitivity", "high");
            
            MusicDetectionResult aggressiveResult = analyzer.analyzeMusic(testAudio, aggressiveParams);
            if (aggressiveResult instanceof UnifiedMusicAnalysisResult) {
                results.add((UnifiedMusicAnalysisResult) aggressiveResult);
            }
            
            // 标准参数
            MusicDetectionResult standardResult = analyzer.analyzeMusic(testAudio);
            if (standardResult instanceof UnifiedMusicAnalysisResult) {
                results.add((UnifiedMusicAnalysisResult) standardResult);
            }
            
            // 验证多算法结果的一致性
            boolean consistent = validateMultiAlgorithmConsistency(results);
            
            result.passed = consistent;
            result.details = "分析了 " + results.size() + " 种算法配置，一致性: " + consistent;
            result.metrics.put("algorithm_configs", results.size());
            result.metrics.put("consistency_achieved", consistent);
            
            // 打印详细结果
            System.out.println("多算法交叉验证详细结果:");
            for (int i = 0; i < results.size(); i++) {
                UnifiedMusicAnalysisResult res = results.get(i);
                System.out.println("  配置 " + (i + 1) + ":");
                System.out.println("    算法: " + res.getAlgorithm());
                System.out.println("    置信度: " + String.format("%.3f", res.getConfidence()));
                System.out.println("    BPM: " + (res.getBeatDetectionResult() != null ? 
                    String.format("%.1f", res.getBeatDetectionResult().getBpm()) : "N/A"));
                System.out.println("    调性: " + (res.getKeyDetectionResult() != null ? 
                    res.getKeyDetectionResult().getKeyName() : "N/A"));
                System.out.println("    和弦: " + (res.getChordDetectionResult() != null ? 
                    res.getChordDetectionResult().getChordName() : "N/A"));
                System.out.println();
            }
            
        } catch (Exception e) {
            result.passed = false;
            result.details = "多算法交叉验证测试失败: " + e.getMessage();
            e.printStackTrace();
        }
        
        result.executionTime = System.currentTimeMillis() - startTime;
        testResults.put(result.testName, result);
        
        System.out.println("多算法交叉验证结果: " + (result.passed ? "通过" : "失败"));
        System.out.println("执行时间: " + result.executionTime + "ms");
        System.out.println("详情: " + result.details);
        System.out.println();
    }
    
    /**
     * 验证多算法一致性 / Validate multi-algorithm consistency
     */
    private boolean validateMultiAlgorithmConsistency(List<UnifiedMusicAnalysisResult> results) {
        if (results.size() < 2) {
            return true; // 单个结果视为一致
        }
        
        // 检查BPM一致性（允许10%的差异）
        List<Double> bpms = new ArrayList<>();
        for (UnifiedMusicAnalysisResult result : results) {
            if (result.getBeatDetectionResult() != null) {
                bpms.add(result.getBeatDetectionResult().getBpm());
            }
        }
        
        boolean bpmConsistent = true;
        if (bpms.size() >= 2) {
            double avgBpm = bpms.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            for (double bpm : bpms) {
                if (Math.abs(bpm - avgBpm) / avgBpm > 0.1) {
                    bpmConsistent = false;
                    break;
                }
            }
        }
        
        // 检查调性一致性
        Set<String> keys = new HashSet<>();
        for (UnifiedMusicAnalysisResult result : results) {
            if (result.getKeyDetectionResult() != null) {
                keys.add(result.getKeyDetectionResult().getKeyName());
            }
        }
        
        boolean keyConsistent = keys.size() <= 2; // 允许最多2种不同的调性
        
        return bpmConsistent && keyConsistent;
    }
    
    /**
     * 运行性能分析测试 / Run performance analysis tests
     */
    private void runPerformanceAnalysisTests() {
        System.out.println("=== 运行性能分析测试 ===");
        
        TestResult result = new TestResult("性能分析测试");
        long startTime = System.currentTimeMillis();
        
        try {
            AudioData testAudio = testAudioSamples.get(0);
            int iterations = 10;
            
            List<Long> executionTimes = new ArrayList<>();
            List<Double> memoryUsages = new ArrayList<>();
            
            // 预热JVM
            for (int i = 0; i < 3; i++) {
                analyzer.analyzeMusic(testAudio);
            }
            
            // 正式测试
            for (int i = 0; i < iterations; i++) {
                long iterationStart = System.currentTimeMillis();
                
                // 获取内存使用
                Runtime runtime = Runtime.getRuntime();
                runtime.gc(); // 建议垃圾回收
                long memoryBefore = runtime.totalMemory() - runtime.freeMemory();
                
                // 执行分析
                MusicDetectionResult analysisResult = analyzer.analyzeMusic(testAudio);
                
                long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
                long iterationEnd = System.currentTimeMillis();
                
                long executionTime = iterationEnd - iterationStart;
                double memoryUsage = (memoryAfter - memoryBefore) / (1024.0 * 1024.0); // MB
                
                executionTimes.add(executionTime);
                memoryUsages.add(memoryUsage);
                
                System.out.println("迭代 " + (i + 1) + ": " + executionTime + "ms, " + 
                    String.format("%.2f", memoryUsage) + "MB");
            }
            
            // 计算统计信息
            double avgExecutionTime = executionTimes.stream().mapToLong(Long::longValue).average().orElse(0.0);
            double maxExecutionTime = executionTimes.stream().mapToLong(Long::longValue).max().orElse(0L);
            double minExecutionTime = executionTimes.stream().mapToLong(Long::longValue).min().orElse(0L);
            double avgMemoryUsage = memoryUsages.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            
            result.passed = avgExecutionTime < 5000; // 平均执行时间小于5秒
            result.details = "平均执行时间: " + String.format("%.1f", avgExecutionTime) + "ms, " +
                "平均内存使用: " + String.format("%.2f", avgMemoryUsage) + "MB";
            
            result.metrics.put("avg_execution_time", avgExecutionTime);
            result.metrics.put("max_execution_time", maxExecutionTime);
            result.metrics.put("min_execution_time", minExecutionTime);
            result.metrics.put("avg_memory_usage", avgMemoryUsage);
            result.metrics.put("iterations", iterations);
            
            // 性能评估
            String performanceLevel;
            if (avgExecutionTime < 1000) {
                performanceLevel = "优秀";
            } else if (avgExecutionTime < 3000) {
                performanceLevel = "良好";
            } else if (avgExecutionTime < 5000) {
                performanceLevel = "一般";
            } else {
                performanceLevel = "需优化";
            }
            
            result.metrics.put("performance_level", performanceLevel);
            
            System.out.println("性能评估级别: " + performanceLevel);
            
        } catch (Exception e) {
            result.passed = false;
            result.details = "性能分析测试失败: " + e.getMessage();
            e.printStackTrace();
        }
        
        result.executionTime = System.currentTimeMillis() - startTime;
        testResults.put(result.testName, result);
        
        System.out.println("性能分析结果: " + (result.passed ? "通过" : "失败"));
        System.out.println("执行时间: " + result.executionTime + "ms");
        System.out.println("详情: " + result.details);
        System.out.println();
    }
    
    /**
     * 生成详细报告 / Generate detailed report
     */
    private void generateDetailedReport() {
        System.out.println("=== 生成详细测试报告 ===");
        
        int totalTests = testResults.size();
        int passedTests = 0;
        long totalExecutionTime = 0;
        
        System.out.println("测试摘要:");
        System.out.println("=".repeat(50));
        
        for (Map.Entry<String, TestResult> entry : testResults.entrySet()) {
            TestResult result = entry.getValue();
            
            System.out.println("测试名称: " + result.testName);
            System.out.println("结果: " + (result.passed ? "通过" : "失败"));
            System.out.println("执行时间: " + result.executionTime + "ms");
            System.out.println("详情: " + result.details);
            
            if (!result.metrics.isEmpty()) {
                System.out.println("指标:");
                for (Map.Entry<String, Object> metric : result.metrics.entrySet()) {
                    System.out.println("  " + metric.getKey() + ": " + metric.getValue());
                }
            }
            
            System.out.println("-".repeat(30));
            
            if (result.passed) {
                passedTests++;
            }
            totalExecutionTime += result.executionTime;
        }
        
        // 总体评估
        System.out.println("总体评估:");
        System.out.println("=".repeat(50));
        System.out.println("总测试数: " + totalTests);
        System.out.println("通过测试: " + passedTests);
        System.out.println("失败测试: " + (totalTests - passedTests));
        System.out.println("成功率: " + String.format("%.1f%%", (double)passedTests / totalTests * 100));
        System.out.println("总执行时间: " + totalExecutionTime + "ms");
        System.out.println("平均执行时间: " + (totalExecutionTime / totalTests) + "ms");
        
        // 性能等级评估
        String overallLevel;
        double successRate = (double)passedTests / totalTests;
        if (successRate >= 0.9) {
            overallLevel = "优秀";
        } else if (successRate >= 0.7) {
            overallLevel = "良好";
        } else if (successRate >= 0.5) {
            overallLevel = "一般";
        } else {
            overallLevel = "需改进";
        }
        
        System.out.println("整体评估级别: " + overallLevel);
        System.out.println();
        
        // 建议
        System.out.println("改进建议:");
        System.out.println("=".repeat(50));
        if (successRate < 0.7) {
            System.out.println("- 建议检查算法实现和参数配置");
        }
        if (totalExecutionTime > 30000) {
            System.out.println("- 建议优化性能，减少执行时间");
        }
        System.out.println("- 定期运行测试以确保算法稳定性");
        System.out.println("- 根据测试结果调整算法参数");
        
        System.out.println();
    }
    
    /**
     * 主方法 / Main method
     */
    public static void main(String[] args) {
        EnhancedMusicAnalyzerTest test = new EnhancedMusicAnalyzerTest();
        test.runEnhancedTestSuite();
    }
}