package com.yishape.lab.music.analysis;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.music.analysis.basic.BeatDetectionResult;
import com.yishape.lab.music.analysis.basic.KeyDetectionResult;
import com.yishape.lab.music.analysis.basic.ChordDetectionResult;

import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 综合音乐分析器 / Comprehensive Music Analyzer
 * <p>
 * 提供全面的音乐分析功能，集成基础音乐分析器和高级音乐分析器的功能。
 * Provides comprehensive music analysis features by integrating BasicMusicAnalyzer and AdvancedMusicAnalyzer.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ComprehensiveMusicAnalyzer implements IMusicAnalyzer {

    private final BasicMusicAnalyzer basicAnalyzer;
    private final AdvancedMusicAnalyzer advancedAnalyzer;
    private final ExecutorService executorService;

    // 默认参数 / Default parameters
    private static final double DEFAULT_CONFIDENCE_THRESHOLD = 0.8;
    private static final int DEFAULT_WINDOW_SIZE = 4096;
    private static final int DEFAULT_HOP_SIZE = 1024;
    private static final boolean DEFAULT_ENABLE_BEAT_DETECTION = true;
    private static final boolean DEFAULT_ENABLE_KEY_DETECTION = true;
    private static final boolean DEFAULT_ENABLE_CHORD_DETECTION = true;
    private static final boolean DEFAULT_ENABLE_STRUCTURE_ANALYSIS = true;
    private static final boolean DEFAULT_ENABLE_GENRE_DETECTION = true;
    private static final boolean DEFAULT_ENABLE_TEMPO_VARIATION = true;
    private static final boolean DEFAULT_ENABLE_HARMONIC_ANALYSIS = true;

    // 当前参数 / Current parameters
    private final Map<String, Object> parameters = new HashMap<>();

    // 分析状态 / Analysis state
    private volatile boolean analysisCancelled = false;
    private volatile boolean analysisPaused = false;
    private volatile double analysisProgress = 0.0;

    /**
     * 构造函数 / Constructor
     */
    public ComprehensiveMusicAnalyzer() {
        this.basicAnalyzer = new BasicMusicAnalyzer();
        this.advancedAnalyzer = new AdvancedMusicAnalyzer();
        this.executorService = Executors.newFixedThreadPool(4);
        initializeDefaultParameters();
    }

    /**
     * 初始化默认参数 / Initialize default parameters
     */
    private void initializeDefaultParameters() {
        parameters.put("confidenceThreshold", DEFAULT_CONFIDENCE_THRESHOLD);
        parameters.put("windowSize", DEFAULT_WINDOW_SIZE);
        parameters.put("hopSize", DEFAULT_HOP_SIZE);
        parameters.put("enableBeatDetection", DEFAULT_ENABLE_BEAT_DETECTION);
        parameters.put("enableKeyDetection", DEFAULT_ENABLE_KEY_DETECTION);
        parameters.put("enableChordDetection", DEFAULT_ENABLE_CHORD_DETECTION);
        parameters.put("enableStructureAnalysis", DEFAULT_ENABLE_STRUCTURE_ANALYSIS);
        parameters.put("enableGenreDetection", DEFAULT_ENABLE_GENRE_DETECTION);
        parameters.put("enableTempoVariation", DEFAULT_ENABLE_TEMPO_VARIATION);
        parameters.put("enableHarmonicAnalysis", DEFAULT_ENABLE_HARMONIC_ANALYSIS);
    }

    @Override
    public MusicDetectionResult analyzeMusic(AudioData audioData) throws AudioProcessingException {
        return analyzeMusic(audioData, parameters);
    }

    @Override
    public MusicDetectionResult analyzeMusic(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }

        // Use a final variable for lambda expressions
        final Map<String, Object> effectiveParameters = parameters != null ? parameters : this.parameters;

        try {
            analysisProgress = 0.0;
            analysisCancelled = false;

            // 并行执行基础分析和高级分析 / Execute basic and advanced analysis in parallel
            CompletableFuture<MusicDetectionResult> basicFuture = CompletableFuture.supplyAsync(() -> {
                try {
                    return basicAnalyzer.analyzeMusic(audioData, effectiveParameters);
                } catch (AudioProcessingException e) {
                    throw new RuntimeException(e);
                }
            }, executorService);

            CompletableFuture<MusicDetectionResult> advancedFuture = CompletableFuture.supplyAsync(() -> {
                try {
                    return advancedAnalyzer.analyzeAdvancedMusic(audioData, effectiveParameters);
                } catch (AudioProcessingException e) {
                    throw new RuntimeException(e);
                }
            }, executorService);

            // 等待所有分析完成 / Wait for all analyses to complete
            MusicDetectionResult basicResult = basicFuture.join();
            MusicDetectionResult advancedResult = advancedFuture.join();

            // 合并结果 / Merge results
            UnifiedMusicAnalysisResult result = mergeAnalysisResults(basicResult, advancedResult);
            result.setAlgorithm("comprehensive_music_analyzer");

            analysisProgress = 100.0;
            return result;

        } catch (Exception e) {
            throw new AudioProcessingException("Comprehensive music analysis failed: " + e.getMessage(), e);
        }
    }

    /**
     * 合并分析结果 / Merge analysis results
     */
    private UnifiedMusicAnalysisResult mergeAnalysisResults(MusicDetectionResult basicResult, MusicDetectionResult advancedResult) {
        UnifiedMusicAnalysisResult result = new UnifiedMusicAnalysisResult();
        
        // 从基础结果复制数据 / Copy data from basic result
        if (basicResult instanceof UnifiedMusicAnalysisResult) {
            UnifiedMusicAnalysisResult basicUnified = (UnifiedMusicAnalysisResult) basicResult;
            result.setBeatDetectionResult(basicUnified.getBeatDetectionResult());
            result.setKeyDetectionResult(basicUnified.getKeyDetectionResult());
            result.setChordDetectionResult(basicUnified.getChordDetectionResult());
        }
        
        // 从高级结果复制数据 / Copy data from advanced result
        if (advancedResult instanceof UnifiedMusicAnalysisResult) {
            UnifiedMusicAnalysisResult advancedUnified = (UnifiedMusicAnalysisResult) advancedResult;
            result.setBeatDetectionResult(advancedUnified.getBeatDetectionResult() != null ? 
                advancedUnified.getBeatDetectionResult() : result.getBeatDetectionResult());
            result.setKeyDetectionResult(advancedUnified.getKeyDetectionResult() != null ? 
                advancedUnified.getKeyDetectionResult() : result.getKeyDetectionResult());
            result.setChordDetectionResult(advancedUnified.getChordDetectionResult() != null ? 
                advancedUnified.getChordDetectionResult() : result.getChordDetectionResult());
            
            // 复制高级分析结果 / Copy advanced analysis results
            result.setEmotionAnalysis(advancedUnified.getEmotionAnalysis());
            result.setGenreAnalysis(advancedUnified.getGenreAnalysis());
            result.setComplexityAnalysis(advancedUnified.getComplexityAnalysis());
            result.setStructuralAnalysis(advancedUnified.getStructuralAnalysis());
            result.setTempoAnalysis(advancedUnified.getTempoAnalysis());
            result.setHarmonicAnalysis(advancedUnified.getHarmonicAnalysis());
        }
        
        // 计算整体置信度 / Calculate overall confidence
        result.setConfidence(calculateOverallConfidence(result));
        
        return result;
    }

    /**
     * 计算整体置信度 / Calculate overall confidence
     */
    private double calculateOverallConfidence(UnifiedMusicAnalysisResult result) {
        double totalConfidence = 0.0;
        int count = 0;

        if (result.getBeatDetectionResult() != null) {
            totalConfidence += result.getBeatDetectionResult().getConfidence();
            count++;
        }

        if (result.getKeyDetectionResult() != null) {
            totalConfidence += result.getKeyDetectionResult().getConfidence();
            count++;
        }

        if (result.getChordDetectionResult() != null) {
            totalConfidence += result.getChordDetectionResult().getConfidence();
            count++;
        }

        // Add confidence from advanced analysis results
        if (result.getEmotionAnalysis() != null && !result.getEmotionAnalysis().isEmpty()) {
            Object confidence = result.getEmotionAnalysis().get("confidence");
            if (confidence instanceof Number) {
                totalConfidence += ((Number) confidence).doubleValue();
                count++;
            }
        }

        if (result.getGenreAnalysis() != null && !result.getGenreAnalysis().isEmpty()) {
            Object confidence = result.getGenreAnalysis().get("confidence");
            if (confidence instanceof Number) {
                totalConfidence += ((Number) confidence).doubleValue();
                count++;
            }
        }

        if (result.getComplexityAnalysis() != null && !result.getComplexityAnalysis().isEmpty()) {
            Object confidence = result.getComplexityAnalysis().get("confidence");
            if (confidence instanceof Number) {
                totalConfidence += ((Number) confidence).doubleValue();
                count++;
            }
        }

        return count > 0 ? totalConfidence / count : 0.0;
    }

    @Override
    public BeatDetectionResult analyzeBeat(AudioData audioData) throws AudioProcessingException {
        return basicAnalyzer.analyzeBeat(audioData);
    }

    @Override
    public BeatDetectionResult analyzeBeat(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        return basicAnalyzer.analyzeBeat(audioData, parameters);
    }

    @Override
    public KeyDetectionResult analyzeKey(AudioData audioData) throws AudioProcessingException {
        return basicAnalyzer.analyzeKey(audioData);
    }

    @Override
    public KeyDetectionResult analyzeKey(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        return basicAnalyzer.analyzeKey(audioData, parameters);
    }

    @Override
    public ChordDetectionResult analyzeChord(AudioData audioData) throws AudioProcessingException {
        return basicAnalyzer.analyzeChord(audioData);
    }

    @Override
    public ChordDetectionResult analyzeChord(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        return basicAnalyzer.analyzeChord(audioData, parameters);
    }

    @Override
    public MusicDetectionResult analyzeTimeRange(AudioData audioData, double startTime, double endTime) throws AudioProcessingException {
        return analyzeTimeRange(audioData, startTime, endTime, parameters);
    }

    @Override
    public MusicDetectionResult analyzeTimeRange(AudioData audioData, double startTime, double endTime, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }
        // 提取时间段的音频数据
        AudioData segmentData = audioData.extractSegment(startTime, endTime);
        return analyzeMusic(segmentData, parameters);
    }

    @Override
    public MusicDetectionResult[] analyzeStream(AudioData audioData, double windowSize, double hopSize) throws AudioProcessingException {
        return analyzeStream(audioData, windowSize, hopSize, parameters);
    }

    @Override
    public MusicDetectionResult[] analyzeStream(AudioData audioData, double windowSize, double hopSize, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }

        double duration = audioData.getDuration();
        List<MusicDetectionResult> results = new ArrayList<>();

        for (double time = 0; time < duration - windowSize; time += hopSize) {
            AudioData windowData = audioData.extractSegment(time, time + windowSize);
            MusicDetectionResult result = analyzeMusic(windowData, parameters);
            results.add(result);
        }

        return results.toArray(new MusicDetectionResult[0]);
    }

    @Override
    public String[] getSupportedParameters() {
        return new String[]{
            "confidenceThreshold", "windowSize", "hopSize",
            "enableBeatDetection", "enableKeyDetection", "enableChordDetection",
            "enableStructureAnalysis", "enableGenreDetection", "enableTempoVariation",
            "enableHarmonicAnalysis"
        };
    }

    @Override
    public Map<String, Object> getDefaultParameters() {
        Map<String, Object> defaults = new HashMap<>();
        defaults.put("confidenceThreshold", DEFAULT_CONFIDENCE_THRESHOLD);
        defaults.put("windowSize", DEFAULT_WINDOW_SIZE);
        defaults.put("hopSize", DEFAULT_HOP_SIZE);
        defaults.put("enableBeatDetection", DEFAULT_ENABLE_BEAT_DETECTION);
        defaults.put("enableKeyDetection", DEFAULT_ENABLE_KEY_DETECTION);
        defaults.put("enableChordDetection", DEFAULT_ENABLE_CHORD_DETECTION);
        defaults.put("enableStructureAnalysis", DEFAULT_ENABLE_STRUCTURE_ANALYSIS);
        defaults.put("enableGenreDetection", DEFAULT_ENABLE_GENRE_DETECTION);
        defaults.put("enableTempoVariation", DEFAULT_ENABLE_TEMPO_VARIATION);
        defaults.put("enableHarmonicAnalysis", DEFAULT_ENABLE_HARMONIC_ANALYSIS);
        return defaults;
    }

    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        if (parameters == null) {
            return true; // null parameters are valid, will use defaults
        }
        
        // Validate parameters
        for (String key : parameters.keySet()) {
            if (!java.util.Arrays.asList(getSupportedParameters()).contains(key)) {
                return false;
            }
        }
        
        return true;
    }

    @Override
    public void setParameters(Map<String, Object> parameters) throws AudioProcessingException {
        if (parameters == null) {
            return;
        }
        
        // Validate parameters
        if (!validateParameters(parameters)) {
            throw new AudioProcessingException("Invalid parameters provided");
        }
        
        // Update parameters
        this.parameters.putAll(parameters);
        
        // Also update parameters in the underlying analyzers
        basicAnalyzer.setParameters(parameters);
        advancedAnalyzer.setAdvancedParameters(parameters);
    }

    @Override
    public Map<String, Object> getCurrentParameters() {
        return new HashMap<>(parameters);
    }

    @Override
    public void resetParameters() {
        parameters.clear();
        initializeDefaultParameters();
        basicAnalyzer.resetParameters();
        advancedAnalyzer.resetAdvancedParameters();
    }

    @Override
    public String getVersion() {
        return "1.0";
    }

    @Override
    public String getName() {
        return "comprehensive_music_analyzer";
    }

    @Override
    public String getDescription() {
        return "Provides comprehensive music analysis features by integrating BasicMusicAnalyzer and AdvancedMusicAnalyzer.";
    }

    @Override
    public boolean supportsAudioFormat(double sampleRate, int channels, int bitDepth) {
        return basicAnalyzer.supportsAudioFormat(sampleRate, channels, bitDepth);
    }

    @Override
    public double getMinimumAudioLength() {
        return basicAnalyzer.getMinimumAudioLength();
    }

    @Override
    public double getMaximumAudioLength() {
        return basicAnalyzer.getMaximumAudioLength();
    }

    @Override
    public double getComplexityEstimate(double audioLength) {
        return basicAnalyzer.getComplexityEstimate(audioLength) + advancedAnalyzer.getComplexityEstimate(audioLength);
    }

    @Override
    public void warmUp() throws AudioProcessingException {
        basicAnalyzer.warmUp();
        advancedAnalyzer.warmUp();
    }

    @Override
    public void cleanup() {
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdown();
        }
        basicAnalyzer.cleanup();
        advancedAnalyzer.cleanup();
    }

    @Override
    public String getStatus() {
        return "ready";
    }

    @Override
    public boolean isReady() {
        return basicAnalyzer.isReady();
    }

    @Override
    public Map<String, Object> getLastAnalysisStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("progress", analysisProgress);
        stats.put("cancelled", analysisCancelled);
        stats.put("paused", analysisPaused);
        return stats;
    }

    @Override
    public Map<String, Object> getPerformanceMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        metrics.put("progress", analysisProgress);
        return metrics;
    }

    @Override
    public void setVerboseLogging(boolean enabled) {
        basicAnalyzer.setVerboseLogging(enabled);
    }

    @Override
    public boolean isVerboseLoggingEnabled() {
        return basicAnalyzer.isVerboseLoggingEnabled();
    }
}