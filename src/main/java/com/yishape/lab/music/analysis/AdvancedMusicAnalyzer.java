package com.yishape.lab.music.analysis;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.music.analysis.advanced.IAdvancedAnalyzer;
import com.yishape.lab.music.analysis.advanced.ComplexityAnalyzer;
import com.yishape.lab.music.analysis.advanced.EmotionAnalyzer;
import com.yishape.lab.music.analysis.advanced.GenreAnalyzer;
import com.yishape.lab.audio.preprocessing.AudioPreprocessor;
import com.yishape.lab.audio.preprocessing.AudioPreprocessingOptions;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.music.analysis.advanced.StructureAnalyzer;

import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 高级音乐分析器实现 / Advanced Music Analyzer Implementation
 * <p>
 * 提供高级的音乐分析功能，代理给advanced包中的具体实现类。 Provides advanced music analysis features by
 * delegating to specific implementation classes in the advanced package.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AdvancedMusicAnalyzer implements IAdvancedAnalyzer {

    private final IAdvancedAnalyzer emotionAnalyzer;
    private final IAdvancedAnalyzer genreAnalyzer;
    private final IAdvancedAnalyzer complexityAnalyzer;
    private final IAdvancedAnalyzer structureAnalyzer;
    private final ExecutorService executorService;
    private Map<String, Object> currentParameters;

    // 结果缓存 / Result cache
    private final ConcurrentHashMap<String, UnifiedMusicAnalysisResult> resultCache;
    private static final int MAX_CACHE_SIZE = 50; // 最大缓存大小 / Maximum cache size

    /**
     * 音频预处理器 / Audio preprocessor
     */
    private final AudioPreprocessor audioPreprocessor;

    /**
     * 预处理选项 / Preprocessing options
     */
    private AudioPreprocessingOptions preprocessingOptions;

    /**
     * 性能监控器 / Performance monitor
     */
    private final PerformanceMonitor performanceMonitor;

    // 默认参数 / Default parameters
    private static final double DEFAULT_EMOTION_THRESHOLD = 0.6;
    private static final double DEFAULT_GENRE_THRESHOLD = 0.7;
    private static final double DEFAULT_COMPLEXITY_THRESHOLD = 0.5;

    public AdvancedMusicAnalyzer() {
        this.emotionAnalyzer = new EmotionAnalyzer();
        this.genreAnalyzer = new GenreAnalyzer();
        this.complexityAnalyzer = new ComplexityAnalyzer();
        this.structureAnalyzer = new StructureAnalyzer();
        this.executorService = Executors.newFixedThreadPool(4); // Reduced thread pool size for advanced-only analysis
        this.currentParameters = getDefaultAdvancedParameters();
        this.resultCache = new ConcurrentHashMap<>();
        this.audioPreprocessor = new AudioPreprocessor();
        this.preprocessingOptions = AudioPreprocessingOptions.getDefault();
        this.performanceMonitor = PerformanceMonitor.getInstance();
    }

    /**
     * 执行高级音乐分析（使用默认参数） / Execute advanced music analysis (with default
     * parameters)
     *
     * @param audioData 音频数据 / Audio data
     * @return 分析结果 / Analysis result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    @Override
    public MusicDetectionResult analyzeAdvancedMusic(AudioData audioData) throws AudioProcessingException {
        return analyzeAdvancedMusic(audioData, getDefaultAdvancedParameters());
    }

    /**
     * 执行高级音乐分析 / Execute advanced music analysis
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 分析参数 / Analysis parameters
     * @return 分析结果 / Analysis result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    @Override
    public MusicDetectionResult analyzeAdvancedMusic(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }

        // Use a final variable for lambda expressions
        final Map<String, Object> effectiveParameters = parameters != null ? parameters : this.currentParameters;

        // 检查缓存 / Check cache
        String cacheKey = generateCacheKey(audioData, effectiveParameters);
        UnifiedMusicAnalysisResult cachedResult = resultCache.get(cacheKey);
        if (cachedResult != null) {
            return cachedResult;
        }

        try {
            // 计算音频时长（秒）/ Calculate audio duration (seconds)
            double audioDuration = audioData.getDuration();

            // 根据音频时长动态调整超时时间 / Dynamically adjust timeout based on audio duration
            long timeoutSeconds = calculateTimeoutForAudioDuration(audioDuration);

            // 记录分析开始信息 / Log analysis start information
            if (audioDuration > 60) {
                System.out.println("Starting advanced music analysis for " + String.format("%.1f", audioDuration) + " second audio file...");
                System.out.println("Estimated timeout: " + timeoutSeconds + " seconds");
            }

            // 并行执行高级分析 / Execute advanced analysis in parallel
            CompletableFuture<Map<String, Object>> emotionFuture = CompletableFuture.supplyAsync(() -> {
                try {
                    if (audioDuration > 60) {
                        System.out.println("Starting emotion analysis...");
                    }
                    Map<String, Object> result = emotionAnalyzer.analyze(audioData, effectiveParameters);
                    if (audioDuration > 60) {
                        System.out.println("Emotion analysis completed");
                    }
                    return result;
                } catch (Exception e) {
                    throw new RuntimeException("Emotion analysis failed", e);
                }
            }, executorService);

            CompletableFuture<Map<String, Object>> genreFuture = CompletableFuture.supplyAsync(() -> {
                try {
                    if (audioDuration > 60) {
                        System.out.println("Starting genre analysis...");
                    }
                    Map<String, Object> result = genreAnalyzer.analyze(audioData, effectiveParameters);
                    if (audioDuration > 60) {
                        System.out.println("Genre analysis completed");
                    }
                    return result;
                } catch (Exception e) {
                    throw new RuntimeException("Genre analysis failed", e);
                }
            }, executorService);

            CompletableFuture<Map<String, Object>> complexityFuture = CompletableFuture.supplyAsync(() -> {
                try {
                    if (audioDuration > 60) {
                        System.out.println("Starting complexity analysis...");
                    }
                    Map<String, Object> result = complexityAnalyzer.analyze(audioData, effectiveParameters);
                    if (audioDuration > 60) {
                        System.out.println("Complexity analysis completed");
                    }
                    return result;
                } catch (Exception e) {
                    throw new RuntimeException("Complexity analysis failed", e);
                }
            }, executorService);

            CompletableFuture<Map<String, Object>> structureFuture = CompletableFuture.supplyAsync(() -> {
                try {
                    if (audioDuration > 60) {
                        System.out.println("Starting structure analysis...");
                    }
                    Map<String, Object> result = structureAnalyzer.analyze(audioData, effectiveParameters);
                    if (audioDuration > 60) {
                        System.out.println("Structure analysis completed");
                    }
                    return result;
                } catch (Exception e) {
                    throw new RuntimeException("Structure analysis failed", e);
                }
            }, executorService);

            // 等待所有分析完成（带超时）/ Wait for all analyses to complete (with timeout)
            CompletableFuture<Void> allFutures = CompletableFuture.allOf(emotionFuture, genreFuture, complexityFuture, structureFuture);

            try {
                allFutures.get(timeoutSeconds, TimeUnit.SECONDS); // 动态超时 / Dynamic timeout
            } catch (TimeoutException e) {
                throw new AudioProcessingException("Advanced music analysis timed out after " + timeoutSeconds + " seconds for audio duration " + String.format("%.1f", audioDuration) + " seconds", e);
            } catch (Exception e) {
                throw new AudioProcessingException("Error in parallel analysis execution", e);
            }

            // 获取结果 / Get results
            Map<String, Object> emotionResult = emotionFuture.getNow(null);
            Map<String, Object> genreResult = genreFuture.getNow(null);
            Map<String, Object> complexityResult = complexityFuture.getNow(null);
            Map<String, Object> structureResult = structureFuture.getNow(null);

            // 创建统一结果 / Create unified result (advanced-only)
            UnifiedMusicAnalysisResult result = new UnifiedMusicAnalysisResult();

            // 仅设置高级分析结果 / Set only advanced analysis results
            result.setEmotionAnalysis(emotionResult);
            result.setGenreAnalysis(genreResult);
            result.setComplexityAnalysis(complexityResult);
            result.setStructuralAnalysis(structureResult);

            // 设置全局置信度 / Set overall confidence (advanced-only)
            double overallConfidence = calculateOverallConfidence(result);
            result.setConfidence(overallConfidence);
            result.setAlgorithm("advanced_music_analyzer_parallel_cached_preprocessed");

            // 缓存结果 / Cache result
            cacheResult(cacheKey, result);

            return result;

        } catch (Exception e) {
            throw new AudioProcessingException("Error in advanced music analysis: " + e.getMessage(), e);
        }
    }

    /**
     * 设置预处理选项 / Set preprocessing options
     */
    public void setPreprocessingOptions(AudioPreprocessingOptions options) {
        this.preprocessingOptions = options;
    }

    /**
     * 获取预处理选项 / Get preprocessing options
     */
    public AudioPreprocessingOptions getPreprocessingOptions() {
        return preprocessingOptions;
    }

    /**
     * 根据音频时长计算超时时间 / Calculate timeout based on audio duration
     *
     * @param audioDuration 音频时长（秒）/ Audio duration (seconds)
     * @return 超时时间（秒）/ Timeout (seconds)
     */
    private long calculateTimeoutForAudioDuration(double audioDuration) {
        // 基础超时时间：15秒 / Base timeout: 15 seconds (reduced from 30 seconds)
        long baseTimeout = 15;

        // 对于短音频（< 30秒），使用基础超时 / For short audio (< 30 seconds), use base timeout
        if (audioDuration <= 30) {
            return baseTimeout;
        }

        // 对于中等长度音频（30-150秒），线性增加超时 / For medium length audio (30-150 seconds), increase timeout linearly
        if (audioDuration <= 150) {
            return baseTimeout + (long) ((audioDuration - 30) * 0.3); // 每增加30秒，增加9秒超时
        }

        // 对于长音频（> 150秒），使用更保守的增长 / For long audio (> 150 seconds), use more conservative growth
        return Math.min(120, baseTimeout + (long) ((audioDuration - 30) * 0.2)); // 最多2分钟超时 (reduced from 5 minutes)
    }

    /**
     * 获取音频预处理器 / Get audio preprocessor
     */
    public AudioPreprocessor getAudioPreprocessor() {
        return audioPreprocessor;
    }

    /**
     * 获取性能监控器 / Get performance monitor
     */
    public PerformanceMonitor getPerformanceMonitor() {
        return performanceMonitor;
    }

    /**
     * 获取性能统计 / Get performance statistics
     */
    public PerformanceMonitor.PerformanceStatistics getPerformanceStatistics() {
        return performanceMonitor.getStatistics();
    }

    /**
     * 重置性能统计 / Reset performance statistics
     */
    public void resetPerformanceStatistics() {
        performanceMonitor.reset();
    }

    /**
     * 获取支持的参数 / Get supported parameters
     *
     * @return 支持的参数数组 / Array of supported parameters
     */
    public String[] getSupportedAdvancedParameters() {
        return new String[]{"emotionThreshold", "genreThreshold", "complexityThreshold",
            "windowSize", "hopSize", "confidenceThreshold"};
    }

    @Override
    public String[] getSupportedParameters() {
        return getSupportedAdvancedParameters();
    }

    /**
     * 获取默认参数 / Get default parameters
     *
     * @return 默认参数映射 / Default parameters map
     */
    public Map<String, Object> getDefaultAdvancedParameters() {
        Map<String, Object> params = new HashMap<>();
        params.put("emotionThreshold", DEFAULT_EMOTION_THRESHOLD);
        params.put("genreThreshold", DEFAULT_GENRE_THRESHOLD);
        params.put("complexityThreshold", DEFAULT_COMPLEXITY_THRESHOLD);
        params.put("windowSize", 4096);
        params.put("hopSize", 2048);
        params.put("confidenceThreshold", 0.6);
        return params;
    }

    @Override
    public Map<String, Object> getDefaultParameters() {
        return getDefaultAdvancedParameters();
    }

    /**
     * 计算高级分析的总体置信度 / Calculate overall confidence for advanced analysis only
     */
    private double calculateOverallConfidence(UnifiedMusicAnalysisResult result) {
        double totalConfidence = 0.0;
        double totalWeight = 0.0;

// 仅为高级分析类型分配权重（移除基础分析权重）
// Assign weights only to advanced analysis types (remove basic analysis weights)
        double emotionWeight = 0.30;
        double genreWeight = 0.35;
        double complexityWeight = 0.20;
        double structureWeight = 0.15;

        // Add confidence from advanced analysis results only
        if (result.getEmotionAnalysis() != null && !result.getEmotionAnalysis().isEmpty()) {
            Object confidence = result.getEmotionAnalysis().get("confidence");
            if (confidence instanceof Number) {
                totalConfidence += ((Number) confidence).doubleValue() * emotionWeight;
                totalWeight += emotionWeight;
            }
        }

        if (result.getGenreAnalysis() != null && !result.getGenreAnalysis().isEmpty()) {
            Object confidence = result.getGenreAnalysis().get("confidence");
            if (confidence instanceof Number) {
                totalConfidence += ((Number) confidence).doubleValue() * genreWeight;
                totalWeight += genreWeight;
            }
        }

        if (result.getComplexityAnalysis() != null && !result.getComplexityAnalysis().isEmpty()) {
            Object confidence = result.getComplexityAnalysis().get("confidence");
            if (confidence instanceof Number) {
                totalConfidence += ((Number) confidence).doubleValue() * complexityWeight;
                totalWeight += complexityWeight;
            }
        }
        if (result.getStructuralAnalysis() != null && !result.getStructuralAnalysis().isEmpty()) {
            Object confidence = result.getStructuralAnalysis().get("confidence");
            if (confidence instanceof Number) {
                totalConfidence += ((Number) confidence).doubleValue() * structureWeight;
                totalWeight += structureWeight;
            }
        }

        return totalWeight > 0 ? totalConfidence / totalWeight : 0.5;
    }

    // AdvancedMusicAnalyzer 不再实现基础分析功能 / AdvancedMusicAnalyzer no longer implements basic analysis functionality
    // 基础分析功能应由 BasicMusicAnalyzer 提供 / Basic analysis functionality should be provided by BasicMusicAnalyzer
    /**
     * 验证高级分析参数 / Validate advanced analysis parameters
     *
     * @param parameters 参数映射 / Parameters map
     * @return 是否有效 / Whether valid
     */
    public boolean validateAdvancedParameters(Map<String, Object> parameters) {
        if (parameters == null) {
            return true; // null parameters are valid, will use defaults
        }

        // Validate parameters
        for (String key : parameters.keySet()) {
            if (!java.util.Arrays.asList(getSupportedAdvancedParameters()).contains(key)) {
                return false;
            }
        }

        return true;
    }

    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        return validateAdvancedParameters(parameters);
    }

    /**
     * 设置高级分析参数 / Set advanced analysis parameters
     *
     * @param parameters 参数映射 / Parameters map
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public void setAdvancedParameters(Map<String, Object> parameters) throws AudioProcessingException {
        if (parameters == null) {
            return;
        }

        // Validate parameters
        if (!validateAdvancedParameters(parameters)) {
            throw new AudioProcessingException("Invalid parameters provided");
        }

        this.currentParameters = parameters;
    }

    @Override
    public void setParameters(Map<String, Object> parameters) throws AudioProcessingException {
        setAdvancedParameters(parameters);
    }

    /**
     * 获取当前高级分析参数 / Get current advanced analysis parameters
     *
     * @return 当前参数映射 / Current parameters map
     */
    public Map<String, Object> getCurrentAdvancedParameters() {
        return this.currentParameters;
    }

    @Override
    public Map<String, Object> getCurrentParameters() {
        return getCurrentAdvancedParameters();
    }

    /**
     * 重置高级分析参数为默认值 / Reset advanced analysis parameters to defaults
     */
    public void resetAdvancedParameters() {
        this.currentParameters = getDefaultAdvancedParameters();
    }

    @Override
    public void resetParameters() {
        resetAdvancedParameters();
    }

    /**
     * 获取分析器版本 / Get analyzer version
     *
     * @return 版本字符串 / Version string
     */
    public String getVersion() {
        return "1.0";
    }

    /**
     * 获取分析器名称 / Get analyzer name
     *
     * @return 名称字符串 / Name string
     */
    public String getName() {
        return "AdvancedMusicAnalyzer";
    }

    /**
     * 生成缓存键 / Generate cache key
     */
    private String generateCacheKey(AudioData audioData, Map<String, Object> parameters) {
        StringBuilder keyBuilder = new StringBuilder();
        keyBuilder.append(audioData.getSampleRate()).append("_");
        keyBuilder.append(audioData.getSamples().length()).append("_");
        keyBuilder.append(audioData.getChannels()).append("_");

        // 添加音频数据的简单哈希（前1000个样本）/ Add simple hash of audio data (first 1000 samples)
        IVector<Double> samples = audioData.getSamples();
        int sampleCount = Math.min(1000, samples.length());
        double hashSum = 0.0;
        for (int i = 0; i < sampleCount; i++) {
            hashSum += Math.abs(samples.get(i));
        }
        keyBuilder.append((int) (hashSum * 1000)).append("_");

        // 添加参数哈希 / Add parameter hash
        if (parameters != null && !parameters.isEmpty()) {
            keyBuilder.append(parameters.hashCode());
        }

        return keyBuilder.toString();
    }

    /**
     * 缓存结果 / Cache result
     */
    private void cacheResult(String cacheKey, UnifiedMusicAnalysisResult result) {
        if (resultCache.size() >= MAX_CACHE_SIZE) {
            // 简单的LRU策略：移除第一个元素 / Simple LRU: remove first element
            String firstKey = resultCache.keySet().iterator().next();
            resultCache.remove(firstKey);
        }
        resultCache.put(cacheKey, result);
    }

    /**
     * 清除缓存 / Clear cache
     */
    public void clearCache() {
        resultCache.clear();
    }

    /**
     * 获取缓存大小 / Get cache size
     */
    public int getCacheSize() {
        return resultCache.size();
    }

    /**
     * 获取分析器描述 / Get analyzer description
     *
     * @return 描述字符串 / Description string
     */
    public String getDescription() {
        return "Advanced music analyzer with emotion, genre and complexity analysis by delegating to advanced package implementations";
    }

    /**
     * 预热分析器 / Warm up analyzer
     */
    public void warmUp() throws AudioProcessingException {
        // 高级分析器预热逻辑可在此添加 / Advanced analyzer warm-up logic can be added here
    }

    /**
     * 清理分析器资源 / Clean up analyzer resources
     */
    public void cleanup() {
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdown();
        }
        // 高级分析器清理逻辑可在此添加 / Advanced analyzer cleanup logic can be added here
    }

    @Override
    public String getStatus() {
        return "Ready";
    }

    @Override
    public boolean isReady() {
        return true;
    }

    @Override
    public Map<String, Object> getLastAnalysisStatistics() {
        // Return empty map for now
        return new HashMap<>();
    }

    @Override
    public Map<String, Object> getPerformanceMetrics() {
        // Return empty map for now
        return new HashMap<>();
    }

    @Override
    public void setVerboseLogging(boolean enabled) {
        // 高级分析器日志设置 / Advanced analyzer logging settings
        // 可以在这里添加高级分析器的日志逻辑 / Add advanced analyzer logging logic here if needed
    }

    @Override
    public boolean isVerboseLoggingEnabled() {
        return false; // 默认不启用详细日志 / Default to no verbose logging
    }

    @Override
    public double getMinimumAudioLength() {
        return 0.1; // 100ms minimum for advanced analysis
    }

    @Override
    public double getMaximumAudioLength() {
        return 3600.0; // 1 hour maximum for advanced analysis
    }

    @Override
    public double getComplexityEstimate(double audioLength) {
        // 高级分析复杂度估计：情感+风格+复杂度+结构分析 / Advanced analysis complexity estimate: emotion + genre + complexity + structure analysis
        double baseComplexity = 4.0; // 四个分析器的开销 / Four analyzers overhead
        double lengthFactor = Math.log10(audioLength + 1) * 0.5;
        return baseComplexity + lengthFactor;
    }

    @Override
    public boolean supportsAudioFormat(double sampleRate, int channels, int bitDepth) {
        // 支持常见的音频格式 / Support common audio formats
        return sampleRate >= 8000 && sampleRate <= 192000
                && channels >= 1 && channels <= 8
                && bitDepth >= 8 && bitDepth <= 32;
    }

    // IAdvancedAnalyzer interface methods
    @Override
    public Map<String, Object> analyze(AudioData audioData) throws AudioProcessingException {
        return analyze(audioData, getDefaultParameters());
    }

    @Override
    public Map<String, Object> analyze(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        // 音频预处理 / Audio preprocessing
        AudioData processedAudio = audioPreprocessor.preprocess(audioData);

        // Delegate to specific analyzers based on the analysis type
        // For now, we'll return a combined result
        MusicDetectionResult result = analyzeAdvancedMusic(processedAudio, parameters);
        Map<String, Object> resultMap = new HashMap<>();
        resultMap.put("result", result);
        return resultMap;
    }

    /**
     * 分析音频数据（从double[]数组） / Analyze audio data (from double[] array)
     *
     * @param audioData 音频样本数据 / Audio sample data
     * @param sampleRate 采样率 / Sample rate
     * @return 统一音乐分析结果 / Unified music analysis result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public UnifiedMusicAnalysisResult analyze(double[] audioData, double sampleRate) throws AudioProcessingException {
        return analyze(audioData, sampleRate, getDefaultParameters());
    }

    /**
     * 分析音频数据（从double[]数组，带参数） / Analyze audio data (from double[] array, with
     * parameters)
     *
     * @param audioData 音频样本数据 / Audio sample data
     * @param sampleRate 采样率 / Sample rate
     * @param parameters 分析参数 / Analysis parameters
     * @return 统一音乐分析结果 / Unified music analysis result
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public UnifiedMusicAnalysisResult analyze(double[] audioData, double sampleRate, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }
        if (sampleRate <= 0) {
            throw new AudioProcessingException("Sample rate must be positive");
        }

        // 将double[]转换为AudioData / Convert double[] to AudioData
        IVector<Double> samples = Linalg.vector(audioData);
        AudioData audioDataObj = new AudioData(samples, sampleRate, 1, 16, AudioFormat.WAV);

        // 执行高级音乐分析 / Perform advanced music analysis
        MusicDetectionResult result = analyzeAdvancedMusic(audioDataObj, parameters);

        // 确保返回UnifiedMusicAnalysisResult / Ensure return UnifiedMusicAnalysisResult
        if (result instanceof UnifiedMusicAnalysisResult) {
            return (UnifiedMusicAnalysisResult) result;
        } else {
            throw new AudioProcessingException("Expected UnifiedMusicAnalysisResult but got " + result.getClass().getSimpleName());
        }
    }

    @Override
    public String getAnalyzerName() {
        return getName();
    }
}
