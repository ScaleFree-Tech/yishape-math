package com.reremouse.lab.music.analysis;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.music.analysis.basic.BeatDetectionResult;
import com.reremouse.lab.music.analysis.basic.KeyDetectionResult;
import com.reremouse.lab.music.analysis.basic.ChordDetectionResult;
import com.reremouse.lab.music.analysis.basic.IBeatAnalyzer;
import com.reremouse.lab.music.analysis.basic.IKeyAnalyzer;
import com.reremouse.lab.music.analysis.basic.IChordAnalyzer;
import com.reremouse.lab.music.analysis.basic.BeatAnalyzerImpl;
import com.reremouse.lab.music.analysis.basic.KeyAnalyzerImpl;
import com.reremouse.lab.music.analysis.basic.ChordAnalyzerImpl;
import com.reremouse.lab.music.analysis.ConfidenceCalculator;
import com.reremouse.lab.music.analysis.StandardizedConfidenceCalculator;

import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * 基础音乐分析器 / Basic Music Analyzer
 * <p>
 * 提供基础的音乐分析功能，代理给basic包中的具体实现类。
 * Provides basic music analysis features by delegating to specific implementation classes in the basic package.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class BasicMusicAnalyzer implements IMusicAnalyzer {

    private final IBeatAnalyzer beatAnalyzer;
    private final IKeyAnalyzer keyAnalyzer;
    private final IChordAnalyzer chordAnalyzer;
    private final ExecutorService executorService;
    private final StandardizedConfidenceCalculator confidenceCalculator = new StandardizedConfidenceCalculator();

    // 默认参数 / Default parameters
    private static final double DEFAULT_CONFIDENCE_THRESHOLD = 0.7;
    private static final int DEFAULT_WINDOW_SIZE = 2048;
    private static final int DEFAULT_HOP_SIZE = 512;
    private static final boolean DEFAULT_ENABLE_BEAT_DETECTION = true;
    private static final boolean DEFAULT_ENABLE_KEY_DETECTION = true;
    private static final boolean DEFAULT_ENABLE_CHORD_DETECTION = true;

    // 当前参数 / Current parameters
    private final Map<String, Object> parameters = new HashMap<>();

    // 分析状态 / Analysis state
    private volatile boolean analysisCancelled = false;
    private volatile boolean analysisPaused = false;
    private volatile double analysisProgress = 0.0;
    private String status = "ready";
    private boolean verboseLogging = false;

    /**
     * 构造函数 / Constructor
     */
    public BasicMusicAnalyzer() {
        this.beatAnalyzer = new BeatAnalyzerImpl();
        this.keyAnalyzer = new KeyAnalyzerImpl();
        this.chordAnalyzer = new ChordAnalyzerImpl();
        // Increase thread pool size to prevent potential deadlocks
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

        // Create a final variable for use in lambda expressions
        final Map<String, Object> effectiveParameters = parameters != null ? parameters : this.parameters;

        try {
            analysisProgress = 0.0;
            analysisCancelled = false;

            // 并行执行节拍和调性分析 / Execute beat and key analyses in parallel
            CompletableFuture<BeatDetectionResult> beatFuture = CompletableFuture.supplyAsync(() -> {
                try {
                    boolean enableBeat = (Boolean) effectiveParameters.getOrDefault("enableBeatDetection", DEFAULT_ENABLE_BEAT_DETECTION);
                    if (enableBeat) {
                        analysisProgress = 20.0;
                        return beatAnalyzer.detectBeats(audioData, effectiveParameters);
                    }
                    return null;
                } catch (AudioProcessingException e) {
                    throw new RuntimeException(e);
                }
            }, executorService);

            CompletableFuture<KeyDetectionResult> keyFuture = CompletableFuture.supplyAsync(() -> {
                try {
                    boolean enableKey = (Boolean) effectiveParameters.getOrDefault("enableKeyDetection", DEFAULT_ENABLE_KEY_DETECTION);
                    if (enableKey) {
                        analysisProgress = 50.0;
                        return keyAnalyzer.detectKey(audioData, effectiveParameters);
                    }
                    return null;
                } catch (AudioProcessingException e) {
                    throw new RuntimeException(e);
                }
            }, executorService);

            // 等待节拍和调性分析完成 / Wait for beat and key analyses to complete
            BeatDetectionResult beatResult = null;
            KeyDetectionResult keyResult = null;
            
            try {
                // Add timeout to prevent indefinite hanging
                beatResult = beatFuture.get(30, TimeUnit.SECONDS);
            } catch (TimeoutException e) {
                System.err.println("Beat analysis timed out");
                beatFuture.cancel(true);
            } catch (InterruptedException e) {
                System.err.println("Beat analysis interrupted");
                Thread.currentThread().interrupt();
            } catch (ExecutionException e) {
                // Handle exception in beat analysis
                if (verboseLogging) {
                    System.err.println("Beat analysis failed: " + e.getCause().getMessage());
                    e.getCause().printStackTrace();
                }
            }
            
            try {
                // Add timeout to prevent indefinite hanging
                keyResult = keyFuture.get(30, TimeUnit.SECONDS);
            } catch (TimeoutException e) {
                System.err.println("Key analysis timed out");
                keyFuture.cancel(true);
            } catch (InterruptedException e) {
                System.err.println("Key analysis interrupted");
                Thread.currentThread().interrupt();
            } catch (ExecutionException e) {
                // Handle exception in key analysis
                if (verboseLogging) {
                    System.err.println("Key analysis failed: " + e.getCause().getMessage());
                    e.getCause().printStackTrace();
                }
            }

            // 准备带调性上下文的和弦分析参数 / Prepare chord analysis parameters with key context
            Map<String, Object> chordParameters = new HashMap<>(effectiveParameters);
            if (keyResult != null && keyResult.getKeyName() != null && keyResult.getScaleType() != null) {
                chordParameters.put("detectedKey", keyResult.getKeyName());
                chordParameters.put("keyScaleType", keyResult.getScaleType());
            }

            // 执行和弦分析（带调性上下文）/ Execute chord analysis with key context
            List<ChordDetectionResult> chordResults = new ArrayList<>();
            try {
                boolean enableChord = (Boolean) effectiveParameters.getOrDefault("enableChordDetection", DEFAULT_ENABLE_CHORD_DETECTION);
                if (enableChord) {
                    analysisProgress = 80.0;
                    chordResults = chordAnalyzer.detectChords(audioData, chordParameters);
                }
            } catch (AudioProcessingException e) {
                if (verboseLogging) {
                    System.err.println("Chord analysis failed: " + e.getMessage());
                    e.printStackTrace();
                }
                chordResults = new ArrayList<>();
            }

            // 创建统一分析结果 / Create unified analysis result
            UnifiedMusicAnalysisResult result = new UnifiedMusicAnalysisResult();
            result.setAlgorithm("basic_music_analyzer");

            // 设置结果 / Set results
            result.setBeatDetectionResult(beatResult);
            result.setKeyDetectionResult(keyResult);
            if (!chordResults.isEmpty()) {
                result.setChordDetectionResult(chordResults.get(0));
            }

            analysisProgress = 100.0;
            
            // Apply validation and improvement mechanisms with multi-algorithm validation
            result = validateAndImproveResultWithMultiAlgorithm(audioData, result);
            
            return result;

        } catch (Exception e) {
            throw new AudioProcessingException("Basic music analysis failed: " + e.getMessage(), e);
        }
    }

    @Override
    public BeatDetectionResult analyzeBeat(AudioData audioData) throws AudioProcessingException {
        return analyzeBeat(audioData, parameters);
    }

    @Override
    public BeatDetectionResult analyzeBeat(AudioData audioData, Map<String, Object> params) throws AudioProcessingException {
        try {
            return beatAnalyzer.detectBeats(audioData, params != null ? params : parameters);
        } catch (Exception e) {
            throw new AudioProcessingException("Beat analysis failed: " + e.getMessage(), e);
        }
    }

    @Override
    public KeyDetectionResult analyzeKey(AudioData audioData) throws AudioProcessingException {
        return analyzeKey(audioData, parameters);
    }

    @Override
    public KeyDetectionResult analyzeKey(AudioData audioData, Map<String, Object> params) throws AudioProcessingException {
        try {
            return keyAnalyzer.detectKey(audioData, params != null ? params : parameters);
        } catch (Exception e) {
            throw new AudioProcessingException("Key analysis failed: " + e.getMessage(), e);
        }
    }

    @Override
    public ChordDetectionResult analyzeChord(AudioData audioData) throws AudioProcessingException {
        return analyzeChord(audioData, parameters);
    }

    @Override
    public ChordDetectionResult analyzeChord(AudioData audioData, Map<String, Object> params) throws AudioProcessingException {
        try {
            List<ChordDetectionResult> results = chordAnalyzer.detectChords(audioData, params != null ? params : parameters);
            return results.isEmpty() ? null : results.get(0);
        } catch (Exception e) {
            throw new AudioProcessingException("Chord analysis failed: " + e.getMessage(), e);
        }
    }

    @Override
    public MusicDetectionResult analyzeTimeRange(AudioData audioData, double startTime, double endTime) throws AudioProcessingException {
        return analyzeTimeRange(audioData, startTime, endTime, parameters);
    }

    @Override
    public MusicDetectionResult analyzeTimeRange(AudioData audioData, double startTime, double endTime, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // Extract audio segment
            AudioData segment = audioData.extractSegment(startTime, endTime);
            return analyzeMusic(segment, parameters);
        } catch (Exception e) {
            throw new AudioProcessingException("Time range analysis failed: " + e.getMessage(), e);
        }
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
        return new String[]{"confidenceThreshold", "windowSize", "hopSize",
                           "enableBeatDetection", "enableKeyDetection", "enableChordDetection"};
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
    public Map<String, Object> getDefaultParameters() {
        Map<String, Object> params = new HashMap<>();
        params.put("confidenceThreshold", DEFAULT_CONFIDENCE_THRESHOLD);
        params.put("windowSize", DEFAULT_WINDOW_SIZE);
        params.put("hopSize", DEFAULT_HOP_SIZE);
        params.put("enableBeatDetection", DEFAULT_ENABLE_BEAT_DETECTION);
        params.put("enableKeyDetection", DEFAULT_ENABLE_KEY_DETECTION);
        params.put("enableChordDetection", DEFAULT_ENABLE_CHORD_DETECTION);
        return params;
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
    }

    @Override
    public Map<String, Object> getCurrentParameters() {
        return new HashMap<>(parameters);
    }

    @Override
    public void resetParameters() {
        parameters.clear();
        initializeDefaultParameters();
    }

    @Override
    public String getVersion() {
        return "1.0";
    }

    @Override
    public String getName() {
        return "BasicMusicAnalyzer";
    }

    @Override
    public String getDescription() {
        return "Basic music analyzer providing beat detection, key analysis, and chord recognition by delegating to basic package implementations";
    }

    @Override
    public boolean supportsAudioFormat(double sampleRate, int channels, int bitDepth) {
        // Support common audio formats
        return sampleRate > 0 && channels > 0 && bitDepth > 0;
    }

    @Override
    public double getMinimumAudioLength() {
        return 0.1; // 100ms minimum
    }

    @Override
    public double getMaximumAudioLength() {
        return 3600.0; // 1 hour maximum
    }

    @Override
    public double getComplexityEstimate(double audioLength) {
        // Simple complexity estimation
        return audioLength * 0.1;
    }

    @Override
    public void warmUp() throws AudioProcessingException {
        // No specific warm-up needed for basic analyzer
        status = "ready";
    }

    @Override
    public void cleanup() {
        // Clean up resources if needed
        parameters.clear();
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdown();
        }
    }

    @Override
    public String getStatus() {
        return status;
    }

    @Override
    public boolean isReady() {
        return "ready".equals(status);
    }

    @Override
    public Map<String, Object> getLastAnalysisStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("analysisProgress", analysisProgress);
        stats.put("analysisCancelled", analysisCancelled);
        return stats;
    }

    @Override
    public Map<String, Object> getPerformanceMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        metrics.put("lastAnalysisProgress", analysisProgress);
        return metrics;
    }

    @Override
    public void setVerboseLogging(boolean enabled) {
        verboseLogging = enabled;
    }

    @Override
    public boolean isVerboseLoggingEnabled() {
        return verboseLogging;
    }

    /**
     * 计算整体置信度 / Calculate overall confidence
     */
    private double calculateOverallConfidence(UnifiedMusicAnalysisResult result) {
        return calculateImprovedOverallConfidence(result);
    }

    /**
     * 改进的整体置信度计算 / Improved overall confidence calculation
     */
    private double calculateImprovedOverallConfidence(UnifiedMusicAnalysisResult result) {
        try {
            // 动态权重分配 / Dynamic weight allocation
            double[] weights = calculateDynamicWeights(result);
            double beatWeight = weights[0];
            double keyWeight = weights[1];
            double chordWeight = weights[2];

            // 基础置信度计算 / Base confidence calculation
            double baseConfidence = calculateBaseConfidence(result, beatWeight, keyWeight, chordWeight);

            // 交叉验证调整 / Cross-validation adjustment
            double crossValidationFactor = calculateCrossValidationFactor(result);

            // 一致性检查 / Consistency check
            double consistencyFactor = calculateConsistencyFactor(result);

            // 综合置信度 / Combined confidence
            double finalConfidence = baseConfidence * crossValidationFactor * consistencyFactor;

            // 使用标准化置信度计算器确保置信度在合理范围内 / Use standardized confidence calculator to ensure confidence is within reasonable range
            return confidenceCalculator.calculateMinimumConfidence(3, finalConfidence);

        } catch (Exception e) {
            // 使用标准化置信度计算器计算最小置信度 / Use standardized confidence calculator for minimum confidence
            return confidenceCalculator.calculateMinimumConfidence(1, 0.5); // 默认中等置信度 / Default medium confidence
        }
    }

    /**
     * 计算基础置信度 / Calculate base confidence
     */
    private double calculateBaseConfidence(UnifiedMusicAnalysisResult result, double beatWeight, double keyWeight, double chordWeight) {
        try {
            double totalConfidence = 0.0;
            double totalWeight = 0.0;

            if (result.getBeatDetectionResult() != null) {
                totalConfidence += result.getBeatDetectionResult().getConfidence() * beatWeight;
                totalWeight += beatWeight;
            }

            if (result.getKeyDetectionResult() != null) {
                totalConfidence += result.getKeyDetectionResult().getConfidence() * keyWeight;
                totalWeight += keyWeight;
            }

            if (result.getChordDetectionResult() != null) {
                totalConfidence += result.getChordDetectionResult().getConfidence() * chordWeight;
                totalWeight += chordWeight;
            }

            double confidence = totalWeight > 0 ? totalConfidence / totalWeight : 0.0;
            
            // 使用标准化置信度计算器确保置信度在合理范围内 / Use standardized confidence calculator to ensure confidence is within reasonable range
            return confidenceCalculator.calculateMinimumConfidence(1, confidence);

        } catch (Exception e) {
            // 使用标准化置信度计算器计算最小置信度 / Use standardized confidence calculator for minimum confidence
            return confidenceCalculator.calculateMinimumConfidence(1, 0.5); // 默认中等置信度 / Default medium confidence
        }
    }

    /**
     * 计算交叉验证因子 / Calculate cross-validation factor
     */
    private double calculateCrossValidationFactor(UnifiedMusicAnalysisResult result) {
        try {
            double factor = 1.0;

            if (result.getBeatDetectionResult() != null && result.getKeyDetectionResult() != null && result.getChordDetectionResult() != null) {
                // 检查节拍和和弦的一致性 / Check consistency between beat and chord
                double beatChordConsistency = calculateBeatChordConsistency(result);
                
                // 检查调性和和弦的一致性 / Check consistency between key and chord
                double keyChordConsistency = calculateKeyChordConsistency(result);

                // 综合一致性因子 / Combined consistency factor
                factor = 0.7 + 0.15 * beatChordConsistency + 0.15 * keyChordConsistency;
            }

            // 使用标准化置信度计算器确保因子在合理范围内 / Use standardized confidence calculator to ensure factor is within reasonable range
            return confidenceCalculator.calculateMinimumConfidence(1, factor);

        } catch (Exception e) {
            // 使用标准化置信度计算器计算最小置信度 / Use standardized confidence calculator for minimum confidence
            return confidenceCalculator.calculateMinimumConfidence(1, 1.0); // 默认中等因子 / Default medium factor
        }
    }

    /**
     * 计算一致性因子 / Calculate consistency factor
     */
    private double calculateConsistencyFactor(UnifiedMusicAnalysisResult result) {
        try {
            double factor = 1.0;
            int validDetections = 0;
            double totalConfidence = 0.0;

            // 统计有效检测的数量和总置信度 / Count valid detections and total confidence
            if (result.getBeatDetectionResult() != null) {
                validDetections++;
                totalConfidence += result.getBeatDetectionResult().getConfidence();
            }
            if (result.getKeyDetectionResult() != null) {
                validDetections++;
                totalConfidence += result.getKeyDetectionResult().getConfidence();
            }
            if (result.getChordDetectionResult() != null) {
                validDetections++;
                totalConfidence += result.getChordDetectionResult().getConfidence();
            }

            if (validDetections > 0) {
                double avgConfidence = totalConfidence / validDetections;
                
                // 如果平均置信度很低，降低一致性因子 / If average confidence is low, reduce consistency factor
                if (avgConfidence < 0.2) {
                    factor = 0.8;
                } else if (avgConfidence < 0.4) {
                    factor = 0.9;
                } else if (avgConfidence > 0.6) {
                    factor = 1.1; // 高置信度时略微提升
                }
            }

            // 使用标准化置信度计算器确保因子在合理范围内 / Use standardized confidence calculator to ensure factor is within reasonable range
            return confidenceCalculator.calculateMinimumConfidence(1, factor);

        } catch (Exception e) {
            // 使用标准化置信度计算器计算最小置信度 / Use standardized confidence calculator for minimum confidence
            return confidenceCalculator.calculateMinimumConfidence(1, 1.0); // 默认中等因子 / Default medium factor
        }
    }

    /**
     * 高级置信度验证机制 / Advanced confidence validation mechanism
     * 实现多算法交叉验证和异常值检测
     */
    private UnifiedMusicAnalysisResult validateAndImproveResult(UnifiedMusicAnalysisResult result) {
        return validateAndImproveResultWithMultiAlgorithm(null, result);
    }
    
    /**
     * 带多算法验证的高级置信度验证机制
     */
    private UnifiedMusicAnalysisResult validateAndImproveResultWithMultiAlgorithm(AudioData audioData, UnifiedMusicAnalysisResult result) {
        // 1. 异常值检测 / Anomaly detection
        result = detectAndFilterAnomalies(result);
        
        // 2. 多算法交叉验证 / Multi-algorithm cross-validation
        result = performCrossValidation(result);
        
        // 3. 多算法验证（如果有音频数据）/ Multi-algorithm validation (if audio data available)
        if (audioData != null) {
            result = performMultiAlgorithmValidation(audioData, result);
        }
        
        // 4. 结果合理性检查 / Result reasonableness check
        result = performReasonablenessCheck(result);
        
        // 5. 重新计算最终置信度 / Recalculate final confidence
        double validatedConfidence = calculateValidatedConfidence(result);
        result.setConfidence(validatedConfidence);
        
        return result;
    }

    /**
     * 异常值检测和过滤 / Anomaly detection and filtering
     */
    private UnifiedMusicAnalysisResult detectAndFilterAnomalies(UnifiedMusicAnalysisResult result) {
        // 检测BPM异常值
        if (result.getBeatDetectionResult() != null) {
            double bpm = result.getBeatDetectionResult().getBpm();
            if (bpm < 60 || bpm > 200) {
                // BPM超出合理范围，降低置信度
                double adjustedConfidence = result.getBeatDetectionResult().getConfidence() * 0.5;
                result.getBeatDetectionResult().setConfidence(adjustedConfidence);
                
                if (verboseLogging) {
                    System.out.println("警告: BPM异常值检测 - BPM: " + bpm + ", 置信度已调整");
                }
            }
        }
        
        // 检测调性检测异常值
        if (result.getKeyDetectionResult() != null) {
            double keyConfidence = result.getKeyDetectionResult().getConfidence();
            if (keyConfidence < 0.1) {
                // 调性置信度过低，可能是误检
                result.getKeyDetectionResult().setConfidence(0.05);
                
                if (verboseLogging) {
                    System.out.println("警告: 调性置信度异常低 - " + keyConfidence);
                }
            }
        }
        
        // 检测和弦检测异常值
        if (result.getChordDetectionResult() != null) {
            String chord = result.getChordDetectionResult().getChordName();
            double chordConfidence = result.getChordDetectionResult().getConfidence();
            
            // 检查和弦名称是否合理
            if (chord == null || chord.trim().isEmpty() || chord.length() > 10) {
                result.getChordDetectionResult().setConfidence(chordConfidence * 0.3);
                
                if (verboseLogging) {
                    System.out.println("警告: 和弦名称异常 - " + chord);
                }
            }
        }
        
        return result;
    }

    /**
     * 多算法交叉验证 / Multi-algorithm cross-validation
     */
    private UnifiedMusicAnalysisResult performCrossValidation(UnifiedMusicAnalysisResult result) {
        // 节拍-调性交叉验证
        if (result.getBeatDetectionResult() != null && result.getKeyDetectionResult() != null) {
            double beatKeyConsistency = calculateBeatKeyConsistency(result);
            
            // 根据一致性调整置信度
            if (beatKeyConsistency < 0.5) {
                // 一致性差，降低两者置信度
                double beatConf = result.getBeatDetectionResult().getConfidence();
                double keyConf = result.getKeyDetectionResult().getConfidence();
                
                result.getBeatDetectionResult().setConfidence(beatConf * 0.9);
                result.getKeyDetectionResult().setConfidence(keyConf * 0.9);
            } else if (beatKeyConsistency > 0.8) {
                // 一致性好，提升置信度
                double beatConf = result.getBeatDetectionResult().getConfidence();
                double keyConf = result.getKeyDetectionResult().getConfidence();
                
                result.getBeatDetectionResult().setConfidence(Math.min(1.0, beatConf * 1.1));
                result.getKeyDetectionResult().setConfidence(Math.min(1.0, keyConf * 1.1));
            }
        }
        
        // 调性-和弦交叉验证（已有实现，进一步增强）
        if (result.getKeyDetectionResult() != null && result.getChordDetectionResult() != null) {
            double keyChordConsistency = calculateEnhancedKeyChordConsistency(result);
            
            if (keyChordConsistency > 0.8) {
                // 高一致性，提升置信度
                double keyConf = result.getKeyDetectionResult().getConfidence();
                double chordConf = result.getChordDetectionResult().getConfidence();
                
                result.getKeyDetectionResult().setConfidence(Math.min(1.0, keyConf * 1.15));
                result.getChordDetectionResult().setConfidence(Math.min(1.0, chordConf * 1.15));
            }
        }
        
        return result;
    }

    /**
     * 计算节拍-调性一致性 / Calculate beat-key consistency
     */
    private double calculateBeatKeyConsistency(UnifiedMusicAnalysisResult result) {
        double bpm = result.getBeatDetectionResult().getBpm();
        String key = result.getKeyDetectionResult().getKeyName();
        
        // 基于音乐理论的一致性检查
        // 某些调性在特定BPM范围内更常见
        if (key != null) {
            if (key.contains("m") || key.contains("minor")) {
                // 小调通常在较慢的BPM下更常见
                if (bpm >= 60 && bpm <= 120) {
                    return 0.9;
                } else if (bpm > 120 && bpm <= 140) {
                    return 0.7;
                } else {
                    return 0.5;
                }
            } else {
                // 大调在各种BPM下都比较常见
                if (bpm >= 80 && bpm <= 160) {
                    return 0.8;
                } else {
                    return 0.6;
                }
            }
        }
        
        return 0.5; // 默认中等一致性
    }

    /**
     * 增强的调性-和弦一致性计算 / Enhanced key-chord consistency calculation
     */
    private double calculateEnhancedKeyChordConsistency(UnifiedMusicAnalysisResult result) {
        String key = result.getKeyDetectionResult().getKeyName();
        String chord = result.getChordDetectionResult().getChordName();
        
        if (key == null || chord == null) {
            return 0.5;
        }
        
        // 提取调性和和弦的根音
        String keyRoot = extractRootNote(key);
        String chordRoot = extractRootNote(chord);
        
        // 检查是否为同一根音
        if (keyRoot.equals(chordRoot)) {
            return 1.0;
        }
        
        // 检查是否为相关调
        double relatedKeyScore = calculateEnhancedRelatedKeyScore(keyRoot, chordRoot, key, chord);
        
        return relatedKeyScore;
    }

    /**
     * 提取根音 / Extract root note
     */
    private String extractRootNote(String noteString) {
        if (noteString == null || noteString.isEmpty()) {
            return "";
        }
        
        // 处理升降号
        if (noteString.length() >= 2 && (noteString.charAt(1) == '#' || noteString.charAt(1) == 'b')) {
            return noteString.substring(0, 2);
        } else {
            return noteString.substring(0, 1);
        }
    }

    /**
     * 增强的相关调评分计算 / Enhanced related key score calculation
     */
    private double calculateEnhancedRelatedKeyScore(String keyRoot, String chordRoot, String fullKey, String fullChord) {
        // 五度圈相关性
        String[] circleOfFifths = {"C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"};
        
        int keyIndex = findNoteIndex(keyRoot, circleOfFifths);
        int chordIndex = findNoteIndex(chordRoot, circleOfFifths);
        
        if (keyIndex != -1 && chordIndex != -1) {
            int distance = Math.min(Math.abs(keyIndex - chordIndex), 
                                  12 - Math.abs(keyIndex - chordIndex));
            
            if (distance == 0) return 1.0;
            if (distance == 1) return 0.9; // 五度关系
            if (distance == 2) return 0.7; // 二度关系
            if (distance == 3) return 0.5; // 三度关系
            if (distance <= 5) return 0.3; // 较远关系
        }
        
        // 检查大小调关系
        if (isRelativeMinorMajor(fullKey, fullChord)) {
            return 0.8;
        }
        
        return 0.2;
    }

    /**
     * 查找音符在圆圈中的索引 / Find note index in circle
     */
    private int findNoteIndex(String note, String[] circle) {
        for (int i = 0; i < circle.length; i++) {
            if (circle[i].equals(note)) {
                return i;
            }
        }
        return -1;
    }

    /**
     * 检查是否为关系大小调 / Check if relative minor/major
     */
    private boolean isRelativeMinorMajor(String key, String chord) {
        // 简化的关系大小调检查
        if (key.contains("m") && !chord.contains("m")) {
            // 小调 vs 大调
            return true;
        } else if (!key.contains("m") && chord.contains("m")) {
            // 大调 vs 小调
            return true;
        }
        return false;
    }

    /**
     * 结果合理性检查 / Result reasonableness check
     */
    private UnifiedMusicAnalysisResult performReasonablenessCheck(UnifiedMusicAnalysisResult result) {
        // 检查整体结果的合理性
        int validResults = 0;
        double totalConfidence = 0.0;
        
        if (result.getBeatDetectionResult() != null && result.getBeatDetectionResult().getConfidence() > 0.3) {
            validResults++;
            totalConfidence += result.getBeatDetectionResult().getConfidence();
        }
        
        if (result.getKeyDetectionResult() != null && result.getKeyDetectionResult().getConfidence() > 0.2) {
            validResults++;
            totalConfidence += result.getKeyDetectionResult().getConfidence();
        }
        
        if (result.getChordDetectionResult() != null && result.getChordDetectionResult().getConfidence() > 0.3) {
            validResults++;
            totalConfidence += result.getChordDetectionResult().getConfidence();
        }
        
        // 如果有效结果太少，降低整体置信度
        if (validResults < 2) {
            double penalty = 0.8;
            if (result.getBeatDetectionResult() != null) {
                result.getBeatDetectionResult().setConfidence(
                    result.getBeatDetectionResult().getConfidence() * penalty);
            }
            if (result.getKeyDetectionResult() != null) {
                result.getKeyDetectionResult().setConfidence(
                    result.getKeyDetectionResult().getConfidence() * penalty);
            }
            if (result.getChordDetectionResult() != null) {
                result.getChordDetectionResult().setConfidence(
                    result.getChordDetectionResult().getConfidence() * penalty);
            }
            
            if (verboseLogging) {
                System.out.println("警告: 有效结果数量不足 (" + validResults + "/3), 应用置信度惩罚");
            }
        }
        
        return result;
    }

    /**
     * 计算验证后的置信度 / Calculate validated confidence
     */
    private double calculateValidatedConfidence(UnifiedMusicAnalysisResult result) {
        try {
            double totalWeight = 0.0;
            double weightedConfidence = 0.0;
            
            // 动态权重分配，基于各检测结果的可靠性
            if (result.getBeatDetectionResult() != null) {
                double beatConf = result.getBeatDetectionResult().getConfidence();
                double beatWeight = beatConf > 0.7 ? 0.4 : (beatConf > 0.5 ? 0.3 : 0.2);
                weightedConfidence += beatConf * beatWeight;
                totalWeight += beatWeight;
            }
            
            if (result.getKeyDetectionResult() != null) {
                double keyConf = result.getKeyDetectionResult().getConfidence();
                double keyWeight = keyConf > 0.6 ? 0.3 : (keyConf > 0.3 ? 0.25 : 0.15);
                weightedConfidence += keyConf * keyWeight;
                totalWeight += keyWeight;
            }
            
            if (result.getChordDetectionResult() != null) {
                double chordConf = result.getChordDetectionResult().getConfidence();
                double chordWeight = chordConf > 0.7 ? 0.3 : (chordConf > 0.5 ? 0.25 : 0.2);
                weightedConfidence += chordConf * chordWeight;
                totalWeight += chordWeight;
            }
            
            double finalConfidence = totalWeight > 0 ? weightedConfidence / totalWeight : 0.0;
            
            // 应用最终的合理性约束，使用标准化置信度计算器 / Apply final reasonableness constraint, use standardized confidence calculator
            return confidenceCalculator.calculateMinimumConfidence(1, finalConfidence);
            
        } catch (Exception e) {
            // 使用标准化置信度计算器计算最小置信度 / Use standardized confidence calculator for minimum confidence
            return confidenceCalculator.calculateMinimumConfidence(1, 0.5); // 默认中等置信度 / Default medium confidence
        }
    }

    /**
     * 执行多算法交叉验证 / Perform multi-algorithm cross-validation
     * <p>
     * 使用多个分析器对同一音频进行分析，比较结果的一致性，选择最可靠的分析结果。
     * Uses multiple analyzers to analyze the same audio, compares result consistency, and selects the most reliable analysis result.
     * </p>
     * 
     * @param audioData 音频数据 / Audio data
     * @param result 当前分析结果 / Current analysis result
     * @return 经过交叉验证优化的结果 / Cross-validation optimized result
     */
    private UnifiedMusicAnalysisResult performMultiAlgorithmValidation(AudioData audioData, UnifiedMusicAnalysisResult result) {
        try {
            // 创建备用分析器进行交叉验证 / Create alternative analyzers for cross-validation
            List<UnifiedMusicAnalysisResult> validationResults = new ArrayList<>();
            validationResults.add(result); // 添加当前结果 / Add current result
            
            // 使用不同参数配置的分析器 / Use analyzers with different parameter configurations
            Map<String, Object> alternativeParams1 = createAlternativeParameters(1);
            Map<String, Object> alternativeParams2 = createAlternativeParameters(2);
            
            // 并行执行备用分析 / Execute alternative analyses in parallel
            CompletableFuture<UnifiedMusicAnalysisResult> alt1Future = CompletableFuture.supplyAsync(() -> {
                try {
                    return performAlternativeAnalysis(audioData, alternativeParams1);
                } catch (Exception e) {
                    if (verboseLogging) {
                        System.err.println("Alternative analysis 1 failed: " + e.getMessage());
                    }
                    return null;
                }
            }, executorService);
            
            CompletableFuture<UnifiedMusicAnalysisResult> alt2Future = CompletableFuture.supplyAsync(() -> {
                try {
                    return performAlternativeAnalysis(audioData, alternativeParams2);
                } catch (Exception e) {
                    if (verboseLogging) {
                        System.err.println("Alternative analysis 2 failed: " + e.getMessage());
                    }
                    return null;
                }
            }, executorService);
            
            // 收集所有有效结果 / Collect all valid results
            try {
                UnifiedMusicAnalysisResult alt1 = alt1Future.get(15, TimeUnit.SECONDS);
                if (alt1 != null) validationResults.add(alt1);
            } catch (Exception e) {
                if (verboseLogging) {
                    System.err.println("Alternative analysis 1 timeout or failed");
                }
            }
            
            try {
                UnifiedMusicAnalysisResult alt2 = alt2Future.get(15, TimeUnit.SECONDS);
                if (alt2 != null) validationResults.add(alt2);
            } catch (Exception e) {
                if (verboseLogging) {
                    System.err.println("Alternative analysis 2 timeout or failed");
                }
            }
            
            // 如果只有一个结果，直接返回 / If only one result, return directly
            if (validationResults.size() == 1) {
                return result;
            }
            
            // 执行一致性分析和结果选择 / Perform consistency analysis and result selection
            return selectBestValidatedResult(validationResults);
            
        } catch (Exception e) {
            if (verboseLogging) {
                System.err.println("Multi-algorithm validation failed: " + e.getMessage());
            }
            return result; // 返回原始结果 / Return original result
        }
    }
    
    /**
     * 创建备用参数配置 / Create alternative parameter configurations
     */
    private Map<String, Object> createAlternativeParameters(int variant) {
        Map<String, Object> params = new HashMap<>(this.parameters);
        
        switch (variant) {
            case 1:
                // 更保守的参数设置 / More conservative parameter settings
                params.put("confidence_threshold", 0.8);
                params.put("window_size", 4096);
                params.put("hop_size", 1024);
                break;
            case 2:
                // 更激进的参数设置 / More aggressive parameter settings
                params.put("confidence_threshold", 0.6);
                params.put("window_size", 1024);
                params.put("hop_size", 256);
                break;
        }
        
        return params;
    }
    
    /**
     * 执行备用分析 / Perform alternative analysis
     */
    private UnifiedMusicAnalysisResult performAlternativeAnalysis(AudioData audioData, Map<String, Object> params) throws AudioProcessingException {
        // 创建新的分析器实例以避免状态冲突 / Create new analyzer instances to avoid state conflicts
        IBeatAnalyzer altBeatAnalyzer = new BeatAnalyzerImpl();
        IKeyAnalyzer altKeyAnalyzer = new KeyAnalyzerImpl();
        IChordAnalyzer altChordAnalyzer = new ChordAnalyzerImpl();
        
        // 执行分析 / Perform analysis
        BeatDetectionResult beatResult = altBeatAnalyzer.detectBeats(audioData, params);
        KeyDetectionResult keyResult = altKeyAnalyzer.detectKey(audioData, params);
        List<ChordDetectionResult> chordResults = altChordAnalyzer.detectChords(audioData, params);
        
        // 创建结果 / Create result
        UnifiedMusicAnalysisResult result = new UnifiedMusicAnalysisResult();
        result.setAlgorithm("basic_music_analyzer_alt");
        result.setBeatDetectionResult(beatResult);
        result.setKeyDetectionResult(keyResult);
        if (!chordResults.isEmpty()) {
            result.setChordDetectionResult(chordResults.get(0));
        }
        
        // 计算置信度 / Calculate confidence
        result.setConfidence(calculateImprovedOverallConfidence(result));
        
        return result;
    }
    
    /**
     * 选择最佳验证结果 / Select best validated result
     */
    private UnifiedMusicAnalysisResult selectBestValidatedResult(List<UnifiedMusicAnalysisResult> results) {
        if (results.isEmpty()) {
            throw new IllegalArgumentException("No results to validate");
        }
        
        if (results.size() == 1) {
            return results.get(0);
        }
        
        // 计算每个结果的综合评分 / Calculate comprehensive score for each result
        double bestScore = -1;
        UnifiedMusicAnalysisResult bestResult = results.get(0);
        
        for (UnifiedMusicAnalysisResult result : results) {
            double score = calculateValidationScore(result, results);
            
            if (verboseLogging) {
                System.out.println("Validation score for " + result.getAlgorithm() + ": " + score);
            }
            
            if (score > bestScore) {
                bestScore = score;
                bestResult = result;
            }
        }
        
        // 应用一致性增强 / Apply consistency enhancement
        return enhanceResultWithConsistency(bestResult, results);
    }
    
    /**
     * 计算验证评分 / Calculate validation score
     */
    private double calculateValidationScore(UnifiedMusicAnalysisResult candidate, List<UnifiedMusicAnalysisResult> allResults) {
        double baseConfidence = candidate.getConfidence();
        double consistencyScore = calculateConsistencyScore(candidate, allResults);
        double stabilityScore = calculateStabilityScore(candidate);
        
        // 综合评分：基础置信度(40%) + 一致性(40%) + 稳定性(20%)
        // Comprehensive score: base confidence(40%) + consistency(40%) + stability(20%)
        return baseConfidence * 0.4 + consistencyScore * 0.4 + stabilityScore * 0.2;
    }
    
    /**
     * 计算一致性评分 / Calculate consistency score
     */
    private double calculateConsistencyScore(UnifiedMusicAnalysisResult candidate, List<UnifiedMusicAnalysisResult> allResults) {
        if (allResults.size() <= 1) {
            return 1.0;
        }
        
        double totalConsistency = 0.0;
        int comparisons = 0;
        
        for (UnifiedMusicAnalysisResult other : allResults) {
            if (other == candidate) continue;
            
            double beatConsistency = calculateBeatConsistency(candidate, other);
            double keyConsistency = calculateKeyConsistency(candidate, other);
            double chordConsistency = calculateChordConsistency(candidate, other);
            
            // 平均一致性 / Average consistency
            double avgConsistency = (beatConsistency + keyConsistency + chordConsistency) / 3.0;
            totalConsistency += avgConsistency;
            comparisons++;
        }
        
        return comparisons > 0 ? totalConsistency / comparisons : 1.0;
    }
    
    /**
     * 计算节拍一致性 / Calculate beat consistency
     */
    private double calculateBeatConsistency(UnifiedMusicAnalysisResult result1, UnifiedMusicAnalysisResult result2) {
        BeatDetectionResult beat1 = result1.getBeatDetectionResult();
        BeatDetectionResult beat2 = result2.getBeatDetectionResult();
        
        if (beat1 == null || beat2 == null) {
            return 0.5;
        }
        
        double bpm1 = beat1.getBpm();
        double bpm2 = beat2.getBpm();
        
        // 计算BPM差异 / Calculate BPM difference
        double bpmDiff = Math.abs(bpm1 - bpm2);
        double maxBpm = Math.max(bpm1, bpm2);
        
        // 允许5%的差异 / Allow 5% difference
        if (maxBpm > 0 && bpmDiff / maxBpm <= 0.05) {
            return 1.0;
        }
        
        // 检查倍数关系 / Check multiple relationships
        if (maxBpm > 0) {
            double ratio = Math.max(bpm1, bpm2) / Math.min(bpm1, bpm2);
            if (Math.abs(ratio - 2.0) < 0.1 || Math.abs(ratio - 0.5) < 0.1) {
                return 0.8; // 倍数关系 / Multiple relationship
            }
        }
        
        // 基于差异计算一致性 / Calculate consistency based on difference
        return Math.max(0.0, 1.0 - (bpmDiff / 200.0));
    }
    
    /**
     * 计算调性一致性 / Calculate key consistency
     */
    private double calculateKeyConsistency(UnifiedMusicAnalysisResult result1, UnifiedMusicAnalysisResult result2) {
        KeyDetectionResult key1 = result1.getKeyDetectionResult();
        KeyDetectionResult key2 = result2.getKeyDetectionResult();
        
        if (key1 == null || key2 == null) {
            return 0.5;
        }
        
        String keyStr1 = key1.getKeyName() + " " + key1.getScaleType();
        String keyStr2 = key2.getKeyName() + " " + key2.getScaleType();
        
        // 完全匹配 / Exact match
        if (keyStr1.equals(keyStr2)) {
            return 1.0;
        }
        
        // 相对大小调关系 / Relative major-minor relationship
        if (isRelativeMinorMajor(keyStr1, keyStr2)) {
            return 0.8;
        }
        
        // 五度圈关系 / Circle of fifths relationship
        double relatedScore = calculateEnhancedRelatedKeyScore(key1.getKeyName(), key2.getKeyName(), keyStr1, keyStr2);
        return Math.max(0.3, relatedScore);
    }
    
    /**
     * 计算和弦一致性 / Calculate chord consistency
     */
    private double calculateChordConsistency(UnifiedMusicAnalysisResult result1, UnifiedMusicAnalysisResult result2) {
        ChordDetectionResult chord1 = result1.getChordDetectionResult();
        ChordDetectionResult chord2 = result2.getChordDetectionResult();
        
        if (chord1 == null || chord2 == null) {
            return 0.5;
        }
        
        String chordName1 = chord1.getChordName();
        String chordName2 = chord2.getChordName();
        
        // 完全匹配 / Exact match
        if (chordName1.equals(chordName2)) {
            return 1.0;
        }
        
        // 根音匹配 / Root note match
        String root1 = extractRootNote(chordName1);
        String root2 = extractRootNote(chordName2);
        
        if (root1.equals(root2)) {
            return 0.7; // 根音相同 / Same root note
        }
        
        // 和声关系 / Harmonic relationship
        return calculateHarmonicRelationship(root1, root2);
    }
    
    /**
     * 计算和声关系 / Calculate harmonic relationship
     */
    private double calculateHarmonicRelationship(String root1, String root2) {
        String[] chromaticScale = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
        
        int index1 = findNoteIndex(root1, chromaticScale);
        int index2 = findNoteIndex(root2, chromaticScale);
        
        if (index1 == -1 || index2 == -1) {
            return 0.3;
        }
        
        int interval = Math.abs(index1 - index2);
        interval = Math.min(interval, 12 - interval); // 最短距离 / Shortest distance
        
        // 和声关系评分 / Harmonic relationship scoring
        switch (interval) {
            case 0: return 1.0;  // 同音 / Unison
            case 7: return 0.8;  // 五度 / Perfect fifth
            case 5: return 0.7;  // 四度 / Perfect fourth
            case 4: return 0.6;  // 大三度 / Major third
            case 3: return 0.6;  // 小三度 / Minor third
            case 2: return 0.4;  // 大二度 / Major second
            case 1: return 0.3;  // 小二度 / Minor second
            default: return 0.3;
        }
    }
    
    /**
     * 计算稳定性评分 / Calculate stability score
     */
    private double calculateStabilityScore(UnifiedMusicAnalysisResult result) {
        double stabilityScore = 1.0;
        
        // 检查BPM稳定性 / Check BPM stability
        BeatDetectionResult beatResult = result.getBeatDetectionResult();
        if (beatResult != null) {
            double bpm = beatResult.getBpm();
            double confidence = beatResult.getConfidence();
            
            // BPM在合理范围内 / BPM within reasonable range
            if (bpm < 60 || bpm > 200) {
                stabilityScore *= 0.8;
            }
            
            // 置信度检查 / Confidence check
            if (confidence < 0.5) {
                stabilityScore *= 0.9;
            }
        }
        
        // 检查调性稳定性 / Check key stability
        KeyDetectionResult keyResult = result.getKeyDetectionResult();
        if (keyResult != null && keyResult.getConfidence() < 0.4) {
            stabilityScore *= 0.9;
        }
        
        // 检查和弦稳定性 / Check chord stability
        ChordDetectionResult chordResult = result.getChordDetectionResult();
        if (chordResult != null && chordResult.getConfidence() < 0.4) {
            stabilityScore *= 0.9;
        }
        
        return Math.max(0.3, stabilityScore);
    }
    
    /**
     * 使用一致性增强结果 / Enhance result with consistency
     */
    private UnifiedMusicAnalysisResult enhanceResultWithConsistency(UnifiedMusicAnalysisResult bestResult, List<UnifiedMusicAnalysisResult> allResults) {
        // 创建增强后的结果副本 / Create enhanced result copy
        UnifiedMusicAnalysisResult enhancedResult = new UnifiedMusicAnalysisResult();
        enhancedResult.setAlgorithm(bestResult.getAlgorithm() + "_validated");
        enhancedResult.setBeatDetectionResult(bestResult.getBeatDetectionResult());
        enhancedResult.setKeyDetectionResult(bestResult.getKeyDetectionResult());
        enhancedResult.setChordDetectionResult(bestResult.getChordDetectionResult());
        
        // 计算增强后的置信度 / Calculate enhanced confidence
        double originalConfidence = bestResult.getConfidence();
        double consistencyBonus = calculateOverallConsistency(bestResult, allResults);
        double enhancedConfidence = originalConfidence + (consistencyBonus * 0.1); // 最多增加10%
        
        enhancedResult.setConfidence(Math.min(1.0, enhancedConfidence));
        
        if (verboseLogging) {
            System.out.println("Enhanced confidence: " + originalConfidence + " -> " + enhancedResult.getConfidence());
        }
        
        return enhancedResult;
    }
    
    /**
     * 计算整体一致性 / Calculate overall consistency
     */
    private double calculateOverallConsistency(UnifiedMusicAnalysisResult result, List<UnifiedMusicAnalysisResult> allResults) {
        if (allResults.size() <= 1) {
            return 0.0;
        }
        
        double totalConsistency = 0.0;
        int validComparisons = 0;
        
        for (UnifiedMusicAnalysisResult other : allResults) {
            if (other == result) continue;
            
            double consistency = calculateConsistencyScore(result, List.of(other));
            totalConsistency += consistency;
            validComparisons++;
        }
        
        return validComparisons > 0 ? totalConsistency / validComparisons : 0.0;
    }
    
    /**
     * 计算动态权重 / Calculate dynamic weights
     */
    private double[] calculateDynamicWeights(UnifiedMusicAnalysisResult result) {
        double beatWeight = 0.35;
        double keyWeight = 0.30;
        double chordWeight = 0.35;

        // 根据各项检测的置信度调整权重 / Adjust weights based on detection confidence
        if (result.getBeatDetectionResult() != null && result.getKeyDetectionResult() != null && result.getChordDetectionResult() != null) {
            double beatConf = result.getBeatDetectionResult().getConfidence();
            double keyConf = result.getKeyDetectionResult().getConfidence();
            double chordConf = result.getChordDetectionResult().getConfidence();

            // 如果某项检测置信度很低，降低其权重 / If detection confidence is very low, reduce its weight
            if (keyConf < 0.1) {
                keyWeight = 0.15;  // 降低调性检测权重
                beatWeight = 0.45; // 增加节拍检测权重
                chordWeight = 0.40; // 增加和弦检测权重
            }

            // 如果节拍检测很好，增加其权重 / If beat detection is good, increase its weight
            if (beatConf > 0.6) {
                beatWeight = 0.45;
                keyWeight = 0.25;
                chordWeight = 0.30;
            }

            // 如果和弦检测很好，增加其权重 / If chord detection is good, increase its weight
            if (chordConf > 0.6) {
                chordWeight = 0.40;
                beatWeight = 0.35;
                keyWeight = 0.25;
            }
        }

        return new double[]{beatWeight, keyWeight, chordWeight};
    }

    /**
     * 计算节拍和和弦的一致性 / Calculate beat-chord consistency
     */
    private double calculateBeatChordConsistency(UnifiedMusicAnalysisResult result) {
        // 简化的一致性检查：如果节拍检测置信度高且和弦检测置信度也高，认为一致性好
        double beatConf = result.getBeatDetectionResult().getConfidence();
        double chordConf = result.getChordDetectionResult().getConfidence();
        
        if (beatConf > 0.5 && chordConf > 0.5) {
            return 1.0;
        } else if (beatConf > 0.3 && chordConf > 0.3) {
            return 0.7;
        } else {
            return 0.3;
        }
    }

    /**
     * 计算调性和和弦的一致性 / Calculate key-chord consistency
     */
    private double calculateKeyChordConsistency(UnifiedMusicAnalysisResult result) {
        String detectedKey = result.getKeyDetectionResult().getKeyName();
        String detectedChord = result.getChordDetectionResult().getChordName();
        
        if (detectedKey == null || detectedChord == null) {
            return 0.5;
        }

        // 简化的调性和弦一致性检查 / Simplified key-chord consistency check
        String keyRoot = detectedKey.replace("#", "").replace("b", "");
        String chordRoot = detectedChord.replaceAll("[^A-G#b]", "").replace("#", "").replace("b", "");
        
        if (keyRoot.equals(chordRoot)) {
            return 1.0; // 根音相同
        } else {
            // 检查是否为相关调
            return calculateRelatedKeyScore(keyRoot, chordRoot);
        }
    }

    /**
     * 计算相关调评分 / Calculate related key score
     */
    private double calculateRelatedKeyScore(String keyRoot, String chordRoot) {
        // 五度圈相关性检查 / Circle of fifths relatedness check
        String[] circleOfFifths = {"C", "G", "D", "A", "E", "B", "F", "C"};
        
        int keyIndex = -1, chordIndex = -1;
        for (int i = 0; i < circleOfFifths.length; i++) {
            if (circleOfFifths[i].equals(keyRoot)) keyIndex = i;
            if (circleOfFifths[i].equals(chordRoot)) chordIndex = i;
        }
        
        if (keyIndex != -1 && chordIndex != -1) {
            int distance = Math.abs(keyIndex - chordIndex);
            if (distance <= 1) return 0.8; // 相邻调
            if (distance <= 2) return 0.6; // 较近调
            if (distance <= 3) return 0.4; // 中等距离
        }
        
        return 0.2; // 不相关
    }
}