package com.reremouse.lab.music.analysis.advanced;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.music.analysis.basic.KeyAnalyzerImpl;
import com.reremouse.lab.music.analysis.basic.BeatAnalyzerImpl;
import com.reremouse.lab.music.analysis.MusicDetectionResult;
import com.reremouse.lab.music.analysis.ConfidenceCalculator;
import com.reremouse.lab.music.analysis.StandardizedConfidenceCalculator;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.core.RereFFT;
import com.reremouse.lab.math.signal.core.Complex;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

/**
 * 音乐复杂度分析器 / Music Complexity Analyzer
 * <p>
 * 分析音乐的复杂度，包括和声复杂度、节奏复杂度、旋律复杂度等。
 * Analyzes music complexity including harmonic complexity, rhythmic complexity, melodic complexity, etc.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class ComplexityAnalyzer implements IAdvancedAnalyzer {
    
    // 复杂度维度 / Complexity dimensions
    public static final String HARMONIC_COMPLEXITY = "harmonicComplexity";
    public static final String RHYTHMIC_COMPLEXITY = "rhythmicComplexity";
    public static final String MELODIC_COMPLEXITY = "melodicComplexity";
    public static final String SPECTRAL_COMPLEXITY = "spectralComplexity";
    public static final String OVERALL_COMPLEXITY = "overallComplexity";
    
    // 默认参数 / Default parameters
    private static final int DEFAULT_WINDOW_SIZE = 4096;
    private static final int DEFAULT_HOP_SIZE = 2048;
    private static final int DEFAULT_FRAME_SIZE = 1024;
    
    private final KeyAnalyzerImpl keyAnalyzer;
    private final BeatAnalyzerImpl beatAnalyzer;
    private final StandardizedConfidenceCalculator confidenceCalculator = new StandardizedConfidenceCalculator();
    
    public ComplexityAnalyzer() {
        this.keyAnalyzer = new KeyAnalyzerImpl();
        this.beatAnalyzer = new BeatAnalyzerImpl();
    }
    
    @Override
    public Map<String, Object> analyze(AudioData audioData) throws AudioProcessingException {
        return analyze(audioData, getDefaultParameters());
    }
    
    @Override
    public MusicDetectionResult analyzeAdvancedMusic(AudioData audioData) throws AudioProcessingException {
        return analyzeAdvancedMusic(audioData, getDefaultParameters());
    }
    
    @Override
    public Map<String, Object> analyze(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }
        
        try {
            Map<String, Object> results = new HashMap<>();
            
            // 计算各种复杂度 / Calculate various complexities
            double harmonicComplexity = calculateHarmonicComplexity(audioData, parameters);
            double rhythmicComplexity = calculateRhythmicComplexity(audioData, parameters);
            double melodicComplexity = calculateMelodicComplexity(audioData, parameters);
            double spectralComplexity = calculateSpectralComplexity(audioData, parameters);
            
            // 计算总体复杂度 / Calculate overall complexity
            double overallComplexity = calculateOverallComplexity(
                harmonicComplexity, rhythmicComplexity, melodicComplexity, spectralComplexity
            );
            
            // 存储结果 / Store results
            results.put(HARMONIC_COMPLEXITY, harmonicComplexity);
            results.put(RHYTHMIC_COMPLEXITY, rhythmicComplexity);
            results.put(MELODIC_COMPLEXITY, melodicComplexity);
            results.put(SPECTRAL_COMPLEXITY, spectralComplexity);
            results.put(OVERALL_COMPLEXITY, overallComplexity);
            
            // 添加复杂度等级 / Add complexity level
            results.put("complexityLevel", classifyComplexity(overallComplexity));
            
            // 添加详细分析 / Add detailed analysis
            results.put("detailedAnalysis", generateDetailedAnalysis(
                harmonicComplexity, rhythmicComplexity, melodicComplexity, spectralComplexity
            ));
            
            return results;
            
        } catch (Exception e) {
            throw new AudioProcessingException("Error in complexity analysis: " + e.getMessage(), e);
        }
    }
    
    @Override
    public MusicDetectionResult analyzeAdvancedMusic(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        Map<String, Object> results = analyze(audioData, parameters);
        
        // Create a ComplexityAnalysisResult from the map results
        ComplexityAnalysisResult complexityResult = new ComplexityAnalysisResult();
        complexityResult.setHarmonicComplexity((Double) results.get(HARMONIC_COMPLEXITY));
        complexityResult.setRhythmicComplexity((Double) results.get(RHYTHMIC_COMPLEXITY));
        complexityResult.setMelodicComplexity((Double) results.get(MELODIC_COMPLEXITY));
        complexityResult.setSpectralComplexity((Double) results.get(SPECTRAL_COMPLEXITY));
        complexityResult.setOverallComplexity((Double) results.get(OVERALL_COMPLEXITY));
        complexityResult.setComplexityLevel((String) results.get("complexityLevel"));
        
        return complexityResult;
    }
    
    @Override
    public String getAnalyzerName() {
        return "complexity_analyzer";
    }
    
    @Override
    public String getName() {
        return getAnalyzerName();
    }
    
    @Override
    public String[] getSupportedParameters() {
        return new String[]{"windowSize", "hopSize", "frameSize"};
    }
    
    @Override
    public Map<String, Object> getDefaultParameters() {
        Map<String, Object> params = new HashMap<>();
        params.put("windowSize", DEFAULT_WINDOW_SIZE);
        params.put("hopSize", DEFAULT_HOP_SIZE);
        params.put("frameSize", DEFAULT_FRAME_SIZE);
        return params;
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
        
        // In this implementation, parameters are passed directly to methods
        // so we don't need to store them in instance variables
    }
    
    @Override
    public Map<String, Object> getCurrentParameters() {
        return getDefaultParameters(); // In this implementation, we always use defaults
    }
    
    @Override
    public void resetParameters() {
        // Nothing to reset in this implementation
    }
    
    @Override
    public String getVersion() {
        return "1.0";
    }
    
    @Override
    public String getDescription() {
        return "Analyzes music complexity including harmonic, rhythmic, melodic, and spectral complexity";
    }
    
    @Override
    public boolean supportsAudioFormat(double sampleRate, int channels, int bitDepth) {
        return sampleRate > 0 && channels > 0 && bitDepth > 0;
    }
    
    @Override
    public double getMinimumAudioLength() {
        return 1.0; // 1 second minimum
    }
    
    @Override
    public double getMaximumAudioLength() {
        return 3600.0; // 1 hour maximum
    }
    
    @Override
    public double getComplexityEstimate(double audioLength) {
        return audioLength * 0.15; // Medium complexity
    }
    
    @Override
    public void warmUp() throws AudioProcessingException {
        // No warm-up needed for this analyzer
    }
    
    @Override
    public void cleanup() {
        // No resources to clean up
    }
    
    @Override
    public String getStatus() {
        return "ready";
    }
    
    @Override
    public boolean isReady() {
        return true;
    }
    
    @Override
    public Map<String, Object> getLastAnalysisStatistics() {
        return new HashMap<>(); // No statistics tracked
    }
    
    @Override
    public Map<String, Object> getPerformanceMetrics() {
        return new HashMap<>(); // No metrics tracked
    }
    
    @Override
    public void setVerboseLogging(boolean enabled) {
        // Verbose logging not implemented
    }
    
    @Override
    public boolean isVerboseLoggingEnabled() {
        return false;
    }
    
    /**
     * 计算和声复杂度 / Calculate harmonic complexity
     */
    private double calculateHarmonicComplexity(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 分析色度特征 / Analyze chroma features
            double[] chromaFeatures = keyAnalyzer.analyzeChromaFeatures(audioData);
            
            // 计算色度特征的熵 / Calculate entropy of chroma features
            double chromaEntropy = calculateEntropy(chromaFeatures);
            
            // 计算色度特征的变化 / Calculate chroma feature variation
            double chromaVariation = calculateVariation(chromaFeatures);
            
            // 分析和弦变化 / Analyze chord changes
            double chordChangeRate = calculateChordChangeRate(audioData);
            
            // 计算频谱复杂度 / Calculate spectral complexity
            double spectralComplexity = calculateSpectralHarmonicComplexity(audioData, parameters);
            
            // 综合和声复杂度 / Combine harmonic complexity
            double harmonicComplexity = (chromaEntropy * 0.3 + chromaVariation * 0.3 + 
                                       chordChangeRate * 0.2 + spectralComplexity * 0.2);
            
            // 使用标准化置信度计算器 / Use standardized confidence calculator
            Map<String, Double> factors = new HashMap<>();
            factors.put("chromaEntropy", chromaEntropy);
            factors.put("chromaVariation", chromaVariation);
            factors.put("chordChangeRate", chordChangeRate);
            factors.put("spectralComplexity", spectralComplexity);
            
            Map<String, Double> weights = new HashMap<>();
            weights.put("chromaEntropy", 0.3);
            weights.put("chromaVariation", 0.3);
            weights.put("chordChangeRate", 0.2);
            weights.put("spectralComplexity", 0.2);
            
            double confidence = confidenceCalculator.calculateWeightedConfidence(factors, weights);
            
            return Math.max(0.0, Math.min(1.0, harmonicComplexity));
            
        } catch (Exception e) {
            // 使用标准化置信度计算器计算最小置信度 / Use standardized confidence calculator for minimum confidence
            return confidenceCalculator.calculateMinimumConfidence(1, 0.5); // 默认中等复杂度 / Default medium complexity
        }
    }
    
    /**
     * 计算节奏复杂度 / Calculate rhythmic complexity
     */
    private double calculateRhythmicComplexity(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 分析节拍稳定性 / Analyze beat stability
            double beatStability = calculateBeatStability(audioData);
            
            // 计算节拍变化 / Calculate tempo variation
            double tempoVariation = calculateTempoVariation(audioData, parameters);
            
            // 分析节奏模式复杂度 / Analyze rhythm pattern complexity
            double rhythmPatternComplexity = calculateRhythmPatternComplexity(audioData, parameters);
            
            // 计算同步性 / Calculate synchrony
            double synchrony = calculateRhythmicSynchrony(audioData, parameters);
            
            // 综合节奏复杂度 / Combine rhythmic complexity
            double rhythmicComplexity = ((1.0 - beatStability) * 0.3 + tempoVariation * 0.3 + 
                                       rhythmPatternComplexity * 0.3 + (1.0 - synchrony) * 0.1);
            
            // 使用标准化置信度计算器 / Use standardized confidence calculator
            Map<String, Double> factors = new HashMap<>();
            factors.put("beatInstability", 1.0 - beatStability);
            factors.put("tempoVariation", tempoVariation);
            factors.put("rhythmPatternComplexity", rhythmPatternComplexity);
            factors.put("asynchrony", 1.0 - synchrony);
            
            Map<String, Double> weights = new HashMap<>();
            weights.put("beatInstability", 0.3);
            weights.put("tempoVariation", 0.3);
            weights.put("rhythmPatternComplexity", 0.3);
            weights.put("asynchrony", 0.1);
            
            double confidence = confidenceCalculator.calculateWeightedConfidence(factors, weights);
            
            return Math.max(0.0, Math.min(1.0, rhythmicComplexity));
            
        } catch (Exception e) {
            // 使用标准化置信度计算器计算最小置信度 / Use standardized confidence calculator for minimum confidence
            return confidenceCalculator.calculateMinimumConfidence(1, 0.5); // 默认中等复杂度 / Default medium complexity
        }
    }
    
    /**
     * 计算旋律复杂度 / Calculate melodic complexity
     */
    private double calculateMelodicComplexity(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 计算频谱重心变化 / Calculate spectral centroid variation
            double centroidVariation = calculateSpectralCentroidVariation(audioData, parameters);
            
            // 计算音高变化 / Calculate pitch variation
            double pitchVariation = calculatePitchVariation(audioData, parameters);
            
            // 分析旋律轮廓复杂度 / Analyze melodic contour complexity
            double contourComplexity = calculateMelodicContourComplexity(audioData, parameters);
            
            // 计算音程复杂度 / Calculate interval complexity
            double intervalComplexity = calculateIntervalComplexity(audioData, parameters);
            
            // 综合旋律复杂度 / Combine melodic complexity
            double melodicComplexity = (centroidVariation * 0.25 + pitchVariation * 0.25 + 
                                      contourComplexity * 0.25 + intervalComplexity * 0.25);
            
            // 使用标准化置信度计算器 / Use standardized confidence calculator
            Map<String, Double> factors = new HashMap<>();
            factors.put("centroidVariation", centroidVariation);
            factors.put("pitchVariation", pitchVariation);
            factors.put("contourComplexity", contourComplexity);
            factors.put("intervalComplexity", intervalComplexity);
            
            Map<String, Double> weights = new HashMap<>();
            weights.put("centroidVariation", 0.25);
            weights.put("pitchVariation", 0.25);
            weights.put("contourComplexity", 0.25);
            weights.put("intervalComplexity", 0.25);
            
            double confidence = confidenceCalculator.calculateWeightedConfidence(factors, weights);
            
            return Math.max(0.0, Math.min(1.0, melodicComplexity));
            
        } catch (Exception e) {
            // 使用标准化置信度计算器计算最小置信度 / Use standardized confidence calculator for minimum confidence
            return confidenceCalculator.calculateMinimumConfidence(1, 0.5); // 默认中等复杂度 / Default medium complexity
        }
    }
    
    /**
     * 计算频谱复杂度 / Calculate spectral complexity
     */
    private double calculateSpectralComplexity(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 计算频谱熵 / Calculate spectral entropy
            double spectralEntropy = calculateSpectralEntropy(audioData, parameters);
            
            // 计算频谱平坦度 / Calculate spectral flatness
            double spectralFlatness = calculateSpectralFlatness(audioData, parameters);
            
            // 计算频谱对比度 / Calculate spectral contrast
            double spectralContrast = calculateSpectralContrast(audioData, parameters);
            
            // 计算频谱变化率 / Calculate spectral flux
            double spectralFlux = calculateSpectralFlux(audioData, parameters);
            
            // 综合频谱复杂度 / Combine spectral complexity
            double spectralComplexity = (spectralEntropy * 0.3 + (1.0 - spectralFlatness) * 0.2 + 
                                       spectralContrast * 0.3 + spectralFlux * 0.2);
            
            // 使用标准化置信度计算器 / Use standardized confidence calculator
            Map<String, Double> factors = new HashMap<>();
            factors.put("spectralEntropy", spectralEntropy);
            factors.put("spectralFlatnessInv", 1.0 - spectralFlatness);
            factors.put("spectralContrast", spectralContrast);
            factors.put("spectralFlux", spectralFlux);
            
            Map<String, Double> weights = new HashMap<>();
            weights.put("spectralEntropy", 0.3);
            weights.put("spectralFlatnessInv", 0.2);
            weights.put("spectralContrast", 0.3);
            weights.put("spectralFlux", 0.2);
            
            double confidence = confidenceCalculator.calculateWeightedConfidence(factors, weights);
            
            return Math.max(0.0, Math.min(1.0, spectralComplexity));
            
        } catch (Exception e) {
            // 使用标准化置信度计算器计算最小置信度 / Use standardized confidence calculator for minimum confidence
            return confidenceCalculator.calculateMinimumConfidence(1, 0.5); // 默认中等复杂度 / Default medium complexity
        }
    }
    
    /**
     * 计算总体复杂度 / Calculate overall complexity
     */
    private double calculateOverallComplexity(double harmonic, double rhythmic, double melodic, double spectral) {
        // 加权平均 / Weighted average
        return harmonic * 0.3 + rhythmic * 0.25 + melodic * 0.25 + spectral * 0.2;
    }
    
    /**
     * 分类复杂度等级 / Classify complexity level
     */
    private String classifyComplexity(double complexity) {
        if (complexity < 0.2) {
            return "very_simple";
        } else if (complexity < 0.4) {
            return "simple";
        } else if (complexity < 0.6) {
            return "moderate";
        } else if (complexity < 0.8) {
            return "complex";
        } else {
            return "very_complex";
        }
    }
    
    /**
     * 生成详细分析 / Generate detailed analysis
     */
    private Map<String, String> generateDetailedAnalysis(double harmonic, double rhythmic, double melodic, double spectral) {
        Map<String, String> analysis = new HashMap<>();
        
        analysis.put("harmonic", getComplexityDescription(harmonic, "和声"));
        analysis.put("rhythmic", getComplexityDescription(rhythmic, "节奏"));
        analysis.put("melodic", getComplexityDescription(melodic, "旋律"));
        analysis.put("spectral", getComplexityDescription(spectral, "频谱"));
        
        return analysis;
    }
    
    private String getComplexityDescription(double complexity, String aspect) {
        String level = classifyComplexity(complexity);
        switch (level) {
            case "very_simple":
                return aspect + "非常简单，结构清晰";
            case "simple":
                return aspect + "相对简单，易于理解";
            case "moderate":
                return aspect + "中等复杂度，平衡适中";
            case "complex":
                return aspect + "较为复杂，富有变化";
            case "very_complex":
                return aspect + "非常复杂，结构丰富";
            default:
                return aspect + "复杂度未知";
        }
    }
    
    // 辅助计算方法 / Helper calculation methods
    
    private double calculateEntropy(double[] values) {
        // 归一化 / Normalize
        double sum = 0.0;
        for (double value : values) {
            sum += Math.abs(value);
        }
        
        if (sum == 0) return 0.0;
        
        double entropy = 0.0;
        for (double value : values) {
            double p = Math.abs(value) / sum;
            if (p > 0) {
                entropy -= p * Math.log(p) / Math.log(2);
            }
        }
        
        return entropy / Math.log(values.length) / Math.log(2); // 归一化到0-1 / Normalize to 0-1
    }
    
    private double calculateVariation(double[] values) {
        if (values.length < 2) return 0.0;
        
        double mean = 0.0;
        for (double value : values) {
            mean += value;
        }
        mean /= values.length;
        
        double variance = 0.0;
        for (double value : values) {
            double diff = value - mean;
            variance += diff * diff;
        }
        variance /= values.length;
        
        return Math.min(1.0, Math.sqrt(variance));
    }
    
    private double calculateChordChangeRate(AudioData audioData) throws AudioProcessingException {
        // 简化的和弦变化率计算 / Simplified chord change rate calculation
        int frameSize = 4096;
        int hopSize = 2048;
        IVector<Double> samples = audioData.getSamples();
        
        List<Double> chromaChanges = new ArrayList<>();
        double[] prevChroma = null;
        
        int frameCount = 0;
        int maxFrames = 100; // 限制帧数以提高性能 / Limit frames for performance
        
        for (int i = 0; i < samples.length() - frameSize && frameCount < maxFrames; i += hopSize, frameCount++) {
            IVector<Double> frame = Linalg.zeros(frameSize);
            for (int j = 0; j < frameSize && i + j < samples.length(); j++) {
                frame.set(j, samples.get(i + j));
            }
            
            try {
                AudioData frameData = new AudioData(Linalg.vector(frame.toDoubleArray()), audioData.getSampleRate(), 
                    audioData.getChannels(), frame.length(), audioData.getFormat());
                double[] chroma = keyAnalyzer.analyzeChromaFeatures(frameData);
                
                if (prevChroma != null) {
                    double change = calculateEuclideanDistance(chroma, prevChroma);
                    chromaChanges.add(change);
                }
                
                prevChroma = chroma;
            } catch (Exception e) {
                // 忽略错误帧 / Ignore error frames
            }
        }
        
        if (chromaChanges.isEmpty()) return 0.0;
        
        double meanChange = chromaChanges.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        return Math.min(1.0, meanChange * 2.0); // 归一化 / Normalize
    }
    
    private double calculateEuclideanDistance(double[] a, double[] b) {
        if (a.length != b.length) return 0.0;
        
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        
        return Math.sqrt(sum);
    }
    
    private double calculateBeatStability(AudioData audioData) throws AudioProcessingException {
        try {
            double tempo = beatAnalyzer.estimateTempo(audioData);
            
            // 分析节拍的一致性 / Analyze beat consistency
            // 这里简化处理，实际应该分析整个音频的节拍变化 / Simplified here, should analyze beat changes throughout audio
            
            // 基于节拍速度的稳定性估计 / Stability estimation based on tempo
            double stability = 1.0;
            
            // 极端节拍速度通常不太稳定 / Extreme tempos are usually less stable
            if (tempo < 60 || tempo > 180) {
                stability *= 0.8;
            }
            
            return stability;
            
        } catch (Exception e) {
            return 0.5; // 默认中等稳定性 / Default medium stability
        }
    }
    
    private double calculateTempoVariation(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        // 简化的节拍变化计算 / Simplified tempo variation calculation
        int frameSize = (Integer) parameters.getOrDefault("frameSize", DEFAULT_FRAME_SIZE);
        int hopSize = frameSize / 2;
        IVector<Double> samples = audioData.getSamples();
        
        List<Double> tempos = new ArrayList<>();
        
        int frameCount = 0;
        int maxFrames = 100; // 限制帧数以提高性能 / Limit frames for performance
        
        for (int i = 0; i < samples.length() - frameSize && frameCount < maxFrames; i += hopSize, frameCount++) {
            IVector<Double> frame = Linalg.zeros(frameSize);
            for (int j = 0; j < frameSize && i + j < samples.length(); j++) {
                frame.set(j, samples.get(i + j));
            }
            
            try {
                AudioData frameData = new AudioData(Linalg.vector(frame.toDoubleArray()), audioData.getSampleRate(), 
                    audioData.getChannels(), frame.length(), audioData.getFormat());
                double tempo = beatAnalyzer.estimateTempo(frameData);
                if (tempo > 0) {
                    tempos.add(tempo);
                }
            } catch (Exception e) {
                // 忽略错误帧 / Ignore error frames
            }
        }
        
        if (tempos.size() < 2) return 0.0;
        
        double[] tempoArray = tempos.stream().mapToDouble(Double::doubleValue).toArray();
        return calculateVariation(tempoArray);
    }
    
    private double calculateRhythmPatternComplexity(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        // 简化的节奏模式复杂度计算 / Simplified rhythm pattern complexity calculation
        IVector<Double> samples = audioData.getSamples();
        
        // 计算能量包络 / Calculate energy envelope
        int frameSize = 512;
        List<Double> energyEnvelope = new ArrayList<>();
        
        for (int i = 0; i < samples.length() - frameSize; i += frameSize / 2) {
            double energy = 0.0;
            for (int j = 0; j < frameSize && i + j < samples.length(); j++) {
                double sample = samples.get(i + j);
                energy += sample * sample;
            }
            energyEnvelope.add(Math.sqrt(energy / frameSize));
        }
        
        // 分析能量包络的复杂度 / Analyze energy envelope complexity
        if (energyEnvelope.size() < 2) return 0.0;
        
        double[] energyArray = energyEnvelope.stream().mapToDouble(Double::doubleValue).toArray();
        return calculateEntropy(energyArray);
    }
    
    private double calculateRhythmicSynchrony(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        // 简化的节奏同步性计算 / Simplified rhythmic synchrony calculation
        // 实际应该分析不同频段的节奏同步性 / Should analyze rhythmic synchrony across different frequency bands
        return 0.7; // 默认值 / Default value
    }
    
    private double calculateSpectralCentroidVariation(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        int windowSize = (Integer) parameters.getOrDefault("windowSize", DEFAULT_WINDOW_SIZE);
        int hopSize = (Integer) parameters.getOrDefault("hopSize", DEFAULT_HOP_SIZE);
        IVector<Double> samples = audioData.getSamples();
        
        List<Double> centroids = new ArrayList<>();
        
        int frameCount = 0;
        int maxFrames = 100; // 限制帧数以提高性能 / Limit frames for performance
        
        for (int i = 0; i < samples.length() - windowSize && frameCount < maxFrames; i += hopSize, frameCount++) {
            IVector<Double> frame = Linalg.zeros(windowSize);
            for (int j = 0; j < windowSize && i + j < samples.length(); j++) {
                frame.set(j, samples.get(i + j));
            }
            
            IVector<Double> windowed = applyWindow(frame, windowSize);
            
            // 转换为Complex数组
            Complex[] complexInput = new Complex[windowed.length()];
            for (int j = 0; j < windowed.length(); j++) {
                complexInput[j] = new Complex(windowed.get(j), 0.0);
            }
            
            // 零填充确保长度为2的幂
            Complex[] paddedInput = RereFFT.zeroPadToPowerOfTwo(complexInput);
            
            Complex[] spectrum = RereFFT.fft(paddedInput);
            
            double centroid = calculateSpectralCentroid(spectrum, audioData.getSampleRate(), windowSize);
            centroids.add(centroid);
        }
        
        if (centroids.size() < 2) return 0.0;
        
        double[] centroidArray = centroids.stream().mapToDouble(Double::doubleValue).toArray();
        return calculateVariation(centroidArray);
    }
    
    private double calculateSpectralCentroid(Complex[] spectrum, double sampleRate, int windowSize) {
        double weightedSum = 0.0;
        double magnitudeSum = 0.0;
        
        for (int i = 0; i < spectrum.length / 2; i++) {
            double magnitude = spectrum[i].magnitude();
            double frequency = (i * sampleRate) / windowSize;
            weightedSum += frequency * magnitude;
            magnitudeSum += magnitude;
        }
        
        return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0.0;
    }
    
    private double calculatePitchVariation(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        // 简化的音高变化计算 / Simplified pitch variation calculation
        // 基于频谱重心变化估计 / Estimate based on spectral centroid variation
        return calculateSpectralCentroidVariation(audioData, parameters);
    }
    
    private double calculateMelodicContourComplexity(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        // 简化的旋律轮廓复杂度计算 / Simplified melodic contour complexity calculation
        return calculateSpectralCentroidVariation(audioData, parameters) * 0.8;
    }
    
    private double calculateIntervalComplexity(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        // 简化的音程复杂度计算 / Simplified interval complexity calculation
        try {
            double[] chromaFeatures = keyAnalyzer.analyzeChromaFeatures(audioData);
            return calculateEntropy(chromaFeatures) * 0.9;
        } catch (Exception e) {
            return 0.5;
        }
    }
    
    private double calculateSpectralHarmonicComplexity(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        int windowSize = (Integer) parameters.getOrDefault("windowSize", DEFAULT_WINDOW_SIZE);
        IVector<Double> windowed = applyWindow(audioData.getSamples(), windowSize);
        
        // 转换为Complex数组
        Complex[] complexInput = new Complex[windowed.length()];
        for (int i = 0; i < windowed.length(); i++) {
            complexInput[i] = new Complex(windowed.get(i), 0.0);
        }
        
        // 零填充确保长度为2的幂
        Complex[] paddedInput = RereFFT.zeroPadToPowerOfTwo(complexInput);
        
        Complex[] spectrum = RereFFT.fft(paddedInput);
        
        // 计算谐波复杂度 / Calculate harmonic complexity
        double harmonicEnergy = 0.0;
        double totalEnergy = 0.0;
        
        for (int i = 1; i < spectrum.length / 2; i++) {
            double magnitude = spectrum[i].magnitude();
            totalEnergy += magnitude;
            
            // 检查是否为谐波 / Check if it's a harmonic
            boolean isHarmonic = false;
            for (int h = 2; h <= 10; h++) {
                if (Math.abs(i % h) < 2) {
                    isHarmonic = true;
                    break;
                }
            }
            
            if (isHarmonic) {
                harmonicEnergy += magnitude;
            }
        }
        
        return totalEnergy > 0 ? 1.0 - (harmonicEnergy / totalEnergy) : 0.5;
    }
    
    private double calculateSpectralEntropy(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        int windowSize = (Integer) parameters.getOrDefault("windowSize", DEFAULT_WINDOW_SIZE);
        IVector<Double> windowed = applyWindow(audioData.getSamples(), windowSize);
        
        // 转换为Complex数组
        Complex[] complexInput = new Complex[windowed.length()];
        for (int i = 0; i < windowed.length(); i++) {
            complexInput[i] = new Complex(windowed.get(i), 0.0);
        }
        
        // 零填充确保长度为2的幂
        Complex[] paddedInput = RereFFT.zeroPadToPowerOfTwo(complexInput);
        
        Complex[] spectrum = RereFFT.fft(paddedInput);
        
        double[] magnitudes = new double[spectrum.length / 2];
        for (int i = 0; i < magnitudes.length; i++) {
            magnitudes[i] = spectrum[i].magnitude();
        }
        
        return calculateEntropy(magnitudes);
    }
    
    private double calculateSpectralFlatness(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        int windowSize = (Integer) parameters.getOrDefault("windowSize", DEFAULT_WINDOW_SIZE);
        IVector<Double> windowed = applyWindow(audioData.getSamples(), windowSize);
        
        // 转换为Complex数组
        Complex[] complexInput = new Complex[windowed.length()];
        for (int i = 0; i < windowed.length(); i++) {
            complexInput[i] = new Complex(windowed.get(i), 0.0);
        }
        
        // 零填充确保长度为2的幂
        Complex[] paddedInput = RereFFT.zeroPadToPowerOfTwo(complexInput);
        
        Complex[] spectrum = RereFFT.fft(paddedInput);
        
        double geometricMean = 1.0;
        double arithmeticMean = 0.0;
        int count = 0;
        
        for (int i = 1; i < spectrum.length / 2; i++) {
            double magnitude = spectrum[i].magnitude();
            if (magnitude > 0) {
                geometricMean *= Math.pow(magnitude, 1.0 / (spectrum.length / 2 - 1));
                arithmeticMean += magnitude;
                count++;
            }
        }
        
        if (count > 0) {
            arithmeticMean /= count;
            return arithmeticMean > 0 ? geometricMean / arithmeticMean : 0.0;
        }
        
        return 0.0;
    }
    
    private double calculateSpectralContrast(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        int windowSize = (Integer) parameters.getOrDefault("windowSize", DEFAULT_WINDOW_SIZE);
        IVector<Double> windowed = applyWindow(audioData.getSamples(), windowSize);
        
        // Convert IVector<Double> to Complex[] for FFT
        Complex[] complexInput = new Complex[windowed.length()];
        for (int i = 0; i < windowed.length(); i++) {
            complexInput[i] = new Complex(windowed.get(i), 0.0);
        }
        // 零填充确保长度为2的幂
        Complex[] paddedInput = RereFFT.zeroPadToPowerOfTwo(complexInput);
        Complex[] spectrum = RereFFT.fft(paddedInput);
        
        int numBands = 6;
        double contrast = 0.0;
        
        for (int band = 0; band < numBands; band++) {
            int startBin = (band * spectrum.length / 2) / numBands;
            int endBin = ((band + 1) * spectrum.length / 2) / numBands;
            
            double maxMagnitude = 0.0;
            double minMagnitude = Double.MAX_VALUE;
            
            for (int i = startBin; i < endBin; i++) {
                double magnitude = spectrum[i].magnitude();
                maxMagnitude = Math.max(maxMagnitude, magnitude);
                minMagnitude = Math.min(minMagnitude, magnitude);
            }
            
            if (minMagnitude > 0) {
                contrast += Math.log(maxMagnitude / minMagnitude);
            }
        }
        
        return contrast / numBands / 10.0; // 归一化 / Normalize
    }
    
    private double calculateSpectralFlux(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        int windowSize = (Integer) parameters.getOrDefault("windowSize", DEFAULT_WINDOW_SIZE);
        int hopSize = (Integer) parameters.getOrDefault("hopSize", DEFAULT_HOP_SIZE);
        IVector<Double> samples = audioData.getSamples();
        
        List<Double> fluxValues = new ArrayList<>();
        Complex[] prevSpectrum = null;
        
        for (int i = 0; i < samples.length() - windowSize; i += hopSize) {
            IVector<Double> frame = Linalg.zeros(windowSize);
            for (int j = 0; j < windowSize && i + j < samples.length(); j++) {
                frame.set(j, samples.get(i + j));
            }
            
            IVector<Double> windowed = applyWindow(frame, windowSize);
            
            // Convert IVector<Double> to Complex[] for FFT
            Complex[] complexInput = new Complex[windowed.length()];
            for (int k = 0; k < windowed.length(); k++) {
                complexInput[k] = new Complex(windowed.get(k), 0.0);
            }
            // 零填充确保长度为2的幂
            Complex[] paddedInput = RereFFT.zeroPadToPowerOfTwo(complexInput);
            Complex[] spectrum = RereFFT.fft(paddedInput);
            
            if (prevSpectrum != null) {
                double flux = 0.0;
                for (int k = 0; k < spectrum.length / 2; k++) {
                    double diff = spectrum[k].magnitude() - prevSpectrum[k].magnitude();
                    flux += Math.max(0, diff); // 只考虑增加的部分 / Only consider increases
                }
                fluxValues.add(flux);
            }
            
            prevSpectrum = spectrum;
        }
        
        if (fluxValues.isEmpty()) return 0.0;
        
        double meanFlux = fluxValues.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        return Math.min(1.0, meanFlux / 1000.0); // 归一化 / Normalize
    }
    
    private IVector<Double> applyWindow(IVector<Double> signal, int windowSize) {
        int length = Math.min(signal.length(), windowSize);
        IVector<Double> windowed = Linalg.zeros(windowSize);
        
        for (int i = 0; i < length; i++) {
            double window = 0.54 - 0.46 * Math.cos(2.0 * Math.PI * i / (length - 1));
            windowed.set(i, signal.get(i) * window);
        }
        
        return windowed;
    }
}