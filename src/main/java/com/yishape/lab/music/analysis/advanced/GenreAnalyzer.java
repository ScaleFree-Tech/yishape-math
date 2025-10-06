package com.yishape.lab.music.analysis.advanced;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioUtil;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.music.analysis.basic.BeatAnalyzerImpl;
import com.yishape.lab.music.analysis.basic.KeyAnalyzerImpl;
import com.yishape.lab.music.analysis.MusicDetectionResult;
import com.yishape.lab.music.analysis.StandardizedConfidenceCalculator;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.core.RereFFT;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.music.core.MusicUtil;

import java.util.Map;


import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

/**
 * 音乐风格分析器 / Music Genre Analyzer
 * <p>
 * 基于音频特征分析音乐的风格类型，包括古典、摇滚、爵士、电子等。
 * Analyzes music genre based on audio features, including classical, rock, jazz, electronic, etc.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class GenreAnalyzer implements IAdvancedAnalyzer {
    
    // 支持的音乐风格 / Supported genres
    public static final String[] SUPPORTED_GENRES = {
        "classical", "rock", "jazz", "electronic", "pop", "blues", "country", "reggae"
    };
    
    // 默认参数 / Default parameters
    private static final int DEFAULT_WINDOW_SIZE = 4096;
    private static final int DEFAULT_HOP_SIZE = 2048;
    private static final int DEFAULT_MFCC_COUNT = 13;
    
    private final BeatAnalyzerImpl beatAnalyzer;
    private final KeyAnalyzerImpl keyAnalyzer;
    private final StandardizedConfidenceCalculator confidenceCalculator = new StandardizedConfidenceCalculator();
    
    public GenreAnalyzer() {
        this.beatAnalyzer = new BeatAnalyzerImpl();
        this.keyAnalyzer = new KeyAnalyzerImpl();
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
            
            // 提取特征 / Extract features
            GenreFeatures features = extractGenreFeatures(audioData, parameters);
            
            // 计算每个风格的概率 / Calculate probability for each genre
            Map<String, Double> genreProbabilities = calculateGenreProbabilities(features);
            
            // 找到最可能的风格 / Find most likely genre
            String predictedGenre = findMostLikelyGenre(genreProbabilities);
            double rawConfidence = genreProbabilities.get(predictedGenre);
            
            // 改进的置信度计算 / Improved confidence calculation
            double confidence = calculateImprovedConfidence(genreProbabilities, predictedGenre, rawConfidence);
            
            // 存储结果 / Store results
            results.put("predictedGenre", predictedGenre);
            results.put("confidence", confidence);
            results.put("genreProbabilities", genreProbabilities);
            results.put("features", features.toMap());
            
            return results;
            
        } catch (Exception e) {
            throw new AudioProcessingException("Error in genre analysis: " + e.getMessage(), e);
        }
    }
    
    @Override
    public MusicDetectionResult analyzeAdvancedMusic(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        Map<String, Object> results = analyze(audioData, parameters);
        
        // Create a GenreAnalysisResult from the map results
        GenreAnalysisResult genreResult = new GenreAnalysisResult();
        genreResult.setPredictedGenre((String) results.get("predictedGenre"));
        genreResult.setConfidence((Double) results.get("confidence"));
        genreResult.setGenreProbabilities((Map<String, Double>) results.get("genreProbabilities"));
        
        return genreResult;
    }
    
    @Override
    public String getAnalyzerName() {
        return "genre_analyzer";
    }
    
    @Override
    public String getName() {
        return getAnalyzerName();
    }
    
    @Override
    public String[] getSupportedParameters() {
        return new String[]{"windowSize", "hopSize", "mfccCount"};
    }
    
    @Override
    public Map<String, Object> getDefaultParameters() {
        Map<String, Object> params = new HashMap<>();
        params.put("windowSize", DEFAULT_WINDOW_SIZE);
        params.put("hopSize", DEFAULT_HOP_SIZE);
        params.put("mfccCount", DEFAULT_MFCC_COUNT);
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
        return "Analyzes music genre based on audio features";
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
        return audioLength * 0.1; // Low complexity
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
     * 提取风格特征 / Extract genre features
     */
    private GenreFeatures extractGenreFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        GenreFeatures features = new GenreFeatures();
        
        // 基本音频特征 / Basic audio features
        IVector<Double> samples = audioData.getSamples();
        features.rms = calculateRMS(samples);
        features.zcr = calculateZeroCrossingRate(samples);
        
        // 计算频谱特征并进行归一化 / Calculate spectral features with normalization
        double rawSpectralCentroid = calculateSpectralCentroid(audioData, parameters);
        double rawSpectralRolloff = calculateSpectralRolloff(audioData, parameters);
        double rawSpectralBandwidth = calculateSpectralBandwidth(audioData, parameters);
        
        // 频谱特征归一化 / Normalize spectral features
        // 基于采样率的归一化因子 / Normalization factors based on sample rate
        double sampleRate = audioData.getSampleRate();
        double nyquist = sampleRate / 2.0;
        
        features.spectralCentroid = rawSpectralCentroid / nyquist;
        features.spectralRolloff = rawSpectralRolloff / nyquist;
        features.spectralBandwidth = rawSpectralBandwidth / nyquist;
        
        // MFCC特征 / MFCC features
        int mfccCount = (Integer) parameters.getOrDefault("mfccCount", DEFAULT_MFCC_COUNT);
        
        // 使用MusicUtility的新MFCC矩阵计算方法，限制帧数以提高性能
        // Use MusicUtil's new MFCC matrix calculation method, limit frames for performance
        double[][] mfccFrames = AudioUtil.calculateMFCCFrames(audioData, parameters, mfccCount);
        
        // 限制处理的帧数以提高性能 / Limit processed frames for performance
        int maxFrames = 100; // 与EmotionAnalyzer保持一致 / Keep consistent with EmotionAnalyzer
        if (mfccFrames.length > maxFrames) {
            // 均匀采样帧 / Uniformly sample frames
            double[][] sampledFrames = new double[maxFrames][mfccFrames[0].length];
            double step = (double) (mfccFrames.length - 1) / (maxFrames - 1);
            for (int i = 0; i < maxFrames; i++) {
                int sourceIndex = (int) Math.round(i * step);
                sampledFrames[i] = mfccFrames[Math.min(sourceIndex, mfccFrames.length - 1)];
            }
            mfccFrames = sampledFrames;
        }
        
        // 计算MFCC统计特征用于风格分析
        features.mfcc = AudioUtil.calculateColumnMeans(mfccFrames);
        features.mfccVariances = AudioUtil.calculateColumnVariances(mfccFrames);
        features.mfccStability = MusicUtil.calculateMFCCStability(mfccFrames);
        
        // 计算MFCC时间序列特征
        double[][] mfccDeltaFrames = AudioUtil.calculateTemporalDelta(mfccFrames);
        features.mfccDeltaMeans = AudioUtil.calculateColumnMeans(mfccDeltaFrames);
        
        // 节拍特征 / Beat features
        try {
            features.tempo = beatAnalyzer.estimateTempo(audioData);
            features.beatStrength = calculateBeatStrength(audioData);
        } catch (Exception e) {
            features.tempo = 120.0;
            features.beatStrength = 0.5;
        }
        
        // 调性特征 / Key features
        try {
            double[] chromaFeatures = keyAnalyzer.analyzeChromaFeatures(audioData);
            features.chromaVariance = calculateVariance(chromaFeatures);
            features.chromaEnergy = calculateMean(chromaFeatures);
        } catch (Exception e) {
            features.chromaVariance = 0.5;
            features.chromaEnergy = 0.5;
        }
        
        // 频谱特征 / Spectral features
        double rawSpectralContrast = calculateSpectralContrast(audioData, parameters);
        double rawSpectralFlatness = calculateSpectralFlatness(audioData, parameters);
        
        // 频谱对比度和平坦度归一化 / Normalize spectral contrast and flatness
        // 频谱对比度通常范围在0-10之间，归一化到[0,1] / Spectral contrast typically ranges 0-10, normalize to [0,1]
        features.spectralContrast = Math.min(1.0, rawSpectralContrast / 5.0);
        // 频谱平坦度已经是[0,1]范围，但进行稳定性处理 / Spectral flatness is already [0,1] but stabilize
        features.spectralFlatness = Math.max(0.0, Math.min(1.0, rawSpectralFlatness));
        
        return features;
    }
    
    /**
     * 计算风格概率 / Calculate genre probabilities
     */
    private Map<String, Double> calculateGenreProbabilities(GenreFeatures features) {
        Map<String, Double> probabilities = new HashMap<>();
        
        // 基于规则的简化分类器 / Rule-based simplified classifier
        // 实际应用中应该使用机器学习模型 / In practice, should use machine learning models
        
        // 首先进行节拍验证 / First perform tempo validation
        Map<String, Double> tempoValidationScores = calculateTempoValidationScores(features.tempo);
        
        for (String genre : SUPPORTED_GENRES) {
            double baseScore = calculateGenreScore(genre, features);
            double tempoScore = tempoValidationScores.getOrDefault(genre, 1.0);
            
            // 结合基础得分和节拍验证得分 / Combine base score with tempo validation score
            double finalScore = baseScore * tempoScore;
            probabilities.put(genre, finalScore);
        }
        
        // 归一化概率 / Normalize probabilities
        double sum = probabilities.values().stream().mapToDouble(Double::doubleValue).sum();
        if (sum > 0) {
            probabilities.replaceAll((k, v) -> v / sum);
        }
        
        return probabilities;
    }
    
    /**
     * 计算节拍验证得分 / Calculate tempo validation scores
     */
    private Map<String, Double> calculateTempoValidationScores(double tempo) {
        Map<String, Double> scores = new HashMap<>();
        
        // 定义各流派的典型节拍范围 / Define typical tempo ranges for each genre
        // 基于音乐理论的实际节拍范围 / Based on music theory and actual tempo ranges
        
        // 古典音乐：60-180 BPM，但主要集中在中等速度
        if (tempo >= 60 && tempo <= 180) {
            scores.put("classical", 1.0 - Math.abs(tempo - 100.0) / 120.0);
        } else {
            scores.put("classical", 0.2); // 超出范围但给予少量分数
        }
        
        // 摇滚音乐：100-180 BPM
        if (tempo >= 100 && tempo <= 180) {
            scores.put("rock", 1.0 - Math.abs(tempo - 140.0) / 80.0);
        } else {
            scores.put("rock", 0.1);
        }
        
        // 爵士音乐：80-200 BPM，但主要集中在中等速度
        if (tempo >= 80 && tempo <= 200) {
            scores.put("jazz", 1.0 - Math.abs(tempo - 120.0) / 120.0);
        } else {
            scores.put("jazz", 0.2);
        }
        
        // 电子音乐：120-140 BPM（主要范围）
        if (tempo >= 120 && tempo <= 140) {
            scores.put("electronic", 1.0 - Math.abs(tempo - 130.0) / 20.0);
        } else if (tempo >= 110 && tempo <= 150) {
            scores.put("electronic", 0.7 - Math.abs(tempo - 130.0) / 40.0);
        } else {
            scores.put("electronic", 0.1);
        }
        
        // 流行音乐：90-130 BPM
        if (tempo >= 90 && tempo <= 130) {
            scores.put("pop", 1.0 - Math.abs(tempo - 110.0) / 40.0);
        } else {
            scores.put("pop", 0.2);
        }
        
        // 蓝调音乐：60-120 BPM
        if (tempo >= 60 && tempo <= 120) {
            scores.put("blues", 1.0 - Math.abs(tempo - 90.0) / 60.0);
        } else {
            scores.put("blues", 0.1);
        }
        
        // 乡村音乐：100-160 BPM
        if (tempo >= 100 && tempo <= 160) {
            scores.put("country", 1.0 - Math.abs(tempo - 130.0) / 60.0);
        } else {
            scores.put("country", 0.1);
        }
        
        // 雷鬼音乐：60-100 BPM（强调慢节奏）
        if (tempo >= 60 && tempo <= 100) {
            scores.put("reggae", 1.0 - Math.abs(tempo - 80.0) / 40.0);
        } else if (tempo >= 50 && tempo <= 110) {
            scores.put("reggae", 0.6 - Math.abs(tempo - 80.0) / 60.0);
        } else {
            scores.put("reggae", 0.05);
        }
        
        // 确保所有分数在[0,1]范围内 / Ensure all scores are in [0,1] range
        scores.replaceAll((k, v) -> Math.max(0.0, Math.min(1.0, v)));
        
        return scores;
    }
    
    /**
     * 计算特定风格的得分 / Calculate score for specific genre
     */
    private double calculateGenreScore(String genre, GenreFeatures features) {
        double score = 0.0;
        
        // 计算MFCC特征得分 / Calculate MFCC feature scores
        double mfccScore = calculateMFCCScore(features);
        double mfccStabilityScore = 1.0 - features.mfccStability; // 稳定性越低，变化越大
        
        switch (genre) {
            case "classical":
                // 古典音乐特征：低节拍强度，高频谱复杂度，中等节拍速度，MFCC变化丰富
                score += (1.0 - features.beatStrength) * 0.25;
                score += features.spectralContrast * 0.25;
                score += (1.0 - Math.abs(features.tempo - 100.0) / 100.0) * 0.15;
                score += features.chromaVariance * 0.15;
                score += mfccStabilityScore * 0.2; // 古典音乐MFCC变化丰富
                break;
                
            case "rock":
                // 摇滚音乐特征：高能量，强节拍，中高频谱重心，MFCC相对稳定
                score += features.rms * 0.25;
                score += features.beatStrength * 0.25;
                score += Math.min(features.spectralCentroid / 3000.0, 1.0) * 0.2;
                score += (features.tempo > 100 ? 1.0 : features.tempo / 100.0) * 0.15;
                score += features.mfccStability * 0.15; // 摇滚音乐MFCC相对稳定
                break;
                
            case "jazz":
                // 爵士音乐特征：复杂和声，中等节拍，高色度变化，MFCC变化丰富
                score += features.chromaVariance * 0.3;
                score += features.spectralContrast * 0.2;
                score += (1.0 - Math.abs(features.tempo - 120.0) / 80.0) * 0.15;
                score += features.spectralBandwidth / 2000.0 * 0.15;
                score += mfccStabilityScore * 0.2; // 爵士音乐MFCC变化丰富
                break;
                
            case "electronic":
                // 电子音乐特征：高频谱平坦度，强节拍，高频能量，MFCC特征独特
                // 优化权重以更好地区分电子音乐与其他流派
                score += features.spectralFlatness * 0.3; // 电子音乐通常有高频谱平坦度
                score += features.beatStrength * 0.25; // 强节拍是电子音乐的重要特征
                score += Math.min(features.spectralRolloff / 8000.0, 1.0) * 0.15; // 高频能量
                score += (features.tempo >= 120 && features.tempo <= 140 ? 1.0 : 
                         Math.max(0.0, 1.0 - Math.abs(features.tempo - 130.0) / 50.0)) * 0.15; // 电子音乐典型节拍范围
                score += mfccScore * 0.15; // 电子音乐有独特的MFCC模式
                break;
                
            case "pop":
                // 流行音乐特征：中等各项指标，平衡的特征，MFCC适中
                score += (1.0 - Math.abs(features.rms - 0.5)) * 0.2;
                score += (1.0 - Math.abs(features.beatStrength - 0.7)) * 0.2;
                score += (1.0 - Math.abs(features.tempo - 120.0) / 60.0) * 0.2;
                score += (1.0 - Math.abs(features.spectralCentroid - 2000.0) / 2000.0) * 0.2;
                score += (1.0 - Math.abs(features.mfccStability - 0.5)) * 0.2; // 流行音乐MFCC适中
                break;
                
            case "blues":
                // 蓝调音乐特征：中低频重心，中等节拍，特定和声特征，MFCC相对稳定
                score += (1.0 - features.spectralCentroid / 4000.0) * 0.25;
                score += (1.0 - Math.abs(features.tempo - 90.0) / 60.0) * 0.25;
                score += features.chromaEnergy * 0.2;
                score += (1.0 - features.spectralFlatness) * 0.15;
                score += features.mfccStability * 0.15; // 蓝调音乐MFCC相对稳定
                break;
                
            case "country":
                // 乡村音乐特征：中频重心，稳定节拍，简单和声，MFCC稳定
                score += (1.0 - Math.abs(features.spectralCentroid - 2500.0) / 2500.0) * 0.25;
                score += (1.0 - Math.abs(features.tempo - 110.0) / 50.0) * 0.25;
                score += (1.0 - features.chromaVariance) * 0.2;
                score += features.beatStrength * 0.15;
                score += features.mfccStability * 0.15; // 乡村音乐MFCC稳定
                break;
                
            case "reggae":
                // 雷鬼音乐特征：特殊节拍模式，中低频重心，MFCC有特定模式
                // 优化权重以更好地区分雷鬼音乐与电子音乐
                score += (1.0 - Math.abs(features.tempo - 80.0) / 30.0) * 0.35; // 雷鬼音乐典型节拍范围更严格
                score += (1.0 - features.spectralCentroid / 0.3) * 0.25; // 更低的频谱重心（归一化后阈值）
                score += features.beatStrength * 0.15; // 降低节拍强度权重
                score += (1.0 - features.spectralFlatness) * 0.1; // 雷鬼音乐通常有较低的频谱平坦度
                score += mfccScore * 0.15; // 雷鬼音乐有特定的MFCC模式
                break;
                
            default:
                score = 0.1; // 默认低分 / Default low score
        }
        
        return Math.max(0.0, Math.min(1.0, score));
    }
    
    /**
     * 找到最可能的风格 / Find most likely genre
     */
    private String findMostLikelyGenre(Map<String, Double> probabilities) {
        return probabilities.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("unknown");
    }
    
    /**
     * 改进的置信度计算 / Improved confidence calculation
     */
    private double calculateImprovedConfidence(Map<String, Double> probabilities, String predictedGenre, double rawConfidence) {
        try {
            // 基础置信度 / Base confidence
            double baseConfidence = rawConfidence;
            
            // 计算概率分布的离散度 / Calculate probability distribution dispersion
            List<Double> sortedProbabilities = probabilities.values().stream()
                    .sorted(java.util.Comparator.reverseOrder())
                    .collect(java.util.stream.Collectors.toList());
            
            double top1 = sortedProbabilities.get(0);
            double top2 = sortedProbabilities.size() > 1 ? sortedProbabilities.get(1) : 0.0;
            double top3 = sortedProbabilities.size() > 2 ? sortedProbabilities.get(2) : 0.0;
            
            // 计算前两名之间的差距 / Calculate gap between top two
            double topGap = top1 - top2;
            
            // 使用标准化置信度计算器计算基于差距的置信度 / Use standardized confidence calculator for gap-based confidence
            double[] topValues = {top1, top2, top3};
            double gapBasedConfidence = confidenceCalculator.calculateGapBasedConfidence(topValues);
            
            // 计算熵值（概率分布的均匀程度）/ Calculate entropy (uniformity of probability distribution)
            Map<String, Double> probabilityMap = new HashMap<>();
            int index = 0;
            for (String genre : probabilities.keySet()) {
                probabilityMap.put(genre, probabilities.get(genre));
                index++;
                if (index >= 5) break; // 限制为前5个流派 / Limit to top 5 genres
            }
            double entropyBasedConfidence = confidenceCalculator.calculateDistributionConfidence(probabilityMap);
            
            // 基于以下因素调整置信度 / Adjust confidence based on:
            // 1. 前两名之间的差距越大，置信度越高 / Larger gap between top two increases confidence
            // 2. 熵值越低（分布越集中），置信度越高 / Lower entropy (more concentrated distribution) increases confidence
            // 3. 前三名之间的差距越大，置信度越高 / Larger gaps among top three increase confidence
            
            double gapFactor = 1.0;
            double entropyFactor = entropyBasedConfidence; // 熵值越低，因子越大 / Lower entropy gives larger factor
            
            // 根据差距调整因子 / Adjust factor based on gaps
            if (topGap > 0.3) {
                gapFactor = 1.2; // 明显区分 / Clear distinction
            } else if (topGap > 0.15) {
                gapFactor = 1.0; // 中等区分 / Moderate distinction
            } else if (topGap > 0.05) {
                gapFactor = 0.8; // 轻微区分 / Slight distinction
            } else {
                gapFactor = 0.5; // 几乎无区分 / Almost no distinction
            }
            
            // 根据前三名之间的关系进一步调整 / Further adjust based on top three relationships
            double thirdGapFactor = 1.0;
            if (top2 - top3 < 0.05) {
                thirdGapFactor = 0.9; // 第二名和第三名很接近 / Top 2 and 3 are very close
            }
            
            // 计算最终置信度 / Calculate final confidence
            double adjustedConfidence = baseConfidence * gapFactor * entropyFactor * thirdGapFactor;
            
            // 使用标准化置信度计算器确保置信度在合理范围内 / Use standardized confidence calculator to ensure confidence is within reasonable range
            return confidenceCalculator.calculateMinimumConfidence(1, adjustedConfidence);
            
        } catch (Exception e) {
            // 使用标准化置信度计算器计算最小置信度 / Use standardized confidence calculator for minimum confidence
            return confidenceCalculator.calculateMinimumConfidence(1, 0.5); // 默认中等置信度 / Default medium confidence
        }
    }
    
    // 辅助方法 / Helper methods
    
    private double calculateRMS(IVector<Double> samples) {
        double sum = 0.0;
        for (int i = 0; i < samples.length(); i++) {
            double sample = samples.get(i);
            sum += sample * sample;
        }
        return Math.sqrt(sum / samples.length());
    }
    
    private double calculateZeroCrossingRate(IVector<Double> samples) {
        int crossings = 0;
        for (int i = 1; i < samples.length(); i++) {
            if ((samples.get(i) >= 0) != (samples.get(i - 1) >= 0)) {
                crossings++;
            }
        }
        return (double) crossings / samples.length();
    }
    
    private double calculateSpectralCentroid(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        int windowSize = (Integer) parameters.getOrDefault("windowSize", DEFAULT_WINDOW_SIZE);
        IVector<Double> windowed = applyWindow(audioData.getSamples(), windowSize);
        
        // Convert IVector<Double> to Complex[] for FFT
        Complex[] complexInput = new Complex[windowed.length()];
        for (int i = 0; i < windowed.length(); i++) {
            complexInput[i] = new Complex(windowed.get(i), 0.0);
        }
        Complex[] spectrum = RereFFT.fft(complexInput);
        
        double weightedSum = 0.0;
        double magnitudeSum = 0.0;
        double sampleRate = audioData.getSampleRate();
        
        for (int i = 0; i < spectrum.length / 2; i++) {
            double magnitude = spectrum[i].magnitude();
            double frequency = (i * sampleRate) / windowSize;
            weightedSum += frequency * magnitude;
            magnitudeSum += magnitude;
        }
        
        return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0.0;
    }
    
    private double calculateSpectralRolloff(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        int windowSize = (Integer) parameters.getOrDefault("windowSize", DEFAULT_WINDOW_SIZE);
        IVector<Double> windowed = applyWindow(audioData.getSamples(), windowSize);
        
        // Convert IVector<Double> to Complex[] for FFT
        Complex[] input = new Complex[windowed.length()];
        for (int i = 0; i < windowed.length(); i++) {
            input[i] = new Complex(windowed.get(i), 0.0);
        }
        Complex[] spectrum = RereFFT.fft(input);
        
        double totalEnergy = 0.0;
        for (int i = 0; i < spectrum.length / 2; i++) {
            totalEnergy += spectrum[i].magnitude();
        }
        
        double threshold = 0.85 * totalEnergy;
        double cumulativeEnergy = 0.0;
        double sampleRate = audioData.getSampleRate();
        
        for (int i = 0; i < spectrum.length / 2; i++) {
            cumulativeEnergy += spectrum[i].magnitude();
            if (cumulativeEnergy >= threshold) {
                return (i * sampleRate) / windowSize;
            }
        }
        
        return sampleRate / 2.0;
    }
    
    private double calculateSpectralBandwidth(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        double centroid = calculateSpectralCentroid(audioData, parameters);
        int windowSize = (Integer) parameters.getOrDefault("windowSize", DEFAULT_WINDOW_SIZE);
        IVector<Double> windowed = applyWindow(audioData.getSamples(), windowSize);
        
        // Convert IVector<Double> to Complex[] for FFT
        Complex[] input = new Complex[windowed.length()];
        for (int i = 0; i < windowed.length(); i++) {
            input[i] = new Complex(windowed.get(i), 0.0);
        }
        Complex[] spectrum = RereFFT.fft(input);
        
        double weightedSum = 0.0;
        double magnitudeSum = 0.0;
        double sampleRate = audioData.getSampleRate();
        
        for (int i = 0; i < spectrum.length / 2; i++) {
            double magnitude = spectrum[i].magnitude();
            double frequency = (i * sampleRate) / windowSize;
            double diff = frequency - centroid;
            weightedSum += diff * diff * magnitude;
            magnitudeSum += magnitude;
        }
        
        return magnitudeSum > 0 ? Math.sqrt(weightedSum / magnitudeSum) : 0.0;
    }
    

    
    private double[] calculateMelFilters(Complex[] spectrum, double sampleRate, int filterCount) {
        double[] filters = new double[filterCount];
        int spectrumLength = spectrum.length / 2;
        
        for (int i = 0; i < filterCount; i++) {
            double sum = 0.0;
            int startBin = (i * spectrumLength) / filterCount;
            int endBin = ((i + 1) * spectrumLength) / filterCount;
            
            for (int j = startBin; j < endBin; j++) {
                sum += spectrum[j].magnitude();
            }
            
            filters[i] = Math.log(sum + 1e-10); // 避免log(0) / Avoid log(0)
        }
        
        return filters;
    }
    
    private double calculateBeatStrength(AudioData audioData) throws AudioProcessingException {
        // 简化的节拍强度计算 / Simplified beat strength calculation
        IVector<Double> samples = audioData.getSamples();
        double energy = calculateRMS(samples);
        
        // 计算能量变化 / Calculate energy variation
        int frameSize = 1024;
        List<Double> frameEnergies = new ArrayList<>();
        
        for (int i = 0; i < samples.length() - frameSize; i += frameSize) {
            double frameEnergy = 0.0;
            for (int j = 0; j < frameSize; j++) {
                double sample = samples.get(i + j);
                frameEnergy += sample * sample;
            }
            frameEnergies.add(Math.sqrt(frameEnergy / frameSize));
        }
        
        // 计算能量变化的方差 / Calculate variance of energy changes
        double mean = frameEnergies.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double variance = frameEnergies.stream()
                .mapToDouble(e -> (e - mean) * (e - mean))
                .average().orElse(0.0);
        
        return Math.min(1.0, variance * 10.0); // 归一化 / Normalize
    }
    
    private double calculateSpectralContrast(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        int windowSize = (Integer) parameters.getOrDefault("windowSize", DEFAULT_WINDOW_SIZE);
        IVector<Double> windowed = applyWindow(audioData.getSamples(), windowSize);
        
        // Convert IVector<Double> to Complex[] for FFT
        Complex[] input = new Complex[windowed.length()];
        for (int i = 0; i < windowed.length(); i++) {
            input[i] = new Complex(windowed.get(i), 0.0);
        }
        Complex[] spectrum = RereFFT.fft(input);
        
        // 计算频谱对比度 / Calculate spectral contrast
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
        
        return contrast / numBands;
    }
    
    private double calculateSpectralFlatness(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        int windowSize = (Integer) parameters.getOrDefault("windowSize", DEFAULT_WINDOW_SIZE);
        IVector<Double> windowed = applyWindow(audioData.getSamples(), windowSize);
        
        // Convert IVector<Double> to Complex[] for FFT
        Complex[] input = new Complex[windowed.length()];
        for (int i = 0; i < windowed.length(); i++) {
            input[i] = new Complex(windowed.get(i), 0.0);
        }
        Complex[] spectrum = RereFFT.fft(input);
        
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
    
    private IVector<Double> applyWindow(IVector<Double> signal, int windowSize) {
        int length = Math.min(signal.length(), windowSize);
        IVector<Double> windowed = Linalg.zeros(windowSize);
        
        for (int i = 0; i < length; i++) {
            double window = 0.54 - 0.46 * Math.cos(2.0 * Math.PI * i / (length - 1));
            windowed.set(i, signal.get(i) * window);
        }
        
        return windowed;
    }
    
    private double calculateVariance(double[] values) {
        double mean = calculateMean(values);
        double variance = 0.0;
        for (double value : values) {
            double diff = value - mean;
            variance += diff * diff;
        }
        return variance / values.length;
    }
    
    private double calculateMean(double[] values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }
    
    /**
     * 计算MFCC特征得分 / Calculate MFCC feature score
     */
    private double calculateMFCCScore(GenreFeatures features) {
        if (features.mfcc == null || features.mfccVariances == null) {
            return 0.5; // 默认中等得分
        }
        
        // 基于MFCC系数特征计算得分 - 针对电子和雷鬼音乐优化
        // 电子音乐：通常在低频MFCC系数有独特模式
        // 雷鬼音乐：在中频MFCC系数有特殊表现
        
        double lowFreqScore = 0.0;  // MFCC 1-4 (低频特征)
        double midFreqScore = 0.0;  // MFCC 5-8 (中频特征)
        double highFreqScore = 0.0; // MFCC 9-12 (高频特征)
        
        int coeffCount = Math.min(features.mfcc.length, 13); // 通常使用13个MFCC系数
        
        for (int i = 0; i < coeffCount; i++) {
            double normalizedCoeff = Math.abs(features.mfcc[i]) / 10.0; // 归一化
            normalizedCoeff = Math.min(1.0, normalizedCoeff); // 限制在[0,1]范围
            
            if (i < 4) {
                lowFreqScore += normalizedCoeff;
            } else if (i < 8) {
                midFreqScore += normalizedCoeff;
            } else {
                highFreqScore += normalizedCoeff;
            }
        }
        
        // 计算各频段得分
        lowFreqScore /= 4.0;
        midFreqScore /= 4.0;
        highFreqScore /= Math.max(1.0, coeffCount - 8);
        
        // 计算方差特征 - 反映频谱变化
        double varianceScore = 0.0;
        int varianceCount = Math.min(features.mfccVariances.length, coeffCount);
        for (int i = 0; i < varianceCount; i++) {
            varianceScore += Math.min(1.0, features.mfccVariances[i] / 5.0);
        }
        varianceScore /= varianceCount;
        
        // 计算稳定性特征 - 反映时间变化
        double stabilityScore = 1.0 - features.mfccStability; // 稳定性越低，变化越大
        
        // 综合得分：低频特征权重更高（对电子和雷鬼音乐更重要）
        return lowFreqScore * 0.4 + midFreqScore * 0.3 + highFreqScore * 0.1 + 
               varianceScore * 0.1 + stabilityScore * 0.1;
    }
    
    /**
     * 风格特征类 / Genre Features Class
     */
    private static class GenreFeatures {
        double rms;
        double zcr;
        double spectralCentroid;
        double spectralRolloff;
        double spectralBandwidth;
        double spectralContrast;
        double spectralFlatness;
        double[] mfcc;                  // MFCC均值 / MFCC means
        double[] mfccVariances;         // MFCC方差 / MFCC variances
        double mfccStability;           // MFCC稳定性 / MFCC stability
        double[] mfccDeltaMeans;        // MFCC Delta均值 / MFCC Delta means
        double tempo;
        double beatStrength;
        double chromaVariance;
        double chromaEnergy;
        
        Map<String, Object> toMap() {
            Map<String, Object> map = new HashMap<>();
            map.put("rms", rms);
            map.put("zcr", zcr);
            map.put("spectralCentroid", spectralCentroid);
            map.put("spectralRolloff", spectralRolloff);
            map.put("spectralBandwidth", spectralBandwidth);
            map.put("spectralContrast", spectralContrast);
            map.put("spectralFlatness", spectralFlatness);
            map.put("mfcc", mfcc);
            map.put("mfccVariances", mfccVariances);
            map.put("mfccStability", mfccStability);
            map.put("mfccDeltaMeans", mfccDeltaMeans);
            map.put("tempo", tempo);
            map.put("beatStrength", beatStrength);
            map.put("chromaVariance", chromaVariance);
            map.put("chromaEnergy", chromaEnergy);
            return map;
        }
    }
}