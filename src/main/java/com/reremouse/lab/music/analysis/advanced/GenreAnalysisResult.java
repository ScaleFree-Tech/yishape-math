package com.reremouse.lab.music.analysis.advanced;

import com.reremouse.lab.music.analysis.MusicDetectionResult;
import java.util.Map;
import java.util.HashMap;

/**
 * 风格分析结果类 / Genre Analysis Result Class
 * <p>
 * 封装音乐风格分析的结果，包括预测的风格类型和置信度。
 * Encapsulates the results of music genre analysis, including predicted genre type and confidence.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class GenreAnalysisResult extends MusicDetectionResult {
    private String predictedGenre;                    // 预测的风格 / Predicted genre
    private double confidence;                        // 置信度 / Confidence
    private Map<String, Double> genreProbabilities;   // 各风格的概率 / Genre probabilities
    private Map<String, Object> features;             // 特征数据 / Feature data

    public GenreAnalysisResult() {
        super();
        this.predictedGenre = "unknown";
        this.confidence = 0.0;
        this.genreProbabilities = new HashMap<>();
        this.features = new HashMap<>();
    }

    // Getters and setters
    public String getPredictedGenre() {
        return predictedGenre;
    }

    public void setPredictedGenre(String predictedGenre) {
        this.predictedGenre = predictedGenre;
    }

    public double getConfidence() {
        return confidence;
    }

    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }

    public Map<String, Double> getGenreProbabilities() {
        return genreProbabilities;
    }

    public void setGenreProbabilities(Map<String, Double> genreProbabilities) {
        this.genreProbabilities = genreProbabilities;
    }

    public Map<String, Object> getFeatures() {
        return features;
    }

    public void setFeatures(Map<String, Object> features) {
        this.features = features;
    }

    @Override
    public String getDescription() {
        return String.format("Genre Analysis Result: %s (Confidence: %.2f)", predictedGenre, confidence);
    }
}