package com.yishape.lab.music.analysis.advanced;

import com.yishape.lab.music.analysis.MusicDetectionResult;
import java.util.Map;
import java.util.HashMap;

/**
 * 情感分析结果类 / Emotion Analysis Result Class
 * <p>
 * 封装音乐情感分析的结果，包括价值度、唤醒度、能量和可舞性等维度。
 * Encapsulates the results of music emotion analysis, including valence, arousal, energy, and danceability dimensions.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class EmotionAnalysisResult extends MusicDetectionResult {
    private double valence;           // 价值度 (正面/负面) / Valence (positive/negative)
    private double arousal;           // 唤醒度 (激动/平静) / Arousal (excited/calm)
    private double energy;            // 能量 / Energy
    private double danceability;      // 可舞性 / Danceability
    private String emotionLabel;      // 情感标签 / Emotion label
    private Map<String, Object> rawFeatures; // 原始特征 / Raw features

    public EmotionAnalysisResult() {
        super();
        this.valence = 0.0;
        this.arousal = 0.0;
        this.energy = 0.0;
        this.danceability = 0.0;
        this.emotionLabel = "neutral";
        this.rawFeatures = new HashMap<>();
    }

    // Getters and setters
    public double getValence() {
        return valence;
    }

    public void setValence(double valence) {
        this.valence = valence;
    }

    public double getArousal() {
        return arousal;
    }

    public void setArousal(double arousal) {
        this.arousal = arousal;
    }

    public double getEnergy() {
        return energy;
    }

    public void setEnergy(double energy) {
        this.energy = energy;
    }

    public double getDanceability() {
        return danceability;
    }

    public void setDanceability(double danceability) {
        this.danceability = danceability;
    }

    public String getEmotionLabel() {
        return emotionLabel;
    }

    public void setEmotionLabel(String emotionLabel) {
        this.emotionLabel = emotionLabel;
    }

    public Map<String, Object> getRawFeatures() {
        return rawFeatures;
    }

    public void setRawFeatures(Map<String, Object> rawFeatures) {
        this.rawFeatures = rawFeatures;
    }

    @Override
    public String getDescription() {
        return String.format("Emotion Analysis Result: %s (Valence: %.2f, Arousal: %.2f)",
                            emotionLabel, valence, arousal);
    }
}