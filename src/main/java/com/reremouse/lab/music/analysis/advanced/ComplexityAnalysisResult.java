package com.reremouse.lab.music.analysis.advanced;

import com.reremouse.lab.music.analysis.MusicDetectionResult;
import java.util.Map;
import java.util.HashMap;

/**
 * 复杂度分析结果类 / Complexity Analysis Result Class
 * <p>
 * 封装音乐复杂度分析的结果，包括和声、节奏、旋律和频谱复杂度。
 * Encapsulates the results of music complexity analysis, including harmonic, rhythmic, melodic, and spectral complexity.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ComplexityAnalysisResult extends MusicDetectionResult {
    private double harmonicComplexity;      // 和声复杂度 / Harmonic complexity
    private double rhythmicComplexity;      // 节奏复杂度 / Rhythmic complexity
    private double melodicComplexity;       // 旋律复杂度 / Melodic complexity
    private double spectralComplexity;      // 频谱复杂度 / Spectral complexity
    private double overallComplexity;       // 总体复杂度 / Overall complexity
    private String complexityLevel;         // 复杂度等级 / Complexity level
    private Map<String, Object> detailedAnalysis; // 详细分析 / Detailed analysis

    public ComplexityAnalysisResult() {
        super();
        this.harmonicComplexity = 0.0;
        this.rhythmicComplexity = 0.0;
        this.melodicComplexity = 0.0;
        this.spectralComplexity = 0.0;
        this.overallComplexity = 0.0;
        this.complexityLevel = "medium";
        this.detailedAnalysis = new HashMap<>();
    }

    // Getters and setters
    public double getHarmonicComplexity() {
        return harmonicComplexity;
    }

    public void setHarmonicComplexity(double harmonicComplexity) {
        this.harmonicComplexity = harmonicComplexity;
    }

    public double getRhythmicComplexity() {
        return rhythmicComplexity;
    }

    public void setRhythmicComplexity(double rhythmicComplexity) {
        this.rhythmicComplexity = rhythmicComplexity;
    }

    public double getMelodicComplexity() {
        return melodicComplexity;
    }

    public void setMelodicComplexity(double melodicComplexity) {
        this.melodicComplexity = melodicComplexity;
    }

    public double getSpectralComplexity() {
        return spectralComplexity;
    }

    public void setSpectralComplexity(double spectralComplexity) {
        this.spectralComplexity = spectralComplexity;
    }

    public double getOverallComplexity() {
        return overallComplexity;
    }

    public void setOverallComplexity(double overallComplexity) {
        this.overallComplexity = overallComplexity;
    }

    public String getComplexityLevel() {
        return complexityLevel;
    }

    public void setComplexityLevel(String complexityLevel) {
        this.complexityLevel = complexityLevel;
    }

    public Map<String, Object> getDetailedAnalysis() {
        return detailedAnalysis;
    }

    public void setDetailedAnalysis(Map<String, Object> detailedAnalysis) {
        this.detailedAnalysis = detailedAnalysis;
    }

    @Override
    public String getDescription() {
        return String.format("Complexity Analysis Result: %s (Overall: %.2f)", complexityLevel, overallComplexity);
    }
}