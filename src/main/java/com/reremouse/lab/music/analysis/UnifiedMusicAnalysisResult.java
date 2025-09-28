package com.reremouse.lab.music.analysis;

import com.reremouse.lab.music.analysis.basic.BeatDetectionResult;
import com.reremouse.lab.music.analysis.basic.ChordDetectionResult;
import com.reremouse.lab.music.analysis.basic.KeyDetectionResult;
import java.util.Map;
import java.util.HashMap;

/**
 * 统一音乐分析结果类 / Unified Music Analysis Result Class
 * <p>
 * 为所有音乐分析器提供统一的结果封装，包含基础分析和高级分析结果。
 * Provides unified result encapsulation for all music analyzers, including basic and advanced analysis results.
 * </p>
 */
public class UnifiedMusicAnalysisResult extends MusicDetectionResult {
    // 基础分析结果 / Basic analysis results
    private BeatDetectionResult beatDetectionResult;
    private KeyDetectionResult keyDetectionResult;
    private ChordDetectionResult chordDetectionResult;

    // 高级分析结果 / Advanced analysis results
    private Map<String, Object> emotionAnalysis;
    private Map<String, Object> genreAnalysis;
    private Map<String, Object> complexityAnalysis;

    // 综合分析结果 / Comprehensive analysis results
    private Map<String, Object> structuralAnalysis;
    private Map<String, Object> tempoAnalysis;
    private Map<String, Object> harmonicAnalysis;

    public UnifiedMusicAnalysisResult() {
        super();
        this.emotionAnalysis = new HashMap<>();
        this.genreAnalysis = new HashMap<>();
        this.complexityAnalysis = new HashMap<>();
        this.structuralAnalysis = new HashMap<>();
        this.tempoAnalysis = new HashMap<>();
        this.harmonicAnalysis = new HashMap<>();
    }

    public UnifiedMusicAnalysisResult(double confidence, String algorithm) {
        super(confidence, algorithm);
        this.emotionAnalysis = new HashMap<>();
        this.genreAnalysis = new HashMap<>();
        this.complexityAnalysis = new HashMap<>();
        this.structuralAnalysis = new HashMap<>();
        this.tempoAnalysis = new HashMap<>();
        this.harmonicAnalysis = new HashMap<>();
    }

    // ========== 基础分析结果 Getters and Setters ==========

    public BeatDetectionResult getBeatDetectionResult() {
        return beatDetectionResult;
    }

    public void setBeatDetectionResult(BeatDetectionResult beatDetectionResult) {
        this.beatDetectionResult = beatDetectionResult;
    }

    public KeyDetectionResult getKeyDetectionResult() {
        return keyDetectionResult;
    }

    public void setKeyDetectionResult(KeyDetectionResult keyDetectionResult) {
        this.keyDetectionResult = keyDetectionResult;
    }

    public ChordDetectionResult getChordDetectionResult() {
        return chordDetectionResult;
    }

    public void setChordDetectionResult(ChordDetectionResult chordDetectionResult) {
        this.chordDetectionResult = chordDetectionResult;
    }

    // ========== 高级分析结果 Getters and Setters ==========

    public Map<String, Object> getEmotionAnalysis() {
        return emotionAnalysis;
    }

    public void setEmotionAnalysis(Map<String, Object> emotionAnalysis) {
        if (emotionAnalysis != null) {
            this.emotionAnalysis = emotionAnalysis;
        }
    }

    public Map<String, Object> getGenreAnalysis() {
        return genreAnalysis;
    }

    public void setGenreAnalysis(Map<String, Object> genreAnalysis) {
        if (genreAnalysis != null) {
            this.genreAnalysis = genreAnalysis;
        }
    }

    public Map<String, Object> getComplexityAnalysis() {
        return complexityAnalysis;
    }

    public void setComplexityAnalysis(Map<String, Object> complexityAnalysis) {
        if (complexityAnalysis != null) {
            this.complexityAnalysis = complexityAnalysis;
        }
    }

    // ========== 综合分析结果 Getters and Setters ==========

    public Map<String, Object> getStructuralAnalysis() {
        return structuralAnalysis;
    }

    public void setStructuralAnalysis(Map<String, Object> structuralAnalysis) {
        if (structuralAnalysis != null) {
            this.structuralAnalysis = structuralAnalysis;
        }
    }

    public Map<String, Object> getTempoAnalysis() {
        return tempoAnalysis;
    }

    public void setTempoAnalysis(Map<String, Object> tempoAnalysis) {
        if (tempoAnalysis != null) {
            this.tempoAnalysis = tempoAnalysis;
        }
    }

    public Map<String, Object> getHarmonicAnalysis() {
        return harmonicAnalysis;
    }

    public void setHarmonicAnalysis(Map<String, Object> harmonicAnalysis) {
        if (harmonicAnalysis != null) {
            this.harmonicAnalysis = harmonicAnalysis;
        }
    }

    @Override
    public String getDescription() {
        return "Unified music analysis result containing all analysis types";
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("UnifiedMusicAnalysisResult{");
        sb.append("confidence=").append(String.format("%.2f", confidence));
        sb.append(", algorithm='").append(algorithm).append('\'');
        sb.append(", timestamp=").append(timestamp);
        if (beatDetectionResult != null) {
            sb.append(", beatDetection=").append(beatDetectionResult.toString());
        }
        if (keyDetectionResult != null) {
            sb.append(", keyDetection=").append(keyDetectionResult.toString());
        }
        if (chordDetectionResult != null) {
            sb.append(", chordDetection=").append(chordDetectionResult.toString());
        }

        // Add more detailed information when available
        if (beatDetectionResult != null && beatDetectionResult.getBeatTimes() != null) {
            sb.append(", beatCount=").append(beatDetectionResult.getBeatTimes().length);
            if (beatDetectionResult.getBeatTimes().length > 0) {
                sb.append(", firstBeat=").append(String.format("%.2f", beatDetectionResult.getBeatTimes()[0]));
                sb.append(", lastBeat=").append(String.format("%.2f", beatDetectionResult.getBeatTimes()[beatDetectionResult.getBeatTimes().length - 1]));
            }
        }
        if (keyDetectionResult != null && keyDetectionResult.getKeyName() != null) {
            sb.append(", detectedKey='").append(keyDetectionResult.getKeyName()).append("'");
        }
        if (keyDetectionResult != null && keyDetectionResult.getScaleType() != null) {
            sb.append(", scale='").append(keyDetectionResult.getScaleType()).append("'");
        }
        if (chordDetectionResult != null && chordDetectionResult.getChordName() != null) {
            sb.append(", detectedChord='").append(chordDetectionResult.getChordName()).append("'");
        }
        if (chordDetectionResult != null) {
            sb.append(", chordDuration=").append(String.format("%.2f", 
                chordDetectionResult.getEndTime() - chordDetectionResult.getStartTime()));
        }

        // Add confidence details
        if (beatDetectionResult != null) {
            sb.append(", beatConfidence=").append(String.format("%.2f", beatDetectionResult.getConfidence()));
        }
        if (keyDetectionResult != null) {
            sb.append(", keyConfidence=").append(String.format("%.2f", keyDetectionResult.getConfidence()));
        }
        if (chordDetectionResult != null) {
            sb.append(", chordConfidence=").append(String.format("%.2f", chordDetectionResult.getConfidence()));
        }

        if (!emotionAnalysis.isEmpty()) {
            sb.append(", emotion=").append(formatMap(emotionAnalysis)).append("}");
        }
        if (!genreAnalysis.isEmpty()) {
            sb.append(", genre=").append(formatMap(genreAnalysis)).append("}");
        }
        if (!complexityAnalysis.isEmpty()) {
            sb.append(", complexity=").append(formatMap(complexityAnalysis)).append("}");
        }
        sb.append('}');
        return sb.toString();
    }

    /**
     * Format a map for display in toString()
     * @param map The map to format
     * @return Formatted string representation
     */
    private String formatMap(Map<String, Object> map) {
        if (map == null || map.isEmpty()) {
            return "{}";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("{");
        boolean first = true;
        for (Map.Entry<String, Object> entry : map.entrySet()) {
            if (!first) {
                sb.append(", ");
            }
            sb.append(entry.getKey()).append("=");
            Object value = entry.getValue();
            if (value instanceof Double) {
                sb.append(String.format("%.2f", value));
            } else if (value instanceof Float) {
                sb.append(String.format("%.2f", value));
            } else {
                sb.append(value);
            }
            first = false;
        }
        sb.append("}");
        return sb.toString();
    }
}