package com.yishape.lab.music.analysis.basic;

import com.yishape.lab.music.analysis.MusicDetectionResult;

/**
 * 和弦检测结果类 / Chord Detection Result Class
 * <p>
 * 包含和弦检测的结果信息，包括和弦片段信息。
 * Contains chord detection result information including chord segment information.
 * </p>
 */
public class ChordDetectionResult extends MusicDetectionResult {
    private double startTime;
    private double endTime;
    private String chordName;
    private double[] chromaFeatures;

    public ChordDetectionResult() {
        super();
        this.startTime = 0.0;
        this.endTime = 0.0;
        this.chordName = "";
    }

    public ChordDetectionResult(double startTime, double endTime, String chordName, double confidence) {
        super(confidence, "chord_detection");
        this.startTime = startTime;
        this.endTime = endTime;
        this.chordName = chordName != null ? chordName : "";
    }

    public ChordDetectionResult(double startTime, double endTime, String chordName, double confidence, String algorithm) {
        super(confidence, algorithm);
        this.startTime = startTime;
        this.endTime = endTime;
        this.chordName = chordName != null ? chordName : "";
    }

    public double getStartTime() {
        return startTime;
    }

    public void setStartTime(double startTime) {
        this.startTime = startTime;
    }

    public double getEndTime() {
        return endTime;
    }

    public void setEndTime(double endTime) {
        this.endTime = endTime;
    }

    public String getChordName() {
        return chordName;
    }

    public void setChordName(String chordName) {
        this.chordName = chordName != null ? chordName : "";
    }

    // Chord method as alias for ChordName method
    public void setChord(String chord) {
        this.chordName = chord != null ? chord : "";
    }

    public double[] getChromaFeatures() {
        return chromaFeatures;
    }

    public void setChromaFeatures(double[] chromaFeatures) {
        this.chromaFeatures = chromaFeatures;
    }

    @Override
    public String getDescription() {
        return String.format("Chord detection result: %s from %.2f to %.2f seconds", chordName, startTime, endTime);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("ChordDetectionResult{");
        sb.append("chord='").append(chordName).append("'");
        sb.append(", start=").append(String.format("%.2f", startTime));
        sb.append(", end=").append(String.format("%.2f", endTime));
        sb.append(", confidence=").append(String.format("%.2f", confidence));
        
        // Add chroma features information if available
        if (chromaFeatures != null && chromaFeatures.length == 12) {
            String[] noteNames = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
            sb.append(", chroma=[");
            for (int i = 0; i < 12; i++) {
                if (i > 0) sb.append(", ");
                sb.append(noteNames[i]).append(":").append(String.format("%.2f", chromaFeatures[i]));
            }
            sb.append("]");
        }
        
        sb.append('}');
        return sb.toString();
    }
}