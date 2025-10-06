package com.yishape.lab.music.analysis.basic;

import com.yishape.lab.music.analysis.MusicDetectionResult;

/**
 * 节拍检测结果类 / Beat Detection Result Class
 * <p>
 * 包含节拍检测的结果信息，包括节拍时间点和BPM。
 * Contains beat detection result information including beat time points and BPM.
 * </p>
 */
public class BeatDetectionResult extends MusicDetectionResult {
    private double[] beatTimes;
    private double bpm;

    public BeatDetectionResult() {
        super();
        this.beatTimes = new double[0];
        this.bpm = 0.0;
    }

    public BeatDetectionResult(double[] beatTimes, double bpm) {
        super();
        this.beatTimes = beatTimes != null ? beatTimes.clone() : new double[0];
        this.bpm = bpm;
    }

    public BeatDetectionResult(double[] beatTimes, double bpm, double confidence, String algorithm) {
        super(confidence, algorithm);
        this.beatTimes = beatTimes != null ? beatTimes.clone() : new double[0];
        this.bpm = bpm;
    }

    public double[] getBeatTimes() {
        return beatTimes != null ? beatTimes.clone() : new double[0];
    }

    public void setBeatTimes(double[] beatTimes) {
        this.beatTimes = beatTimes != null ? beatTimes.clone() : new double[0];
    }

    public double getBpm() {
        return bpm;
    }

    public void setBpm(double bpm) {
        this.bpm = bpm;
    }

    // Tempo methods as aliases for BPM methods
    public double getTempo() {
        return bpm;
    }

    public void setTempo(double tempo) {
        this.bpm = tempo;
    }

    @Override
    public String getDescription() {
        return String.format("Beat detection result with %d beats at %.2f BPM",
                            beatTimes != null ? beatTimes.length : 0, bpm);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("BeatDetectionResult{");
        sb.append("beats=").append(beatTimes != null ? beatTimes.length : 0);
        sb.append(", bpm=").append(String.format("%.2f", bpm));
        sb.append(", confidence=").append(String.format("%.2f", confidence));
        
        // Add more detailed information about beat times if available
        if (beatTimes != null && beatTimes.length > 0) {
            sb.append(", beatTimes=[");
            // Only show first few and last few beats to keep output reasonable
            int maxBeatsToShow = Math.min(5, beatTimes.length);
            for (int i = 0; i < maxBeatsToShow; i++) {
                if (i > 0) sb.append(", ");
                sb.append(String.format("%.2f", beatTimes[i]));
            }
            if (beatTimes.length > maxBeatsToShow) {
                sb.append(", ...");
                // Show last beat time
                sb.append(", ").append(String.format("%.2f", beatTimes[beatTimes.length - 1]));
            }
            sb.append("]");
        }
        
        sb.append('}');
        return sb.toString();
    }
}