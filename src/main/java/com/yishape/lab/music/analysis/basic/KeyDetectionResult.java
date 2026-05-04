package com.yishape.lab.music.analysis.basic;

import com.yishape.lab.music.analysis.MusicDetectionResult;

/**
 * 调性检测结果类 / Key Detection Result Class
 * <p>
 * 包含调性检测的结果信息，包括调性名称和置信度。
 * Contains key detection result information including key name and confidence.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class KeyDetectionResult extends MusicDetectionResult {
    private String keyName;
    private String scaleType;
    private double[] pitchClassProfile;

    public KeyDetectionResult() {
        super(0.5, "key_detection"); // Default confidence of 0.5 instead of 0.0
        this.keyName = "";
        this.scaleType = "";
        this.pitchClassProfile = new double[0];
    }

    public KeyDetectionResult(String keyName, String scaleType, double confidence, double[] pitchClassProfile) {
        super(confidence > 0 ? confidence : 0.1, "key_detection"); // Ensure minimum confidence of 0.1
        this.keyName = keyName != null ? keyName : "";
        this.scaleType = scaleType != null ? scaleType : "";
        this.pitchClassProfile = pitchClassProfile != null ? pitchClassProfile.clone() : new double[0];
    }

    public KeyDetectionResult(String keyName, String scaleType, double confidence, double[] pitchClassProfile, String algorithm) {
        super(confidence > 0 ? confidence : 0.1, algorithm); // Ensure minimum confidence of 0.1
        this.keyName = keyName != null ? keyName : "";
        this.scaleType = scaleType != null ? scaleType : "";
        this.pitchClassProfile = pitchClassProfile != null ? pitchClassProfile.clone() : new double[0];
    }

    public String getKeyName() {
        return keyName;
    }

    public void setKeyName(String keyName) {
        this.keyName = keyName != null ? keyName : "";
    }

    public String getScaleType() {
        return scaleType;
    }

    public void setScaleType(String scaleType) {
        this.scaleType = scaleType != null ? scaleType : "";
    }

    /**
     * 设置调式 / Set mode (alias for setScaleType)
     */
    public void setMode(String mode) {
        setScaleType(mode);
    }

    public double[] getPitchClassProfile() {
        return pitchClassProfile != null ? pitchClassProfile.clone() : new double[0];
    }

    public void setPitchClassProfile(double[] pitchClassProfile) {
        this.pitchClassProfile = pitchClassProfile != null ? pitchClassProfile.clone() : new double[0];
    }

    /**
     * 获取色度特征 / Get chroma features
     */
    public double[] getChromaFeatures() {
        return pitchClassProfile != null ? pitchClassProfile.clone() : new double[0];
    }

    /**
     * 设置色度特征 / Set chroma features
     */
    public void setChromaFeatures(double[] chromaFeatures) {
        this.pitchClassProfile = chromaFeatures != null ? chromaFeatures.clone() : new double[0];
        // When setting chroma features, also update confidence if it's zero
        if (this.confidence <= 0.0 && chromaFeatures != null && chromaFeatures.length > 0) {
            // Calculate a basic confidence based on chroma features
            double totalEnergy = 0.0;
            for (double value : chromaFeatures) {
                totalEnergy += Math.abs(value);
            }
            // Set confidence based on energy (but ensure minimum of 0.1)
            double newConfidence = Math.max(0.1, Math.min(1.0, totalEnergy / 10.0));
            this.confidence = newConfidence;
        }
    }

    @Override
    public String getDescription() {
        return String.format("Key detection result: %s %s (confidence: %.2f)", keyName, scaleType, confidence);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("KeyDetectionResult{");
        sb.append("key='").append(keyName).append("'");
        sb.append(", scale='").append(scaleType).append("'");
        sb.append(", confidence=").append(String.format("%.3f", confidence));
        
        // Add chroma note information if available
        if (pitchClassProfile != null && pitchClassProfile.length == 12) {
            String[] noteNames = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
            sb.append(", chroma=[");
            for (int i = 0; i < 12; i++) {
                if (i > 0) sb.append(", ");
                sb.append(noteNames[i]).append(":").append(String.format("%.2f", pitchClassProfile[i]));
            }
            sb.append("]");
        }
        
        sb.append('}');
        return sb.toString();
    }
}