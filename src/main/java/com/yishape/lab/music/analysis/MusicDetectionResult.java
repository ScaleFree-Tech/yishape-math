package com.yishape.lab.music.analysis;

/**
 * 统一的音乐检测结果基类 / Unified Base Class for Music Detection Results
 * <p>
 * 所有音乐检测结果类的基类，提供通用的属性和方法
 * Base class for all music detection result classes, providing common properties and methods.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public abstract class MusicDetectionResult {
    protected double confidence;
    protected long timestamp;
    protected String algorithm;

    public MusicDetectionResult() {
        this.confidence = 0.0;
        this.timestamp = System.currentTimeMillis();
        this.algorithm = "unknown";
    }

    public MusicDetectionResult(double confidence, String algorithm) {
        this.confidence = confidence;
        this.timestamp = System.currentTimeMillis();
        this.algorithm = algorithm != null ? algorithm : "unknown";
    }

    /**
     * 获取置信度 / Get confidence
     * @return 置信度值(0.0-1.0) / Confidence value (0.0-1.0)
     */
    public double getConfidence() {
        return confidence;
    }

    /**
     * 设置置信度 / Set confidence
     * @param confidence 置信度值(0.0-1.0) / Confidence value (0.0-1.0)
     */
    public void setConfidence(double confidence) {
        this.confidence = Math.max(0.0, Math.min(1.0, confidence));
    }

    /**
     * 获取时间戳 / Get timestamp
     * @return 时间戳 / Timestamp
     */
    public long getTimestamp() {
        return timestamp;
    }

    /**
     * 设置时间戳 / Set timestamp
     * @param timestamp 时间戳 / Timestamp
     */
    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    /**
     * 获取算法名称 / Get algorithm name
     * @return 算法名称 / Algorithm name
     */
    public String getAlgorithm() {
        return algorithm;
    }

    /**
     * 设置算法名称 / Set algorithm name
     * @param algorithm 算法名称 / Algorithm name
     */
    public void setAlgorithm(String algorithm) {
        this.algorithm = algorithm != null ? algorithm : "unknown";
    }

    /**
     * 获取结果的详细描述 / Get detailed description of the result
     * @return 结果描述 / Result description
     */
    public abstract String getDescription();

    @Override
    public String toString() {
        return String.format("MusicDetectionResult{confidence=%.2f, algorithm='%s', timestamp=%d}",
                            confidence, algorithm, timestamp);
    }
}