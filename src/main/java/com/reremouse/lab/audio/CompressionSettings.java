package com.reremouse.lab.audio;

/**
 * 压缩器设置类 / Compressor Settings Class
 * <p>
 * 存储动态范围压缩器的配置参数。
 * Stores dynamic range compressor configuration parameters.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class CompressionSettings {
    
    /** 压缩阈值 (dB) / Compression threshold (dB) */
    private final double threshold;
    
    /** 压缩比 / Compression ratio */
    private final double ratio;
    
    /** 启动时间 (ms) / Attack time (ms) */
    private final double attack;
    
    /** 释放时间 (ms) / Release time (ms) */
    private final double release;
    
    /** 输出增益 (dB) / Output gain (dB) */
    private final double outputGain;
    
    /** 软拐点 / Soft knee */
    private final boolean softKnee;
    
    /**
     * 构造函数 / Constructor
     *
     * @param threshold 压缩阈值 / Compression threshold
     * @param ratio 压缩比 / Compression ratio
     * @param attack 启动时间 / Attack time
     * @param release 释放时间 / Release time
     * @param outputGain 输出增益 / Output gain
     * @param softKnee 软拐点 / Soft knee
     */
    public CompressionSettings(double threshold, double ratio, double attack, double release, 
                             double outputGain, boolean softKnee) {
        this.threshold = threshold;
        this.ratio = ratio;
        this.attack = attack;
        this.release = release;
        this.outputGain = outputGain;
        this.softKnee = softKnee;
    }
    
    /**
     * 默认构造函数 / Default constructor
     */
    public CompressionSettings() {
        this(-12.0, 4.0, 10.0, 100.0, 0.0, true);
    }
    
    /**
     * 获取压缩阈值 / Get compression threshold
     *
     * @return 压缩阈值 (dB) / Compression threshold (dB)
     */
    public double getThreshold() {
        return threshold;
    }
    
    /**
     * 获取压缩比 / Get compression ratio
     *
     * @return 压缩比 / Compression ratio
     */
    public double getRatio() {
        return ratio;
    }
    
    /**
     * 获取启动时间 / Get attack time
     *
     * @return 启动时间 (ms) / Attack time (ms)
     */
    public double getAttack() {
        return attack;
    }
    
    /**
     * 获取释放时间 / Get release time
     *
     * @return 释放时间 (ms) / Release time (ms)
     */
    public double getRelease() {
        return release;
    }
    
    /**
     * 获取输出增益 / Get output gain
     *
     * @return 输出增益 (dB) / Output gain (dB)
     */
    public double getOutputGain() {
        return outputGain;
    }
    
    /**
     * 是否使用软拐点 / Whether to use soft knee
     *
     * @return 如果使用软拐点返回true / True if using soft knee
     */
    public boolean isSoftKnee() {
        return softKnee;
    }
    
    /**
     * 创建预设压缩器 / Create preset compressor
     *
     * @param preset 预设类型 / Preset type
     * @return 压缩器设置 / Compressor settings
     */
    public static CompressionSettings createPreset(CompressionPreset preset) {
        switch (preset) {
            case LIGHT:
                return new CompressionSettings(-18.0, 2.0, 20.0, 200.0, 2.0, true);
                
            case MEDIUM:
                return new CompressionSettings(-12.0, 4.0, 10.0, 100.0, 0.0, true);
                
            case HEAVY:
                return new CompressionSettings(-6.0, 8.0, 5.0, 50.0, -2.0, false);
                
            case VOCAL:
                return new CompressionSettings(-15.0, 3.0, 15.0, 150.0, 1.0, true);
                
            case DRUM:
                return new CompressionSettings(-8.0, 6.0, 3.0, 30.0, 0.0, false);
                
            case MASTER:
                return new CompressionSettings(-3.0, 2.0, 30.0, 300.0, 0.0, true);
                
            default:
                return new CompressionSettings();
        }
    }
    
    /**
     * 检查设置是否有效 / Check if settings are valid
     *
     * @return 如果设置有效返回true / True if settings are valid
     */
    public boolean isValid() {
        return threshold <= 0 && ratio >= 1.0 && attack > 0 && release > 0;
    }
    
    @Override
    public String toString() {
        return String.format("CompressionSettings{threshold=%.1fdB, ratio=%.1f:1, attack=%.1fms, " +
                           "release=%.1fms, outputGain=%.1fdB, softKnee=%s}",
                           threshold, ratio, attack, release, outputGain, softKnee);
    }
    
    /**
     * 压缩器预设枚举 / Compressor Preset Enum
     */
    public enum CompressionPreset {
        LIGHT,   // 轻度压缩 / Light compression
        MEDIUM,  // 中度压缩 / Medium compression
        HEAVY,   // 重度压缩 / Heavy compression
        VOCAL,   // 人声压缩 / Vocal compression
        DRUM,    // 鼓声压缩 / Drum compression
        MASTER   // 母带压缩 / Master compression
    }
}
