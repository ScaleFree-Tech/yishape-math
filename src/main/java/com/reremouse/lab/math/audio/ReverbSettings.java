package com.reremouse.lab.math.audio;

/**
 * 混响设置类 / Reverb Settings Class
 * <p>
 * 存储混响效果的配置参数。
 * Stores reverb effect configuration parameters.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ReverbSettings {
    
    /** 延迟时间 (ms) / Delay time (ms) */
    private final double delay;
    
    /** 反馈量 (0-1) / Feedback amount (0-1) */
    private final double feedback;
    
    /** 湿信号电平 (0-1) / Wet signal level (0-1) */
    private final double wetLevel;
    
    /** 干信号电平 (0-1) / Dry signal level (0-1) */
    private final double dryLevel;
    
    /** 房间大小 (0-1) / Room size (0-1) */
    private final double roomSize;
    
    /** 阻尼 (0-1) / Damping (0-1) */
    private final double damping;
    
    /**
     * 构造函数 / Constructor
     *
     * @param delay 延迟时间 / Delay time
     * @param feedback 反馈量 / Feedback amount
     * @param wetLevel 湿信号电平 / Wet signal level
     * @param dryLevel 干信号电平 / Dry signal level
     * @param roomSize 房间大小 / Room size
     * @param damping 阻尼 / Damping
     */
    public ReverbSettings(double delay, double feedback, double wetLevel, double dryLevel, 
                         double roomSize, double damping) {
        this.delay = delay;
        this.feedback = Math.max(0, Math.min(1, feedback));
        this.wetLevel = Math.max(0, Math.min(1, wetLevel));
        this.dryLevel = Math.max(0, Math.min(1, dryLevel));
        this.roomSize = Math.max(0, Math.min(1, roomSize));
        this.damping = Math.max(0, Math.min(1, damping));
    }
    
    /**
     * 默认构造函数 / Default constructor
     */
    public ReverbSettings() {
        this(50.0, 0.3, 0.3, 0.7, 0.5, 0.5);
    }
    
    /**
     * 获取延迟时间 / Get delay time
     *
     * @return 延迟时间 (ms) / Delay time (ms)
     */
    public double getDelay() {
        return delay;
    }
    
    /**
     * 获取反馈量 / Get feedback amount
     *
     * @return 反馈量 (0-1) / Feedback amount (0-1)
     */
    public double getFeedback() {
        return feedback;
    }
    
    /**
     * 获取湿信号电平 / Get wet signal level
     *
     * @return 湿信号电平 (0-1) / Wet signal level (0-1)
     */
    public double getWetLevel() {
        return wetLevel;
    }
    
    /**
     * 获取干信号电平 / Get dry signal level
     *
     * @return 干信号电平 (0-1) / Dry signal level (0-1)
     */
    public double getDryLevel() {
        return dryLevel;
    }
    
    /**
     * 获取房间大小 / Get room size
     *
     * @return 房间大小 (0-1) / Room size (0-1)
     */
    public double getRoomSize() {
        return roomSize;
    }
    
    /**
     * 获取阻尼 / Get damping
     *
     * @return 阻尼 (0-1) / Damping (0-1)
     */
    public double getDamping() {
        return damping;
    }
    
    /**
     * 创建预设混响 / Create preset reverb
     *
     * @param preset 预设类型 / Preset type
     * @return 混响设置 / Reverb settings
     */
    public static ReverbSettings createPreset(ReverbPreset preset) {
        switch (preset) {
            case ROOM:
                return new ReverbSettings(30.0, 0.2, 0.2, 0.8, 0.3, 0.7);
                
            case HALL:
                return new ReverbSettings(80.0, 0.4, 0.4, 0.6, 0.8, 0.3);
                
            case CHURCH:
                return new ReverbSettings(120.0, 0.5, 0.5, 0.5, 0.9, 0.2);
                
            case CAVE:
                return new ReverbSettings(100.0, 0.6, 0.6, 0.4, 0.7, 0.1);
                
            case PLATE:
                return new ReverbSettings(40.0, 0.3, 0.3, 0.7, 0.4, 0.8);
                
            case SPRING:
                return new ReverbSettings(20.0, 0.4, 0.4, 0.6, 0.2, 0.9);
                
            case AMBIENT:
                return new ReverbSettings(60.0, 0.2, 0.1, 0.9, 0.6, 0.6);
                
            default:
                return new ReverbSettings();
        }
    }
    
    /**
     * 检查设置是否有效 / Check if settings are valid
     *
     * @return 如果设置有效返回true / True if settings are valid
     */
    public boolean isValid() {
        return delay > 0 && feedback >= 0 && feedback <= 1 && 
               wetLevel >= 0 && wetLevel <= 1 && dryLevel >= 0 && dryLevel <= 1 &&
               roomSize >= 0 && roomSize <= 1 && damping >= 0 && damping <= 1;
    }
    
    @Override
    public String toString() {
        return String.format("ReverbSettings{delay=%.1fms, feedback=%.2f, wet=%.2f, dry=%.2f, " +
                           "roomSize=%.2f, damping=%.2f}",
                           delay, feedback, wetLevel, dryLevel, roomSize, damping);
    }
    
    /**
     * 混响预设枚举 / Reverb Preset Enum
     */
    public enum ReverbPreset {
        ROOM,     // 房间 / Room
        HALL,     // 大厅 / Hall
        CHURCH,   // 教堂 / Church
        CAVE,     // 洞穴 / Cave
        PLATE,    // 板式 / Plate
        SPRING,   // 弹簧 / Spring
        AMBIENT   // 环境 / Ambient
    }
}
