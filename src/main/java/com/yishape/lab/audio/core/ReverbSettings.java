package com.yishape.lab.audio.core;

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
    
    /** 早期反射延迟 (ms) / Early reflection delay (ms) */
    private final double earlyReflectionDelay;
    
    /** 早期反射电平 (0-1) / Early reflection level (0-1) */
    private final double earlyReflectionLevel;
    
    /** 扩散度 (0-1) / Diffusion (0-1) */
    private final double diffusion;
    
    /** 高频衰减 (0-1) / High frequency decay (0-1) */
    private final double highFreqDecay;
    
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
        this(delay, feedback, wetLevel, dryLevel, roomSize, damping, 
             delay * 0.3, 0.2, 0.5, 0.7);
    }
    
    /**
     * 完整构造函数 / Full constructor
     *
     * @param delay 延迟时间 / Delay time
     * @param feedback 反馈量 / Feedback amount
     * @param wetLevel 湿信号电平 / Wet signal level
     * @param dryLevel 干信号电平 / Dry signal level
     * @param roomSize 房间大小 / Room size
     * @param damping 阻尼 / Damping
     * @param earlyReflectionDelay 早期反射延迟 / Early reflection delay
     * @param earlyReflectionLevel 早期反射电平 / Early reflection level
     * @param diffusion 扩散度 / Diffusion
     * @param highFreqDecay 高频衰减 / High frequency decay
     */
    public ReverbSettings(double delay, double feedback, double wetLevel, double dryLevel,
                          double roomSize, double damping, double earlyReflectionDelay,
                          double earlyReflectionLevel, double diffusion, double highFreqDecay) {
        this.delay = Math.max(0, delay);
        this.feedback = Math.max(0, Math.min(1, feedback));
        this.wetLevel = Math.max(0, Math.min(1, wetLevel));
        this.dryLevel = Math.max(0, Math.min(1, dryLevel));
        this.roomSize = Math.max(0, Math.min(1, roomSize));
        this.damping = Math.max(0, Math.min(1, damping));
        this.earlyReflectionDelay = Math.max(0, earlyReflectionDelay);
        this.earlyReflectionLevel = Math.max(0, Math.min(1, earlyReflectionLevel));
        this.diffusion = Math.max(0, Math.min(1, diffusion));
        this.highFreqDecay = Math.max(0, Math.min(1, highFreqDecay));
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
     * 获取早期反射延迟 / Get early reflection delay
     *
     * @return 早期反射延迟 (ms) / Early reflection delay (ms)
     */
    public double getEarlyReflectionDelay() {
        return earlyReflectionDelay;
    }
    
    /**
     * 获取早期反射电平 / Get early reflection level
     *
     * @return 早期反射电平 (0-1) / Early reflection level (0-1)
     */
    public double getEarlyReflectionLevel() {
        return earlyReflectionLevel;
    }
    
    /**
     * 获取扩散度 / Get diffusion
     *
     * @return 扩散度 (0-1) / Diffusion (0-1)
     */
    public double getDiffusion() {
        return diffusion;
    }
    
    /**
     * 获取高频衰减 / Get high frequency decay
     *
     * @return 高频衰减 (0-1) / High frequency decay (0-1)
     */
    public double getHighFreqDecay() {
        return highFreqDecay;
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
                return new ReverbSettings(30.0, 0.2, 0.2, 0.8, 0.3, 0.7, 
                                        10.0, 0.15, 0.4, 0.8);
                
            case HALL:
                return new ReverbSettings(80.0, 0.4, 0.4, 0.6, 0.8, 0.3,
                                        25.0, 0.25, 0.7, 0.6);
                
            case CHURCH:
                return new ReverbSettings(120.0, 0.5, 0.5, 0.5, 0.9, 0.2,
                                        40.0, 0.3, 0.8, 0.4);
                
            case CAVE:
                return new ReverbSettings(100.0, 0.6, 0.6, 0.4, 0.7, 0.1,
                                        30.0, 0.4, 0.9, 0.3);
                
            case PLATE:
                return new ReverbSettings(40.0, 0.3, 0.3, 0.7, 0.4, 0.8,
                                        5.0, 0.1, 0.3, 0.9);
                
            case SPRING:
                return new ReverbSettings(20.0, 0.4, 0.4, 0.6, 0.2, 0.9,
                                        3.0, 0.05, 0.2, 0.95);
                
            case AMBIENT:
                return new ReverbSettings(60.0, 0.2, 0.1, 0.9, 0.6, 0.6,
                                        20.0, 0.08, 0.6, 0.7);
                
            case VOCAL:
                return new ReverbSettings(35.0, 0.25, 0.25, 0.75, 0.4, 0.6,
                                        8.0, 0.12, 0.5, 0.75);
                
            case DRUM:
                return new ReverbSettings(25.0, 0.15, 0.15, 0.85, 0.3, 0.8,
                                        5.0, 0.2, 0.3, 0.85);
                
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
        return delay >= 0 && feedback >= 0 && feedback <= 1 &&
               wetLevel >= 0 && wetLevel <= 1 && dryLevel >= 0 && dryLevel <= 1 &&
               roomSize >= 0 && roomSize <= 1 && damping >= 0 && damping <= 1 &&
               earlyReflectionDelay >= 0 && earlyReflectionLevel >= 0 && earlyReflectionLevel <= 1 &&
               diffusion >= 0 && diffusion <= 1 && highFreqDecay >= 0 && highFreqDecay <= 1;
    }
    
    /**
     * 计算混响时间 (RT60) / Calculate reverb time (RT60)
     *
     * @return 混响时间 (ms) / Reverb time (ms)
     */
    public double getReverbTime() {
        // 基于房间大小和阻尼计算混响时间
        // Calculate reverb time based on room size and damping
        double baseTime = roomSize * 2000; // 基础时间 / Base time
        double dampingFactor = 1.0 - damping; // 阻尼因子 / Damping factor
        return baseTime * dampingFactor;
    }
    
    /**
     * 创建自定义设置的构建器 / Create builder for custom settings
     *
     * @return 设置构建器 / Settings builder
     */
    public static Builder builder() {
        return new Builder();
    }
    
    @Override
    public String toString() {
        return String.format("ReverbSettings{delay=%.1fms, feedback=%.2f, wet=%.2f, dry=%.2f, " +
                           "roomSize=%.2f, damping=%.2f, rt60=%.1fms}",
                           delay, feedback, wetLevel, dryLevel, roomSize, damping, getReverbTime());
    }
    
    /**
     * 混响预设枚举 / Reverb Preset Enum
     */
    public enum ReverbPreset {
        ROOM("房间", "Room"),         // 房间 / Room
        HALL("大厅", "Hall"),         // 大厅 / Hall
        CHURCH("教堂", "Church"),     // 教堂 / Church
        CAVE("洞穴", "Cave"),         // 洞穴 / Cave
        PLATE("板式", "Plate"),       // 板式 / Plate
        SPRING("弹簧", "Spring"),     // 弹簧 / Spring
        AMBIENT("环境", "Ambient"),   // 环境 / Ambient
        VOCAL("人声", "Vocal"),       // 人声 / Vocal
        DRUM("鼓", "Drum");           // 鼓 / Drum
        
        private final String chineseName;
        private final String englishName;
        
        ReverbPreset(String chineseName, String englishName) {
            this.chineseName = chineseName;
            this.englishName = englishName;
        }
        
        public String getChineseName() {
            return chineseName;
        }
        
        public String getEnglishName() {
            return englishName;
        }
        
        @Override
        public String toString() {
            return String.format("%s (%s)", chineseName, englishName);
        }
    }
    
    /**
     * 混响设置构建器 / Reverb Settings Builder
     */
    public static class Builder {
        private double delay = 50.0;
        private double feedback = 0.3;
        private double wetLevel = 0.3;
        private double dryLevel = 0.7;
        private double roomSize = 0.5;
        private double damping = 0.5;
        private double earlyReflectionDelay = 15.0;
        private double earlyReflectionLevel = 0.2;
        private double diffusion = 0.5;
        private double highFreqDecay = 0.7;
        
        public Builder delay(double delay) {
            this.delay = delay;
            return this;
        }
        
        public Builder feedback(double feedback) {
            this.feedback = feedback;
            return this;
        }
        
        public Builder wetLevel(double wetLevel) {
            this.wetLevel = wetLevel;
            return this;
        }
        
        public Builder dryLevel(double dryLevel) {
            this.dryLevel = dryLevel;
            return this;
        }
        
        public Builder roomSize(double roomSize) {
            this.roomSize = roomSize;
            return this;
        }
        
        public Builder damping(double damping) {
            this.damping = damping;
            return this;
        }
        
        public Builder earlyReflectionDelay(double earlyReflectionDelay) {
            this.earlyReflectionDelay = earlyReflectionDelay;
            return this;
        }
        
        public Builder earlyReflectionLevel(double earlyReflectionLevel) {
            this.earlyReflectionLevel = earlyReflectionLevel;
            return this;
        }
        
        public Builder diffusion(double diffusion) {
            this.diffusion = diffusion;
            return this;
        }
        
        public Builder highFreqDecay(double highFreqDecay) {
            this.highFreqDecay = highFreqDecay;
            return this;
        }
        
        public ReverbSettings build() {
            return new ReverbSettings(delay, feedback, wetLevel, dryLevel, roomSize, damping,
                                    earlyReflectionDelay, earlyReflectionLevel, diffusion, highFreqDecay);
        }
    }
}