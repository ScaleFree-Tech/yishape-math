package com.reremouse.lab.math.audio;

/**
 * 音频质量等级枚举 / Audio Quality Level Enum
 * <p>
 * 定义音频质量的不同等级，用于评估音频数据的质量。
 * Defines different levels of audio quality for evaluating audio data quality.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public enum AudioQuality {
    
    /** 静音 / Silent */
    SILENT("静音", "Silent", "音频信号过小或完全静音 / Audio signal too small or completely silent"),
    
    /** 过载 / Clipped */
    CLIPPED("过载", "Clipped", "音频信号过载，存在削波失真 / Audio signal clipped, contains distortion"),
    
    /** 差 / Poor */
    POOR("差", "Poor", "音频质量较差，信噪比低 / Poor audio quality, low signal-to-noise ratio"),
    
    /** 一般 / Fair */
    FAIR("一般", "Fair", "音频质量一般，可接受 / Fair audio quality, acceptable"),
    
    /** 好 / Good */
    GOOD("好", "Good", "音频质量良好 / Good audio quality"),
    
    /** 优秀 / Excellent */
    EXCELLENT("优秀", "Excellent", "音频质量优秀，高信噪比 / Excellent audio quality, high signal-to-noise ratio");
    
    private final String chineseName;
    private final String englishName;
    private final String description;
    
    /**
     * 构造函数 / Constructor
     *
     * @param chineseName 中文名称 / Chinese name
     * @param englishName 英文名称 / English name
     * @param description 描述 / Description
     */
    AudioQuality(String chineseName, String englishName, String description) {
        this.chineseName = chineseName;
        this.englishName = englishName;
        this.description = description;
    }
    
    /**
     * 获取中文名称 / Get Chinese name
     *
     * @return 中文名称 / Chinese name
     */
    public String getChineseName() {
        return chineseName;
    }
    
    /**
     * 获取英文名称 / Get English name
     *
     * @return 英文名称 / English name
     */
    public String getEnglishName() {
        return englishName;
    }
    
    /**
     * 获取描述 / Get description
     *
     * @return 描述 / Description
     */
    public String getDescription() {
        return description;
    }
    
    /**
     * 获取质量分数 (0-100) / Get quality score (0-100)
     *
     * @return 质量分数 / Quality score
     */
    public int getScore() {
        switch (this) {
            case SILENT: return 0;
            case CLIPPED: return 10;
            case POOR: return 30;
            case FAIR: return 50;
            case GOOD: return 75;
            case EXCELLENT: return 100;
            default: return 0;
        }
    }
    
    /**
     * 检查是否为高质量 / Check if high quality
     *
     * @return 如果是高质量返回true / True if high quality
     */
    public boolean isHighQuality() {
        return this == GOOD || this == EXCELLENT;
    }
    
    /**
     * 检查是否为低质量 / Check if low quality
     *
     * @return 如果是低质量返回true / True if low quality
     */
    public boolean isLowQuality() {
        return this == SILENT || this == CLIPPED || this == POOR;
    }
    
    @Override
    public String toString() {
        return String.format("%s (%s) - %s", chineseName, englishName, description);
    }
}
