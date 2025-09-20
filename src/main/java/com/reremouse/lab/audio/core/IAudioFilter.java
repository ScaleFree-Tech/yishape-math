package com.reremouse.lab.audio.core;

import com.reremouse.lab.audio.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;

/**
 * 音频滤波器接口 / Audio Filter Interface
 * <p>
 * 定义音频滤波器的基本操作，包括低通、高通、带通、带阻等滤波器。
 * 所有音频滤波器都应该实现此接口，确保一致的API设计。
 * </p>
 * <p>
 * Defines basic operations for audio filters, including low-pass, high-pass, band-pass, band-stop filters.
 * All audio filters should implement this interface to ensure consistent API design.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IAudioFilter extends IAudioProcessor {
    
    /**
     * 滤波器类型枚举 / Filter Type Enum
     */
    enum FilterType {
        LOW_PASS("低通", "Low Pass"),
        HIGH_PASS("高通", "High Pass"),
        BAND_PASS("带通", "Band Pass"),
        BAND_STOP("带阻", "Band Stop"),
        NOTCH("陷波", "Notch"),
        ALL_PASS("全通", "All Pass");
        
        private final String chineseName;
        private final String englishName;
        
        FilterType(String chineseName, String englishName) {
            this.chineseName = chineseName;
            this.englishName = englishName;
        }
        
        public String getChineseName() { return chineseName; }
        public String getEnglishName() { return englishName; }
        
        @Override
        public String toString() {
            return chineseName + " / " + englishName;
        }
    }
    
    /**
     * 应用滤波器 / Apply filter
     * <p>
     * 对音频数据应用滤波器处理。
     * Apply filter processing to audio data.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 滤波后的音频数据 / Filtered audio data
     * @throws AudioProcessingException 当滤波过程中发生错误时抛出 / Thrown when error occurs during filtering
     */
    AudioData filter(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 设置截止频率 / Set cutoff frequency
     * <p>
     * 设置滤波器的截止频率。
     * Set cutoff frequency of the filter.
     * </p>
     *
     * @param frequency 截止频率 (Hz) / Cutoff frequency (Hz)
     * @throws IllegalArgumentException 当频率值无效时抛出 / Thrown when frequency value is invalid
     */
    void setCutoffFrequency(double frequency) throws IllegalArgumentException;
    
    /**
     * 获取截止频率 / Get cutoff frequency
     * <p>
     * 获取当前设置的截止频率。
     * Get currently set cutoff frequency.
     * </p>
     *
     * @return 截止频率 (Hz) / Cutoff frequency (Hz)
     */
    double getCutoffFrequency();
    
    /**
     * 设置滤波器类型 / Set filter type
     * <p>
     * 设置滤波器的类型（低通、高通等）。
     * Set type of the filter (low-pass, high-pass, etc.).
     * </p>
     *
     * @param filterType 滤波器类型 / Filter type
     */
    void setFilterType(FilterType filterType);
    
    /**
     * 获取滤波器类型 / Get filter type
     * <p>
     * 获取当前设置的滤波器类型。
     * Get currently set filter type.
     * </p>
     *
     * @return 滤波器类型 / Filter type
     */
    FilterType getFilterType();
    
    /**
     * 设置滤波器阶数 / Set filter order
     * <p>
     * 设置滤波器的阶数，阶数越高，滤波效果越陡峭。
     * Set order of the filter. Higher order provides steeper filtering.
     * </p>
     *
     * @param order 滤波器阶数 / Filter order
     * @throws IllegalArgumentException 当阶数无效时抛出 / Thrown when order is invalid
     */
    void setOrder(int order) throws IllegalArgumentException;
    
    /**
     * 获取滤波器阶数 / Get filter order
     * <p>
     * 获取当前设置的滤波器阶数。
     * Get currently set filter order.
     * </p>
     *
     * @return 滤波器阶数 / Filter order
     */
    int getOrder();
    
    /**
     * 设置质量因子 / Set quality factor
     * <p>
     * 设置滤波器的质量因子（Q值），影响滤波器的频率响应。
     * Set quality factor (Q value) of the filter, affecting frequency response.
     * </p>
     *
     * @param q 质量因子 / Quality factor
     * @throws IllegalArgumentException 当Q值无效时抛出 / Thrown when Q value is invalid
     */
    void setQualityFactor(double q) throws IllegalArgumentException;
    
    /**
     * 获取质量因子 / Get quality factor
     * <p>
     * 获取当前设置的质量因子。
     * Get currently set quality factor.
     * </p>
     *
     * @return 质量因子 / Quality factor
     */
    double getQualityFactor();
    
    /**
     * 获取频率响应 / Get frequency response
     * <p>
     * 计算滤波器在指定频率的响应。
     * Calculate filter response at specified frequency.
     * </p>
     *
     * @param frequency 频率 (Hz) / Frequency (Hz)
     * @return 频率响应幅度 / Frequency response magnitude
     */
    double getFrequencyResponse(double frequency);
    
    /**
     * 获取群延迟 / Get group delay
     * <p>
     * 计算滤波器在指定频率的群延迟。
     * Calculate group delay of filter at specified frequency.
     * </p>
     *
     * @param frequency 频率 (Hz) / Frequency (Hz)
     * @return 群延迟 (秒) / Group delay (seconds)
     */
    double getGroupDelay(double frequency);
}