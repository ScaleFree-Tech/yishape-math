package com.reremouse.lab.audio.filter;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;

/**
 * 高级音频滤波器接口 / Advanced Audio Filter Interface
 * <p>
 * 扩展基础音频滤波器接口，提供更高级的滤波功能。
 * Extends the base audio filter interface to provide more advanced filtering capabilities.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public interface IAdvancedAudioFilter extends IBaseAudioFilter {
    
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
    
    /**
     * 设置带宽 / Set bandwidth
     * <p>
     * 设置滤波器的带宽。
     * Set bandwidth of the filter.
     * </p>
     *
     * @param bandwidth 带宽 (Hz) / Bandwidth (Hz)
     * @throws IllegalArgumentException 当带宽值无效时抛出 / Thrown when bandwidth value is invalid
     */
    void setBandwidth(double bandwidth) throws IllegalArgumentException;
    
    /**
     * 获取带宽 / Get bandwidth
     * <p>
     * 获取当前设置的带宽。
     * Get currently set bandwidth.
     * </p>
     *
     * @return 带宽 (Hz) / Bandwidth (Hz)
     */
    double getBandwidth();
}