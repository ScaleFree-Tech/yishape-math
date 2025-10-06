package com.yishape.lab.audio.core;

/**
 * 均衡器频段类 / Equalizer Band Class
 * <p>
 * 表示均衡器的一个频段，包含中心频率、增益和Q值。
 * Represents one band of an equalizer, including center frequency, gain, and Q value.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class EqualizerBand {
    
    /** 中心频率 (Hz) / Center frequency (Hz) */
    private final double frequency;
    
    /** 增益 (dB) / Gain (dB) */
    private final double gain;
    
    /** Q值 / Q value */
    private final double q;
    
    /** 频段类型 / Band type */
    private final BandType type;
    
    /**
     * 构造函数 / Constructor
     *
     * @param frequency 中心频率 / Center frequency
     * @param gain 增益 / Gain
     * @param q Q值 / Q value
     */
    public EqualizerBand(double frequency, double gain, double q) {
        this(frequency, gain, q, BandType.PEAK);
    }
    
    /**
     * 构造函数 / Constructor
     *
     * @param frequency 中心频率 / Center frequency
     * @param gain 增益 / Gain
     * @param q Q值 / Q value
     * @param type 频段类型 / Band type
     */
    public EqualizerBand(double frequency, double gain, double q, BandType type) {
        this.frequency = frequency;
        this.gain = gain;
        this.q = q;
        this.type = type;
    }
    
    /**
     * 获取中心频率 / Get center frequency
     *
     * @return 中心频率 (Hz) / Center frequency (Hz)
     */
    public double getFrequency() {
        return frequency;
    }
    
    /**
     * 获取增益 / Get gain
     *
     * @return 增益 (dB) / Gain (dB)
     */
    public double getGain() {
        return gain;
    }
    
    /**
     * 获取Q值 / Get Q value
     *
     * @return Q值 / Q value
     */
    public double getQ() {
        return q;
    }
    
    /**
     * 获取频段类型 / Get band type
     *
     * @return 频段类型 / Band type
     */
    public BandType getType() {
        return type;
    }
    
    /**
     * 获取带宽 / Get bandwidth
     * <p>
     * 根据Q值计算频段的带宽。
     * Calculate band bandwidth based on Q value.
     * </p>
     *
     * @return 带宽 (Hz) / Bandwidth (Hz)
     */
    public double getBandwidth() {
        return frequency / q;
    }
    
    /**
     * 获取低截止频率 / Get low cutoff frequency
     *
     * @return 低截止频率 (Hz) / Low cutoff frequency (Hz)
     */
    public double getLowCutoff() {
        return frequency - getBandwidth() / 2;
    }
    
    /**
     * 获取高截止频率 / Get high cutoff frequency
     *
     * @return 高截止频率 (Hz) / High cutoff frequency (Hz)
     */
    public double getHighCutoff() {
        return frequency + getBandwidth() / 2;
    }
    
    /**
     * 检查频段是否有效 / Check if band is valid
     *
     * @return 如果频段有效返回true / True if band is valid
     */
    public boolean isValid() {
        return frequency > 0 && q > 0 && Math.abs(gain) <= 20; // 限制增益范围 / Limit gain range
    }
    
    @Override
    public String toString() {
        return String.format("EqualizerBand{freq=%.1fHz, gain=%.1fdB, q=%.2f, type=%s}", 
                           frequency, gain, q, type);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        EqualizerBand that = (EqualizerBand) obj;
        return Double.compare(that.frequency, frequency) == 0 &&
               Double.compare(that.gain, gain) == 0 &&
               Double.compare(that.q, q) == 0 &&
               type == that.type;
    }
    
    @Override
    public int hashCode() {
        int result = 17;
        long temp = Double.doubleToLongBits(frequency);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(gain);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(q);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        result = 31 * result + type.hashCode();
        return result;
    }
    
    /**
     * 频段类型枚举 / Band Type Enum
     */
    public enum BandType {
        LOW_SHELF,  // 低架 / Low shelf
        HIGH_SHELF, // 高架 / High shelf
        PEAK,       // 峰值 / Peak
        LOW_PASS,   // 低通 / Low pass
        HIGH_PASS,  // 高通 / High pass
        BAND_PASS,  // 带通 / Band pass
        BAND_STOP   // 带阻 / Band stop
    }
}