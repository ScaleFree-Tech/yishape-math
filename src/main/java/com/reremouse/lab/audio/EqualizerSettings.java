package com.reremouse.lab.audio;

import java.util.ArrayList;
import java.util.List;

/**
 * 均衡器设置类 / Equalizer Settings Class
 * <p>
 * 存储多频段均衡器的配置信息。
 * Stores multi-band equalizer configuration information.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class EqualizerSettings {
    
    /** 均衡器频段列表 / Equalizer bands list */
    private final List<EqualizerBand> bands;
    
    /** 全局增益 (dB) / Global gain (dB) */
    private final double globalGain;
    
    /**
     * 构造函数 / Constructor
     *
     * @param globalGain 全局增益 / Global gain
     */
    public EqualizerSettings(double globalGain) {
        this.bands = new ArrayList<>();
        this.globalGain = globalGain;
    }
    
    /**
     * 默认构造函数 / Default constructor
     */
    public EqualizerSettings() {
        this(0.0);
    }
    
    /**
     * 添加均衡器频段 / Add equalizer band
     *
     * @param band 均衡器频段 / Equalizer band
     */
    public void addBand(EqualizerBand band) {
        bands.add(band);
    }
    
    /**
     * 添加均衡器频段 / Add equalizer band
     *
     * @param frequency 中心频率 (Hz) / Center frequency (Hz)
     * @param gain 增益 (dB) / Gain (dB)
     * @param q Q值 / Q value
     */
    public void addBand(double frequency, double gain, double q) {
        addBand(new EqualizerBand(frequency, gain, q));
    }
    
    /**
     * 获取均衡器频段列表 / Get equalizer bands list
     *
     * @return 频段列表 / Bands list
     */
    public List<EqualizerBand> getBands() {
        return new ArrayList<>(bands);
    }
    
    /**
     * 获取全局增益 / Get global gain
     *
     * @return 全局增益 (dB) / Global gain (dB)
     */
    public double getGlobalGain() {
        return globalGain;
    }
    
    /**
     * 获取频段数量 / Get number of bands
     *
     * @return 频段数量 / Number of bands
     */
    public int getBandCount() {
        return bands.size();
    }
    
    /**
     * 创建预设均衡器 / Create preset equalizer
     *
     * @param preset 预设类型 / Preset type
     * @return 均衡器设置 / Equalizer settings
     */
    public static EqualizerSettings createPreset(EqualizerPreset preset) {
        EqualizerSettings settings = new EqualizerSettings();
        
        switch (preset) {
            case FLAT:
                // 平坦响应 / Flat response
                break;
                
            case ROCK:
                // 摇滚预设 / Rock preset
                settings.addBand(60, 4, 1.0);
                settings.addBand(170, -2, 1.0);
                settings.addBand(310, 3, 1.0);
                settings.addBand(600, 2, 1.0);
                settings.addBand(1000, 0, 1.0);
                settings.addBand(3000, 2, 1.0);
                settings.addBand(6000, 3, 1.0);
                settings.addBand(12000, 4, 1.0);
                settings.addBand(14000, 3, 1.0);
                settings.addBand(16000, 2, 1.0);
                break;
                
            case POP:
                // 流行预设 / Pop preset
                settings.addBand(60, 2, 1.0);
                settings.addBand(170, 1, 1.0);
                settings.addBand(310, 0, 1.0);
                settings.addBand(600, 1, 1.0);
                settings.addBand(1000, 2, 1.0);
                settings.addBand(3000, 3, 1.0);
                settings.addBand(6000, 2, 1.0);
                settings.addBand(12000, 1, 1.0);
                settings.addBand(14000, 0, 1.0);
                settings.addBand(16000, -1, 1.0);
                break;
                
            case JAZZ:
                // 爵士预设 / Jazz preset
                settings.addBand(60, 1, 1.0);
                settings.addBand(170, 0, 1.0);
                settings.addBand(310, 1, 1.0);
                settings.addBand(600, 2, 1.0);
                settings.addBand(1000, 1, 1.0);
                settings.addBand(3000, 0, 1.0);
                settings.addBand(6000, 1, 1.0);
                settings.addBand(12000, 2, 1.0);
                settings.addBand(14000, 1, 1.0);
                settings.addBand(16000, 0, 1.0);
                break;
                
            case CLASSICAL:
                // 古典预设 / Classical preset
                settings.addBand(60, 0, 1.0);
                settings.addBand(170, 0, 1.0);
                settings.addBand(310, 0, 1.0);
                settings.addBand(600, 0, 1.0);
                settings.addBand(1000, 0, 1.0);
                settings.addBand(3000, 0, 1.0);
                settings.addBand(6000, 1, 1.0);
                settings.addBand(12000, 2, 1.0);
                settings.addBand(14000, 2, 1.0);
                settings.addBand(16000, 1, 1.0);
                break;
                
            case VOCAL:
                // 人声预设 / Vocal preset
                settings.addBand(60, -2, 1.0);
                settings.addBand(170, -1, 1.0);
                settings.addBand(310, 0, 1.0);
                settings.addBand(600, 1, 1.0);
                settings.addBand(1000, 2, 1.0);
                settings.addBand(3000, 3, 1.0);
                settings.addBand(6000, 2, 1.0);
                settings.addBand(12000, 1, 1.0);
                settings.addBand(14000, 0, 1.0);
                settings.addBand(16000, -1, 1.0);
                break;
        }
        
        return settings;
    }
    
    @Override
    public String toString() {
        return String.format("EqualizerSettings{bands=%d, globalGain=%.1fdB}", 
                           bands.size(), globalGain);
    }
    
    /**
     * 均衡器预设枚举 / Equalizer Preset Enum
     */
    public enum EqualizerPreset {
        FLAT,       // 平坦 / Flat
        ROCK,       // 摇滚 / Rock
        POP,        // 流行 / Pop
        JAZZ,       // 爵士 / Jazz
        CLASSICAL,  // 古典 / Classical
        VOCAL       // 人声 / Vocal
    }
}
