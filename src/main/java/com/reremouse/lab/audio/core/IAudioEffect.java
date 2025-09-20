package com.reremouse.lab.audio.core;

import com.reremouse.lab.audio.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;

/**
 * 音频效果器接口 / Audio Effect Interface
 * <p>
 * 定义音频效果器的基本操作，包括混响、延迟、失真、合唱等效果。
 * 所有音频效果器都应该实现此接口，确保一致的API设计。
 * </p>
 * <p>
 * Defines basic operations for audio effects, including reverb, delay, distortion, chorus, etc.
 * All audio effects should implement this interface to ensure consistent API design.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IAudioEffect extends IAudioProcessor {
    
    /**
     * 效果类型枚举 / Effect Type Enum
     */
    enum EffectType {
        REVERB("混响", "Reverb"),
        DELAY("延迟", "Delay"),
        ECHO("回声", "Echo"),
        CHORUS("合唱", "Chorus"),
        FLANGER("镶边", "Flanger"),
        PHASER("相位器", "Phaser"),
        DISTORTION("失真", "Distortion"),
        OVERDRIVE("过载", "Overdrive"),
        COMPRESSOR("压缩", "Compressor"),
        LIMITER("限制器", "Limiter"),
        TREMOLO("颤音", "Tremolo"),
        VIBRATO("颤动", "Vibrato"),
        AUTO_WAH("自动哇音", "Auto Wah"),
        PITCH_SHIFT("变调", "Pitch Shift"),
        TIME_STRETCH("时间拉伸", "Time Stretch"),
        HARMONIZER("和声器", "Harmonizer");
        
        private final String chineseName;
        private final String englishName;
        
        EffectType(String chineseName, String englishName) {
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
     * 应用音效 / Apply effect
     * <p>
     * 对音频数据应用效果处理。
     * Apply effect processing to audio data.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 添加效果后的音频数据 / Audio data with effect applied
     * @throws AudioProcessingException 当效果处理过程中发生错误时抛出 / Thrown when error occurs during effect processing
     */
    AudioData applyEffect(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 设置效果类型 / Set effect type
     * <p>
     * 设置要应用的效果类型。
     * Set type of effect to apply.
     * </p>
     *
     * @param effectType 效果类型 / Effect type
     */
    void setEffectType(EffectType effectType);
    
    /**
     * 获取效果类型 / Get effect type
     * <p>
     * 获取当前设置的效果类型。
     * Get currently set effect type.
     * </p>
     *
     * @return 效果类型 / Effect type
     */
    EffectType getEffectType();
    
    /**
     * 设置干湿比 / Set dry/wet mix
     * <p>
     * 设置原声信号和效果信号的混合比例。
     * Set mixing ratio between dry signal and wet signal.
     * </p>
     *
     * @param mix 干湿比 (0.0-1.0, 0.0为全干声，1.0为全湿声) / Dry/wet mix (0.0-1.0, 0.0 is all dry, 1.0 is all wet)
     * @throws IllegalArgumentException 当混合比例无效时抛出 / Thrown when mix ratio is invalid
     */
    void setDryWetMix(double mix) throws IllegalArgumentException;
    
    /**
     * 获取干湿比 / Get dry/wet mix
     * <p>
     * 获取当前设置的干湿比。
     * Get currently set dry/wet mix.
     * </p>
     *
     * @return 干湿比 / Dry/wet mix
     */
    double getDryWetMix();
    
    /**
     * 设置效果强度 / Set effect intensity
     * <p>
     * 设置效果的强度或深度。
     * Set intensity or depth of the effect.
     * </p>
     *
     * @param intensity 效果强度 (0.0-1.0) / Effect intensity (0.0-1.0)
     * @throws IllegalArgumentException 当强度值无效时抛出 / Thrown when intensity value is invalid
     */
    void setIntensity(double intensity) throws IllegalArgumentException;
    
    /**
     * 获取效果强度 / Get effect intensity
     * <p>
     * 获取当前设置的效果强度。
     * Get currently set effect intensity.
     * </p>
     *
     * @return 效果强度 / Effect intensity
     */
    double getIntensity();
    
    /**
     * 启用/禁用效果 / Enable/disable effect
     * <p>
     * 控制效果的启用状态，禁用时直接输出原始信号。
     * Control enable state of effect. When disabled, output original signal directly.
     * </p>
     *
     * @param enabled 是否启用效果 / Whether to enable effect
     */
    void setEnabled(boolean enabled);
    
    /**
     * 检查效果是否启用 / Check if effect is enabled
     * <p>
     * 检查效果当前是否处于启用状态。
     * Check if effect is currently enabled.
     * </p>
     *
     * @return 如果启用返回true / Return true if enabled
     */
    boolean isEnabled();
    
    /**
     * 获取效果预设 / Get effect presets
     * <p>
     * 返回可用的效果预设列表。
     * Return list of available effect presets.
     * </p>
     *
     * @return 预设名称数组 / Preset name array
     */
    String[] getPresets();
    
    /**
     * 加载预设 / Load preset
     * <p>
     * 加载指定的效果预设。
     * Load specified effect preset.
     * </p>
     *
     * @param presetName 预设名称 / Preset name
     * @throws IllegalArgumentException 当预设不存在时抛出 / Thrown when preset doesn't exist
     */
    void loadPreset(String presetName) throws IllegalArgumentException;
    
    /**
     * 保存预设 / Save preset
     * <p>
     * 保存当前设置为新的预设。
     * Save current settings as new preset.
     * </p>
     *
     * @param presetName 预设名称 / Preset name
     * @throws IllegalArgumentException 当预设名称无效时抛出 / Thrown when preset name is invalid
     */
    void savePreset(String presetName) throws IllegalArgumentException;
}