package com.reremouse.lab.audio.effect;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.audio.processing.IBaseAudioProcessor;

import java.util.Map;

/**
 * 统一音频效果器接口 / Unified Audio Effect Interface
 * <p>
 * 定义音频效果处理操作的统一接口，支持不同类型的效果器。
 * 所有音频效果算法都应实现此接口以确保一致的行为和可扩展性。
 * </p>
 * <p>
 * Defines unified interface for audio effect processing operations, supporting different types of effects.
 * All audio effect algorithms should implement this interface to ensure consistent behavior and extensibility.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public interface IAudioEffect extends IBaseAudioProcessor {
    
    /**
     * 效果器类型枚举 / Effect Type Enum
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
     * 处理音频数据 / Process audio data
     * <p>
     * 对输入的音频数据执行特定的处理操作。
     * Perform specific processing operation on input audio data.
     * </p>
     *
     * @param input 输入音频数据 / Input audio data
     * @return 处理后的音频数据 / Processed audio data
     * @throws AudioProcessingException 当处理过程中发生错误时抛出 / Thrown when error occurs during processing
     */
    @Override
    AudioData process(AudioData input) throws AudioProcessingException;
    
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
     * 使用参数应用音频效果 / Apply Audio Effect with Parameters
     * <p>
     * 对输入音频应用效果，使用指定的参数。
     * Applies effect to input audio with specified parameters.
     * </p>
     * 
     * @param input 输入音频 / Input audio
     * @param parameters 效果参数 / Effect parameters
     * @return 应用效果后的音频 / Audio with applied effect
     * @throws AudioProcessingException 效果应用过程中发生错误 / Error occurred during effect application
     */
    AudioData applyEffect(AudioData input, Map<String, Object> parameters) throws AudioProcessingException;
    
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
    
    // getName() is already defined in IBaseAudioProcessor
    
    // getDescription() is already defined in IBaseAudioProcessor
    
    /**
     * 获取支持的参数 / Get Supported Parameters
     * 
     * @return 支持的参数名称列表 / List of supported parameter names
     */
    String[] getSupportedParameters();
    
    // getDefaultParameters() is already defined in IBaseAudioProcessor
    
    /**
     * 验证输入音频 / Validate Input Audio
     * <p>
     * 验证输入音频是否符合效果器的要求。
     * Validates if input audio meets effect requirements.
     * </p>
     * 
     * @param input 输入音频 / Input audio
     * @return 验证结果 / Validation result
     */
    boolean validateInput(AudioData input);
    
    // validateParameters() is already defined in IBaseAudioProcessor
    
    // clone() is already defined in IBaseAudioProcessor (returns IAudioProcessor)
    // We'll override it to return IAudioEffect
    IAudioEffect clone();
    
    /**
     * 获取效果器版本 / Get Effect Version
     * 
     * @return 版本字符串 / Version string
     */
    default String getVersion() {
        return "1.0";
    }
}