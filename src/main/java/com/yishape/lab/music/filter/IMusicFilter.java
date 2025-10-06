package com.yishape.lab.music.filter;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.audio.filter.IBaseAudioFilter;
import com.yishape.lab.music.theory.ChordTheory;
import com.yishape.lab.music.theory.KeyTheory;
import com.yishape.lab.music.theory.ScaleTheory;

/**
 * 音乐滤波器接口 / Music Filter Interface
 * <p>
 * 扩展IBaseAudioFilter接口，专门用于音乐相关的滤波处理。
 * 包括音乐频带滤波、和声滤波、音符滤波等。
 * </p>
 * <p>
 * Extends IBaseAudioFilter interface specifically for music-related filtering processing,
 * including musical frequency band filtering, harmonic filtering, note filtering, etc.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IMusicFilter extends IBaseAudioFilter {
    
    /**
     * 音符滤波 / Note filtering
     * <p>
     * 只保留指定音符的频率成分，过滤其他频率。
     * Only preserve frequency components of specified notes, filter out other frequencies.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param notes 要保留的音符数组 (0-11) / Notes to preserve (0-11)
     * @return 滤波后的音频数据 / Filtered audio data
     * @throws AudioProcessingException 当滤波过程中发生错误时抛出 / Thrown when error occurs during filtering
     */
    AudioData filterByNotes(AudioData audioData, int[] notes) throws AudioProcessingException;
    
    /**
     * 音阶滤波 / Scale filtering
     * <p>
     * 只保留指定音阶中的音符频率，过滤音阶外的音符。
     * Only preserve note frequencies within specified scale, filter out notes outside the scale.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param rootNote 根音 (0-11) / Root note (0-11)
     * @param scaleType 音阶类型 / Scale type
     * @return 滤波后的音频数据 / Filtered audio data
     * @throws AudioProcessingException 当滤波过程中发生错误时抛出 / Thrown when error occurs during filtering
     */
    AudioData filterByScale(AudioData audioData, int rootNote, ScaleTheory.ScaleType scaleType) throws AudioProcessingException;
    
    /**
     * 和声滤波 / Harmonic filtering
     * <p>
     * 增强或抑制指定和弦的和声成分。
     * Enhance or suppress harmonic components of specified chord.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param chord 和弦定义 / Chord definition
     * @param enhance 是否增强 (true=增强, false=抑制) / Whether to enhance (true=enhance, false=suppress)
     * @return 滤波后的音频数据 / Filtered audio data
     * @throws AudioProcessingException 当滤波过程中发生错误时抛出 / Thrown when error occurs during filtering
     */
    AudioData filterByChord(AudioData audioData, ChordTheory chord, boolean enhance) throws AudioProcessingException;
    
    /**
     * 八度滤波 / Octave filtering
     * <p>
     * 只保留指定八度范围内的频率成分。
     * Only preserve frequency components within specified octave range.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param minOctave 最小八度 / Minimum octave
     * @param maxOctave 最大八度 / Maximum octave
     * @return 滤波后的音频数据 / Filtered audio data
     * @throws AudioProcessingException 当滤波过程中发生错误时抛出 / Thrown when error occurs during filtering
     */
    AudioData filterByOctaveRange(AudioData audioData, int minOctave, int maxOctave) throws AudioProcessingException;
    
    /**
     * 乐器分离滤波 / Instrument separation filtering
     * <p>
     * 尝试分离出指定乐器的音频成分。
     * Attempt to separate audio components of specified instrument.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param instrumentType 乐器类型 / Instrument type
     * @return 分离后的音频数据 / Separated audio data
     * @throws AudioProcessingException 当分离过程中发生错误时抛出 / Thrown when error occurs during separation
     */
    AudioData separateInstrument(AudioData audioData, String instrumentType) throws AudioProcessingException;
    
    /**
     * 人声分离滤波 / Vocal separation filtering
     * <p>
     * 分离人声和伴奏，可选择保留人声或伴奏。
     * Separate vocals and accompaniment, option to keep vocals or accompaniment.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param keepVocals 是否保留人声 (true=保留人声, false=保留伴奏) / Whether to keep vocals (true=keep vocals, false=keep accompaniment)
     * @return 分离后的音频数据 / Separated audio data
     * @throws AudioProcessingException 当分离过程中发生错误时抛出 / Thrown when error occurs during separation
     */
    AudioData separateVocals(AudioData audioData, boolean keepVocals) throws AudioProcessingException;
    
    /**
     * 频率谱减法滤波 / Spectral subtraction filtering
     * <p>
     * 基于音乐理论的频谱减法，移除不协和的频率成分。
     * Music theory-based spectral subtraction, remove dissonant frequency components.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param referenceKey 参考调性 / Reference key
     * @param aggressiveness 激进程度 (0-1) / Aggressiveness (0-1)
     * @return 滤波后的音频数据 / Filtered audio data
     * @throws AudioProcessingException 当滤波过程中发生错误时抛出 / Thrown when error occurs during filtering
     */
    AudioData spectralSubtraction(AudioData audioData, KeyTheory referenceKey, double aggressiveness) throws AudioProcessingException;
    
    /**
     * 节拍同步滤波 / Beat-synchronized filtering
     * <p>
     * 根据检测到的节拍应用滤波处理。
     * Apply filtering based on detected beats.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param filterOnBeat 在节拍点应用滤波 / Apply filter on beat points
     * @return 滤波后的音频数据 / Filtered audio data
     * @throws AudioProcessingException 当滤波过程中发生错误时抛出 / Thrown when error occurs during filtering
     */
    AudioData beatSynchronizedFilter(AudioData audioData, boolean filterOnBeat) throws AudioProcessingException;
    
    /**
     * 动态频率滤波 / Dynamic frequency filtering
     * <p>
     * 根据音乐内容动态调整滤波参数。
     * Dynamically adjust filtering parameters based on music content.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param adaptiveness 自适应程度 (0-1) / Adaptiveness (0-1)
     * @return 滤波后的音频数据 / Filtered audio data
     * @throws AudioProcessingException 当滤波过程中发生错误时抛出 / Thrown when error occurs during filtering
     */
    AudioData dynamicFrequencyFilter(AudioData audioData, double adaptiveness) throws AudioProcessingException;
    
    /**
     * 设置音乐滤波参数 / Set music filtering parameters
     *
     * @param parameterName 参数名称 / Parameter name
     * @param value 参数值 / Parameter value
     * @throws IllegalArgumentException 当参数无效时抛出 / Thrown when parameter is invalid
     */
    void setMusicFilterParameter(String parameterName, Object value) throws IllegalArgumentException;
    
    /**
     * 获取音乐滤波参数 / Get music filtering parameter
     *
     * @param parameterName 参数名称 / Parameter name
     * @return 参数值 / Parameter value
     * @throws IllegalArgumentException 当参数不存在时抛出 / Thrown when parameter doesn't exist
     */
    Object getMusicFilterParameter(String parameterName) throws IllegalArgumentException;
    
    /**
     * 获取支持的乐器类型 / Get supported instrument types for separation
     *
     * @return 乐器类型数组 / Instrument type array
     */
    String[] getSupportedInstrumentTypes();
    
    /**
     * 获取支持的滤波模式 / Get supported filtering modes
     *
     * @return 滤波模式数组 / Filtering mode array
     */
    String[] getSupportedFilteringModes();
}