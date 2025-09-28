package com.reremouse.lab.music.generation;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;

import java.util.Map;

/**
 * 音乐生成器接口 / Music Generator Interface
 * <p>
 * 定义音乐生成的基本功能，包括音阶、和弦、音程等音乐元素的生成。
 * Defines basic music generation functionality, including generation of scales, chords, intervals and other musical elements.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IMusicGenerator {
    
    /**
     * 生成音阶 / Generate scale
     *
     * @param rootNote 根音 / Root note (0-11, C=0, C#=1, D=2, ...)
     * @param scaleType 音阶类型 / Scale type (major, minor, dorian, etc.)
     * @param octave 八度 / Octave
     * @param duration 持续时间（秒） / Duration in seconds
     * @param sampleRate 采样率 / Sample rate
     * @return 生成的音频数据 / Generated audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData generateScale(int rootNote, String scaleType, int octave, double duration, double sampleRate) throws AudioProcessingException;
    
    /**
     * 生成和弦 / Generate chord
     *
     * @param rootNote 根音 / Root note (0-11)
     * @param chordType 和弦类型 / Chord type (major, minor, dim, aug, etc.)
     * @param inversion 转位 / Inversion (0=root position, 1=first inversion, etc.)
     * @param octave 八度 / Octave
     * @param duration 持续时间（秒） / Duration in seconds
     * @param sampleRate 采样率 / Sample rate
     * @return 生成的音频数据 / Generated audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData generateChord(int rootNote, String chordType, int inversion, int octave, double duration, double sampleRate) throws AudioProcessingException;
    
    /**
     * 生成音程 / Generate interval
     *
     * @param rootNote 根音 / Root note (0-11)
     * @param intervalType 音程类型 / Interval type (unison, minor2nd, major2nd, etc.)
     * @param octave 八度 / Octave
     * @param duration 持续时间（秒） / Duration in seconds
     * @param sampleRate 采样率 / Sample rate
     * @return 生成的音频数据 / Generated audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData generateInterval(int rootNote, String intervalType, int octave, double duration, double sampleRate) throws AudioProcessingException;
    
    /**
     * 生成旋律 / Generate melody
     *
     * @param notes 音符序列 / Note sequence (array of note numbers)
     * @param durations 持续时间序列 / Duration sequence (array of durations in seconds)
     * @param octave 八度 / Octave
     * @param sampleRate 采样率 / Sample rate
     * @return 生成的音频数据 / Generated audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData generateMelody(int[] notes, double[] durations, int octave, double sampleRate) throws AudioProcessingException;
    
    /**
     * 生成和弦进行 / Generate chord progression
     *
     * @param chordProgression 和弦进行 / Chord progression (array of chord specifications)
     * @param key 调性 / Key (0-11, C=0, C#=1, D=2, ...)
     * @param octave 八度 / Octave
     * @param chordDuration 每个和弦的持续时间 / Duration of each chord
     * @param sampleRate 采样率 / Sample rate
     * @return 生成的音频数据 / Generated audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData generateChordProgression(String[] chordProgression, int key, int octave, double chordDuration, double sampleRate) throws AudioProcessingException;
    
    /**
     * 生成单个音符 / Generate single note
     *
     * @param note 音符 / Note (0-11)
     * @param octave 八度 / Octave
     * @param duration 持续时间（秒） / Duration in seconds
     * @param sampleRate 采样率 / Sample rate
     * @return 生成的音频数据 / Generated audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData generateNote(int note, int octave, double duration, double sampleRate) throws AudioProcessingException;
    
    /**
     * 生成带参数的音符 / Generate note with parameters
     *
     * @param note 音符 / Note (0-11)
     * @param octave 八度 / Octave
     * @param duration 持续时间（秒） / Duration in seconds
     * @param sampleRate 采样率 / Sample rate
     * @param parameters 生成参数 / Generation parameters (amplitude, waveform, etc.)
     * @return 生成的音频数据 / Generated audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData generateNote(int note, int octave, double duration, double sampleRate, Map<String, Object> parameters) throws AudioProcessingException;
    
    /**
     * 获取支持的音阶类型 / Get supported scale types
     *
     * @return 支持的音阶类型数组 / Array of supported scale types
     */
    String[] getSupportedScaleTypes();
    
    /**
     * 获取支持的和弦类型 / Get supported chord types
     *
     * @return 支持的和弦类型数组 / Array of supported chord types
     */
    String[] getSupportedChordTypes();
    
    /**
     * 获取支持的音程类型 / Get supported interval types
     *
     * @return 支持的音程类型数组 / Array of supported interval types
     */
    String[] getSupportedIntervalTypes();
    
    /**
     * 获取支持的波形类型 / Get supported waveform types
     *
     * @return 支持的波形类型数组 / Array of supported waveform types
     */
    String[] getSupportedWaveforms();
    
    /**
     * 获取生成器名称 / Get generator name
     *
     * @return 生成器名称 / Generator name
     */
    String getGeneratorName();
    
    /**
     * 获取支持的参数 / Get supported parameters
     *
     * @return 支持的参数名称数组 / Array of supported parameter names
     */
    String[] getSupportedParameters();
    
    /**
     * 获取默认参数 / Get default parameters
     *
     * @return 默认参数映射 / Default parameters map
     */
    Map<String, Object> getDefaultParameters();
    
    /**
     * 验证参数 / Validate parameters
     *
     * @param parameters 参数映射 / Parameters map
     * @return 参数是否有效 / Whether parameters are valid
     */
    boolean validateParameters(Map<String, Object> parameters);
    
    /**
     * 设置生成器参数 / Set generator parameters
     *
     * @param parameters 参数映射 / Parameters map
     * @throws AudioProcessingException 参数无效时抛出异常 / Thrown when parameters are invalid
     */
    void setParameters(Map<String, Object> parameters) throws AudioProcessingException;
    
    /**
     * 获取当前参数 / Get current parameters
     *
     * @return 当前参数映射 / Current parameters map
     */
    Map<String, Object> getCurrentParameters();
}