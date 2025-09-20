package com.reremouse.lab.audio.core;

import com.reremouse.lab.audio.AudioData;
import com.reremouse.lab.audio.MusicTheory;
import com.reremouse.lab.audio.exception.AudioProcessingException;

/**
 * 音乐处理器接口 / Music Processor Interface
 * <p>
 * 扩展IAudioProcessor接口，专门用于音乐理论和音乐生成相关的处理功能，
 * 包括音阶生成、和弦生成、音乐合成、音乐变换等。
 * </p>
 * <p>
 * Extends IAudioProcessor interface specifically for music theory and music generation
 * related processing functions, including scale generation, chord generation, 
 * music synthesis, music transformation, etc.
 * </p>
 *
 * @author Qoder AI
 * @version 1.0
 * @since 1.0
 */
public interface IMusicProcessor extends IAudioProcessor {
    
    /**
     * 生成音阶音频 / Generate scale audio
     * <p>
     * 根据指定的音阶类型和参数生成音阶的音频数据。
     * Generate audio data for scale based on specified scale type and parameters.
     * </p>
     *
     * @param rootNote 根音 (0-11) / Root note (0-11)
     * @param scaleType 音阶类型 / Scale type
     * @param octave 八度 / Octave
     * @param noteDuration 每个音符的持续时间 (秒) / Duration of each note (seconds)
     * @return 音阶音频数据 / Scale audio data
     * @throws AudioProcessingException 当生成过程中发生错误时抛出 / Thrown when error occurs during generation
     */
    AudioData generateScale(int rootNote, MusicTheory.ScaleType scaleType, int octave, double noteDuration) throws AudioProcessingException;
    
    /**
     * 生成和弦音频 / Generate chord audio
     * <p>
     * 根据指定的和弦类型和参数生成和弦的音频数据。
     * Generate audio data for chord based on specified chord type and parameters.
     * </p>
     *
     * @param rootNote 根音 (0-11) / Root note (0-11)
     * @param chordType 和弦类型 / Chord type
     * @param octave 八度 / Octave
     * @param duration 持续时间 (秒) / Duration (seconds)
     * @return 和弦音频数据 / Chord audio data
     * @throws AudioProcessingException 当生成过程中发生错误时抛出 / Thrown when error occurs during generation
     */
    AudioData generateChord(int rootNote, MusicTheory.ChordType chordType, int octave, double duration) throws AudioProcessingException;
    
    /**
     * 生成音程音频 / Generate interval audio
     * <p>
     * 生成两个音符之间指定音程的音频。
     * Generate audio for specified interval between two notes.
     * </p>
     *
     * @param note1 第一个音符 (0-11) / First note (0-11)
     * @param note2 第二个音符 (0-11) / Second note (0-11)
     * @param octave 八度 / Octave
     * @param duration 持续时间 (秒) / Duration (seconds)
     * @param simultaneous 是否同时播放 / Whether to play simultaneously
     * @return 音程音频数据 / Interval audio data
     * @throws AudioProcessingException 当生成过程中发生错误时抛出 / Thrown when error occurs during generation
     */
    AudioData generateInterval(int note1, int note2, int octave, double duration, boolean simultaneous) throws AudioProcessingException;
    
    /**
     * 转调处理 / Transpose processing
     * <p>
     * 将输入音频转换到指定的调性。
     * Transpose input audio to specified key.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param semitones 转调的半音数 / Semitones to transpose
     * @return 转调后的音频数据 / Transposed audio data
     * @throws AudioProcessingException 当转调过程中发生错误时抛出 / Thrown when error occurs during transposition
     */
    AudioData transpose(AudioData audioData, int semitones) throws AudioProcessingException;
    
    /**
     * 调性转换 / Key conversion
     * <p>
     * 将音频从源调性转换到目标调性。
     * Convert audio from source key to target key.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param sourceKey 源调性 / Source key
     * @param targetKey 目标调性 / Target key
     * @return 转换后的音频数据 / Converted audio data
     * @throws AudioProcessingException 当转换过程中发生错误时抛出 / Thrown when error occurs during conversion
     */
    AudioData convertKey(AudioData audioData, MusicTheory.Key sourceKey, MusicTheory.Key targetKey) throws AudioProcessingException;
    
    /**
     * 和声化处理 / Harmonization processing
     * <p>
     * 为单声部旋律添加和声声部。
     * Add harmonic voices to monophonic melody.
     * </p>
     *
     * @param audioData 输入单声部音频 / Input monophonic audio
     * @param harmonyType 和声类型 / Harmony type
     * @return 和声化后的音频数据 / Harmonized audio data
     * @throws AudioProcessingException 当和声化过程中发生错误时抛出 / Thrown when error occurs during harmonization
     */
    AudioData harmonize(AudioData audioData, String harmonyType) throws AudioProcessingException;
    
    /**
     * 节拍量化 / Beat quantization
     * <p>
     * 将音频的节拍对齐到指定的网格。
     * Align audio beats to specified grid.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param beatGrid 节拍网格 (如1/4, 1/8, 1/16) / Beat grid (e.g., 1/4, 1/8, 1/16)
     * @return 量化后的音频数据 / Quantized audio data
     * @throws AudioProcessingException 当量化过程中发生错误时抛出 / Thrown when error occurs during quantization
     */
    AudioData quantizeBeats(AudioData audioData, String beatGrid) throws AudioProcessingException;
    
    /**
     * 音乐拼接 / Music concatenation
     * <p>
     * 将多个音乐片段按照音乐理论规则拼接在一起。
     * Concatenate multiple music segments according to music theory rules.
     * </p>
     *
     * @param audioSegments 音频片段数组 / Audio segment array
     * @param transitionType 过渡类型 / Transition type
     * @return 拼接后的音频数据 / Concatenated audio data
     * @throws AudioProcessingException 当拼接过程中发生错误时抛出 / Thrown when error occurs during concatenation
     */
    AudioData concatenateMusically(AudioData[] audioSegments, String transitionType) throws AudioProcessingException;
    
    /**
     * 音色变换 / Timbre transformation
     * <p>
     * 改变音频的音色特性，模拟不同乐器的音色。
     * Transform timbre characteristics of audio, simulate different instrument timbres.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param targetInstrument 目标乐器 / Target instrument
     * @return 音色变换后的音频数据 / Timbre transformed audio data
     * @throws AudioProcessingException 当变换过程中发生错误时抛出 / Thrown when error occurs during transformation
     */
    AudioData transformTimbre(AudioData audioData, String targetInstrument) throws AudioProcessingException;
    
    /**
     * 音乐风格转换 / Music style conversion
     * <p>
     * 将音频转换为指定的音乐风格。
     * Convert audio to specified music style.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param targetStyle 目标风格 / Target style
     * @return 风格转换后的音频数据 / Style converted audio data
     * @throws AudioProcessingException 当转换过程中发生错误时抛出 / Thrown when error occurs during conversion
     */
    AudioData convertMusicStyle(AudioData audioData, String targetStyle) throws AudioProcessingException;
    
    /**
     * 节拍匹配 / Tempo matching
     * <p>
     * 调整音频的节拍以匹配目标BPM，保持音高不变。
     * Adjust audio tempo to match target BPM while preserving pitch.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param targetBpm 目标BPM / Target BPM
     * @return 节拍匹配后的音频数据 / Tempo matched audio data
     * @throws AudioProcessingException 当匹配过程中发生错误时抛出 / Thrown when error occurs during matching
     */
    AudioData matchTempo(AudioData audioData, double targetBpm) throws AudioProcessingException;
    
    /**
     * 音高校正 / Pitch correction
     * <p>
     * 将音频中的音高校正到最近的音乐音符。
     * Correct pitch in audio to nearest musical notes.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param strength 校正强度 (0-1) / Correction strength (0-1)
     * @return 音高校正后的音频数据 / Pitch corrected audio data
     * @throws AudioProcessingException 当校正过程中发生错误时抛出 / Thrown when error occurs during correction
     */
    AudioData correctPitch(AudioData audioData, double strength) throws AudioProcessingException;
    
    /**
     * 设置音乐处理参数 / Set music processing parameters
     *
     * @param parameterName 参数名称 / Parameter name
     * @param value 参数值 / Parameter value
     * @throws IllegalArgumentException 当参数无效时抛出 / Thrown when parameter is invalid
     */
    void setMusicParameter(String parameterName, Object value) throws IllegalArgumentException;
    
    /**
     * 获取音乐处理参数 / Get music processing parameter
     *
     * @param parameterName 参数名称 / Parameter name
     * @return 参数值 / Parameter value
     * @throws IllegalArgumentException 当参数不存在时抛出 / Thrown when parameter doesn't exist
     */
    Object getMusicParameter(String parameterName) throws IllegalArgumentException;
    
    /**
     * 获取支持的乐器类型 / Get supported instrument types
     *
     * @return 乐器类型数组 / Instrument type array
     */
    String[] getSupportedInstruments();
    
    /**
     * 获取支持的音乐风格 / Get supported music styles
     *
     * @return 音乐风格数组 / Music style array
     */
    String[] getSupportedMusicStyles();
    
    /**
     * 获取支持的和声类型 / Get supported harmony types
     *
     * @return 和声类型数组 / Harmony type array
     */
    String[] getSupportedHarmonyTypes();
}