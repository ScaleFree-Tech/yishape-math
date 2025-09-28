package com.reremouse.lab.music.processing;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.audio.processing.IAdvancedAudioProcessor;
import com.reremouse.lab.music.theory.ScaleTheory;
import com.reremouse.lab.music.theory.ChordTheory;

import java.util.Map;

/**
 * 音乐处理器接口 / Music Processor Interface
 * <p>
 * 继承高级音频处理功能，专门定义音乐处理的专业功能，包括转调、和声化、量化等操作
 * Extends advanced audio processing functionality and specifically defines professional music processing 
 * including transposition, harmonization, quantization, etc.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IMusicProcessor extends IAdvancedAudioProcessor {

    // ========== 音乐专业处理方法 / Music Professional Processing Methods ==========

    /**
     * 转调处理 / Transpose audio
     * 
     * @param audioData 输入音频数据 / Input audio data
     * @param semitones 半音变化量 / Semitone shift amount
     * @return 转调后的音频数据 / Transposed audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData transpose(AudioData audioData, int semitones) throws AudioProcessingException;

    /**
     * 和声化处理 / Harmonize audio
     * 
     * @param audioData 输入音频数据 / Input audio data
     * @param harmonyType 和声类型 / Harmony type
     * @param voiceCount 声部数量 / Number of voices
     * @return 和声化后的音频数据 / Harmonized audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData harmonize(AudioData audioData, String harmonyType, int voiceCount) throws AudioProcessingException;

    /**
     * 量化处理 / Quantize audio
     * 
     * @param audioData 输入音频数据 / Input audio data
     * @param gridSize 量化网格大小 / Quantization grid size
     * @return 量化后的音频数据 / Quantized audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData quantize(AudioData audioData, double gridSize) throws AudioProcessingException;

    /**
     * 生成音阶 / Generate scale
     * 
     * @param scale 音阶定义 / Scale definition
     * @param rootNote 根音 / Root note
     * @param octave 八度 / Octave
     * @param duration 持续时间 / Duration
     * @return 生成的音阶音频 / Generated scale audio
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData generateScale(ScaleTheory scale, int rootNote, int octave, double duration) throws AudioProcessingException;

    /**
     * 生成和弦 / Generate chord
     * 
     * @param chord 和弦定义 / Chord definition
     * @param rootNote 根音 / Root note
     * @param octave 八度 / Octave
     * @param duration 持续时间 / Duration
     * @return 生成的和弦音频 / Generated chord audio
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData generateChord(ChordTheory chord, int rootNote, int octave, double duration) throws AudioProcessingException;

    /**
     * 应用音乐理论变换 / Apply music theory transformation
     * 
     * @param audioData 输入音频数据 / Input audio data
     * @param transformation 变换类型 / Transformation type
     * @param parameters 变换参数 / Transformation parameters
     * @return 变换后的音频数据 / Transformed audio data
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    AudioData applyMusicTransformation(AudioData audioData, String transformation, Map<String, Object> parameters) throws AudioProcessingException;

    /**
     * 获取支持的音乐变换类型 / Get supported music transformation types
     * 
     * @return 支持的变换类型数组 / Array of supported transformation types
     */
    String[] getSupportedMusicTransformations();

    /**
     * 获取支持的和声类型 / Get supported harmony types
     * 
     * @return 支持的和声类型数组 / Array of supported harmony types
     */
    String[] getSupportedHarmonyTypes();

    /**
     * 获取当前处理质量级别 / Get current processing quality level
     * 
     * @return 当前质量级别 / Current quality level
     */
    String getCurrentQualityLevel();

    /**
     * 估计处理时间 / Estimate processing time
     * 
     * @param audioLength 音频长度（秒） / Audio length in seconds
     * @return 估计处理时间（秒） / Estimated processing time in seconds
     */
    double estimateProcessingTime(double audioLength);

    /**
     * 获取处理进度 / Get processing progress
     * 
     * @return 进度百分比（0-100） / Progress percentage (0-100)
     */
    double getProcessingProgress();

    /**
     * 取消当前处理 / Cancel current processing
     * 
     * @return 是否成功取消 / Whether successfully cancelled
     */
    boolean cancelProcessing();

    /**
     * 检查处理是否被取消 / Check if processing is cancelled
     * 
     * @return 是否被取消 / Whether cancelled
     */
    boolean isProcessingCancelled();

    /**
     * 暂停处理 / Pause processing
     * 
     * @return 是否成功暂停 / Whether successfully paused
     */
    boolean pauseProcessing();

    /**
     * 恢复处理 / Resume processing
     * 
     * @return 是否成功恢复 / Whether successfully resumed
     */
    boolean resumeProcessing();

    /**
     * 检查处理是否暂停 / Check if processing is paused
     * 
     * @return 是否暂停 / Whether paused
     */
    boolean isProcessingPaused();
}