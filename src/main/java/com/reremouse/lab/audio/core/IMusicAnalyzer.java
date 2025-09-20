package com.reremouse.lab.audio.core;

import com.reremouse.lab.audio.AudioData;
import com.reremouse.lab.audio.MusicAnalyzer;
import com.reremouse.lab.audio.MusicTheory;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;

/**
 * 音乐分析器接口 / Music Analyzer Interface
 * <p>
 * 扩展IAudioAnalyzer接口，专门用于音乐分析功能，包括节拍检测、调性识别、
 * 和弦识别、音乐特征提取、风格分类等高级音乐分析功能。
 * </p>
 * <p>
 * Extends IAudioAnalyzer interface specifically for music analysis functionality,
 * including beat detection, key identification, chord recognition, music feature extraction,
 * genre classification and other advanced music analysis functions.
 * </p>
 *
 * @author Qoder AI
 * @version 1.0
 * @since 1.0
 */
public interface IMusicAnalyzer extends IAudioAnalyzer {
    
    /**
     * 检测音乐节拍 / Detect music beats
     * <p>
     * 分析音频中的节拍模式，返回BPM和节拍时间点。
     * Analyze beat patterns in audio, return BPM and beat time points.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 节拍检测结果 / Beat detection result
     * @throws AudioProcessingException 当检测过程中发生错误时抛出 / Thrown when error occurs during detection
     */
    MusicAnalyzer.BeatDetectionResult detectBeats(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 检测音乐节拍（指定BPM范围） / Detect music beats (with BPM range)
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param minBpm 最小BPM / Minimum BPM
     * @param maxBpm 最大BPM / Maximum BPM
     * @return 节拍检测结果 / Beat detection result
     * @throws AudioProcessingException 当检测过程中发生错误时抛出 / Thrown when error occurs during detection
     */
    MusicAnalyzer.BeatDetectionResult detectBeats(AudioData audioData, double minBpm, double maxBpm) throws AudioProcessingException;
    
    /**
     * 检测音乐调性 / Detect music key
     * <p>
     * 分析音频的调性特征，识别主调和调式。
     * Analyze key characteristics of audio, identify root key and mode.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 检测到的调性 / Detected key
     * @throws AudioProcessingException 当检测过程中发生错误时抛出 / Thrown when error occurs during detection
     */
    MusicTheory.Key detectKey(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 检测和弦 / Detect chord
     * <p>
     * 分析音频中的和弦结构，识别和弦类型。
     * Analyze chord structure in audio, identify chord type.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 检测到的和弦 / Detected chord
     * @throws AudioProcessingException 当检测过程中发生错误时抛出 / Thrown when error occurs during detection
     */
    MusicTheory.Chord detectChord(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 提取音乐特征 / Extract music features
     * <p>
     * 提取综合的音乐特征，包括节拍、调性、情感等多维度特征。
     * Extract comprehensive music features including tempo, key, emotion and other multi-dimensional features.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 音乐特征对象 / Music features object
     * @throws AudioProcessingException 当提取过程中发生错误时抛出 / Thrown when error occurs during extraction
     */
    MusicAnalyzer.MusicFeatures extractMusicFeatures(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 检测音乐风格 / Detect music genre
     * <p>
     * 基于音乐特征分析音乐风格类型。
     * Analyze music genre type based on music features.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 检测到的音乐风格 / Detected music genre
     * @throws AudioProcessingException 当检测过程中发生错误时抛出 / Thrown when error occurs during detection
     */
    String detectMusicGenre(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 分析音乐结构 / Analyze music structure
     * <p>
     * 分析音乐的结构组成，识别前奏、主歌、副歌等部分。
     * Analyze music structure composition, identify intro, verse, chorus and other sections.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 结构分析结果 / Structure analysis result
     * @throws AudioProcessingException 当分析过程中发生错误时抛出 / Thrown when error occurs during analysis
     */
    String analyzeMusicStructure(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 计算音乐相似度 / Calculate music similarity
     * <p>
     * 比较两个音频片段的音乐相似度。
     * Compare music similarity between two audio segments.
     * </p>
     *
     * @param audioData1 第一个音频数据 / First audio data
     * @param audioData2 第二个音频数据 / Second audio data
     * @return 相似度 (0-1) / Similarity (0-1)
     * @throws AudioProcessingException 当计算过程中发生错误时抛出 / Thrown when error occurs during calculation
     */
    double calculateMusicSimilarity(AudioData audioData1, AudioData audioData2) throws AudioProcessingException;
    
    /**
     * 检测音乐情感 / Detect music emotion
     * <p>
     * 分析音乐的情感特征，如快乐、悲伤、平静、激动等。
     * Analyze emotional characteristics of music such as happy, sad, calm, excited, etc.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 情感标签 / Emotion label
     * @throws AudioProcessingException 当检测过程中发生错误时抛出 / Thrown when error occurs during detection
     */
    String detectMusicEmotion(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 分析音乐复杂度 / Analyze music complexity
     * <p>
     * 计算音乐的复杂度指标，包括和声复杂度、节奏复杂度等。
     * Calculate music complexity metrics including harmonic complexity, rhythmic complexity, etc.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 复杂度评分 (0-1) / Complexity score (0-1)
     * @throws AudioProcessingException 当分析过程中发生错误时抛出 / Thrown when error occurs during analysis
     */
    double analyzeMusicComplexity(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 检测音乐乐器 / Detect musical instruments
     * <p>
     * 识别音频中存在的乐器类型。
     * Identify instrument types present in the audio.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 检测到的乐器列表 / List of detected instruments
     * @throws AudioProcessingException 当检测过程中发生错误时抛出 / Thrown when error occurs during detection
     */
    String[] detectInstruments(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 分析音乐动态变化 / Analyze music dynamics
     * <p>
     * 分析音乐的动态变化模式，如强弱变化、音量包络等。
     * Analyze dynamic change patterns in music such as loudness variations, volume envelope, etc.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @return 动态变化向量 / Dynamics variation vector
     * @throws AudioProcessingException 当分析过程中发生错误时抛出 / Thrown when error occurs during analysis
     */
    IVector<Double> analyzeMusicDynamics(AudioData audioData) throws AudioProcessingException;
    
    /**
     * 设置音乐分析参数 / Set music analysis parameters
     * <p>
     * 设置音乐分析的专用参数，如节拍检测灵敏度、和弦识别阈值等。
     * Set specialized parameters for music analysis such as beat detection sensitivity, chord recognition threshold, etc.
     * </p>
     *
     * @param parameterName 参数名称 / Parameter name
     * @param value 参数值 / Parameter value
     * @throws IllegalArgumentException 当参数无效时抛出 / Thrown when parameter is invalid
     */
    void setMusicParameter(String parameterName, Object value) throws IllegalArgumentException;
    
    /**
     * 获取音乐分析参数 / Get music analysis parameter
     *
     * @param parameterName 参数名称 / Parameter name
     * @return 参数值 / Parameter value
     * @throws IllegalArgumentException 当参数不存在时抛出 / Thrown when parameter doesn't exist
     */
    Object getMusicParameter(String parameterName) throws IllegalArgumentException;
    
    /**
     * 获取支持的音乐特征类型 / Get supported music feature types
     *
     * @return 音乐特征类型数组 / Music feature type array
     */
    String[] getSupportedMusicFeatureTypes();
    
    /**
     * 检查是否支持指定音乐特征 / Check if supports specified music feature
     *
     * @param featureType 特征类型 / Feature type
     * @return 如果支持返回true / Return true if supported
     */
    boolean supportsMusicFeatureType(String featureType);
}