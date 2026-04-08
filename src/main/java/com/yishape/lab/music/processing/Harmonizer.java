package com.yishape.lab.music.processing;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.music.theory.ChordTheory;
import com.yishape.lab.music.theory.ScaleTheory;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.audio.processing.IAudioProcessor;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Set;
import java.util.HashSet;
import java.util.Locale;

/**
 * 音乐和声器 / Music Harmonizer
 * <p>
 * 实现音乐的和声化功能，可以为单声部音乐添加和声。
 * Implements music harmonization functionality to add harmony to monophonic music.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class Harmonizer implements IMusicProcessor {

    private static final Logger log = LoggerFactory.getLogger(Harmonizer.class);

    private static final String NAME = "Music Harmonizer";
    private static final String VERSION = "1.0.0";
    private static final String DESCRIPTION = "Adds harmony to monophonic music";

    // 默认参数 / Default parameters
    private static final Map<String, Object> DEFAULT_PARAMETERS = new HashMap<>();
    private static final Set<String> SUPPORTED_PARAMETERS = new HashSet<>();
    
    static {
        DEFAULT_PARAMETERS.put("harmonyType", "thirds");      // 和声类型 / Harmony type
        DEFAULT_PARAMETERS.put("voiceCount", 2);              // 声部数量 / Number of voices
        DEFAULT_PARAMETERS.put("key", "C");                   // 调性 / Key
        DEFAULT_PARAMETERS.put("mode", "major");              // 调式 / Mode
        DEFAULT_PARAMETERS.put("chordProgression", "I-V-vi-IV"); // 和弦进行 / Chord progression
        DEFAULT_PARAMETERS.put("voicing", "close");           // 声部排列 / Voicing
        DEFAULT_PARAMETERS.put("rhythmPattern", "follow");    // 节奏模式 / Rhythm pattern
        DEFAULT_PARAMETERS.put("harmonicRhythm", 1.0);       // 和声节奏 / Harmonic rhythm
        DEFAULT_PARAMETERS.put("dissonanceLevel", "low");     // 不协和度 / Dissonance level
        DEFAULT_PARAMETERS.put("qualityLevel", "high");       // 质量级别 / Quality level
        
        SUPPORTED_PARAMETERS.addAll(DEFAULT_PARAMETERS.keySet());
    }

    private Map<String, Object> currentParameters;
    private boolean verboseLogging = false;
    private double processingProgress = 0.0;
    private boolean processingCancelled = false;
    private boolean processingPaused = false;
    private Map<String, Object> lastStatistics = new HashMap<>();
    private Map<String, Object> performanceMetrics = new HashMap<>();
    private boolean isReady = true;

    /**
     * 构造函数 / Constructor
     */
    public Harmonizer() {
        this.currentParameters = new HashMap<>(DEFAULT_PARAMETERS);
        initializePerformanceMetrics();
    }

    // IAudioProcessor methods
    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public String getVersion() {
        return VERSION;
    }

    @Override
    public void setParameter(String key, Object value) throws IllegalArgumentException {
        if (key == null) {
            throw new IllegalArgumentException("Parameter key cannot be null");
        }
        currentParameters.put(key, value);
    }

    @Override
    public Object getParameter(String key) throws IllegalArgumentException {
        if (key == null) {
            throw new IllegalArgumentException("Parameter key cannot be null");
        }
        return currentParameters.get(key);
    }

    @Override
    public void reset() {
        resetParameters();
    }

    @Override
    public IAudioProcessor clone() {
        Harmonizer cloned = new Harmonizer();
        cloned.currentParameters = new HashMap<>(this.currentParameters);
        return cloned;
    }

    @Override
    public boolean supportsFormat(AudioData audioData) {
        if (audioData == null) {
            return false;
        }
        return supportsAudioFormat(audioData.getSampleRate(), audioData.getChannels(), audioData.getBitDepth());
    }

    @Override
    public int getLatency() {
        // Return estimated latency in samples
        return 0; // No additional latency in this implementation
    }

    // IBaseAudioProcessor methods
    @Override
    public String getDescription() {
        return DESCRIPTION;
    }

    @Override
    public String[] getSupportedParameters() {
        return SUPPORTED_PARAMETERS.toArray(new String[0]);
    }

    @Override
    public Map<String, Object> getDefaultParameters() {
        return Collections.unmodifiableMap(DEFAULT_PARAMETERS);
    }

    @Override
    public boolean isReady() {
        return isReady;
    }

    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        if (parameters == null) {
            return true; // null parameters means use defaults
        }

        try {
            // 验证和声类型
            String harmonyType = (String) parameters.get("harmonyType");
            if (harmonyType != null && !isValidHarmonyType(harmonyType)) {
                return false;
            }

            // 验证声部数量
            Integer voiceCount = (Integer) parameters.get("voiceCount");
            if (voiceCount != null && (voiceCount < 1 || voiceCount > 8)) {
                return false;
            }
            
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public void setParameters(Map<String, Object> parameters) throws AudioProcessingException {
        if (parameters != null) {
            if (!validateParameters(parameters)) {
                throw new AudioProcessingException("Invalid parameters provided");
            }
            currentParameters.putAll(parameters);
        }
    }

    @Override
    public Map<String, Object> getCurrentParameters() {
        return new HashMap<>(currentParameters);
    }

    @Override
    public void resetParameters() {
        currentParameters.clear();
        currentParameters.putAll(DEFAULT_PARAMETERS);
    }

    @Override
    public boolean supportsAudioFormat(double sampleRate, int channels, int bitDepth) {
        // Support common audio formats
        return sampleRate > 0 && channels > 0 && bitDepth > 0;
    }

    @Override
    public String getStatus() {
        if (processingCancelled) return "cancelled";
        if (processingPaused) return "paused";
        return "ready";
    }

    // IMusicProcessor methods
    @Override
    public AudioData process(AudioData audioData) throws AudioProcessingException {
        return process(audioData, currentParameters);
    }

    public AudioData process(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }

        if (!validateParameters(parameters)) {
            throw new AudioProcessingException("Invalid parameters provided");
        }

        try {
            processingProgress = 0.0;
            processingCancelled = false;

            String harmonyType = (String) parameters.get("harmonyType");
            int voiceCount = (Integer) parameters.get("voiceCount");
            String key = (String) parameters.get("key");
            String mode = (String) parameters.get("mode");

            if (verboseLogging) {
                log.debug("Harmonizing audio with {} voices in {} {}", voiceCount, key, mode);
            }

            AudioData result = performHarmonization(audioData, harmonyType, voiceCount, key, mode, parameters);

            processingProgress = 100.0;
            updateStatistics(audioData, result);

            return result;

        } catch (Exception e) {
            throw new AudioProcessingException("Failed to harmonize audio: " + e.getMessage(), e);
        }
    }

    @Override
    public AudioData processTimeRange(AudioData audioData, double startTime, double endTime) throws AudioProcessingException {
        return processTimeRange(audioData, startTime, endTime, currentParameters);
    }

    @Override
    public AudioData processTimeRange(AudioData audioData, double startTime, double endTime, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }

        if (startTime < 0 || endTime <= startTime || endTime > audioData.getDuration()) {
            throw new AudioProcessingException("Invalid time range");
        }

        // 提取指定时间段的音频
        AudioData segment = extractTimeSegment(audioData, startTime, endTime);

        // 处理音频段
        AudioData processedSegment = process(segment, parameters);

        // 将处理后的段重新插入原音频
        return insertProcessedSegment(audioData, processedSegment, startTime, endTime);
    }

    @Override
    public AudioData processStream(AudioData audioData, double windowSize, double hopSize) throws AudioProcessingException {
        return processStream(audioData, windowSize, hopSize, currentParameters);
    }

    @Override
    public AudioData processStream(AudioData audioData, double windowSize, double hopSize, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }

        if (!validateParameters(parameters)) {
            throw new AudioProcessingException("Invalid parameters provided");
        }

        try {
            processingProgress = 0.0;
            processingCancelled = false;

            double sampleRate = audioData.getSampleRate();
            int windowSamples = (int) (windowSize * sampleRate);
            int hopSamples = (int) (hopSize * sampleRate);

            return performStreamHarmonization(audioData, windowSamples, hopSamples, parameters);

        } catch (Exception e) {
            throw new AudioProcessingException("Failed to process audio stream: " + e.getMessage(), e);
        }
    }

    @Override
    public AudioData[] processBatch(AudioData[] audioDataArray) throws AudioProcessingException {
        return processBatch(audioDataArray, currentParameters);
    }

    @Override
    public AudioData[] processBatch(AudioData[] audioDataArray, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioDataArray == null || audioDataArray.length == 0) {
            throw new AudioProcessingException("Audio data array cannot be null or empty");
        }

        AudioData[] results = new AudioData[audioDataArray.length];

        for (int i = 0; i < audioDataArray.length; i++) {
            if (processingCancelled) {
                break;
            }

            results[i] = process(audioDataArray[i], parameters);
            processingProgress = ((double) (i + 1) / audioDataArray.length) * 100.0;
        }

        return results;
    }

    /**
     * 执行和声化操作 / Perform harmonization
     */
    private AudioData performHarmonization(AudioData audioData, String harmonyType, int voiceCount,
                                          String key, String mode, Map<String, Object> parameters) throws AudioProcessingException {

        // 分析原始音频的音高
        double[] pitches = extractPitches(audioData);

        // 生成和声声部
        List<double[]> harmonyVoices = generateHarmonyVoices(pitches, harmonyType, voiceCount, key, mode, parameters);

        // 合成和声音频
        return synthesizeHarmony(audioData, harmonyVoices, parameters);
    }

    /**
     * 提取音高信息 / Extract pitch information
     */
    private double[] extractPitches(AudioData audioData) throws AudioProcessingException {
        double[] samples = audioData.getSamples().toDoubleArray();
        double sampleRate = audioData.getSampleRate();

        int windowSize = 2048;
        int hopSize = 512;
        int numFrames = (samples.length - windowSize) / hopSize + 1;

        double[] pitches = new double[numFrames];

        for (int frame = 0; frame < numFrames; frame++) {
            if (processingCancelled) break;

            int startSample = frame * hopSize;

            // 使用自相关法检测音高
            double pitch = detectPitchAutocorrelation(samples, startSample, windowSize, sampleRate);
            pitches[frame] = pitch;

            processingProgress = ((double) frame / numFrames) * 30.0; // 30%用于音高检测
        }

        return pitches;
    }

    /**
     * 自相关音高检测 / Autocorrelation pitch detection
     */
    private double detectPitchAutocorrelation(double[] samples, int start, int windowSize, double sampleRate) {
        double[] window = new double[windowSize];
        System.arraycopy(samples, start, window, 0, Math.min(windowSize, samples.length - start));

        // 计算自相关
        double[] autocorr = new double[windowSize / 2];
        for (int lag = 1; lag < autocorr.length; lag++) {
            double sum = 0.0;
            for (int i = 0; i < windowSize - lag; i++) {
                sum += window[i] * window[i + lag];
            }
            autocorr[lag] = sum;
        }

        // 找到最大峰值
        int maxLag = 1;
        double maxValue = autocorr[1];
        for (int lag = 2; lag < autocorr.length; lag++) {
            if (autocorr[lag] > maxValue) {
                maxValue = autocorr[lag];
                maxLag = lag;
            }
        }

        // 转换为频率
        return sampleRate / maxLag;
    }

    /**
     * 生成和声声部 / Generate harmony voices
     */
    private List<double[]> generateHarmonyVoices(double[] pitches, String harmonyType, int voiceCount,
                                                String key, String mode, Map<String, Object> parameters) throws AudioProcessingException {

        List<double[]> voices = new ArrayList<>();
        String voicing = (String) parameters.get("voicing");
        String chordProgression = (String) parameters.get("chordProgression");

        for (int voice = 0; voice < voiceCount; voice++) {
            double[] harmonyPitches = new double[pitches.length];

            for (int frame = 0; frame < pitches.length; frame++) {
                if (processingCancelled) break;

                double originalPitch = pitches[frame];
                if (originalPitch > 0) { // 有效音高
                    double harmonyPitch = calculateHarmonyPitch(originalPitch, voice, harmonyType, key, mode, voicing);
                    harmonyPitches[frame] = harmonyPitch;
                } else {
                    harmonyPitches[frame] = 0; // 静音
                }
            }

            voices.add(harmonyPitches);
            processingProgress = 30.0 + ((double) (voice + 1) / voiceCount) * 40.0; // 30-70%用于生成和声
        }

        return voices;
    }

    /**
     * 计算和声音高 / Calculate harmony pitch
     */
    private double calculateHarmonyPitch(double originalPitch, int voiceIndex, String harmonyType,
                                        String key, String mode, String voicing) {

        // 将频率转换为MIDI音符
        double midiNote = 69 + 12 * Math.log(originalPitch / 440.0) / Math.log(2);

        // 根据和声类型计算间隔
        int interval = 0;
        switch (harmonyType.toLowerCase()) {
            case "thirds":
                interval = (voiceIndex + 1) * 3; // 三度叠置
                break;
            case "fourths":
                interval = (voiceIndex + 1) * 4; // 四度叠置
                break;
            case "fifths":
                interval = (voiceIndex + 1) * 7; // 五度叠置
                break;
            case "octaves":
                interval = (voiceIndex + 1) * 12; // 八度叠置
                break;
            case "custom":
                interval = calculateCustomInterval(midiNote, voiceIndex, key, mode);
                break;
            default:
                interval = (voiceIndex + 1) * 3; // 默认三度
        }

        // 根据声部排列调整
        if ("open".equals(voicing) && voiceIndex > 0) {
            interval += 12; // 开放排列，增加八度
        }

        // 计算和声音高
        double harmonyMidi = midiNote + interval;
        return 440.0 * Math.pow(2, (harmonyMidi - 69) / 12.0);
    }

    /**
     * 计算自定义间隔 / Calculate custom interval
     */
    private int calculateCustomInterval(double midiNote, int voiceIndex, String key, String mode) {
        // 根据调性和调式计算合适的间隔
        // 这里简化处理，实际应该根据和弦理论

        int[] majorIntervals = {3, 7, 10}; // 大三度、纯五度、大七度
        int[] minorIntervals = {3, 7, 10}; // 小三度、纯五度、小七度

        int[] intervals = "major".equals(mode) ? majorIntervals : minorIntervals;

        if (voiceIndex < intervals.length) {
            return intervals[voiceIndex];
        } else {
            return intervals[voiceIndex % intervals.length] + 12 * (voiceIndex / intervals.length);
        }
    }

    /**
     * 合成和声音频 / Synthesize harmony audio
     */
    private AudioData synthesizeHarmony(AudioData originalAudio, List<double[]> harmonyVoices,
                                       Map<String, Object> parameters) throws AudioProcessingException {

        double[] originalSamples = originalAudio.getSamples().toDoubleArray();
        double sampleRate = originalAudio.getSampleRate();

        // 创建输出样本数组
        double[] outputSamples = new double[originalSamples.length];

        // 添加原始音频
        System.arraycopy(originalSamples, 0, outputSamples, 0, originalSamples.length);

        // 为每个和声声部生成音频并混合
        for (int voiceIndex = 0; voiceIndex < harmonyVoices.size(); voiceIndex++) {
            if (processingCancelled) break;

            double[] voicePitches = harmonyVoices.get(voiceIndex);
            double[] voiceSamples = synthesizeVoice(voicePitches, originalSamples.length, sampleRate);

            // 混合到输出中
            double voiceGain = 0.3 / (voiceIndex + 1); // 递减音量
            for (int i = 0; i < outputSamples.length; i++) {
                outputSamples[i] += voiceSamples[i] * voiceGain;
            }

            processingProgress = 70.0 + ((double) (voiceIndex + 1) / harmonyVoices.size()) * 30.0; // 70-100%用于合成
        }

        // 归一化输出
        normalizeAudio(outputSamples);

        return new AudioData(Linalg.vector(outputSamples), sampleRate, originalAudio.getChannels(),
                            outputSamples.length, originalAudio.getFormat());
    }

    /**
     * 合成单个声部 / Synthesize single voice
     */
    private double[] synthesizeVoice(double[] pitches, int outputLength, double sampleRate) {
        double[] samples = new double[outputLength];

        int hopSize = 512;
        double phase = 0.0;

        for (int frame = 0; frame < pitches.length; frame++) {
            double pitch = pitches[frame];
            int startSample = frame * hopSize;
            int endSample = Math.min(startSample + hopSize, outputLength);

            if (pitch > 0) { // 有效音高
                double frequency = pitch;
                double phaseIncrement = 2 * Math.PI * frequency / sampleRate;

                for (int i = startSample; i < endSample; i++) {
                    // 使用正弦波合成
                    samples[i] = Math.sin(phase);
                    phase += phaseIncrement;

                    // 防止相位溢出
                    if (phase > 2 * Math.PI) {
                        phase -= 2 * Math.PI;
                    }
                }
            } else {
                // 静音段
                for (int i = startSample; i < endSample; i++) {
                    samples[i] = 0.0;
                }
            }
        }

        return samples;
    }

    /**
     * 归一化音频 / Normalize audio
     */
    private void normalizeAudio(double[] samples) {
        double maxValue = 0.0;
        for (double sample : samples) {
            maxValue = Math.max(maxValue, Math.abs(sample));
        }

        if (maxValue > 0) {
            double scale = 0.95 / maxValue; // 留一点余量
            for (int i = 0; i < samples.length; i++) {
                samples[i] *= scale;
            }
        }
    }

    /**
     * 流式和声化处理 / Stream harmonization processing
     */
    private AudioData performStreamHarmonization(AudioData audioData, int windowSamples, int hopSamples,
                                                 Map<String, Object> parameters) throws AudioProcessingException {

        double[] inputSamples = audioData.getSamples().toDoubleArray();
        double[] outputSamples = new double[inputSamples.length];

        // 分块处理
        for (int pos = 0; pos < inputSamples.length; pos += hopSamples) {
            if (processingCancelled) break;

            int blockSize = Math.min(windowSamples, inputSamples.length - pos);

            // 提取当前块
            double[] blockSamples = new double[blockSize];
            System.arraycopy(inputSamples, pos, blockSamples, 0, blockSize);

            // 创建临时音频数据
            AudioData blockAudio = new AudioData(Linalg.vector(blockSamples), audioData.getSampleRate(),
                                                audioData.getChannels(), blockSamples.length, audioData.getFormat());

            // 处理当前块
            AudioData processedBlock = process(blockAudio, parameters);

            // 将结果复制到输出
            double[] processedSamples = processedBlock.getSamples().toDoubleArray();
            System.arraycopy(processedSamples, 0, outputSamples, pos,
                            Math.min(processedSamples.length, outputSamples.length - pos));

            processingProgress = ((double) pos / inputSamples.length) * 100.0;
        }

        return new AudioData(Linalg.vector(outputSamples), audioData.getSampleRate(),
                            audioData.getChannels(), outputSamples.length, audioData.getFormat());
    }

    /**
     * 提取时间段 / Extract time segment
     */
    private AudioData extractTimeSegment(AudioData audioData, double startTime, double endTime) {
        double sampleRate = audioData.getSampleRate();
        int startSample = (int) (startTime * sampleRate);
        int endSample = (int) (endTime * sampleRate);
        int segmentLength = endSample - startSample;

        double[] originalSamples = audioData.getSamples().toDoubleArray();
        double[] segmentSamples = new double[segmentLength];

        System.arraycopy(originalSamples, startSample, segmentSamples, 0, segmentLength);

        return new AudioData(Linalg.vector(segmentSamples), sampleRate, audioData.getChannels(),
                            segmentLength, audioData.getFormat());
    }

    /**
     * 插入处理后的段 / Insert processed segment
     */
    private AudioData insertProcessedSegment(AudioData originalAudio, AudioData processedSegment,
                                           double startTime, double endTime) {
        double sampleRate = originalAudio.getSampleRate();
        int startSample = (int) (startTime * sampleRate);

        double[] originalSamples = originalAudio.getSamples().toDoubleArray();
        double[] processedSamples = processedSegment.getSamples().toDoubleArray();
        double[] outputSamples = new double[originalSamples.length];

        // 复制原始音频
        System.arraycopy(originalSamples, 0, outputSamples, 0, originalSamples.length);

        // 插入处理后的段
        System.arraycopy(processedSamples, 0, outputSamples, startSample,
                        Math.min(processedSamples.length, outputSamples.length - startSample));

        return new AudioData(Linalg.vector(outputSamples), sampleRate, originalAudio.getChannels(),
                            outputSamples.length, originalAudio.getFormat());
    }

    /**
     * 检查和声类型是否有效 / Check if harmony type is valid
     */
    private boolean isValidHarmonyType(String harmonyType) {
        return harmonyType.equals("thirds") || harmonyType.equals("fourths") ||
               harmonyType.equals("fifths") || harmonyType.equals("octaves") ||
               harmonyType.equals("custom");
    }

    /**
     * 初始化性能指标 / Initialize performance metrics
     */
    private void initializePerformanceMetrics() {
        performanceMetrics.put("processingTime", 0.0);
        performanceMetrics.put("memoryUsage", 0.0);
        performanceMetrics.put("cpuUsage", 0.0);
    }

    /**
     * 更新统计信息 / Update statistics
     */
    private void updateStatistics(AudioData input, AudioData output) {
        lastStatistics.put("inputLength", input.getSamples().size());
        lastStatistics.put("outputLength", output.getSamples().size());
        lastStatistics.put("sampleRate", input.getSampleRate());
        lastStatistics.put("channels", input.getChannels());
    }

    @Override
    public AudioData transpose(AudioData audioData, int semitones) throws AudioProcessingException {
        return new Transposer().transpose(audioData, semitones);
    }

    @Override
    public AudioData harmonize(AudioData audioData, String harmonyType, int voiceCount) throws AudioProcessingException {
        Map<String, Object> params = new HashMap<>(currentParameters);
        params.put("harmonyType", harmonyType);
        params.put("voiceCount", voiceCount);
        return process(audioData, params);
    }

    @Override
    public AudioData quantize(AudioData audioData, double gridSize) throws AudioProcessingException {
        return new Quantizer().quantize(audioData, gridSize);
    }

    @Override
    public AudioData generateScale(ScaleTheory.ScaleType scaleType, int rootNote, int octave, double duration) throws AudioProcessingException {
        return new MusicTheoryProcessor().generateScale(scaleType, rootNote, octave, duration);
    }

    @Override
    public AudioData generateChord(ChordTheory.ChordType chordType, int rootNote, int octave, double duration) throws AudioProcessingException {
        return new MusicTheoryProcessor().generateChord(chordType, rootNote, octave, duration);
    }

    @Override
    public AudioData applyMusicTransformation(AudioData audioData, String transformation, Map<String, Object> parameters) throws AudioProcessingException {
        if (transformation == null) {
            throw new AudioProcessingException("transformation 不能为空 / transformation cannot be null");
        }
        if (parameters == null) {
            parameters = new HashMap<>();
        }
        String t = transformation.toLowerCase(Locale.ROOT);
        return switch (t) {
            case "transpose" -> transpose(audioData, ((Number) parameters.getOrDefault("semitones", 0)).intValue());
            case "harmonize" -> harmonize(audioData,
                    (String) parameters.getOrDefault("harmonyType", "thirds"),
                    ((Number) parameters.getOrDefault("voiceCount", 2)).intValue());
            case "quantize" -> quantize(audioData, ((Number) parameters.getOrDefault("gridSize", 0.25)).doubleValue());
            default -> process(audioData, parameters);
        };
    }

    @Override
    public String[] getSupportedMusicTransformations() {
        return new String[]{"transpose", "harmonize", "quantize"};
    }

    @Override
    public String[] getSupportedHarmonyTypes() {
        return new String[]{"thirds", "fourths", "fifths", "sixths", "octaves"};
    }

    @Override
    public String getCurrentQualityLevel() {
        return (String) currentParameters.get("qualityLevel");
    }

    @Override
    public double estimateProcessingTime(double audioLength) {
        // Simple estimation: 1.5x the audio length
        return audioLength * 1.5;
    }

    @Override
    public double getProcessingProgress() {
        return processingProgress;
    }

    @Override
    public boolean cancelProcessing() {
        processingCancelled = true;
        return true;
    }

    @Override
    public boolean isProcessingCancelled() {
        return processingCancelled;
    }

    @Override
    public boolean pauseProcessing() {
        processingPaused = true;
        return true;
    }

    @Override
    public boolean resumeProcessing() {
        processingPaused = false;
        return true;
    }

    @Override
    public boolean isProcessingPaused() {
        return processingPaused;
    }

    // IAdvancedAudioProcessor methods
    @Override
    public double getMinimumAudioLength() {
        return 0.1; // 100ms minimum
    }

    @Override
    public double getMaximumAudioLength() {
        return 3600.0; // 1 hour maximum
    }

    @Override
    public double getComplexityEstimate(double audioLength) {
        // Simple complexity estimation
        return audioLength * 0.2;
    }

    @Override
    public void warmUp() throws AudioProcessingException {
        // No specific warm-up needed for harmonizer
    }

    @Override
    public void cleanup() {
        // Clean up resources if needed
        currentParameters.clear();
        lastStatistics.clear();
        performanceMetrics.clear();
    }

    @Override
    public Map<String, Object> getLastProcessingStatistics() {
        return new HashMap<>(lastStatistics);
    }

    @Override
    public Map<String, Object> getPerformanceMetrics() {
        return new HashMap<>(performanceMetrics);
    }

    @Override
    public void setVerboseLogging(boolean enabled) {
        verboseLogging = enabled;
    }

    @Override
    public boolean isVerboseLoggingEnabled() {
        return verboseLogging;
    }

    @Override
    public String[] getSupportedProcessingTypes() {
        return new String[]{"harmonization", "transposition", "quantization"};
    }

    @Override
    public boolean supportsProcessingType(String processingType) {
        return "harmonization".equals(processingType) || 
               "transposition".equals(processingType) || 
               "quantization".equals(processingType);
    }

    @Override
    public String[] getQualityLevels() {
        return new String[]{"low", "medium", "high"};
    }

    @Override
    public void setQualityLevel(String qualityLevel) throws AudioProcessingException {
        if (qualityLevel == null || !("low".equals(qualityLevel) || "medium".equals(qualityLevel) || "high".equals(qualityLevel))) {
            throw new AudioProcessingException("Invalid quality level: " + qualityLevel);
        }
        currentParameters.put("qualityLevel", qualityLevel);
    }
}