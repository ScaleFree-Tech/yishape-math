package com.reremouse.lab.music.processing;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioFormat;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.audio.processing.IAudioProcessor;

import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Set;
import java.util.HashSet;

/**
 * 音乐量化器 / Music Quantizer
 * <p>
 * 实现音乐的量化功能，将音符时值对齐到网格。
 * Implements music quantization functionality to align note timing to grid.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class Quantizer implements IMusicProcessor {
    
    private static final String NAME = "Music Quantizer";
    private static final String VERSION = "1.0.0";
    private static final String DESCRIPTION = "Quantizes music timing to grid";
    
    // 默认参数 / Default parameters
    private static final Map<String, Object> DEFAULT_PARAMETERS = new HashMap<>();
    private static final Set<String> SUPPORTED_PARAMETERS = new HashSet<>();
    
    static {
        DEFAULT_PARAMETERS.put("gridResolution", "1/16");     // 网格分辨率 / Grid resolution
        DEFAULT_PARAMETERS.put("strength", 100);             // 量化强度(0-100) / Quantization strength
        DEFAULT_PARAMETERS.put("swing", 0);                  // 摇摆量(0-100) / Swing amount
        DEFAULT_PARAMETERS.put("humanize", 0);               // 人性化(0-100) / Humanize amount
        DEFAULT_PARAMETERS.put("preserveGroove", false);     // 保持律动 / Preserve groove
        DEFAULT_PARAMETERS.put("adaptiveQuantization", true); // 自适应量化 / Adaptive quantization
        DEFAULT_PARAMETERS.put("onsetDetectionThreshold", 0.3); // 起始检测阈值 / Onset detection threshold
        DEFAULT_PARAMETERS.put("tempoEstimation", "auto");   // 速度估计 / Tempo estimation
        DEFAULT_PARAMETERS.put("beatTracking", true);        // 节拍跟踪 / Beat tracking
        DEFAULT_PARAMETERS.put("qualityLevel", "high");      // 质量级别 / Quality level
        
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
    public Quantizer() {
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
        Quantizer cloned = new Quantizer();
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
            // 验证量化强度
            Integer strength = (Integer) parameters.get("strength");
            if (strength != null && (strength < 0 || strength > 100)) {
                return false;
            }

            // 验证摇摆量
            Integer swing = (Integer) parameters.get("swing");
            if (swing != null && (swing < 0 || swing > 100)) {
                return false;
            }

            // 验证人性化
            Integer humanize = (Integer) parameters.get("humanize");
            if (humanize != null && (humanize < 0 || humanize > 100)) {
                return false;
            }

            // 验证起始检测阈值
            Double threshold = (Double) parameters.get("onsetDetectionThreshold");
            if (threshold != null && (threshold < 0 || threshold > 1)) {
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
            
            String gridResolution = (String) parameters.get("gridResolution");
            int strength = (Integer) parameters.get("strength");
            int swing = (Integer) parameters.get("swing");
            
            if (verboseLogging) {
                System.out.println("Quantizing audio to " + gridResolution + " grid with " + strength + "% strength");
            }
            
            AudioData result = performQuantization(audioData, gridResolution, strength, swing, parameters);
            
            processingProgress = 100.0;
            updateStatistics(audioData, result);
            
            return result;
            
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to quantize audio: " + e.getMessage(), e);
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
            
            return performStreamQuantization(audioData, windowSamples, hopSamples, parameters);
            
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
     * 执行量化操作 / Perform quantization
     */
    private AudioData performQuantization(AudioData audioData, String gridResolution, int strength,
                                        int swing, Map<String, Object> parameters) throws AudioProcessingException {
        
        // 检测音符起始点
        List<OnsetEvent> onsets = detectOnsets(audioData, parameters);
        
        // 估计速度和节拍
        TempoInfo tempoInfo = estimateTempo(audioData, onsets, parameters);
        
        // 计算量化网格
        QuantizationGrid grid = createQuantizationGrid(tempoInfo, gridResolution, swing);
        
        // 量化起始点
        List<OnsetEvent> quantizedOnsets = quantizeOnsets(onsets, grid, strength, parameters);
        
        // 重建音频
        return reconstructAudio(audioData, onsets, quantizedOnsets, parameters);
    }
    
    /**
     * 检测音符起始点 / Detect note onsets
     */
    private List<OnsetEvent> detectOnsets(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        
        double[] samples = audioData.getSamples().toDoubleArray();
        double sampleRate = audioData.getSampleRate();
        double threshold = (Double) parameters.get("onsetDetectionThreshold");
        
        List<OnsetEvent> onsets = new ArrayList<>();
        
        int windowSize = 1024;
        int hopSize = 256;
        int numFrames = (samples.length - windowSize) / hopSize + 1;
        
        double[] spectralFlux = new double[numFrames];
        double[] previousSpectrum = new double[windowSize / 2];
        
        // 计算频谱流量
        for (int frame = 0; frame < numFrames; frame++) {
            if (processingCancelled) break;
            
            int startSample = frame * hopSize;
            double[] currentSpectrum = computeSpectrum(samples, startSample, windowSize);
            
            // 计算频谱差异
            double flux = 0.0;
            for (int bin = 0; bin < currentSpectrum.length; bin++) {
                double diff = currentSpectrum[bin] - previousSpectrum[bin];
                if (diff > 0) {
                    flux += diff;
                }
            }
            spectralFlux[frame] = flux;
            
            // 更新前一帧频谱
            System.arraycopy(currentSpectrum, 0, previousSpectrum, 0, currentSpectrum.length);
            
            processingProgress = ((double) frame / numFrames) * 30.0; // 30%用于起始检测
        }
        
        // 峰值检测
        for (int frame = 1; frame < spectralFlux.length - 1; frame++) {
            if (spectralFlux[frame] > threshold &&
                spectralFlux[frame] > spectralFlux[frame - 1] &&
                spectralFlux[frame] > spectralFlux[frame + 1]) {
                
                double timeStamp = (double) frame * hopSize / sampleRate;
                double strength = spectralFlux[frame];
                onsets.add(new OnsetEvent(timeStamp, strength));
            }
        }
        
        return onsets;
    }
    
    /**
     * 计算频谱 / Compute spectrum
     */
    private double[] computeSpectrum(double[] samples, int start, int windowSize) {
        double[] window = new double[windowSize];
        int end = Math.min(start + windowSize, samples.length);
        
        // 提取窗口并应用汉宁窗
        for (int i = 0; i < windowSize; i++) {
            if (start + i < end) {
                double hannWindow = 0.5 * (1 - Math.cos(2 * Math.PI * i / (windowSize - 1)));
                window[i] = samples[start + i] * hannWindow;
            } else {
                window[i] = 0.0;
            }
        }
        
        // 简化的FFT（实际应该使用真正的FFT）
        double[] spectrum = new double[windowSize / 2];
        for (int k = 0; k < spectrum.length; k++) {
            double real = 0.0, imag = 0.0;
            for (int n = 0; n < windowSize; n++) {
                double angle = -2 * Math.PI * k * n / windowSize;
                real += window[n] * Math.cos(angle);
                imag += window[n] * Math.sin(angle);
            }
            spectrum[k] = Math.sqrt(real * real + imag * imag);
        }
        
        return spectrum;
    }
    
    /**
     * 估计速度 / Estimate tempo
     */
    private TempoInfo estimateTempo(AudioData audioData, List<OnsetEvent> onsets, Map<String, Object> parameters) throws AudioProcessingException {
        
        String tempoEstimation = (String) parameters.get("tempoEstimation");
        
        if ("auto".equals(tempoEstimation)) {
            return estimateTempoFromOnsets(onsets);
        } else {
            // 使用指定的速度
            double bpm = Double.parseDouble(tempoEstimation);
            return new TempoInfo(bpm, 4, 4); // 默认4/4拍
        }
    }
    
    /**
     * 从起始点估计速度 / Estimate tempo from onsets
     */
    private TempoInfo estimateTempoFromOnsets(List<OnsetEvent> onsets) {
        if (onsets.size() < 4) {
            return new TempoInfo(120.0, 4, 4); // 默认120 BPM
        }
        
        // 计算相邻起始点间隔
        List<Double> intervals = new ArrayList<>();
        for (int i = 1; i < onsets.size(); i++) {
            double interval = onsets.get(i).timestamp - onsets.get(i - 1).timestamp;
            if (interval > 0.1 && interval < 2.0) { // 过滤异常值
                intervals.add(interval);
            }
        }
        
        if (intervals.isEmpty()) {
            return new TempoInfo(120.0, 4, 4);
        }
        
        // 计算平均间隔
        double avgInterval = intervals.stream().mapToDouble(Double::doubleValue).average().orElse(0.5);
        
        // 转换为BPM（假设间隔是四分音符）
        double bpm = 60.0 / avgInterval;
        
        // 调整到合理范围
        while (bpm < 60) bpm *= 2;
        while (bpm > 200) bpm /= 2;
        
        return new TempoInfo(bpm, 4, 4);
    }
    
    /**
     * 创建量化网格 / Create quantization grid
     */
    private QuantizationGrid createQuantizationGrid(TempoInfo tempoInfo, String gridResolution, int swing) {
        
        double beatDuration = 60.0 / tempoInfo.bpm;
        double gridInterval = calculateGridInterval(beatDuration, gridResolution);
        
        return new QuantizationGrid(gridInterval, swing, tempoInfo);
    }
    
    /**
     * 计算网格间隔 / Calculate grid interval
     */
    private double calculateGridInterval(double beatDuration, String gridResolution) {
        switch (gridResolution) {
            case "1/4": return beatDuration;
            case "1/8": return beatDuration / 2;
            case "1/16": return beatDuration / 4;
            case "1/32": return beatDuration / 8;
            case "1/8T": return beatDuration / 3; // 三连音
            case "1/16T": return beatDuration / 6;
            default: return beatDuration / 4; // 默认1/16
        }
    }
    
    /**
     * 量化起始点 / Quantize onsets
     */
    private List<OnsetEvent> quantizeOnsets(List<OnsetEvent> onsets, QuantizationGrid grid,
                                          int strength, Map<String, Object> parameters) throws AudioProcessingException {
        
        List<OnsetEvent> quantizedOnsets = new ArrayList<>();
        double strengthFactor = strength / 100.0;
        int humanize = (Integer) parameters.get("humanize");
        boolean preserveGroove = (Boolean) parameters.get("preserveGroove");
        
        for (int i = 0; i < onsets.size(); i++) {
            if (processingCancelled) break;
            
            OnsetEvent onset = onsets.get(i);
            double originalTime = onset.timestamp;
            
            // 找到最近的网格点
            double gridTime = findNearestGridPoint(originalTime, grid);
            
            // 应用量化强度
            double quantizedTime = originalTime + (gridTime - originalTime) * strengthFactor;
            
            // 应用人性化
            if (humanize > 0) {
                double humanizeAmount = (humanize / 100.0) * grid.interval * 0.1;
                double randomOffset = (Math.random() - 0.5) * 2 * humanizeAmount;
                quantizedTime += randomOffset;
            }
            
            // 保持律动
            if (preserveGroove && i > 0) {
                double originalInterval = originalTime - onsets.get(i - 1).timestamp;
                double quantizedInterval = quantizedTime - quantizedOnsets.get(i - 1).timestamp;
                
                // 如果间隔变化太大，减少量化强度
                if (Math.abs(quantizedInterval - originalInterval) > originalInterval * 0.2) {
                    quantizedTime = originalTime + (gridTime - originalTime) * strengthFactor * 0.5;
                }
            }
            
            quantizedOnsets.add(new OnsetEvent(quantizedTime, onset.strength));
            
            processingProgress = 50.0 + ((double) i / onsets.size()) * 30.0; // 50-80%用于量化
        }
        
        return quantizedOnsets;
    }
    
    /**
     * 找到最近的网格点 / Find nearest grid point
     */
    private double findNearestGridPoint(double time, QuantizationGrid grid) {
        double gridIndex = time / grid.interval;
        double nearestIndex = Math.round(gridIndex);
        
        double gridTime = nearestIndex * grid.interval;
        
        // 应用摇摆
        if (grid.swing > 0 && nearestIndex % 2 == 1) { // 奇数网格点
            double swingAmount = (grid.swing / 100.0) * grid.interval * 0.1;
            gridTime += swingAmount;
        }
        
        return gridTime;
    }
    
    /**
     * 重建音频 / Reconstruct audio
     */
    private AudioData reconstructAudio(AudioData originalAudio, List<OnsetEvent> originalOnsets,
                                     List<OnsetEvent> quantizedOnsets, Map<String, Object> parameters) throws AudioProcessingException {
        
        double[] originalSamples = originalAudio.getSamples().toDoubleArray();
        double sampleRate = originalAudio.getSampleRate();
        double[] outputSamples = new double[originalSamples.length];
        
        // 使用时间拉伸重新排列音频段
        for (int i = 0; i < originalOnsets.size() - 1; i++) {
            if (processingCancelled) break;
            
            double originalStart = originalOnsets.get(i).timestamp;
            double originalEnd = originalOnsets.get(i + 1).timestamp;
            double quantizedStart = quantizedOnsets.get(i).timestamp;
            double quantizedEnd = quantizedOnsets.get(i + 1).timestamp;
            
            int originalStartSample = (int) (originalStart * sampleRate);
            int originalEndSample = (int) (originalEnd * sampleRate);
            int quantizedStartSample = (int) (quantizedStart * sampleRate);
            int quantizedEndSample = (int) (quantizedEnd * sampleRate);
            
            // 提取原始段
            int originalLength = originalEndSample - originalStartSample;
            int quantizedLength = quantizedEndSample - quantizedStartSample;
            
            if (originalLength > 0 && quantizedLength > 0 &&
                originalStartSample >= 0 && originalEndSample <= originalSamples.length &&
                quantizedStartSample >= 0 && quantizedEndSample <= outputSamples.length) {
                
                // 时间拉伸或压缩
                double timeRatio = (double) quantizedLength / originalLength;
                
                for (int j = 0; j < quantizedLength; j++) {
                    double sourceIndex = j / timeRatio;
                    int sourceIndexInt = (int) sourceIndex;
                    double fraction = sourceIndex - sourceIndexInt;
                    
                    if (sourceIndexInt >= 0 && sourceIndexInt < originalLength - 1) {
                        double sample1 = originalSamples[originalStartSample + sourceIndexInt];
                        double sample2 = originalSamples[originalStartSample + sourceIndexInt + 1];
                        outputSamples[quantizedStartSample + j] = sample1 + fraction * (sample2 - sample1);
                    }
                }
            }
            
            processingProgress = 80.0 + ((double) i / (originalOnsets.size() - 1)) * 20.0; // 80-100%用于重建
        }
        
        return new AudioData(Linalg.vector(outputSamples), sampleRate, originalAudio.getChannels(),
                            outputSamples.length, originalAudio.getFormat());
    }
    
    // ========== IMusicProcessor 接口实现 / IMusicProcessor Interface Implementation ==========
    
    @Override
    public AudioData transpose(AudioData audioData, int semitones) throws AudioProcessingException {
        // Quantizer doesn't support transposition
        return audioData;
    }
    
    @Override
    public AudioData harmonize(AudioData audioData, String harmonyType, int voiceCount) throws AudioProcessingException {
        // Quantizer doesn't support harmonization
        return audioData;
    }
    
    @Override
    public AudioData quantize(AudioData audioData, double gridSize) throws AudioProcessingException {
        Map<String, Object> params = new HashMap<>(currentParameters);
        params.put("gridResolution", "1/" + (int)(1/gridSize));
        return process(audioData, params);
    }
    
    @Override
    public AudioData generateScale(com.reremouse.lab.music.theory.ScaleTheory scale, int rootNote, int octave, double duration) throws AudioProcessingException {
        // Quantizer doesn't support scale generation
        double[] samples = new double[(int)(duration * 44100)];
        return new AudioData(Linalg.zeros(samples.length), 44100, 1, samples.length, AudioFormat.WAV);
    }
    
    @Override
    public AudioData generateChord(com.reremouse.lab.music.theory.ChordTheory chord, int rootNote, int octave, double duration) throws AudioProcessingException {
        // Quantizer doesn't support chord generation
        double[] samples = new double[(int)(duration * 44100)];
        return new AudioData(Linalg.zeros(samples.length), 44100, 1, samples.length, AudioFormat.WAV);
    }
    
    @Override
    public AudioData applyMusicTransformation(AudioData audioData, String transformation, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }
        
        Map<String, Object> params = new HashMap<>(parameters);
        params.put("operation", transformation);
        return process(audioData, params);
    }
    
    @Override
    public String[] getSupportedMusicTransformations() {
        return new String[]{"quantize"};
    }
    
    @Override
    public String[] getSupportedHarmonyTypes() {
        return new String[]{}; // Quantizer doesn't support harmony
    }
    
    @Override
    public String getCurrentQualityLevel() {
        return (String) currentParameters.get("qualityLevel");
    }
    
    @Override
    public double estimateProcessingTime(double audioLength) {
        // Simple estimation: 1.2x the audio length
        return audioLength * 1.2;
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
    
    // ========== IAdvancedAudioProcessor 接口实现 / IAdvancedAudioProcessor Interface Implementation ==========
    
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
        return audioLength * 0.15;
    }
    
    @Override
    public void warmUp() throws AudioProcessingException {
        // No specific warm-up needed for quantizer
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
        return new String[]{"quantization"};
    }
    
    @Override
    public boolean supportsProcessingType(String processingType) {
        return "quantization".equals(processingType);
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
    
    // 辅助类定义
    private static class OnsetEvent {
        public final double timestamp;
        public final double strength;
        
        public OnsetEvent(double timestamp, double strength) {
            this.timestamp = timestamp;
            this.strength = strength;
        }
    }
    
    private static class TempoInfo {
        public final double bpm;
        public final int numerator;
        public final int denominator;
        
        public TempoInfo(double bpm, int numerator, int denominator) {
            this.bpm = bpm;
            this.numerator = numerator;
            this.denominator = denominator;
        }
    }
    
    private static class QuantizationGrid {
        public final double interval;
        public final int swing;
        public final TempoInfo tempoInfo;
        
        public QuantizationGrid(double interval, int swing, TempoInfo tempoInfo) {
            this.interval = interval;
            this.swing = swing;
            this.tempoInfo = tempoInfo;
        }
    }
    
    // 实用方法实现
    private void initializePerformanceMetrics() {
        performanceMetrics.put("processingTime", 0.0);
        performanceMetrics.put("memoryUsage", 0.0);
        performanceMetrics.put("cpuUsage", 0.0);
    }
    
    private void updateStatistics(AudioData input, AudioData output) {
        lastStatistics.put("inputLength", input.getSamples().size());
        lastStatistics.put("outputLength", output.getSamples().size());
        lastStatistics.put("sampleRate", input.getSampleRate());
        lastStatistics.put("channels", input.getChannels());
    }
    
    private AudioData extractTimeSegment(AudioData audioData, double startTime, double endTime) throws AudioProcessingException {
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
    
    private AudioData insertProcessedSegment(AudioData originalAudio, AudioData processedSegment,
                                           double startTime, double endTime) throws AudioProcessingException {
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
    
    private AudioData performStreamQuantization(AudioData audioData, int windowSamples, int hopSamples, Map<String, Object> parameters) throws AudioProcessingException {
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
}