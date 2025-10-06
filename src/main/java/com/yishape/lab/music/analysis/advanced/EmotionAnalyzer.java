package com.yishape.lab.music.analysis.advanced;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioUtil;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.music.analysis.MusicDetectionResult;
import com.yishape.lab.music.analysis.StandardizedConfidenceCalculator;
import com.yishape.lab.music.analysis.basic.BeatAnalyzerImpl;
import com.yishape.lab.music.analysis.basic.KeyAnalyzerImpl;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.signal.core.RereFFT;
import com.yishape.lab.math.signal.core.Complex;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

/**
 * 音乐情感分析器 / Music Emotion Analyzer
 * <p>
 * 基于音频特征分析音乐的情感特征，包括能量、价值度、唤醒度等。
 * Analyzes emotional characteristics of music based on audio features, including energy, valence, arousal, etc.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class EmotionAnalyzer implements IAdvancedAnalyzer {
    
    // 情感维度 / Emotion dimensions
    public static final String VALENCE = "valence";     // 价值度 (正面/负面) / Valence (positive/negative)
    public static final String AROUSAL = "arousal";     // 唤醒度 (激动/平静) / Arousal (excited/calm)
    public static final String ENERGY = "energy";       // 能量 / Energy
    public static final String DANCEABILITY = "danceability"; // 可舞性 / Danceability
    
    // 默认参数 / Default parameters
    private static final int DEFAULT_WINDOW_SIZE = 4096;
    private static final int DEFAULT_HOP_SIZE = 2048;
    
    private final BeatAnalyzerImpl beatAnalyzer;
    private final KeyAnalyzerImpl keyAnalyzer;
    private final StandardizedConfidenceCalculator confidenceCalculator = new StandardizedConfidenceCalculator();
    
    public EmotionAnalyzer() {
        this.beatAnalyzer = new BeatAnalyzerImpl();
        this.keyAnalyzer = new KeyAnalyzerImpl();
    }
    
    @Override
    public Map<String, Object> analyze(AudioData audioData) throws AudioProcessingException {
        return analyze(audioData, getDefaultParameters());
    }
    
    @Override
    public MusicDetectionResult analyzeAdvancedMusic(AudioData audioData) throws AudioProcessingException {
        return analyzeAdvancedMusic(audioData, getDefaultParameters());
    }
    
    @Override
    public Map<String, Object> analyze(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }
        
        try {
            Map<String, Object> results = new HashMap<>();
            
            // 计算基本音频特征 / Calculate basic audio features
            AudioFeatures features = extractAudioFeatures(audioData, parameters);
            
            // 计算情感维度 / Calculate emotion dimensions
            double valence = calculateValence(features);
            double arousal = calculateArousal(features);
            double energy = calculateEnergy(features);
            double danceability = calculateDanceability(features);
            
            // 存储结果 / Store results
            results.put(VALENCE, valence);
            results.put(AROUSAL, arousal);
            results.put(ENERGY, energy);
            results.put(DANCEABILITY, danceability);
            
            // 添加原始特征 / Add raw features
            results.put("rawFeatures", features.toMap());
            
            // 添加情感标签 / Add emotion label
            results.put("emotionLabel", classifyEmotion(valence, arousal));
            
            return results;
            
        } catch (Exception e) {
            throw new AudioProcessingException("Error in emotion analysis: " + e.getMessage(), e);
        }
    }
    
    @Override
    public MusicDetectionResult analyzeAdvancedMusic(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }
        
        try {
            // 计算基本音频特征 / Calculate basic audio features
            AudioFeatures features = extractAudioFeatures(audioData, parameters);
            
            // 计算情感维度 / Calculate emotion dimensions
            double valence = calculateValence(features);
            double arousal = calculateArousal(features);
            double energy = calculateEnergy(features);
            double danceability = calculateDanceability(features);
            
            // 创建结构化结果对象 / Create structured result object
            EmotionAnalysisResult result = new EmotionAnalysisResult();
            result.setValence(valence);
            result.setArousal(arousal);
            result.setEnergy(energy);
            result.setDanceability(danceability);
            result.setEmotionLabel(classifyEmotion(valence, arousal));
            result.setRawFeatures(features.toMap());
            
            // Improved confidence calculation based on multiple factors
            double confidence = calculateEmotionConfidence(features, valence, arousal, energy, danceability);
            result.setConfidence(confidence);
            result.setAlgorithm("emotion_analyzer");
            
            return result;
            
        } catch (Exception e) {
            throw new AudioProcessingException("Error in emotion analysis: " + e.getMessage(), e);
        }
    }
    
    /**
     * Improved confidence calculation for emotion analysis
     * Takes into account multiple factors to determine the reliability of emotion predictions
     */
    private double calculateEmotionConfidence(AudioFeatures features, double valence, double arousal, double energy, double danceability) {
        try {
            // Base confidence on the consistency of features
            double featureConsistency = 0.0;
            int validFeatures = 0;
            
            // Check RMS consistency (should be in reasonable range)
            if (features.rms >= 0.0 && features.rms <= 1.0) {
                featureConsistency += 1.0;
                validFeatures++;
            }
            
            // Check ZCR consistency (typically 0-1 for normalized audio)
            if (features.zcr >= 0.0 && features.zcr <= 1.0) {
                featureConsistency += 1.0;
                validFeatures++;
            }
            
            // Check spectral centroid (reasonable range for music)
            if (features.spectralCentroid >= 100.0 && features.spectralCentroid <= 8000.0) {
                featureConsistency += 1.0;
                validFeatures++;
            }
            
            // Check spectral rolloff (reasonable range for music)
            if (features.spectralRolloff >= 500.0 && features.spectralRolloff <= 10000.0) {
                featureConsistency += 1.0;
                validFeatures++;
            }
            
            // Check spectral bandwidth (reasonable range for music)
            if (features.spectralBandwidth >= 100.0 && features.spectralBandwidth <= 5000.0) {
                featureConsistency += 1.0;
                validFeatures++;
            }
            
            // 使用标准化置信度计算器计算统计置信度 / Use standardized confidence calculator for statistical confidence
            double[] featureValues = {
                features.rms, features.zcr, features.spectralCentroid,
                features.spectralRolloff, features.spectralBandwidth
            };
            double normalizedConsistency = confidenceCalculator.calculateStatisticalConfidence(featureValues);
            
            // Confidence based on how extreme the emotion values are (more extreme = more confident)
            double extremityScore = (Math.abs(valence - 0.5) + Math.abs(arousal - 0.5)) / 2.0;
            
            // Confidence based on energy level (moderate energy often gives better predictions)
            double energyScore = 1.0 - Math.abs(energy - 0.5);
            
            // Combine all factors with weighted average using standardized confidence calculator
            Map<String, Double> factors = new HashMap<>();
            factors.put("featureConsistency", normalizedConsistency);
            factors.put("extremityScore", extremityScore);
            factors.put("energyScore", energyScore);
            
            Map<String, Double> weights = new HashMap<>();
            weights.put("featureConsistency", 0.4);
            weights.put("extremityScore", 0.4);
            weights.put("energyScore", 0.2);
            
            double confidence = confidenceCalculator.calculateWeightedConfidence(factors, weights);
            
            // Ensure confidence is in 0-1 range using standardized confidence calculator
            return confidenceCalculator.calculateMinimumConfidence(1, confidence);
            
        } catch (Exception e) {
            // 使用标准化置信度计算器计算最小置信度 / Use standardized confidence calculator for minimum confidence
            return confidenceCalculator.calculateMinimumConfidence(1, 0.5); // 默认中等置信度 / Default medium confidence
        }
    }
    
    @Override
    public String getAnalyzerName() {
        return "emotion_analyzer";
    }
    
    @Override
    public String getName() {
        return "Emotion Analyzer";
    }
    
    @Override
    public String[] getSupportedParameters() {
        return new String[]{"windowSize", "hopSize"};
    }
    
    @Override
    public Map<String, Object> getDefaultParameters() {
        Map<String, Object> params = new HashMap<>();
        params.put("windowSize", DEFAULT_WINDOW_SIZE);
        params.put("hopSize", DEFAULT_HOP_SIZE);
        return params;
    }
    
    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        if (parameters == null) return true;
        
        for (String key : parameters.keySet()) {
            if (!java.util.Arrays.asList(getSupportedParameters()).contains(key)) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * 设置分析器参数 / Set analyzer parameters
     * 
     * @param parameters 要设置的参数 / Parameters to set
     * @throws AudioProcessingException 参数无效时抛出异常 / Thrown when parameters are invalid
     */
    public void setParameters(Map<String, Object> parameters) throws AudioProcessingException {
        if (parameters == null) {
            return;
        }
        
        // 验证参数有效性 / Validate parameters
        for (String key : parameters.keySet()) {
            if (!java.util.Arrays.asList(getSupportedParameters()).contains(key)) {
                throw new AudioProcessingException("Unsupported parameter: " + key);
            }
        }
        
        // 这里可以添加参数验证逻辑，但由于当前实现使用方法参数传递，
        // 暂时只做基本验证 / Basic validation for now since current implementation uses method parameters
    }
    
    @Override
    public Map<String, Object> getCurrentParameters() {
        return getDefaultParameters();
    }
    
    @Override
    public void resetParameters() {
        // 重置为默认参数 / Reset to default parameters
    }
    
    @Override
    public String getVersion() {
        return "1.0.0";
    }
    
    @Override
    public String getDescription() {
        return "Analyzes emotional characteristics of music based on audio features";
    }
    
    @Override
    public boolean supportsAudioFormat(double sampleRate, int channels, int bitDepth) {
        return sampleRate > 0 && channels > 0 && bitDepth > 0;
    }
    
    @Override
    public double getMinimumAudioLength() {
        return 1.0; // 1 second minimum
    }
    
    @Override
    public double getMaximumAudioLength() {
        return 3600.0; // 1 hour maximum
    }
    
    @Override
    public double getComplexityEstimate(double audioLength) {
        return audioLength * 0.1; // Low complexity
    }
    
    @Override
    public void warmUp() throws AudioProcessingException {
        // 预热分析器 / Warm up analyzer
    }
    
    @Override
    public void cleanup() {
        // 清理资源 / Clean up resources
    }
    
    @Override
    public String getStatus() {
        return "ready";
    }
    
    @Override
    public boolean isReady() {
        return true;
    }
    
    @Override
    public Map<String, Object> getLastAnalysisStatistics() {
        return new HashMap<>();
    }
    
    @Override
    public Map<String, Object> getPerformanceMetrics() {
        return new HashMap<>();
    }
    
    @Override
    public void setVerboseLogging(boolean enabled) {
        // 设置详细日志 / Set verbose logging
    }
    
    @Override
    public boolean isVerboseLoggingEnabled() {
        return false;
    }
    
    /**
     * 提取音频特征 / Extract audio features
     */
    private AudioFeatures extractAudioFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        AudioFeatures features = new AudioFeatures();
        
        // Add timeout check to prevent hanging
        long startTime = System.currentTimeMillis();
        long maxDuration = 5000; // 5 seconds max for emotion analysis (reduced from 15 seconds)
        
        // 限制处理的样本数量以提高性能 / Limit number of samples processed for performance
        IVector<Double> samples = audioData.getSamples();
        int maxSamples = Math.min(500000, samples.length()); // Process at most ~11 seconds at 44.1kHz (reduced from 45 seconds)
        
        try {
            // 基本统计特征 / Basic statistical features
            features.rms = AudioUtil.calculateRMS(samples.slice(0, Math.min(100000, samples.length()))); // Process smaller chunk for RMS
            features.zcr = AudioUtil.calculateZeroCrossingRate(samples.slice(0, Math.min(100000, samples.length()))); // Process smaller chunk for ZCR
            
            // 使用帧处理来计算频谱特征 / Use frame processing for spectral features
            SpectralFeatures spectralFeatures = calculateSpectralFeaturesFrameBased(audioData, parameters, maxSamples, startTime, maxDuration);
            features.spectralCentroid = spectralFeatures.centroid;
            features.spectralRolloff = spectralFeatures.rolloff;
            features.spectralBandwidth = spectralFeatures.bandwidth;
            
            // 节拍和调性特征 / Beat and key features
            try {
                // Check for timeout
                if ((System.currentTimeMillis() - startTime) > maxDuration) {
                    throw new AudioProcessingException("Emotion analysis timed out");
                }
                
                features.tempo = beatAnalyzer.estimateTempo(audioData);
                double[] chromaFeatures = keyAnalyzer.analyzeChromaFeatures(audioData);
                features.chromaVariance = calculateVariance(chromaFeatures);
            } catch (Exception e) {
                features.tempo = 120.0; // 默认值 / Default value
                features.chromaVariance = 0.5;
            }
            
            return features;
            
        } catch (Exception e) {
            // Fallback to basic features if advanced analysis fails
            features.rms = AudioUtil.calculateRMS(samples.slice(0, Math.min(50000, samples.length()))); // Smaller chunk
            features.zcr = AudioUtil.calculateZeroCrossingRate(samples.slice(0, Math.min(50000, samples.length()))); // Smaller chunk
            features.spectralCentroid = 1000.0; // Default value
            features.spectralRolloff = 4000.0;  // Default value
            features.spectralBandwidth = 2000.0; // Default value
            features.tempo = 120.0; // Default value
            features.chromaVariance = 0.5;
            return features;
        }
    }
    
    /**
     * 使用帧处理计算频谱特征 / Calculate spectral features using frame processing
     */
    private SpectralFeatures calculateSpectralFeaturesFrameBased(AudioData audioData, Map<String, Object> parameters,
                                                               int maxSamples, long startTime, long maxDuration) throws AudioProcessingException {
        SpectralFeatures features = new SpectralFeatures();
        
        int windowSize = (Integer) parameters.getOrDefault("windowSize", DEFAULT_WINDOW_SIZE);
        int hopSize = (Integer) parameters.getOrDefault("hopSize", DEFAULT_HOP_SIZE);
        
        IVector<Double> samples = audioData.getSamples();
        double sampleRate = audioData.getSampleRate();
        
        // 限制处理的样本数量 / Limit number of samples processed
        int processSamples = Math.min(maxSamples, samples.length());
        
        // 计算帧数 / Calculate number of frames
        int numFrames = Math.max(1, (processSamples - windowSize) / hopSize + 1);
        // 限制帧数以提高性能 / Limit number of frames for performance
        numFrames = Math.min(numFrames, 30); // Process at most 30 frames (reduced from 100 for better performance)
        
        double totalCentroid = 0.0;
        double totalRolloff = 0.0;
        double totalBandwidth = 0.0;
        int validFrames = 0;
        
        for (int frame = 0; frame < numFrames; frame++) {
            // Check for timeout
            if ((System.currentTimeMillis() - startTime) > maxDuration) {
                break;
            }
            
            int start = frame * hopSize;
            if (start + windowSize > processSamples) {
                break;
            }
            
            // 提取帧数据 / Extract frame data
            IVector<Double> frameData = samples.slice(start, start + windowSize);
            
            try {
                // 计算频谱特征 / Calculate spectral features for this frame
                double[] centroidAndRolloff = calculateFrameSpectralCentroidAndRolloff(frameData, sampleRate, windowSize);
                double bandwidth = calculateFrameSpectralBandwidth(frameData, sampleRate, windowSize, centroidAndRolloff[0]);
                
                totalCentroid += centroidAndRolloff[0];
                totalRolloff += centroidAndRolloff[1];
                totalBandwidth += bandwidth;
                validFrames++;
            } catch (Exception e) {
                // Skip frame if calculation fails
                continue;
            }
        }
        
        // 计算平均值 / Calculate averages
        if (validFrames > 0) {
            features.centroid = totalCentroid / validFrames;
            features.rolloff = totalRolloff / validFrames;
            features.bandwidth = totalBandwidth / validFrames;
        } else {
            // 默认值 / Default values
            features.centroid = 1000.0;
            features.rolloff = 4000.0;
            features.bandwidth = 2000.0;
        }
        
        return features;
    }
    
    /**
     * 计算帧的频谱质心和滚降点 / Calculate spectral centroid and rolloff for a frame
     */
    private double[] calculateFrameSpectralCentroidAndRolloff(IVector<Double> frameData, double sampleRate, int windowSize) throws AudioProcessingException {
        // 应用窗函数 / Apply window function
        IVector<Double> windowed = AudioUtil.applyWindow(frameData, windowSize);
        
        // 转换为复数数组进行FFT / Convert to Complex array for FFT
        Complex[] complexInput = AudioUtil.convertToComplex(windowed);
        
        // 使用零填充确保长度为2的幂 / Zero-pad to ensure length is power of 2
        Complex[] paddedInput = RereFFT.zeroPadToPowerOfTwo(complexInput);
        Complex[] spectrum = RereFFT.fft(paddedInput);
        
        // 计算质心和滚降点 / Calculate centroid and rolloff
        double centroid = AudioUtil.calculateSpectralCentroid(spectrum, sampleRate, paddedInput.length);
        double rolloff = AudioUtil.calculateSpectralRolloff(spectrum, sampleRate, paddedInput.length);
        
        return new double[]{centroid, rolloff};
    }
    
    /**
     * 计算帧的频谱带宽 / Calculate spectral bandwidth for a frame
     */
    private double calculateFrameSpectralBandwidth(IVector<Double> frameData, double sampleRate, int windowSize, double centroid) throws AudioProcessingException {
        // 应用窗函数 / Apply window function
        IVector<Double> windowed = AudioUtil.applyWindow(frameData, windowSize);
        
        // 转换为复数数组进行FFT / Convert to Complex array for FFT
        Complex[] complexInput = AudioUtil.convertToComplex(windowed);
        
        // 使用零填充确保长度为2的幂 / Zero-pad to ensure length is power of 2
        Complex[] paddedInput = RereFFT.zeroPadToPowerOfTwo(complexInput);
        Complex[] spectrum = RereFFT.fft(paddedInput);
        
        // 计算带宽 / Calculate bandwidth
        return AudioUtil.calculateSpectralBandwidth(spectrum, sampleRate, paddedInput.length, centroid);
    }
    
    /**
     * 计算价值度 (正面/负面情感) / Calculate valence (positive/negative emotion)
     */
    private double calculateValence(AudioFeatures features) {
        // 基于多个特征计算价值度 / Calculate valence based on multiple features
        double valence = 0.0;
        
        // 频谱质心：较高的质心通常对应更明亮、更正面的音乐 / Higher centroid usually corresponds to brighter, more positive music
        double centroidScore = Math.min(1.0, features.spectralCentroid / 4000.0);
        
        // 节拍：适中的节拍通常更正面 / Moderate tempo is usually more positive
        double tempoScore = 1.0 - Math.abs(features.tempo - 120.0) / 120.0;
        tempoScore = Math.max(0.0, Math.min(1.0, tempoScore));
        
        // 色度方差：适中的方差表示和谐 / Moderate chroma variance indicates harmony
        double chromaScore = 1.0 - Math.abs(features.chromaVariance - 0.5);
        
        // 加权平均 / Weighted average
        valence = (centroidScore * 0.4 + tempoScore * 0.3 + chromaScore * 0.3);
        
        return Math.max(0.0, Math.min(1.0, valence));
    }
    
    /**
     * 计算唤醒度 (激动/平静) / Calculate arousal (excited/calm)
     */
    private double calculateArousal(AudioFeatures features) {
        // 基于能量和动态特征计算唤醒度 / Calculate arousal based on energy and dynamic features
        double arousal = 0.0;
        
        // RMS能量：更高的能量对应更高的唤醒度 / Higher RMS energy corresponds to higher arousal
        double energyScore = Math.min(1.0, features.rms * 2.0);
        
        // 零交叉率：更高的ZCR通常对应更高的唤醒度 / Higher ZCR usually corresponds to higher arousal
        double zcrScore = Math.min(1.0, features.zcr * 2.0);
        
        // 节拍：更快的节拍对应更高的唤醒度 / Faster tempo corresponds to higher arousal
        double tempoScore = Math.min(1.0, features.tempo / 180.0);
        
        // 频谱带宽：更宽的带宽通常对应更高的唤醒度 / Wider bandwidth usually corresponds to higher arousal
        double bandwidthScore = Math.min(1.0, features.spectralBandwidth / 3000.0);
        
        // 加权平均 / Weighted average
        arousal = (energyScore * 0.3 + zcrScore * 0.2 + tempoScore * 0.3 + bandwidthScore * 0.2);
        
        return Math.max(0.0, Math.min(1.0, arousal));
    }
    
    /**
     * 计算能量 / Calculate energy
     */
    private double calculateEnergy(AudioFeatures features) {
        // 主要基于RMS能量，结合其他特征 / Primarily based on RMS energy, combined with other features
        double energy = features.rms;
        
        // 结合频谱滚降点 / Combine with spectral rolloff
        double rolloffContribution = Math.min(1.0, features.spectralRolloff / 8000.0);
        
        // 加权组合 / Weighted combination
        energy = (energy * 0.7 + rolloffContribution * 0.3);
        
        return Math.max(0.0, Math.min(1.0, energy));
    }
    
    /**
     * 计算可舞性 / Calculate danceability
     */
    private double calculateDanceability(AudioFeatures features) {
        // 基于节拍稳定性和能量 / Based on beat stability and energy
        double danceability = 0.0;
        
        // 节拍在舞蹈范围内 (90-140 BPM) / Tempo in dance range (90-140 BPM)
        double tempoScore = 0.0;
        if (features.tempo >= 90.0 && features.tempo <= 140.0) {
            tempoScore = 1.0 - Math.abs(features.tempo - 115.0) / 25.0;
        }
        tempoScore = Math.max(0.0, Math.min(1.0, tempoScore));
        
        // 适中的能量 / Moderate energy
        double energyScore = 1.0 - Math.abs(features.rms - 0.5);
        
        // 稳定的色度特征 / Stable chroma features
        double stabilityScore = 1.0 - features.chromaVariance;
        
        // 加权平均 / Weighted average
        danceability = (tempoScore * 0.5 + energyScore * 0.3 + stabilityScore * 0.2);
        
        return Math.max(0.0, Math.min(1.0, danceability));
    }
    
    /**
     * 分类情感标签 / Classify emotion label
     */
    private String classifyEmotion(double valence, double arousal) {
        // 基于价值度和唤醒度的四象限模型 / Four-quadrant model based on valence and arousal
        if (valence >= 0.5 && arousal >= 0.5) {
            return "happy"; // 高价值度，高唤醒度 / High valence, high arousal
        } else if (valence >= 0.5 && arousal < 0.5) {
            return "peaceful"; // 高价值度，低唤醒度 / High valence, low arousal
        } else if (valence < 0.5 && arousal >= 0.5) {
            return "angry"; // 低价值度，高唤醒度 / Low valence, high arousal
        } else {
            return "sad"; // 低价值度，低唤醒度 / Low valence, low arousal
        }
    }
    
    /**
     * 计算方差 / Calculate variance
     */
    private double calculateVariance(double[] values) {
        if (values == null || values.length == 0) {
            return 0.0;
        }
        
        double mean = 0.0;
        for (double value : values) {
            mean += value;
        }
        mean /= values.length;
        
        double variance = 0.0;
        for (double value : values) {
            variance += Math.pow(value - mean, 2);
        }
        variance /= values.length;
        
        return variance;
    }
    
    /**
     * 音频特征类 / Audio features class
     */
    private static class AudioFeatures {
        double rms;
        double zcr;
        double spectralCentroid;
        double spectralRolloff;
        double spectralBandwidth;
        double tempo;
        double chromaVariance;
        
        Map<String, Object> toMap() {
            Map<String, Object> map = new HashMap<>();
            map.put("rms", rms);
            map.put("zcr", zcr);
            map.put("spectralCentroid", spectralCentroid);
            map.put("spectralRolloff", spectralRolloff);
            map.put("spectralBandwidth", spectralBandwidth);
            map.put("tempo", tempo);
            map.put("chromaVariance", chromaVariance);
            return map;
        }
    }
    
    /**
     * 频谱特征类 / Spectral features class
     */
    private static class SpectralFeatures {
        double centroid;
        double rolloff;
        double bandwidth;
    }
    
}