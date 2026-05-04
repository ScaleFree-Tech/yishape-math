package com.yishape.lab.music.analysis.basic;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.music.analysis.StandardizedConfidenceCalculator;

import java.util.Map;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

/**
 * 节拍分析器实现 / Beat Analyzer Implementation
 * <p>
 * 基于能量检测和自相关的节拍检测实现。 Beat detection implementation based on energy detection
 * and autocorrelation.
 * </p>
 * <p>
 * 该分析器通过计算音频能量包络、检测峰值并估算节拍速度来识别音乐中的节拍。
 * This analyzer identifies beats in music by calculating audio energy envelope, detecting peaks, and estimating tempo.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class BeatAnalyzerImpl implements IBeatAnalyzer {

    private static final Logger log = LoggerFactory.getLogger(BeatAnalyzerImpl.class);


    // 标准化置信度计算器 / Standardized confidence calculator
    private final StandardizedConfidenceCalculator confidenceCalculator = new StandardizedConfidenceCalculator();

    // 默认参数 / Default parameters
    private static final double DEFAULT_MIN_BPM = 60.0;
    private static final double DEFAULT_MAX_BPM = 200.0;
    private static final double DEFAULT_BEAT_SENSITIVITY = 0.5;
    private static final int DEFAULT_WINDOW_SIZE = 1024;
    private static final int DEFAULT_HOP_SIZE = 512;

    /**
     * 构造节拍分析器 / Construct a Beat Analyzer
     * <p>
     * 初始化节拍分析器，使用默认参数。
     * Initializes the beat analyzer with default parameters.
     * </p>
     */
    public BeatAnalyzerImpl() {
        // Default constructor
    }

    @Override
    public BeatDetectionResult detectBeats(AudioData audioData) throws AudioProcessingException {
        return detectBeats(audioData, getDefaultParameters());
    }

    @Override
    public BeatDetectionResult detectBeats(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }

        // 获取参数 / Get parameters
        double minBpm = (Double) parameters.getOrDefault("minBpm", DEFAULT_MIN_BPM);
        double maxBpm = (Double) parameters.getOrDefault("maxBpm", DEFAULT_MAX_BPM);
        double sensitivity = (Double) parameters.getOrDefault("beatSensitivity", DEFAULT_BEAT_SENSITIVITY);
        int windowSize = (Integer) parameters.getOrDefault("windowSize", DEFAULT_WINDOW_SIZE);
        int hopSize = (Integer) parameters.getOrDefault("hopSize", DEFAULT_HOP_SIZE);

        // 添加参数验证以防止潜在的无限循环或计算问题
        // Add parameter validation to prevent potential infinite loops or calculation issues
        if (windowSize <= 0) {
            windowSize = DEFAULT_WINDOW_SIZE;
        }
        if (hopSize <= 0) {
            hopSize = DEFAULT_HOP_SIZE;
        }
        // 确保hopSize不会大于windowSize，防止计算问题
        // Ensure hopSize is not greater than windowSize to prevent calculation issues
        if (hopSize > windowSize) {
            hopSize = windowSize / 2;
        }

        // 验证BPM范围
        // Validate BPM range
        if (minBpm <= 0) {
            minBpm = DEFAULT_MIN_BPM;
        }
        if (maxBpm <= 0 || maxBpm < minBpm) {
            maxBpm = DEFAULT_MAX_BPM;
        }

        // 验证敏感度参数
        // Validate sensitivity parameter
        if (sensitivity < 0) {
            sensitivity = DEFAULT_BEAT_SENSITIVITY;
        }

        try {
            // 计算能量包络 / Calculate energy envelope
            IVector<Double> energyEnvelope = calculateEnergyEnvelope(audioData.getSamples(), windowSize, hopSize);

            // 检测峰值 / Detect peaks
            double audioDuration = audioData.getDuration();
            List<Double> beatTimes = detectPeaks(energyEnvelope, audioData.getSampleRate(), hopSize, sensitivity, audioDuration);

            // 估算节拍速度 / Estimate tempo
            double tempo = estimateTempoFromBeats(beatTimes, minBpm, maxBpm);

            // 创建结果 / Create result
            BeatDetectionResult result = new BeatDetectionResult();
            result.setBeatTimes(beatTimes.stream().mapToDouble(Double::doubleValue).toArray());
            result.setTempo(tempo);
            result.setConfidence(calculateConfidence(beatTimes, tempo));
            result.setAlgorithm("energy_autocorr");

            return result;

        } catch (Exception e) {
            throw new AudioProcessingException("Error in beat detection: " + e.getMessage(), e);
        }
    }

    @Override
    public double estimateTempo(AudioData audioData) throws AudioProcessingException {
        return estimateTempo(audioData, getDefaultParameters());
    }

    @Override
    public double estimateTempo(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        BeatDetectionResult result = detectBeats(audioData, parameters);
        return result.getTempo();
    }

    @Override
    public String[] getSupportedParameters() {
        return new String[]{"minBpm", "maxBpm", "beatSensitivity", "windowSize", "hopSize"};
    }

    @Override
    public Map<String, Object> getDefaultParameters() {
        Map<String, Object> params = new HashMap<>();
        params.put("minBpm", DEFAULT_MIN_BPM);
        params.put("maxBpm", DEFAULT_MAX_BPM);
        params.put("beatSensitivity", DEFAULT_BEAT_SENSITIVITY);
        params.put("windowSize", DEFAULT_WINDOW_SIZE);
        params.put("hopSize", DEFAULT_HOP_SIZE);
        return params;
    }

    /**
     * 设置分析器参数 / Set analyzer parameters
     *
     * @param parameters 要设置的参数 / Parameters to set
     * @throws AudioProcessingException 参数无效时抛出异常 / Thrown when parameters are
     * invalid
     */
    public void setParameters(Map<String, Object> parameters) throws AudioProcessingException {
        if (parameters == null) {
            return;
        }

        // 验证参数有效性 / Validate parameters
        for (String key : parameters.keySet()) {
            if (!Arrays.asList(getSupportedParameters()).contains(key)) {
                throw new AudioProcessingException("Unsupported parameter: " + key);
            }
        }

        // 这里可以添加参数验证逻辑，但由于当前实现使用方法参数传递，
        // 暂时只做基础验证 / Basic validation for now since current implementation uses method parameters
    }

    /**
     * 计算能量包络 / Calculate energy envelope
     */
    private IVector<Double> calculateEnergyEnvelope(IVector<Double> signal, int windowSize, int hopSize) {
        // 添加保护性检查以防止除零错误和负数计算
        // Add protective checks to prevent division by zero and negative calculations
        if (windowSize <= 0) {
            windowSize = DEFAULT_WINDOW_SIZE;
        }
        if (hopSize <= 0) {
            hopSize = DEFAULT_HOP_SIZE;
        }
        if (hopSize > windowSize) {
            hopSize = windowSize / 2;
        }

        // 确保我们有足够的样本进行处理
        // Ensure we have enough samples for processing
        if (signal.length() < windowSize) {
            // 如果信号太短，返回一个最小的能量包络
            // If signal is too short, return a minimal energy envelope
            IVector<Double> minimalEnvelope = Linalg.zeros(1);
            minimalEnvelope.set(0, 0.0);
            return minimalEnvelope;
        }

        int numFrames = Math.max(1, (signal.length() - windowSize) / hopSize + 1);
        IVector<Double> envelope = Linalg.zeros(numFrames);

        for (int i = 0; i < numFrames; i++) {
            int start = i * hopSize;
            double energy = 0.0;

            // 确保我们不会超出信号边界
            // Ensure we don't go beyond signal boundaries
            int actualWindowSize = Math.min(windowSize, signal.length() - start);
            for (int j = 0; j < actualWindowSize; j++) {
                double sample = signal.get(start + j);
                energy += sample * sample;
            }

            // 防止除零错误
            // Prevent division by zero
            if (actualWindowSize > 0) {
                envelope.set(i, Math.sqrt(energy / actualWindowSize));
            } else {
                envelope.set(i, 0.0);
            }
        }

        return envelope;
    }

    /**
     * 检测峰值 / Detect peaks
     */
    private List<Double> detectPeaks(IVector<Double> envelope, double sampleRate, int hopSize, double sensitivity, double audioDuration) {
        // 添加保护性检查
        // Add protective checks
        if (envelope == null || envelope.length() < 3) {
            return new ArrayList<>();
        }

        if (sampleRate <= 0) {
            sampleRate = 44100.0; // 默认采样率
        }

        if (hopSize <= 0) {
            hopSize = DEFAULT_HOP_SIZE;
        }

        // 使用改进的峰值检测算法
        return detectPeaksImproved(envelope, sampleRate, hopSize, sensitivity, audioDuration);
    }

    /**
     * 改进的峰值检测算法 / Improved peak detection algorithm
     */
    private List<Double> detectPeaksImproved(IVector<Double> envelope, double sampleRate, int hopSize, double sensitivity, double audioDuration) {
        List<Double> peaks = new ArrayList<>();

        // 计算自适应阈值
        double adaptiveThreshold = calculateAdaptiveThreshold(envelope, sensitivity);

        // 多尺度峰值检测
        List<Integer> peakIndices = findPeakIndices(envelope, adaptiveThreshold);

        // 过滤过于接近的峰值
        peakIndices = filterClosePeaks(peakIndices, sampleRate, hopSize);

        // 使用传入的正确音频时长（而不是从envelope计算）
        // Use the correct audio duration passed in (instead of calculating from envelope)

        // 转换为时间并验证
        for (int index : peakIndices) {
            double timeInSeconds = (index * hopSize) / sampleRate;

            // 验证时间在有效范围内
            if (timeInSeconds >= 0 && timeInSeconds <= audioDuration) {
                peaks.add(timeInSeconds);
            } else {
//                log.warn("Warning: Invalid beat time " + timeInSeconds
//                        + "s detected (audio duration: " + audioDuration + "s)");
            }
        }

        // 添加调试信息
        if (!peaks.isEmpty()) {
            double firstBeat = peaks.get(0);
            double lastBeat = peaks.get(peaks.size() - 1);

            log.debug("Beat Detection Debug:");
            log.debug("  Total beats: " + peaks.size());
            log.debug("  First beat: " + String.format("%.3f", firstBeat) + "s");
            log.debug("  Last beat: " + String.format("%.3f", lastBeat) + "s");
            log.debug("  Audio duration: " + String.format("%.3f", audioDuration) + "s");
            log.debug("  Beats per second: " + String.format("%.2f", peaks.size() / audioDuration));
        }

        return peaks;
    }

    /**
     * 计算自适应阈值 / Calculate adaptive threshold
     */
    private double calculateAdaptiveThreshold(IVector<Double> envelope, double sensitivity) {
        if (envelope == null || envelope.length() == 0) {
            return 0.0;
        }

        // 计算统计信息
        double mean = 0.0;
        double max = Double.MIN_VALUE;
        for (int i = 0; i < envelope.length(); i++) {
            double value = envelope.get(i);
            mean += value;
            max = Math.max(max, value);
        }
        mean /= envelope.length();

        // 计算标准差
        double variance = 0.0;
        for (int i = 0; i < envelope.length(); i++) {
            double diff = envelope.get(i) - mean;
            variance += diff * diff;
        }
        variance /= envelope.length();
        double stdDev = Math.sqrt(variance);

        // 自适应阈值计算
        double dynamicRange = max - mean;
        double adaptiveFactor = 0.3 + (1.0 - sensitivity) * 0.4; // 0.3 到 0.7 之间

        return mean + adaptiveFactor * stdDev + 0.1 * dynamicRange;
    }

    /**
     * 查找峰值索引 / Find peak indices
     */
    private List<Integer> findPeakIndices(IVector<Double> envelope, double threshold) {
        List<Integer> peaks = new ArrayList<>();
        int windowSize = 3; // 局部窗口大小

        for (int i = windowSize; i < envelope.length() - windowSize; i++) {
            boolean isPeak = true;
            double centerValue = envelope.get(i);

            // 检查是否超过阈值
            if (centerValue <= threshold) {
                continue;
            }

            // 检查是否为局部最大值
            for (int j = -windowSize; j <= windowSize; j++) {
                if (j != 0 && envelope.get(i + j) >= centerValue) {
                    isPeak = false;
                    break;
                }
            }

            if (isPeak) {
                peaks.add(i);
            }
        }

        return peaks;
    }

    /**
     * 过滤过于接近的峰值 / Filter peaks that are too close
     */
    private List<Integer> filterClosePeaks(List<Integer> peakIndices, double sampleRate, int hopSize) {
        if (peakIndices.isEmpty()) {
            return peakIndices;
        }

        List<Integer> filtered = new ArrayList<>();
        filtered.add(peakIndices.get(0));

        // 最小节拍间隔：对应240 BPM（0.25秒）
        int minPeakDistance = (int) (0.25 * sampleRate / hopSize);

        for (int i = 1; i < peakIndices.size(); i++) {
            int currentPeak = peakIndices.get(i);
            int lastPeak = filtered.get(filtered.size() - 1);

            if (currentPeak - lastPeak >= minPeakDistance) {
                filtered.add(currentPeak);
            }
        }

        return filtered;
    }

    /**
     * 计算阈值 / Calculate threshold
     */
    private double calculateThreshold(IVector<Double> envelope, double sensitivity) {
        // 添加保护性检查
        // Add protective checks
        if (envelope == null || envelope.length() == 0) {
            return 0.0;
        }

        // 限制敏感度参数范围
        // Limit sensitivity parameter range
        sensitivity = Math.max(0.0, Math.min(1.0, sensitivity));

        double mean = 0.0;
        for (int i = 0; i < envelope.length(); i++) {
            // 添加保护性检查以确保不会访问无效索引
            // Add protective checks to ensure we don't access invalid indices
            if (i >= 0 && i < envelope.length()) {
                mean += envelope.get(i);
            }
        }
        mean /= envelope.length();

        double variance = 0.0;
        for (int i = 0; i < envelope.length(); i++) {
            // 添加保护性检查以确保不会访问无效索引
            // Add protective checks to ensure we don't access invalid indices
            if (i >= 0 && i < envelope.length()) {
                double diff = envelope.get(i) - mean;
                variance += diff * diff;
            }
        }
        variance /= envelope.length();
        double stdDev = Math.sqrt(variance);

        return mean + sensitivity * stdDev;
    }

    /**
     * 从节拍时间估算速度 / Estimate tempo from beat times
     */
    private double estimateTempoFromBeats(List<Double> beatTimes, double minBpm, double maxBpm) {
        // 添加保护性检查
        // Add protective checks
        if (beatTimes == null || beatTimes.size() < 2) {
            return 120.0; // 默认值 / Default value
        }

        // 验证BPM范围
        // Validate BPM range
        if (minBpm <= 0) {
            minBpm = DEFAULT_MIN_BPM;
        }
        if (maxBpm <= 0 || maxBpm < minBpm) {
            maxBpm = DEFAULT_MAX_BPM;
        }

        // 使用改进的速度估算算法
        return estimateTempoImproved(beatTimes, minBpm, maxBpm);
    }

    /**
     * 改进的速度估算算法 / Improved tempo estimation algorithm
     */
    private double estimateTempoImproved(List<Double> beatTimes, double minBpm, double maxBpm) {
        // 计算所有间隔
        List<Double> intervals = calculateIntervals(beatTimes);
        if (intervals.isEmpty()) {
            return 120.0;
        }

        // 使用多种方法估算BPM
        double medianBpm = estimateFromMedianInterval(intervals, minBpm, maxBpm);
        double modeBpm = estimateFromModeInterval(intervals, minBpm, maxBpm);
        double autocorrBpm = estimateFromAutocorrelation(beatTimes, minBpm, maxBpm);

        // 选择最可信的BPM
        return selectBestTempo(medianBpm, modeBpm, autocorrBpm, intervals);
    }

    /**
     * 计算节拍间隔 / Calculate beat intervals
     */
    private List<Double> calculateIntervals(List<Double> beatTimes) {
        List<Double> intervals = new ArrayList<>();
        for (int i = 1; i < beatTimes.size(); i++) {
            double interval = beatTimes.get(i) - beatTimes.get(i - 1);
            if (interval > 0.1 && interval < 2.0) { // 过滤异常间隔
                intervals.add(interval);
            }
        }
        return intervals;
    }

    /**
     * 基于中位数间隔估算BPM / Estimate BPM from median interval
     */
    private double estimateFromMedianInterval(List<Double> intervals, double minBpm, double maxBpm) {
        if (intervals.isEmpty()) {
            return 120.0;
        }

        List<Double> sortedIntervals = new ArrayList<>(intervals);
        sortedIntervals.sort(Double::compareTo);
        double medianInterval = sortedIntervals.get(sortedIntervals.size() / 2);

        double bpm = 60.0 / medianInterval;
        return Math.max(minBpm, Math.min(maxBpm, bpm));
    }

    /**
     * 基于众数间隔估算BPM / Estimate BPM from mode interval
     */
    private double estimateFromModeInterval(List<Double> intervals, double minBpm, double maxBpm) {
        if (intervals.isEmpty()) {
            return 120.0;
        }

        // 创建间隔直方图
        Map<Integer, Integer> histogram = new HashMap<>();
        double binSize = 0.05; // 50ms bins

        for (double interval : intervals) {
            int bin = (int) (interval / binSize);
            histogram.put(bin, histogram.getOrDefault(bin, 0) + 1);
        }

        // 找到最频繁的间隔
        int maxCount = 0;
        int modeBin = 0;
        for (Map.Entry<Integer, Integer> entry : histogram.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                modeBin = entry.getKey();
            }
        }

        double modeInterval = modeBin * binSize + binSize / 2;
        if (modeInterval > 0) {
            double bpm = 60.0 / modeInterval;
            return Math.max(minBpm, Math.min(maxBpm, bpm));
        }

        return 120.0;
    }

    /**
     * 基于自相关估算BPM / Estimate BPM from autocorrelation
     */
    private double estimateFromAutocorrelation(List<Double> beatTimes, double minBpm, double maxBpm) {
        if (beatTimes.size() < 4) {
            return 120.0;
        }

        double minInterval = 60.0 / maxBpm;
        double maxInterval = 60.0 / minBpm;
        double bestBpm = 120.0;
        double maxCorrelation = 0.0;

        // 测试不同的BPM候选值
        for (double testBpm = minBpm; testBpm <= maxBpm; testBpm += 1.0) {
            double testInterval = 60.0 / testBpm;
            double correlation = calculateTempoCorrelation(beatTimes, testInterval);

            if (correlation > maxCorrelation) {
                maxCorrelation = correlation;
                bestBpm = testBpm;
            }
        }

        return bestBpm;
    }

    /**
     * 计算节拍速度相关性 / Calculate tempo correlation
     */
    private double calculateTempoCorrelation(List<Double> beatTimes, double testInterval) {
        double correlation = 0.0;
        int matches = 0;
        double tolerance = testInterval * 0.1; // 10% tolerance

        for (int i = 0; i < beatTimes.size(); i++) {
            double expectedTime = beatTimes.get(0) + i * testInterval;

            // 查找最接近的实际节拍
            double minDistance = Double.MAX_VALUE;
            for (double actualTime : beatTimes) {
                double distance = Math.abs(actualTime - expectedTime);
                minDistance = Math.min(minDistance, distance);
            }

            if (minDistance <= tolerance) {
                correlation += 1.0 - (minDistance / tolerance);
                matches++;
            }
        }

        return matches > 0 ? correlation / beatTimes.size() : 0.0;
    }

    /**
     * 选择最佳速度 / Select best tempo 改进版：添加多重假设验证和细分节拍过滤
     */
    private double selectBestTempo(double medianBpm, double modeBpm, double autocorrBpm, List<Double> intervals) {
        // 创建候选BPM列表，包括原始值和可能的细分/倍数
        List<Double> candidates = new ArrayList<>();
        candidates.add(medianBpm);
        candidates.add(modeBpm);
        candidates.add(autocorrBpm);

        // 添加细分和倍数候选（解决细分节拍误判问题）
        addSubdivisionCandidates(candidates, medianBpm);
        addSubdivisionCandidates(candidates, modeBpm);
        addSubdivisionCandidates(candidates, autocorrBpm);

        // 过滤不合理的BPM值
        candidates = filterUnreasonableBpm(candidates);

        // 计算每个候选的综合分数
        double bestBpm = 120.0;
        double bestScore = 0.0;

        for (double candidate : candidates) {
            double score = calculateComprehensiveScore(candidate, intervals);
            if (score > bestScore) {
                bestScore = score;
                bestBpm = candidate;
            }
        }

        return bestBpm;
    }

    /**
     * 添加细分和倍数候选BPM / Add subdivision and multiple candidates
     */
    private void addSubdivisionCandidates(List<Double> candidates, double bpm) {
        if (bpm <= 0) {
            return;
        }

        // 添加1/2倍数（解决双倍速度误判）
        double halfBpm = bpm / 2.0;
        if (halfBpm >= 50.0 && halfBpm <= 180.0) {
            candidates.add(halfBpm);
        }

        // 添加1/3倍数（解决三连音误判）
        double thirdBpm = bpm / 3.0;
        if (thirdBpm >= 50.0 && thirdBpm <= 180.0) {
            candidates.add(thirdBpm);
        }

        // 添加2倍数（但要谨慎，通常是误判）
        double doubleBpm = bpm * 2.0;
        if (doubleBpm >= 60.0 && doubleBpm <= 160.0) {
            candidates.add(doubleBpm);
        }
    }

    /**
     * 过滤不合理的BPM值 / Filter unreasonable BPM values
     */
    private List<Double> filterUnreasonableBpm(List<Double> candidates) {
        List<Double> filtered = new ArrayList<>();

        for (double bpm : candidates) {
            // 流行音乐合理BPM范围：50-180
            if (bpm >= 50.0 && bpm <= 180.0) {
                // 额外检查：避免过于极端的值
                if (!(bpm > 160.0 && bpm < 180.0)) { // 避免160-180的高速区间
                    filtered.add(bpm);
                }
            }
        }

        // 如果过滤后为空，返回默认值
        if (filtered.isEmpty()) {
            filtered.add(120.0);
        }

        return filtered;
    }

    /**
     * 计算综合分数 / Calculate comprehensive score
     */
    private double calculateComprehensiveScore(double bpm, List<Double> intervals) {
        if (intervals.isEmpty() || bpm <= 0) {
            return 0.0;
        }

        // 1. 一致性分数（原有逻辑）
        double consistencyScore = calculateConsistencyScore(bpm, intervals);

        // 2. 流行音乐BPM偏好分数
        double preferenceScore = calculateBpmPreferenceScore(bpm);

        // 3. 间隔分布合理性分数
        double distributionScore = calculateIntervalDistributionScore(bpm, intervals);

        // 综合权重：一致性50%，偏好30%，分布20%
        return consistencyScore * 0.5 + preferenceScore * 0.3 + distributionScore * 0.2;
    }

    /**
     * 计算BPM偏好分数 / Calculate BPM preference score 基于流行音乐常见BPM范围给出偏好分数
     */
    private double calculateBpmPreferenceScore(double bpm) {
        // 流行音乐最常见BPM范围
        if (bpm >= 60.0 && bpm <= 90.0) {
            return 1.0; // 慢歌、抒情歌曲
        } else if (bpm >= 90.0 && bpm <= 130.0) {
            return 0.9; // 中等节拍流行歌
        } else if (bpm >= 130.0 && bpm <= 150.0) {
            return 0.7; // 快节拍流行歌
        } else if (bpm >= 50.0 && bpm <= 60.0) {
            return 0.8; // 很慢的抒情歌
        } else if (bpm >= 150.0 && bpm <= 180.0) {
            return 0.4; // 很快的歌曲，较少见
        } else {
            return 0.1; // 不太可能的BPM范围
        }
    }

    /**
     * 计算间隔分布合理性分数 / Calculate interval distribution reasonableness score
     */
    private double calculateIntervalDistributionScore(double bpm, List<Double> intervals) {
        double expectedInterval = 60.0 / bpm;

        // 计算间隔的变异系数
        double mean = intervals.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        if (mean == 0) {
            return 0.0;
        }

        double variance = intervals.stream()
                .mapToDouble(interval -> Math.pow(interval - expectedInterval, 2))
                .average().orElse(0.0);

        double stdDev = Math.sqrt(variance);
        double coefficientOfVariation = stdDev / expectedInterval;

        // 变异系数越小，分布越合理
        return Math.max(0.0, 1.0 - Math.min(1.0, coefficientOfVariation));
    }

    /**
     * 计算一致性分数 / Calculate consistency score
     */
    private double calculateConsistencyScore(double bpm, List<Double> intervals) {
        if (intervals.isEmpty() || bpm <= 0) {
            return 0.0;
        }

        double expectedInterval = 60.0 / bpm;
        double totalError = 0.0;

        for (double interval : intervals) {
            double error = Math.abs(interval - expectedInterval) / expectedInterval;
            totalError += Math.min(error, 1.0); // Cap error at 100%
        }

        double avgError = totalError / intervals.size();
        return Math.max(0.0, 1.0 - avgError);
    }

    /**
     * 计算置信度 / Calculate confidence
     */
    private double calculateConfidence(List<Double> beatTimes, double tempo) {
        // 添加保护性检查
        // Add protective checks
        if (beatTimes == null || beatTimes.size() < 3) {
            // 如果节拍数量很少，根据数量返回一个低置信度
            // If there are very few beats, return a low confidence based on the count
            if (beatTimes != null) {
                return Math.min(0.5, beatTimes.size() / 10.0);
            }
            return 0.1;
        }

        // 验证节拍速度
        // Validate tempo
        if (tempo <= 0) {
            tempo = 120.0; // 默认值 / Default value
        }

        // 使用改进的置信度计算
        return calculateImprovedConfidence(beatTimes, tempo);
    }

    /**
     * 改进的置信度计算 / Improved confidence calculation
     */
    private double calculateImprovedConfidence(List<Double> beatTimes, double tempo) {
        // 多维度置信度评估
        double consistencyScore = calculateTempoConsistency(beatTimes, tempo);
        double regularityScore = calculateBeatRegularity(beatTimes);
        double densityScore = calculateBeatDensity(beatTimes);
        double stabilityScore = calculateTempoStability(beatTimes, tempo);

        // 使用标准化置信度计算器 / Use standardized confidence calculator
        java.util.Map<String, Double> factors = new java.util.HashMap<>();
        factors.put("consistency", consistencyScore);
        factors.put("regularity", regularityScore);
        factors.put("density", densityScore);
        factors.put("stability", stabilityScore);

        java.util.Map<String, Double> weights = new java.util.HashMap<>();
        weights.put("consistency", 0.35);
        weights.put("regularity", 0.25);
        weights.put("density", 0.2);
        weights.put("stability", 0.2);

        return confidenceCalculator.calculateWeightedConfidence(factors, weights);
    }

    /**
     * 计算节拍速度一致性 / Calculate tempo consistency
     */
    private double calculateTempoConsistency(List<Double> beatTimes, double tempo) {
        double expectedInterval = 60.0 / tempo;
        double totalError = 0.0;
        int validIntervals = 0;

        for (int i = 1; i < beatTimes.size(); i++) {
            double actualInterval = beatTimes.get(i) - beatTimes.get(i - 1);
            if (actualInterval > 0 && expectedInterval > 0) {
                double error = Math.abs(actualInterval - expectedInterval) / expectedInterval;
                totalError += Math.min(error, 2.0); // Cap error at 200%
                validIntervals++;
            }
        }

        if (validIntervals == 0) {
            return 0.1;
        }

        double avgError = totalError / validIntervals;
        return Math.max(0.0, 1.0 - (avgError / 2.0)); // Normalize by max error
    }

    /**
     * 计算节拍规律性 / Calculate beat regularity
     */
    private double calculateBeatRegularity(List<Double> beatTimes) {
        if (beatTimes.size() < 4) {
            return 0.5;
        }

        List<Double> intervals = calculateIntervals(beatTimes);
        if (intervals.isEmpty()) {
            return 0.1;
        }

        // 计算间隔的变异系数
        double mean = intervals.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        if (mean == 0) {
            return 0.1;
        }

        double variance = intervals.stream()
                .mapToDouble(interval -> Math.pow(interval - mean, 2))
                .average().orElse(0.0);

        double stdDev = Math.sqrt(variance);
        double coefficientOfVariation = stdDev / mean;

        // 变异系数越小，规律性越好
        return Math.max(0.0, 1.0 - Math.min(1.0, coefficientOfVariation));
    }

    /**
     * 计算节拍密度 / Calculate beat density
     */
    private double calculateBeatDensity(List<Double> beatTimes) {
        if (beatTimes.size() < 2) {
            return 0.1;
        }

        double totalDuration = beatTimes.get(beatTimes.size() - 1) - beatTimes.get(0);
        if (totalDuration <= 0) {
            return 0.1;
        }

        double beatsPerSecond = (beatTimes.size() - 1) / totalDuration;

        // 理想的节拍密度范围：1-4 beats/second (60-240 BPM)
        double idealMin = 1.0;
        double idealMax = 4.0;

        if (beatsPerSecond >= idealMin && beatsPerSecond <= idealMax) {
            return 1.0;
        } else if (beatsPerSecond < idealMin) {
            return Math.max(0.2, beatsPerSecond / idealMin);
        } else {
            return Math.max(0.2, idealMax / beatsPerSecond);
        }
    }

    /**
     * 计算节拍速度稳定性 / Calculate tempo stability
     */
    private double calculateTempoStability(List<Double> beatTimes, double globalTempo) {
        if (beatTimes.size() < 6) {
            return 0.5;
        }

        // 分段计算局部速度
        int segmentSize = Math.max(3, beatTimes.size() / 4);
        List<Double> localTempos = new ArrayList<>();

        for (int i = 0; i <= beatTimes.size() - segmentSize; i += segmentSize / 2) {
            int endIndex = Math.min(i + segmentSize, beatTimes.size());
            List<Double> segment = beatTimes.subList(i, endIndex);

            if (segment.size() >= 3) {
                double localTempo = estimateLocalTempo(segment);
                if (localTempo > 0) {
                    localTempos.add(localTempo);
                }
            }
        }

        if (localTempos.isEmpty()) {
            return 0.5;
        }

        // 计算局部速度与全局速度的一致性
        double totalDeviation = 0.0;
        for (double localTempo : localTempos) {
            double deviation = Math.abs(localTempo - globalTempo) / globalTempo;
            totalDeviation += Math.min(deviation, 1.0); // Cap at 100% deviation
        }

        double avgDeviation = totalDeviation / localTempos.size();
        return Math.max(0.0, 1.0 - avgDeviation);
    }

    /**
     * 估算局部速度 / Estimate local tempo
     */
    private double estimateLocalTempo(List<Double> segmentTimes) {
        if (segmentTimes.size() < 2) {
            return 0.0;
        }

        List<Double> intervals = new ArrayList<>();
        for (int i = 1; i < segmentTimes.size(); i++) {
            double interval = segmentTimes.get(i) - segmentTimes.get(i - 1);
            if (interval > 0) {
                intervals.add(interval);
            }
        }

        if (intervals.isEmpty()) {
            return 0.0;
        }

        double avgInterval = intervals.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        return avgInterval > 0 ? 60.0 / avgInterval : 0.0;
    }

    /**
     * 计算动态权重 / Calculate dynamic weights
     */
    private double[] calculateDynamicWeights(int beatCount, double consistencyScore) {
        double[] weights = new double[4]; // [consistency, regularity, density, stability]

        // 基础权重
        weights[0] = 0.4; // consistency
        weights[1] = 0.25; // regularity
        weights[2] = 0.2; // density
        weights[3] = 0.15; // stability

        // 根据节拍数量调整权重
        if (beatCount < 10) {
            // 节拍少时，更重视密度
            weights[2] += 0.1;
            weights[3] -= 0.1;
        } else if (beatCount > 50) {
            // 节拍多时，更重视稳定性
            weights[3] += 0.1;
            weights[2] -= 0.1;
        }

        // 根据一致性分数调整权重
        if (consistencyScore < 0.5) {
            // 一致性差时，降低其权重
            weights[0] -= 0.1;
            weights[1] += 0.1;
        }

        // 确保权重和为1
        double sum = weights[0] + weights[1] + weights[2] + weights[3];
        for (int i = 0; i < weights.length; i++) {
            weights[i] /= sum;
        }

        return weights;
    }
}
