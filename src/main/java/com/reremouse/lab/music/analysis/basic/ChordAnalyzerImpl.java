package com.reremouse.lab.music.analysis.basic;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.music.theory.ChordTheory;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.music.analysis.StandardizedConfidenceCalculator;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.util.HashSet;

/**
 * 和弦分析器实现 / Chord Analyzer Implementation
 * <p>
 * 基于色度特征和模板匹配的和弦检测实现。
 * Chord detection implementation based on chroma features and template matching.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ChordAnalyzerImpl implements IChordAnalyzer {

    // 默认参数 / Default parameters
    private static final int DEFAULT_WINDOW_SIZE = 4096;
    private static final int DEFAULT_HOP_SIZE = 2048;
    private static final double DEFAULT_CHORD_THRESHOLD = 0.5;
    private static final double DEFAULT_SEGMENT_LENGTH = 1.0; // 秒 / seconds

    // 标准化置信度计算器 / Standardized confidence calculator
    private final StandardizedConfidenceCalculator confidenceCalculator = new StandardizedConfidenceCalculator();
    
    // 色度特征维度 / Chroma feature dimensions
    private static final int CHROMA_BINS = 12;

    // 基本和弦模板 / Basic chord templates
    private static final Map<String, double[]> CHORD_TEMPLATES = new HashMap<>();

    static {
        // 大三和弦模板 / Major triad templates
        CHORD_TEMPLATES.put("major", new double[]{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
        // 小三和弦模板 / Minor triad templates
        CHORD_TEMPLATES.put("minor", new double[]{1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
        // 属七和弦模板 / Dominant 7th chord templates
        CHORD_TEMPLATES.put("dom7", new double[]{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0});
        // 大七和弦模板 / Major 7th chord templates
        CHORD_TEMPLATES.put("maj7", new double[]{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
        // 小七和弦模板 / Minor 7th chord templates
        CHORD_TEMPLATES.put("min7", new double[]{1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0});
        
        // 流行音乐常用和弦模板 / Popular music chord templates
        // 流行大三和弦（增强根音和五度音） / Pop major triad (enhanced root and fifth)
        CHORD_TEMPLATES.put("pop_major", new double[]{1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0});
        // 流行小三和弦（增强根音和五度音） / Pop minor triad (enhanced root and fifth)
        CHORD_TEMPLATES.put("pop_minor", new double[]{1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0});
        // 流行属七和弦（常见于流行音乐） / Pop dominant 7th (common in pop music)
        CHORD_TEMPLATES.put("pop_dom7", new double[]{1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.9, 0.0, 0.0, 0.7, 0.0});
        // 流行小七和弦（常见于R&B和流行音乐） / Pop minor 7th (common in R&B and pop)
        CHORD_TEMPLATES.put("pop_min7", new double[]{1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.7, 0.0});
        
        // 新增和弦类型 / Additional chord types
        // 减三和弦模板 / Diminished triad templates
        CHORD_TEMPLATES.put("dim", new double[]{1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        // 增三和弦模板 / Augmented triad templates
        CHORD_TEMPLATES.put("aug", new double[]{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0});
        // 挂四和弦模板 / Suspended 4th chord templates
        CHORD_TEMPLATES.put("sus4", new double[]{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
        // 挂二和弦模板 / Suspended 2nd chord templates
        CHORD_TEMPLATES.put("sus2", new double[]{1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
        // 九和弦模板 / 9th chord templates
        CHORD_TEMPLATES.put("9", new double[]{1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0});
        // 大九和弦模板 / Major 9th chord templates
        CHORD_TEMPLATES.put("maj9", new double[]{1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
        // 小九和弦模板 / Minor 9th chord templates
        CHORD_TEMPLATES.put("min9", new double[]{1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0});
        // 十一和弦模板 / 11th chord templates
        CHORD_TEMPLATES.put("11", new double[]{1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0});
        // 十三和弦模板 / 13th chord templates
        CHORD_TEMPLATES.put("13", new double[]{1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0});
        
        // 流行音乐特殊和弦 / Special pop music chords
        // add9和弦（流行音乐中常见） / add9 chord (common in pop music)
        CHORD_TEMPLATES.put("add9", new double[]{1.0, 0.0, 0.6, 0.0, 0.8, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0});
        // 6和弦（爵士和流行音乐） / 6th chord (jazz and pop music)
        CHORD_TEMPLATES.put("6", new double[]{1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.9, 0.0, 0.7, 0.0, 0.0});
        // m6和弦 / minor 6th chord
        CHORD_TEMPLATES.put("m6", new double[]{1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.9, 0.0, 0.7, 0.0, 0.0});
    }

    private final KeyAnalyzerImpl keyAnalyzer;

    public ChordAnalyzerImpl() {
        this.keyAnalyzer = new KeyAnalyzerImpl();
    }

    @Override
    public List<ChordDetectionResult> detectChords(AudioData audioData) throws AudioProcessingException {
        return detectChords(audioData, getDefaultParameters());
    }

    @Override
    public List<ChordDetectionResult> detectChords(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }

        // 获取参数 / Get parameters
        // Fix: Safely convert parameter to double to prevent ClassCastException
        double segmentLength = getDoubleParameter(parameters, "segmentLength", DEFAULT_SEGMENT_LENGTH);

        try {
            List<ChordDetectionResult> results = new ArrayList<>();
            double duration = audioData.getDuration();

            // 分段分析 / Analyze in segments
            // Fix: Added check to prevent infinite loop when segmentLength is 0 or negative
            if (segmentLength <= 0) {
                segmentLength = DEFAULT_SEGMENT_LENGTH; // Use default if invalid
            }
            
            for (double startTime = 0; startTime < duration && duration > 0; startTime += segmentLength) {
                double endTime = Math.min(startTime + segmentLength, duration);
                ChordDetectionResult result = detectChordInSegment(audioData, startTime, endTime, parameters);
                if (result != null) {
                    results.add(result);
                }
                // Fix: Prevent infinite loop by ensuring progress when startTime doesn't advance
                if (endTime <= startTime) {
                    break;
                }
            }

            return results;

        } catch (Exception e) {
            throw new AudioProcessingException("Error in chord detection: " + e.getMessage(), e);
        }
    }

    @Override
    public ChordDetectionResult detectChordInSegment(AudioData audioData, double startTime, double endTime) throws AudioProcessingException {
        return detectChordInSegment(audioData, startTime, endTime, getDefaultParameters());
    }

    @Override
    public ChordDetectionResult detectChordInSegment(AudioData audioData, double startTime, double endTime, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }

        try {
            // 提取音频段 / Extract audio segment
            AudioData segment = extractAudioSegment(audioData, startTime, endTime);

            // 计算色度特征 / Calculate chroma features
            double[] chromaFeatures = keyAnalyzer.analyzeChromaFeatures(segment, parameters);

            // 获取调性上下文参数 / Get key context parameters
            String detectedKey = null;
            String keyScaleType = null;
            if (parameters != null) {
                detectedKey = (String) parameters.get("detectedKey");
                keyScaleType = (String) parameters.get("keyScaleType");
            }
            
            // 检测和弦 / Detect chord
            ChordMatchResult bestMatch = findBestChordMatch(chromaFeatures, detectedKey, keyScaleType);

            // 创建结果 / Create result
            // Fix: Safely convert parameter to double to prevent ClassCastException
            double chordThreshold = getDoubleParameter(parameters, "chordThreshold", DEFAULT_CHORD_THRESHOLD);
            if (bestMatch.confidence >= chordThreshold) {
                ChordDetectionResult result = new ChordDetectionResult();
                result.setChord(bestMatch.chordName);
                result.setStartTime(startTime);
                result.setEndTime(endTime);
                result.setConfidence(bestMatch.confidence);
                result.setChromaFeatures(chromaFeatures);
                result.setAlgorithm("chroma_template");
                return result;
            }

            return null; // 置信度不够 / Confidence too low

        } catch (Exception e) {
            throw new AudioProcessingException("Error in chord segment detection: " + e.getMessage(), e);
        }
    }

    @Override
    public String[] getSupportedParameters() {
        return new String[]{"windowSize", "hopSize", "chordThreshold", "segmentLength", "detectedKey", "keyScaleType"};
    }

    @Override
    public Map<String, Object> getDefaultParameters() {
        Map<String, Object> params = new HashMap<>();
        params.put("windowSize", DEFAULT_WINDOW_SIZE);
        params.put("hopSize", DEFAULT_HOP_SIZE);
        params.put("chordThreshold", DEFAULT_CHORD_THRESHOLD);
        params.put("segmentLength", DEFAULT_SEGMENT_LENGTH);
        return params;
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
        // 暂时只做基础验证 / Basic validation for now since current implementation uses method parameters
    }

    /**
     * 提取音频段 / Extract audio segment
     */
    private AudioData extractAudioSegment(AudioData audioData, double startTime, double endTime) {
        double sampleRate = audioData.getSampleRate();
        int startSample = (int) (startTime * sampleRate);
        int endSample = (int) (endTime * sampleRate);

        IVector<Double> originalSamples = audioData.getSamples();
        
        // 添加保护性检查
        // Add protective checks
        if (startSample < 0) {
            startSample = 0;
        }
        if (endSample > originalSamples.length()) {
            endSample = originalSamples.length();
        }
        if (startSample >= endSample) {
            // 返回一个最小的音频段
            // Return a minimal audio segment
            return new AudioData(Linalg.zeros(1), sampleRate, audioData.getChannels(),
                            audioData.getBitDepth(), audioData.getFormat());
        }

        int segmentLength = endSample - startSample;

        IVector<Double> segmentSamples = Linalg.zeros(segmentLength);
        for (int i = 0; i < segmentLength; i++) {
            segmentSamples.set(i, originalSamples.get(startSample + i));
        }

        return new AudioData(segmentSamples, sampleRate, audioData.getChannels(),
                        audioData.getBitDepth(), audioData.getFormat());
    }

    /**
     * 寻找最佳和弦匹配 / Find best chord match
     */
    private ChordMatchResult findBestChordMatch(double[] chromaFeatures) {
        return findBestChordMatch(chromaFeatures, null, null);
    }

    /**
     * 寻找最佳和弦匹配（带调性上下文）/ Find best chord match with key context
     */
    private ChordMatchResult findBestChordMatch(double[] chromaFeatures, String detectedKey, String keyScaleType) {
        // 添加保护性检查
        // Add protective checks
        if (chromaFeatures == null || chromaFeatures.length != CHROMA_BINS) {
            return new ChordMatchResult("N", 0.0); // N表示无和弦 / N means no chord
        }
        
        // 检查色度特征的总能量
        // Check total energy of chroma features
        double totalEnergy = 0.0;
        for (double value : chromaFeatures) {
            totalEnergy += value;
        }
        
        // 如果总能量太低，返回低置信度
        // If total energy is too low, return low confidence
        if (totalEnergy < 0.1) {
            return new ChordMatchResult("N", 0.0);
        }
        
        String[] noteNames = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

        double bestScore = -1.0;
        String bestChord = "N"; // N表示无和弦 / N means no chord
        String bestChordType = "";

        // 获取调性的自然音阶和弦（如果提供了调性上下文）/ Get diatonic chords for key if key context provided
        Set<String> diatonicChords = (detectedKey != null && keyScaleType != null) ? 
            getDiatonicChordsForKey(detectedKey, keyScaleType) : new HashSet<>();

        // 测试所有根音和和弦类型 / Test all roots and chord types
        for (String chordType : CHORD_TEMPLATES.keySet()) {
            double[] template = CHORD_TEMPLATES.get(chordType);
            
            // 添加保护性检查
            // Add protective checks
            if (template == null || template.length != CHROMA_BINS) {
                continue;
            }

            for (int root = 0; root < 12; root++) {
                String chordName = noteNames[root] + getChordSuffix(chordType);
                
                // 计算基础匹配分数 / Calculate base matching score
                double score = calculateImprovedTemplateMatch(chromaFeatures, template, root);
                
                // 应用调性上下文权重（如果可用）/ Apply key context weighting if available
                if (!diatonicChords.isEmpty()) {
                    if (diatonicChords.contains(chordName)) {
                        // 自然音阶和弦获得奖励权重 / Diatonic chords get bonus weight
                        score *= 1.3;
                    } else {
                        // 非自然音阶和弦获得惩罚 / Non-diatonic chords get penalty
                        score *= 0.8;
                    }
                }
                
                if (score > bestScore) {
                    bestScore = score;
                    bestChord = chordName;
                    bestChordType = chordType;
                }
            }
        }

        // 改进置信度计算
        // Improved confidence calculation
        double improvedConfidence = calculateImprovedChordConfidence(chromaFeatures, bestScore, bestChordType);
        
        // 和弦后处理：优化复杂和弦识别
        // Chord post-processing: optimize complex chord recognition
        ChordMatchResult postProcessedResult = postProcessChordSelection(bestChord, bestScore, improvedConfidence, chromaFeatures);
        
        return postProcessedResult;
    }

    /**
     * 和弦后处理选择 / Post-process chord selection
     * 优化流行音乐和弦识别和复杂和弦简化
     */
    private ChordMatchResult postProcessChordSelection(String bestChord, double bestScore, double confidence, double[] chromaFeatures) {
        // 如果是流行音乐和弦，直接返回但简化名称
        if (bestChord.contains("pop_")) {
            String simplifiedChord = bestChord.replace("pop_", "");
            return new ChordMatchResult(simplifiedChord, confidence * 1.1); // 提升流行音乐和弦置信度
        }
        
        // 检查是否应该简化复杂和弦
        if (shouldSimplifyChord(bestChord, confidence)) {
            String simplifiedChord = simplifyComplexChord(bestChord);
            if (!simplifiedChord.equals(bestChord)) {
                // 重新计算简化和弦的置信度
                double simplifiedConfidence = confidence * 0.9; // 略微降低置信度以反映简化
                return new ChordMatchResult(simplifiedChord, simplifiedConfidence);
            }
        }
        
        // 检查是否有更适合的流行音乐替代和弦
        String popAlternative = findPopMusicAlternative(bestChord, chromaFeatures);
        if (popAlternative != null && !popAlternative.equals(bestChord)) {
            return new ChordMatchResult(popAlternative, confidence * 1.05);
        }
        
        return new ChordMatchResult(bestChord, confidence);
    }
    
    /**
     * 判断是否应该简化复杂和弦 / Determine if complex chord should be simplified
     */
    private boolean shouldSimplifyChord(String chord, double confidence) {
        // 如果置信度较低且是复杂和弦，考虑简化
        if (confidence < 0.6) {
            return chord.contains("11") || chord.contains("13") || 
                   chord.contains("maj9") || chord.contains("min9");
        }
        return false;
    }
    
    /**
     * 简化复杂和弦 / Simplify complex chord
     */
    private String simplifyComplexChord(String chord) {
        // 将复杂和弦简化为基础和弦
        if (chord.contains("maj9")) {
            return chord.replace("maj9", "maj7");
        } else if (chord.contains("min9")) {
            return chord.replace("min9", "m7");
        } else if (chord.contains("11")) {
            return chord.replace("11", "7");
        } else if (chord.contains("13")) {
            return chord.replace("13", "7");
        }
        return chord;
    }
    
    /**
     * 寻找流行音乐替代和弦 / Find pop music alternative chord
     */
    private String findPopMusicAlternative(String chord, double[] chromaFeatures) {
        // 如果是基础大三和弦，检查是否更适合流行音乐版本
        if (chord.endsWith("") && !chord.contains("m") && !chord.contains("7")) {
            // 检查色度特征是否符合流行音乐特征
            if (hasPopMusicCharacteristics(chromaFeatures)) {
                return chord; // 保持简洁，不添加"pop"前缀
            }
        }
        // 如果是小三和弦
        else if (chord.endsWith("m") && !chord.contains("7")) {
            if (hasPopMusicCharacteristics(chromaFeatures)) {
                return chord; // 保持简洁
            }
        }
        return null;
    }
    
    /**
     * 检查是否具有流行音乐特征 / Check if has pop music characteristics
     */
    private boolean hasPopMusicCharacteristics(double[] chromaFeatures) {
        // 检查根音和五度音是否突出（流行音乐特征）
        double maxValue = 0.0;
        int maxIndex = 0;
        for (int i = 0; i < chromaFeatures.length; i++) {
            if (chromaFeatures[i] > maxValue) {
                maxValue = chromaFeatures[i];
                maxIndex = i;
            }
        }
        
        // 检查五度音（距离根音7个半音）是否也比较突出
        int fifthIndex = (maxIndex + 7) % 12;
        double fifthValue = chromaFeatures[fifthIndex];
        
        // 如果根音和五度音都比较突出，认为具有流行音乐特征
        return maxValue > 0.3 && fifthValue > 0.2 && (fifthValue / maxValue) > 0.5;
    }

    /**
     * 改进的模板匹配计算 / Improved template match calculation
     */
    private double calculateImprovedTemplateMatch(double[] chroma, double[] template, int rootShift) {
        // 添加保护性检查
        // Add protective checks
        if (chroma == null || template == null) {
            return 0.0;
        }
        
        if (chroma.length != CHROMA_BINS || template.length != CHROMA_BINS) {
            return 0.0;
        }
        
        // 检查rootShift是否有效
        // Check if rootShift is valid
        if (rootShift < 0 || rootShift >= CHROMA_BINS) {
            rootShift = 0;
        }

        // 归一化色度特征
        // Normalize chroma features
        double[] normalizedChroma = normalizeChromaVector(chroma);
        
        double dotProduct = 0.0;
        double chromaNorm = 0.0;
        double templateNorm = 0.0;

        for (int i = 0; i < CHROMA_BINS; i++) {
            int templateIndex = (i - rootShift + CHROMA_BINS) % CHROMA_BINS;
            dotProduct += normalizedChroma[i] * template[templateIndex];
            chromaNorm += normalizedChroma[i] * normalizedChroma[i];
            templateNorm += template[templateIndex] * template[templateIndex];
        }

        // 计算余弦相似度 / Calculate cosine similarity
        // 添加保护性检查以防止除零错误
        // Add protective checks to prevent division by zero
        if (chromaNorm > 0 && templateNorm > 0) {
            double cosineSimilarity = dotProduct / (Math.sqrt(chromaNorm) * Math.sqrt(templateNorm));
            
            // 添加权重调整，考虑和弦复杂度
            // Add weight adjustment considering chord complexity
            double complexityWeight = getChordComplexityWeight(template);
            
            return cosineSimilarity * complexityWeight;
        }

        return 0.0;
    }

    /**
     * 归一化色度向量 / Normalize chroma vector
     */
    private double[] normalizeChromaVector(double[] chroma) {
        double[] normalized = new double[chroma.length];
        double sum = 0.0;
        
        for (double value : chroma) {
            sum += value;
        }
        
        if (sum > 0) {
            for (int i = 0; i < chroma.length; i++) {
                normalized[i] = chroma[i] / sum;
            }
        } else {
            // 如果总和为0，返回均匀分布
            // If sum is 0, return uniform distribution
            for (int i = 0; i < chroma.length; i++) {
                normalized[i] = 1.0 / chroma.length;
            }
        }
        
        return normalized;
    }

    /**
     * 获取和弦复杂度权重 / Get chord complexity weight
     */
    private double getChordComplexityWeight(double[] template) {
        int noteCount = 0;
        for (double value : template) {
            if (value > 0.5) {
                noteCount++;
            }
        }
        
        // 根据和弦音符数量调整权重
        // Adjust weight based on number of notes in chord
        switch (noteCount) {
            case 3: return 1.0;  // 三和弦 / Triads
            case 4: return 0.95; // 七和弦 / 7th chords
            case 5: return 0.9;  // 九和弦 / 9th chords
            case 6: return 0.85; // 十一和弦 / 11th chords
            default: return 0.8; // 复杂和弦 / Complex chords
        }
    }

    /**
     * 改进的和弦置信度计算 / Improved chord confidence calculation
     */
    private double calculateImprovedChordConfidence(double[] chromaFeatures, double matchScore, String chordType) {
        // 使用标准化置信度计算器 / Use standardized confidence calculator
        java.util.Map<String, Double> factors = new java.util.HashMap<>();
        factors.put("matchScore", Math.max(0.0, Math.min(1.0, matchScore)));
        factors.put("clarity", calculateChromaClarity(chromaFeatures));
        factors.put("concentration", calculateEnergyConcentration(chromaFeatures));
        factors.put("typeConfidence", getChordTypeConfidence(chordType));
        
        java.util.Map<String, Double> weights = new java.util.HashMap<>();
        weights.put("matchScore", 0.5);
        weights.put("clarity", 0.2);
        weights.put("concentration", 0.2);
        weights.put("typeConfidence", 0.1);
        
        return confidenceCalculator.calculateWeightedConfidence(factors, weights);
    }

    /**
     * 计算色度特征清晰度 / Calculate chroma clarity
     */
    private double calculateChromaClarity(double[] chromaFeatures) {
        if (chromaFeatures == null || chromaFeatures.length == 0) {
            return 0.0;
        }
        
        // 计算最大值与平均值的比率
        // Calculate ratio of max value to average
        double max = 0.0;
        double sum = 0.0;
        
        for (double value : chromaFeatures) {
            max = Math.max(max, value);
            sum += value;
        }
        
        double average = sum / chromaFeatures.length;
        
        if (average > 0) {
            return Math.min(1.0, max / average / 3.0); // 归一化到[0,1]
        }
        
        return 0.0;
    }

    /**
     * 计算能量集中度 / Calculate energy concentration
     */
    private double calculateEnergyConcentration(double[] chromaFeatures) {
        if (chromaFeatures == null || chromaFeatures.length == 0) {
            return 0.0;
        }
        
        // 计算前3个最强音符的能量占比
        // Calculate energy ratio of top 3 strongest notes
        double[] sorted = chromaFeatures.clone();
        java.util.Arrays.sort(sorted);
        
        double topThreeSum = 0.0;
        double totalSum = 0.0;
        
        for (int i = 0; i < sorted.length; i++) {
            totalSum += sorted[i];
            if (i >= sorted.length - 3) {
                topThreeSum += sorted[i];
            }
        }
        
        if (totalSum > 0) {
            return topThreeSum / totalSum;
        }
        
        return 0.0;
    }

    /**
     * 获取和弦类型置信度 / Get chord type confidence
     */
    private double getChordTypeConfidence(String chordType) {
        // 根据和弦类型的常见程度返回置信度调整
        // Return confidence adjustment based on chord type commonality
        switch (chordType) {
            // 流行音乐和弦（最高权重） / Pop music chords (highest weight)
            case "pop_major":
            case "pop_minor":
                return 1.1;  // 流行音乐最常见 / Most common in pop music
            case "pop_dom7":
            case "pop_min7":
                return 1.05; // 流行音乐常见七和弦 / Common 7th chords in pop
            case "add9":
            case "6":
                return 1.0;  // 流行音乐特色和弦 / Characteristic pop chords
                
            // 基础和弦 / Basic chords
            case "major":
            case "minor":
                return 0.95; // 基础三和弦 / Basic triads
            case "dom7":
            case "maj7":
            case "min7":
                return 0.9;  // 常见七和弦 / Common 7th chords
            case "sus4":
            case "sus2":
                return 0.85; // 挂和弦 / Suspended chords
            case "m6":
                return 0.8;  // 小六和弦 / Minor 6th chord
            case "dim":
            case "aug":
                return 0.7;  // 不太常见 / Less common
            case "9":
            case "maj9":
            case "min9":
                return 0.6;  // 复杂九和弦 / Complex 9th chords
            case "11":
            case "13":
                return 0.5;  // 非常复杂 / Very complex
            default:
                return 0.4;  // 未知和弦类型 / Unknown chord types
        }
    }

    /**
     * 获取调性的自然音阶和弦 / Get diatonic chords for a key
     */
    private Set<String> getDiatonicChordsForKey(String keyName, String scaleType) {
        Set<String> diatonicChords = new HashSet<>();
        
        if (keyName == null || scaleType == null) {
            return diatonicChords;
        }
        
        // 将调性名称转换为音符索引 / Convert key name to note index
        String[] noteNames = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
        int keyIndex = -1;
        
        // 处理等音调（D# -> Eb）/ Handle enharmonic equivalents
        if (keyName.equals("D#") && scaleType.equals("major")) {
            keyName = "Eb";
        } else if (keyName.equals("A#") && scaleType.equals("major")) {
            keyName = "Bb";
        } else if (keyName.equals("G#") && scaleType.equals("major")) {
            keyName = "Ab";
        } else if (keyName.equals("C#") && scaleType.equals("major")) {
            keyName = "Db";
        }
        
        for (int i = 0; i < noteNames.length; i++) {
            if (noteNames[i].equals(keyName)) {
                keyIndex = i;
                break;
            }
        }
        
        if (keyIndex == -1) {
            return diatonicChords;
        }
        
        if (scaleType.equals("major")) {
            // 大调的自然音阶和弦（I, ii, iii, IV, V, vi, vii°）
            int[] diatonicChordRoots = {0, 2, 4, 5, 7, 9, 11}; // 相对于调性的音级
            String[] chordTypes = {"", "m", "m", "", "", "m", "dim"};
            
            for (int i = 0; i < diatonicChordRoots.length; i++) {
                int chordRootIndex = (keyIndex + diatonicChordRoots[i]) % 12;
                String chordRoot = noteNames[chordRootIndex];
                String chordType = chordTypes[i];
                diatonicChords.add(chordRoot + chordType);
                
                // 也添加七和弦版本 / Also add 7th chord versions
                if (i == 0 || i == 3) { // I和IV的大七和弦
                    diatonicChords.add(chordRoot + "maj7");
                } else if (i == 4) { // V的属七和弦
                    diatonicChords.add(chordRoot + "7");
                } else if (i == 1 || i == 2 || i == 5) { // ii, iii, vi的小七和弦
                    diatonicChords.add(chordRoot + "m7");
                }
            }
        } else if (scaleType.equals("minor")) {
            // 小调的自然音阶和弦（i, ii°, III, iv, v, VI, VII）
            int[] diatonicChordRoots = {0, 2, 3, 5, 7, 8, 10};
            String[] chordTypes = {"m", "dim", "", "m", "m", "", ""};
            
            for (int i = 0; i < diatonicChordRoots.length; i++) {
                int chordRootIndex = (keyIndex + diatonicChordRoots[i]) % 12;
                String chordRoot = noteNames[chordRootIndex];
                String chordType = chordTypes[i];
                diatonicChords.add(chordRoot + chordType);
                
                // 也添加七和弦版本
                if (i == 0) { // i的小小七和弦
                    diatonicChords.add(chordRoot + "m7");
                } else if (i == 3 || i == 4) { // iv和v的小七和弦
                    diatonicChords.add(chordRoot + "m7");
                } else if (i == 5) { // VI的大七和弦
                    diatonicChords.add(chordRoot + "maj7");
                } else if (i == 6) { // VII的属七和弦
                    diatonicChords.add(chordRoot + "7");
                }
            }
        }
        
        return diatonicChords;
    }

    /**
     * 获取和弦后缀 / Get chord suffix
     */
    private String getChordSuffix(String chordType) {
        switch (chordType) {
            // 基础和弦 / Basic chords
            case "major": return "";
            case "minor": return "m";
            case "dom7": return "7";
            case "maj7": return "maj7";
            case "min7": return "m7";
            case "dim": return "dim";
            case "aug": return "aug";
            case "sus4": return "sus4";
            case "sus2": return "sus2";
            case "9": return "9";
            case "maj9": return "maj9";
            case "min9": return "min9";
            case "11": return "11";
            case "13": return "13";
            
            // 流行音乐和弦（简化后缀） / Pop music chords (simplified suffixes)
            case "pop_major": return "";
            case "pop_minor": return "m";
            case "pop_dom7": return "7";
            case "pop_min7": return "m7";
            case "add9": return "add9";
            case "6": return "6";
            case "m6": return "m6";
            
            default: return "";
        }
    }

    /**
     * 和弦匹配结果内部类 / Chord match result inner class
     */
    private static class ChordMatchResult {
        final String chordName;
        final double confidence;

        ChordMatchResult(String chordName, double confidence) {
            this.chordName = chordName;
            this.confidence = confidence;
        }
    }

    /**
     * 安全地从参数中获取double值 / Safely get double value from parameters
     */
    private double getDoubleParameter(Map<String, Object> parameters, String key, double defaultValue) {
        if (parameters == null) {
            return defaultValue;
        }
        
        Object value = parameters.get(key);
        if (value == null) {
            return defaultValue;
        }
        
        // Handle different numeric types
        if (value instanceof Double) {
            return (Double) value;
        } else if (value instanceof Integer) {
            return ((Integer) value).doubleValue();
        } else if (value instanceof Float) {
            return ((Float) value).doubleValue();
        } else if (value instanceof Long) {
            return ((Long) value).doubleValue();
        } else if (value instanceof String) {
            try {
                return Double.parseDouble((String) value);
            } catch (NumberFormatException e) {
                return defaultValue;
            }
        } else {
            // Try to convert to double
            try {
                return ((Number) value).doubleValue();
            } catch (Exception e) {
                return defaultValue;
            }
        }
    }

    /**
     * 安全地从参数中获取int值 / Safely get int value from parameters
     */
    private int getIntegerParameter(Map<String, Object> parameters, String key, int defaultValue) {
        if (parameters == null) {
            return defaultValue;
        }
        
        Object value = parameters.get(key);
        if (value == null) {
            return defaultValue;
        }
        
        // Handle different numeric types
        if (value instanceof Integer) {
            return (Integer) value;
        } else if (value instanceof Double) {
            return ((Double) value).intValue();
        } else if (value instanceof Float) {
            return ((Float) value).intValue();
        } else if (value instanceof Long) {
            return ((Long) value).intValue();
        } else if (value instanceof String) {
            try {
                return Integer.parseInt((String) value);
            } catch (NumberFormatException e) {
                return defaultValue;
            }
        } else {
            // Try to convert to int
            try {
                return ((Number) value).intValue();
            } catch (Exception e) {
                return defaultValue;
            }
        }
    }
}