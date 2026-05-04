package com.yishape.lab.music.analysis.advanced;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioUtil;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.music.analysis.MusicDetectionResult;
import com.yishape.lab.music.analysis.UnifiedMusicAnalysisResult;
import com.yishape.lab.music.analysis.StandardizedConfidenceCalculator;
import com.yishape.lab.music.analysis.basic.BeatAnalyzerImpl;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.RereFFT;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

/**
 * 音乐结构分析器 / Music Structure Analyzer
 * <p>
 * 分析音乐的结构特征，包括段落划分、重复性分析、新颖性检测等。
 * Analyzes music structure features including section segmentation, repetitiveness analysis, novelty detection, etc.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class StructureAnalyzer implements IAdvancedAnalyzer {
    
    // 结构特征键 / Structure feature keys
    public static final String SECTIONS = "sections";
    public static final String REPETITIVENESS = "repetitiveness";
    public static final String NOVELTY_CURVE = "noveltyCurve";
    public static final String SEGMENT_BOUNDARIES = "segmentBoundaries";
    public static final String AVERAGE_SEGMENT_LENGTH = "averageSegmentLength";
    public static final String STRUCTURAL_COMPLEXITY = "structuralComplexity";
    public static final String CONFIDENCE = "confidence";
    
    // 默认参数 / Default parameters
    private static final int DEFAULT_WINDOW_SIZE = 4096;
    private static final int DEFAULT_HOP_SIZE = 2048;
    private static final int DEFAULT_SEGMENT_LENGTH = 30; // 段落长度（秒）
    private static final double DEFAULT_NOVELTY_THRESHOLD = 0.4;
    
    private final BeatAnalyzerImpl beatAnalyzer;
    private final StandardizedConfidenceCalculator confidenceCalculator = new StandardizedConfidenceCalculator();
    
    /**
     * 默认构造函数 / Default constructor
     * <p>
     * 初始化结构分析器，创建节拍分析器实例。
     * Initializes the structure analyzer, creating beat analyzer instance.
     * </p>
     */
    public StructureAnalyzer() {
        this.beatAnalyzer = new BeatAnalyzerImpl();
    }
    
    @Override
    public Map<String, Object> analyze(AudioData audioData) throws AudioProcessingException {
        return analyze(audioData, getDefaultParameters());
    }
    
    /**
     * 分析音频数据获取结构特征 / Analyze audio data to get structure features
     * <p>
     * 使用指定参数分析音频数据的音乐结构特征。
     * Analyzes audio data for music structure features using specified parameters.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 分析参数 / Analysis parameters
     * @return 结构分析结果映射 / Structure analysis result map
     * @throws AudioProcessingException 处理异常 / Processing exception
     */
    @Override
    public Map<String, Object> analyze(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }
        
        try {
            Map<String, Object> results = new HashMap<>();
            
            double duration = audioData.getDuration();
            int windowSize = getIntParameter(parameters, "windowSize", DEFAULT_WINDOW_SIZE);
            int hopSize = getIntParameter(parameters, "hopSize", DEFAULT_HOP_SIZE);
            double noveltyThreshold = getDoubleParameter(parameters, "noveltyThreshold", DEFAULT_NOVELTY_THRESHOLD);
            
            // 1. 计算新颖性函数 / Calculate novelty function
            double[] noveltyFunction = calculateNoveltyFunction(audioData, windowSize, hopSize);
            
            // 2. 检测段落边界 / Detect segment boundaries
            List<Double> boundaries = detectSegmentBoundaries(noveltyFunction, hopSize, audioData.getSampleRate(), noveltyThreshold);
            
            // 3. 计算段落数量 / Calculate number of sections
            int numSections = boundaries.size() + 1;
            
            // 4. 计算平均段落长度 / Calculate average segment length
            double avgSegmentLength = duration / numSections;
            
            // 5. 计算重复性 / Calculate repetitiveness
            double repetitiveness = calculateRepetitiveness(audioData, boundaries, windowSize, hopSize);
            
            // 6. 计算结构复杂度 / Calculate structural complexity
            double structuralComplexity = calculateStructuralComplexity(numSections, repetitiveness, avgSegmentLength);
            
            // 7. 计算置信度 / Calculate confidence
            double confidence = calculateStructureConfidence(noveltyFunction, boundaries, duration);
            
            // 存储结果 / Store results
            results.put(SECTIONS, numSections);
            results.put(REPETITIVENESS, repetitiveness);
            results.put(NOVELTY_CURVE, noveltyFunction);
            results.put(SEGMENT_BOUNDARIES, boundaries);
            results.put(AVERAGE_SEGMENT_LENGTH, avgSegmentLength);
            results.put(STRUCTURAL_COMPLEXITY, structuralComplexity);
            results.put(CONFIDENCE, confidence);
            
            // 添加结构类型分类 / Add structure type classification
            results.put("structureType", classifyStructureType(numSections, repetitiveness));
            
            return results;
            
        } catch (Exception e) {
            throw new AudioProcessingException("Structure analysis failed: " + e.getMessage(), e);
        }
    }
    
    /**
     * 计算新颖性函数 / Calculate novelty function
     * <p>
     * 通过分析音频特征的时间变化来检测音乐结构的变化点。
     * Detects structural change points by analyzing temporal variations in audio features.
     * </p>
     */
    private double[] calculateNoveltyFunction(AudioData audioData, int windowSize, int hopSize) throws AudioProcessingException {
        IVector<Double> samples = audioData.getSamples();
        double sampleRate = audioData.getSampleRate();
        
        // 计算帧数 / Calculate number of frames
        int numFrames = (samples.length() - windowSize) / hopSize + 1;
        if (numFrames <= 0) {
            return new double[]{0.0};
        }
        
        double[] novelty = new double[numFrames];
        double[] prevSpectralFeatures = null;
        
        for (int frame = 0; frame < numFrames; frame++) {
            int startIdx = frame * hopSize;
            int endIdx = Math.min(startIdx + windowSize, samples.length());
            
            // 提取帧样本 / Extract frame samples
            double[] frameSamples = new double[endIdx - startIdx];
            for (int i = 0; i < frameSamples.length; i++) {
                frameSamples[i] = samples.get(startIdx + i);
            }
            
            // 计算频谱特征 / Calculate spectral features
            Complex[] spectrum = RereFFT.fft(AudioUtil.convertToComplex(frameSamples, windowSize));
            double[] spectralFeatures = extractSpectralFeatures(spectrum, sampleRate, windowSize);
            
            // 计算与前一帧的差异 / Calculate difference with previous frame
            if (prevSpectralFeatures != null) {
                double distance = calculateEuclideanDistance(spectralFeatures, prevSpectralFeatures);
                novelty[frame] = distance;
            } else {
                novelty[frame] = 0.0;
            }
            
            prevSpectralFeatures = spectralFeatures;
        }
        
        // 归一化新颖性函数 / Normalize novelty function
        double maxNovelty = 0.0;
        for (double value : novelty) {
            maxNovelty = Math.max(maxNovelty, value);
        }
        
        if (maxNovelty > 0) {
            for (int i = 0; i < novelty.length; i++) {
                novelty[i] /= maxNovelty;
            }
        }
        
        return novelty;
    }
    
    /**
     * 提取频谱特征向量 / Extract spectral feature vector
     * <p>
     * 从频谱中提取多个特征形成特征向量，包括频谱质心、频谱滚降、频谱带宽、频谱对比度和频谱平坦度。
     * Extracts multiple features from spectrum to form feature vector, including spectral centroid, rolloff, bandwidth, contrast and flatness.
     * </p>
     *
     * @param spectrum 频谱数据 / Spectrum data
     * @param sampleRate 采样率 / Sample rate
     * @param windowSize 窗口大小 / Window size
     * @return 频谱特征向量 / Spectral feature vector
     */
    private double[] extractSpectralFeatures(Complex[] spectrum, double sampleRate, int windowSize) {
        // 提取多个频谱特征形成特征向量
        double[] features = new double[5];
        
        // 1. 频谱质心 / Spectral centroid
        features[0] = AudioUtil.calculateSpectralCentroid(spectrum, sampleRate, windowSize);
        
        // 2. 频谱滚降 / Spectral rolloff
        features[1] = AudioUtil.calculateSpectralRolloff(spectrum, sampleRate, windowSize);
        
        // 3. 频谱带宽 / Spectral bandwidth (requires centroid as 4th parameter)
        features[2] = AudioUtil.calculateSpectralBandwidth(spectrum, sampleRate, windowSize, features[0]);
        
        // 4. 频谱对比度 / Spectral contrast
        features[3] = AudioUtil.calculateSpectralContrast(spectrum);
        
        // 5. 频谱平坦度 / Spectral flatness
        features[4] = AudioUtil.calculateSpectralFlatness(spectrum);
        
        // 归一化特征 / Normalize features
        features[0] /= (sampleRate / 2.0);
        features[1] /= (sampleRate / 2.0);
        features[2] /= (sampleRate / 2.0);
        
        return features;
    }
    
    /**
     * 计算欧氏距离 / Calculate Euclidean distance
     * <p>
     * 计算两个向量之间的欧氏距离。
     * Calculates Euclidean distance between two vectors.
     * </p>
     *
     * @param vec1 第一个向量 / First vector
     * @param vec2 第二个向量 / Second vector
     * @return 欧氏距离 / Euclidean distance
     */
    private double calculateEuclideanDistance(double[] vec1, double[] vec2) {
        double sum = 0.0;
        int len = Math.min(vec1.length, vec2.length);
        
        for (int i = 0; i < len; i++) {
            double diff = vec1[i] - vec2[i];
            sum += diff * diff;
        }
        
        return Math.sqrt(sum);
    }
    
    /**
     * 检测段落边界 / Detect segment boundaries
     * <p>
     * 在新颖性函数中寻找峰值作为段落边界。
     * Finds peaks in novelty function as segment boundaries.
     * </p>
     */
    private List<Double> detectSegmentBoundaries(double[] noveltyFunction, int hopSize, double sampleRate, double threshold) {
        List<Double> boundaries = new ArrayList<>();
        
        if (noveltyFunction.length < 3) {
            return boundaries;
        }
        
        // 寻找局部最大值 / Find local maxima
        for (int i = 1; i < noveltyFunction.length - 1; i++) {
            if (noveltyFunction[i] > threshold &&
                noveltyFunction[i] > noveltyFunction[i - 1] &&
                noveltyFunction[i] > noveltyFunction[i + 1]) {
                
                // 转换为时间（秒）/ Convert to time (seconds)
                double timeInSeconds = (i * hopSize) / sampleRate;
                boundaries.add(timeInSeconds);
            }
        }
        
        return boundaries;
    }
    
    /**
     * 计算重复性 / Calculate repetitiveness
     * <p>
     * 通过自相似矩阵分析音乐的重复性。
     * Analyzes music repetitiveness through self-similarity matrix.
     * </p>
     */
    private double calculateRepetitiveness(AudioData audioData, List<Double> boundaries, int windowSize, int hopSize) {
        try {
            IVector<Double> samples = audioData.getSamples();
            int numSegments = boundaries.size() + 1;
            
            if (numSegments < 2) {
                return 0.5; // 默认中等重复性
            }
            
            // 提取每个段落的特征 / Extract features for each segment
            List<double[]> segmentFeatures = new ArrayList<>();
            double prevBoundary = 0.0;
            
            for (int i = 0; i <= boundaries.size(); i++) {
                double currentBoundary = (i < boundaries.size()) ? boundaries.get(i) : audioData.getDuration();
                
                // 提取段落 / Extract segment
                int startSample = (int)(prevBoundary * audioData.getSampleRate());
                int endSample = (int)(currentBoundary * audioData.getSampleRate());
                
                if (endSample > startSample && startSample < samples.length()) {
                    endSample = Math.min(endSample, samples.length());
                    
                    // 计算段落的平均频谱特征 / Calculate average spectral features for segment
                    double[] segmentFeature = extractSegmentFeature(samples, startSample, endSample, windowSize);
                    segmentFeatures.add(segmentFeature);
                }
                
                prevBoundary = currentBoundary;
            }
            
            // 计算段落之间的相似度 / Calculate similarity between segments
            if (segmentFeatures.size() < 2) {
                return 0.5;
            }
            
            double totalSimilarity = 0.0;
            int comparisons = 0;
            
            for (int i = 0; i < segmentFeatures.size(); i++) {
                for (int j = i + 1; j < segmentFeatures.size(); j++) {
                    double similarity = calculateCosineSimilarity(segmentFeatures.get(i), segmentFeatures.get(j));
                    totalSimilarity += similarity;
                    comparisons++;
                }
            }
            
            double avgSimilarity = comparisons > 0 ? totalSimilarity / comparisons : 0.5;
            
            // 相似度越高，重复性越高 / Higher similarity means higher repetitiveness
            return Math.max(0.0, Math.min(1.0, avgSimilarity));
            
        } catch (Exception e) {
            return 0.5; // 默认中等重复性
        }
    }
    
    /**
     * 提取段落特征 / Extract segment feature
     * <p>
     * 从音频段落中提取频谱包络作为段落特征。
     * Extracts spectral envelope from audio segment as segment feature.
     * </p>
     *
     * @param samples 音频样本 / Audio samples
     * @param startIdx 起始索引 / Start index
     * @param endIdx 结束索引 / End index
     * @param windowSize 窗口大小 / Window size
     * @return 段落特征向量 / Segment feature vector
     */
    private double[] extractSegmentFeature(IVector<Double> samples, int startIdx, int endIdx, int windowSize) {
        try {
            // 提取段落中间部分的样本 / Extract samples from middle of segment
            int segmentLength = endIdx - startIdx;
            int midPoint = startIdx + segmentLength / 2;
            int extractStart = Math.max(startIdx, midPoint - windowSize / 2);
            int extractEnd = Math.min(endIdx, extractStart + windowSize);
            
            double[] segmentSamples = new double[extractEnd - extractStart];
            for (int i = 0; i < segmentSamples.length; i++) {
                segmentSamples[i] = samples.get(extractStart + i);
            }
            
            // 计算FFT / Calculate FFT
            Complex[] spectrum = RereFFT.fft(AudioUtil.convertToComplex(segmentSamples, windowSize));
            
            // 提取频谱包络作为特征 / Extract spectral envelope as feature
            int numBins = Math.min(50, spectrum.length / 2);
            double[] feature = new double[numBins];
            
            for (int i = 0; i < numBins; i++) {
                int binStart = i * spectrum.length / (2 * numBins);
                int binEnd = (i + 1) * spectrum.length / (2 * numBins);
                
                double binEnergy = 0.0;
                for (int j = binStart; j < binEnd && j < spectrum.length; j++) {
                    binEnergy += spectrum[j].magnitude();
                }
                feature[i] = binEnergy / (binEnd - binStart);
            }
            
            // 归一化 / Normalize
            double maxValue = 0.0;
            for (double value : feature) {
                maxValue = Math.max(maxValue, value);
            }
            
            if (maxValue > 0) {
                for (int i = 0; i < feature.length; i++) {
                    feature[i] /= maxValue;
                }
            }
            
            return feature;
            
        } catch (Exception e) {
            return new double[50]; // 返回零向量
        }
    }
    
    /**
     * 计算余弦相似度 / Calculate cosine similarity
     * <p>
     * 计算两个向量之间的余弦相似度。
     * Calculates cosine similarity between two vectors.
     * </p>
     *
     * @param vec1 第一个向量 / First vector
     * @param vec2 第二个向量 / Second vector
     * @return 余弦相似度 / Cosine similarity
     */
    private double calculateCosineSimilarity(double[] vec1, double[] vec2) {
        if (vec1.length != vec2.length) {
            return 0.0;
        }
        
        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        
        for (int i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }
        
        norm1 = Math.sqrt(norm1);
        norm2 = Math.sqrt(norm2);
        
        if (norm1 == 0 || norm2 == 0) {
            return 0.0;
        }
        
        return dotProduct / (norm1 * norm2);
    }
    
    /**
     * 计算结构复杂度 / Calculate structural complexity
     * <p>
     * 综合考虑段落数量、重复性和段落长度计算结构复杂度。
     * Calculates structural complexity by comprehensively considering number of sections, repetitiveness and segment length.
     * </p>
     *
     * @param numSections 段落数量 / Number of sections
     * @param repetitiveness 重复性 / Repetitiveness
     * @param avgSegmentLength 平均段落长度 / Average segment length
     * @return 结构复杂度值(0.0-1.0) / Structural complexity value (0.0-1.0)
     */
    private double calculateStructuralComplexity(int numSections, double repetitiveness, double avgSegmentLength) {
        // 段落数量贡献 / Section count contribution
        double sectionScore = Math.min(1.0, numSections / 10.0);
        
        // 重复性贡献（重复性低 = 复杂度高）/ Repetitiveness contribution (low repetitiveness = high complexity)
        double repetitivenessScore = 1.0 - repetitiveness;
        
        // 段落长度变化贡献 / Segment length variation contribution
        double lengthScore = avgSegmentLength < 20 ? 0.7 : (avgSegmentLength > 60 ? 0.3 : 0.5);
        
        // 综合计算 / Combined calculation
        double complexity = sectionScore * 0.4 + repetitivenessScore * 0.4 + lengthScore * 0.2;
        
        return Math.max(0.0, Math.min(1.0, complexity));
    }
    
    /**
     * 计算结构分析置信度 / Calculate structure analysis confidence
     * <p>
     * 基于新颖性函数的质量和边界数量计算置信度。
     * Calculates confidence based on novelty function quality and boundary count.
     * </p>
     *
     * @param noveltyFunction 新颖性函数 / Novelty function
     * @param boundaries 段落边界列表 / Segment boundaries list
     * @param duration 音频持续时间 / Audio duration
     * @return 置信度值(0.0-1.0) / Confidence value (0.0-1.0)
     */
    private double calculateStructureConfidence(double[] noveltyFunction, List<Double> boundaries, double duration) {
        // 基于新颖性函数的质量和边界数量计算置信度
        
        // 1. 新颖性函数的动态范围 / Dynamic range of novelty function
        double maxNovelty = 0.0;
        double minNovelty = 1.0;
        for (double value : noveltyFunction) {
            maxNovelty = Math.max(maxNovelty, value);
            minNovelty = Math.min(minNovelty, value);
        }
        double dynamicRange = maxNovelty - minNovelty;
        double rangeScore = Math.min(1.0, dynamicRange * 2.0);
        
        // 2. 边界数量的合理性 / Reasonableness of boundary count
        double expectedBoundaries = duration / 30.0; // 预期每30秒一个边界
        double boundaryRatio = boundaries.size() / expectedBoundaries;
        double boundaryScore = boundaryRatio > 0.5 && boundaryRatio < 2.0 ? 0.8 : 0.5;
        
        // 3. 音频长度因素 / Audio length factor
        double lengthScore = duration > 30 ? 0.8 : 0.6;
        
        // 综合置信度 / Combined confidence
        double confidence = rangeScore * 0.4 + boundaryScore * 0.4 + lengthScore * 0.2;
        
        return Math.max(0.3, Math.min(0.9, confidence));
    }
    
    /**
     * 分类结构类型 / Classify structure type
     * <p>
     * 根据段落数量和重复性分类音乐结构类型。
     * Classifies music structure type based on number of sections and repetitiveness.
     * </p>
     *
     * @param numSections 段落数量 / Number of sections
     * @param repetitiveness 重复性 / Repetitiveness
     * @return 结构类型字符串 / Structure type string
     */
    private String classifyStructureType(int numSections, double repetitiveness) {
        if (numSections <= 2) {
            return "Simple";
        } else if (numSections <= 4) {
            return repetitiveness > 0.6 ? "Verse-Chorus" : "Through-Composed";
        } else if (numSections <= 7) {
            return repetitiveness > 0.5 ? "Complex-Repetitive" : "Complex-Progressive";
        } else {
            return "Highly-Complex";
        }
    }
    
    @Override
    public MusicDetectionResult analyzeAdvancedMusic(AudioData audioData) throws AudioProcessingException {
        return analyzeAdvancedMusic(audioData, getDefaultParameters());
    }
    
    /**
     * 分析音频数据获取高级音乐结果 / Analyze audio data to get advanced music result
     * <p>
     * 使用指定参数分析音频数据并返回高级音乐检测结果。
     * Analyzes audio data using specified parameters and returns advanced music detection result.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 分析参数 / Analysis parameters
     * @return 音乐检测结果 / Music detection result
     * @throws AudioProcessingException 处理异常 / Processing exception
     */
    @Override
    public MusicDetectionResult analyzeAdvancedMusic(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        Map<String, Object> analysisResult = analyze(audioData, parameters);
        
        UnifiedMusicAnalysisResult result = new UnifiedMusicAnalysisResult();
        result.setStructuralAnalysis(analysisResult);
        result.setConfidence(getDoubleParameter(analysisResult, CONFIDENCE, 0.5));
        result.setAlgorithm("structure_analyzer");
        
        return result;
    }
    
    /**
     * 获取分析器名称 / Get analyzer name
     *
     * @return 分析器名称 / Analyzer name
     */
    @Override
    public String getAnalyzerName() {
        return "StructureAnalyzer";
    }
    
    /**
     * 获取支持的参数列表 / Get supported parameters list
     *
     * @return 支持的参数名称数组 / Array of supported parameter names
     */
    @Override
    public String[] getSupportedParameters() {
        return new String[]{"windowSize", "hopSize", "segmentLength", "noveltyThreshold"};
    }
    
    /**
     * 获取默认参数 / Get default parameters
     *
     * @return 默认参数映射 / Default parameters map
     */
    @Override
    public Map<String, Object> getDefaultParameters() {
        Map<String, Object> params = new HashMap<>();
        params.put("windowSize", DEFAULT_WINDOW_SIZE);
        params.put("hopSize", DEFAULT_HOP_SIZE);
        params.put("segmentLength", DEFAULT_SEGMENT_LENGTH);
        params.put("noveltyThreshold", DEFAULT_NOVELTY_THRESHOLD);
        return params;
    }
    
    /**
     * 验证参数 / Validate parameters
     * <p>
     * 验证提供的参数是否在支持列表中。
     * Validates whether the provided parameters are in the supported list.
     * </p>
     *
     * @param parameters 待验证的参数 / Parameters to validate
     * @return 是否有效 / Whether valid
     */
    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        if (parameters == null) {
            return true;
        }
        
        for (String key : parameters.keySet()) {
            if (!java.util.Arrays.asList(getSupportedParameters()).contains(key)) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * 设置参数 / Set parameters
     * <p>
     * 设置分析器参数。如果参数无效则抛出异常。
     * Sets analyzer parameters. Throws exception if parameters are invalid.
     * </p>
     *
     * @param parameters 要设置的参数 / Parameters to set
     * @throws AudioProcessingException 参数无效 / Parameters invalid
     */
    @Override
    public void setParameters(Map<String, Object> parameters) throws AudioProcessingException {
        if (!validateParameters(parameters)) {
            throw new AudioProcessingException("Invalid parameters");
        }
    }
    
    /**
     * 获取当前参数 / Get current parameters
     *
     * @return 当前参数映射 / Current parameters map
     */
    @Override
    public Map<String, Object> getCurrentParameters() {
        return getDefaultParameters();
    }
    
    /**
     * 重置参数 / Reset parameters
     * <p>
     * 重置分析器参数到默认状态。此实现无需重置任何状态。
     * Resets analyzer parameters to default state. This implementation has no state to reset.
     */
    @Override
    public void resetParameters() {
        // No state to reset
    }
    
    /**
     * 获取名称 / Get name
     *
     * @return 分析器名称 / Analyzer name
     */
    @Override
    public String getName() {
        return "StructureAnalyzer";
    }
    
    /**
     * 获取描述 / Get description
     *
     * @return 分析器描述 / Analyzer description
     */
    @Override
    public String getDescription() {
        return "Analyzes music structure including sections, repetitiveness, and novelty detection";
    }
    
    /**
     * 检查是否支持音频格式 / Check if audio format is supported
     *
     * @param sampleRate 采样率 / Sample rate
     * @param channels 声道数 / Number of channels
     * @param bitDepth 位深度 / Bit depth
     * @return 是否支持 / Whether supported
     */
    @Override
    public boolean supportsAudioFormat(double sampleRate, int channels, int bitDepth) {
        return sampleRate >= 8000 && sampleRate <= 192000 && 
               channels >= 1 && channels <= 8 && 
               bitDepth >= 8 && bitDepth <= 32;
    }
    
    /**
     * 获取最小音频长度 / Get minimum audio length
     *
     * @return 最小音频长度（秒）/ Minimum audio length (seconds)
     */
    @Override
    public double getMinimumAudioLength() {
        return 5.0; // 至少5秒才能进行结构分析
    }
    
    /**
     * 获取最大音频长度 / Get maximum audio length
     *
     * @return 最大音频长度（秒）/ Maximum audio length (seconds)
     */
    @Override
    public double getMaximumAudioLength() {
        return 3600.0; // 最多1小时
    }
    
    /**
     * 估算复杂度 / Estimate complexity
     *
     * @param audioLength 音频长度（秒）/ Audio length (seconds)
     * @return 估算的复杂度 / Estimated complexity
     */
    @Override
    public double getComplexityEstimate(double audioLength) {
        return Math.log10(audioLength + 1) * 2.0;
    }
    
    /**
     * 预热分析器 / Warm up analyzer
     * <p>
     * 预热分析器以提高首次分析的准确性。此实现无需预热。
     * Warms up the analyzer to improve accuracy of first analysis. This implementation requires no warm-up.
     * </p>
     *
     * @throws AudioProcessingException 预热失败 / Warm-up failed
     */
    @Override
    public void warmUp() throws AudioProcessingException {
        // No warm-up needed
    }
    
    /**
     * 清理资源 / Cleanup resources
     * <p>
     * 释放分析器占用的资源。此实现无需清理。
     * Releases resources occupied by the analyzer. This implementation requires no cleanup.
     */
    @Override
    public void cleanup() {
        // No cleanup needed
    }
    
    /**
     * 获取状态 / Get status
     *
     * @return 分析器当前状态 / Current analyzer status
     */
    @Override
    public String getStatus() {
        return "Ready";
    }
    
    /**
     * 检查是否就绪 / Check if ready
     *
     * @return 是否就绪 / Whether ready
     */
    @Override
    public boolean isReady() {
        return true;
    }
    
    /**
     * 获取上次分析统计 / Get last analysis statistics
     *
     * @return 统计信息映射 / Statistics map
     */
    @Override
    public Map<String, Object> getLastAnalysisStatistics() {
        return new HashMap<>();
    }
    
    /**
     * 获取性能指标 / Get performance metrics
     *
     * @return 性能指标映射 / Performance metrics map
     */
    @Override
    public Map<String, Object> getPerformanceMetrics() {
        return new HashMap<>();
    }
    
    /**
     * 设置详细日志 / Set verbose logging
     *
     * @param enabled 是否启用 / Whether to enable
     */
    @Override
    public void setVerboseLogging(boolean enabled) {
        // No logging to configure
    }
    
    /**
     * 检查详细日志是否启用 / Check if verbose logging is enabled
     *
     * @return 是否启用 / Whether enabled
     */
    @Override
    public boolean isVerboseLoggingEnabled() {
        return false;
    }
    
    // 辅助方法 / Helper methods
    
    /**
     * 获取整型参数值 / Get integer parameter value
     * <p>
     * 从参数映射中安全获取整型值，如果不存在则返回默认值。
     * Safely gets integer value from parameters map, returning default if not exists.
     * </p>
     *
     * @param parameters 参数映射 / Parameters map
     * @param key 参数键名 / Parameter key
     * @param defaultValue 默认值 / Default value
     * @return 参数值或默认值 / Parameter value or default
     */
    private int getIntParameter(Map<String, Object> parameters, String key, int defaultValue) {
        if (parameters == null || !parameters.containsKey(key)) {
            return defaultValue;
        }
        
        Object value = parameters.get(key);
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        
        return defaultValue;
    }
    
    /**
     * 获取双精度浮点型参数值 / Get double parameter value
     * <p>
     * 从参数映射中安全获取双精度浮点型值，如果不存在则返回默认值。
     * Safely gets double value from parameters map, returning default if not exists.
     * </p>
     *
     * @param parameters 参数映射 / Parameters map
     * @param key 参数键名 / Parameter key
     * @param defaultValue 默认值 / Default value
     * @return 参数值或默认值 / Parameter value or default
     */
    private double getDoubleParameter(Map<String, Object> parameters, String key, double defaultValue) {
        if (parameters == null || !parameters.containsKey(key)) {
            return defaultValue;
        }
        
        Object value = parameters.get(key);
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        
        return defaultValue;
    }
}
