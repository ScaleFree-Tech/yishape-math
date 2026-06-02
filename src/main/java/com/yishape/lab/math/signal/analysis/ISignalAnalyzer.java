package com.yishape.lab.math.signal.analysis;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.signal.core.ISignalProcessor;
import com.yishape.lab.math.signal.core.SignalProcessingException;

/**
 * 信号分析器接口 / Signal Analyzer Interface
 * <p>
 * 定义所有信号分析操作的基础接口，支持各种信号分析方法。
 * 使用策略模式支持不同的分析算法实现。
 * </p>
 * <p>
 * Defines the base interface for all signal analysis operations supporting various analysis methods.
 * Uses Strategy pattern to support different analysis algorithm implementations.
 * </p>
 *
 * @param <T> 信号数据类型 / Signal data type
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface ISignalAnalyzer<T extends Number> extends ISignalProcessor<T> {
    
    /**
     * 分析类型枚举 / Analysis Type Enum
     */
    enum AnalysisType {
        SPECTRUM("频谱分析", "Spectrum Analysis"),
        POWER_SPECTRUM("功率谱分析", "Power Spectrum Analysis"),
        AUTOCORRELATION("自相关分析", "Autocorrelation Analysis"),
        CROSS_CORRELATION("互相关分析", "Cross-correlation Analysis"),
        COHERENCE("相干性分析", "Coherence Analysis"),
        ENVELOPE("包络分析", "Envelope Analysis"),
        INSTANTANEOUS_FREQUENCY("瞬时频率分析", "Instantaneous Frequency Analysis"),
        TIME_FREQUENCY("时频分析", "Time-Frequency Analysis"),
        STATISTICAL("统计分析", "Statistical Analysis"),
        PEAK_DETECTION("峰值检测", "Peak Detection"),
        ENERGY("能量分析", "Energy Analysis"),
        SNR("信噪比分析", "SNR Analysis"),
        THD("总谐波失真分析", "THD Analysis"),
        CREST_FACTOR("峰值因子分析", "Crest Factor Analysis");
        
        private final String chineseName;
        private final String englishName;
        
        AnalysisType(String chineseName, String englishName) {
            this.chineseName = chineseName;
            this.englishName = englishName;
        }
        
        public String getChineseName() { return chineseName; }
        public String getEnglishName() { return englishName; }
    }
    
    /**
     * 分析结果类 / Analysis Result Class
     */
    class AnalysisResult<R> {
        private final AnalysisType analysisType;
        private final R result;
        private final String[] resultNames;
        private final String description;
        private final double confidenceLevel;
        
        public AnalysisResult(AnalysisType analysisType, R result, String[] resultNames, String description, double confidenceLevel) {
            this.analysisType = analysisType;
            this.result = result;
            this.resultNames = resultNames.clone();
            this.description = description;
            this.confidenceLevel = confidenceLevel;
        }
        
        public AnalysisType getAnalysisType() { return analysisType; }
        public R getResult() { return result; }
        public String[] getResultNames() { return resultNames.clone(); }
        public String getDescription() { return description; }
        public double getConfidenceLevel() { return confidenceLevel; }
    }
    
    /**
     * 分析参数类 / Analysis Parameters Class
     */
    class AnalysisParameters {
        private double samplingRate = 1000.0;      // 采样率 / Sampling rate
        private int windowSize = 256;              // 窗口大小 / Window size
        private double overlap = 0.5;              // 重叠比例 / Overlap ratio
        private String windowType = "hanning";     // 窗函数类型 / Window type
        private int nfft = 512;                    // FFT点数 / FFT points
        private double frequencyRange[] = {0, 500}; // 频率范围 / Frequency range
        private double confidenceLevel = 0.95;     // 置信水平 / Confidence level
        private int maxPeaks = 10;                 // 最大峰值数 / Maximum peaks
        private double peakThreshold = 0.1;        // 峰值阈值 / Peak threshold
        private double minPeakDistance = 1.0;      // 最小峰值距离 / Minimum peak distance
        
        // Builder pattern methods
        public AnalysisParameters samplingRate(double samplingRate) { this.samplingRate = samplingRate; return this; }
        public AnalysisParameters windowSize(int windowSize) { this.windowSize = windowSize; return this; }
        public AnalysisParameters overlap(double overlap) { this.overlap = overlap; return this; }
        public AnalysisParameters windowType(String windowType) { this.windowType = windowType; return this; }
        public AnalysisParameters nfft(int nfft) { this.nfft = nfft; return this; }
        public AnalysisParameters frequencyRange(double minFreq, double maxFreq) { 
            this.frequencyRange = new double[]{minFreq, maxFreq}; return this; 
        }
        public AnalysisParameters confidenceLevel(double confidenceLevel) { this.confidenceLevel = confidenceLevel; return this; }
        public AnalysisParameters maxPeaks(int maxPeaks) { this.maxPeaks = maxPeaks; return this; }
        public AnalysisParameters peakThreshold(double peakThreshold) { this.peakThreshold = peakThreshold; return this; }
        public AnalysisParameters minPeakDistance(double minPeakDistance) { this.minPeakDistance = minPeakDistance; return this; }
        
        // Getters
        public double getSamplingRate() { return samplingRate; }
        public int getWindowSize() { return windowSize; }
        public double getOverlap() { return overlap; }
        public String getWindowType() { return windowType; }
        public int getNfft() { return nfft; }
        public double[] getFrequencyRange() { return frequencyRange.clone(); }
        public double getConfidenceLevel() { return confidenceLevel; }
        public int getMaxPeaks() { return maxPeaks; }
        public double getPeakThreshold() { return peakThreshold; }
        public double getMinPeakDistance() { return minPeakDistance; }
    }
    
    /**
     * 分析信号 / Analyze signal
     * <p>
     * 对输入信号进行指定类型的分析。
     * Perform specified type of analysis on input signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param analysisType 分析类型 / Analysis type
     * @param parameters 分析参数 / Analysis parameters
     * @return 分析结果 / Analysis result
     * @throws SignalProcessingException 分析过程中发生错误时抛出 / Thrown when errors occur during analysis
     */
    <R> AnalysisResult<R> analyze(IVector<T> signal, AnalysisType analysisType, AnalysisParameters parameters) throws SignalProcessingException;
    
    /**
     * 分析信号（简化版） / Analyze signal (simplified version)
     * <p>
     * 使用默认参数对输入信号进行指定类型的分析。
     * Perform specified type of analysis on input signal using default parameters.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param analysisType 分析类型 / Analysis type
     * @return 分析结果 / Analysis result
     * @throws SignalProcessingException 分析过程中发生错误时抛出 / Thrown when errors occur during analysis
     */
    default <R> AnalysisResult<R> analyze(IVector<T> signal, AnalysisType analysisType) throws SignalProcessingException {
        return analyze(signal, analysisType, new AnalysisParameters());
    }
    
    /**
     * 批量分析 / Batch analysis
     * <p>
     * 对输入信号进行多种类型的分析。
     * Perform multiple types of analysis on input signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param analysisTypes 分析类型数组 / Analysis type array
     * @param parameters 分析参数 / Analysis parameters
     * @return 分析结果数组 / Analysis result array
     * @throws SignalProcessingException 分析过程中发生错误时抛出 / Thrown when errors occur during analysis
     */
    AnalysisResult<?>[] batchAnalyze(IVector<T> signal, AnalysisType[] analysisTypes, AnalysisParameters parameters) throws SignalProcessingException;
    
    /**
     * 比较分析 / Comparative analysis
     * <p>
     * 比较两个信号的指定特征。
     * Compare specified features of two signals.
     * </p>
     *
     * @param signal1 第一个信号 / First signal
     * @param signal2 第二个信号 / Second signal
     * @param analysisType 分析类型 / Analysis type
     * @param parameters 分析参数 / Analysis parameters
     * @return 比较分析结果 / Comparative analysis result
     * @throws SignalProcessingException 分析过程中发生错误时抛出 / Thrown when errors occur during analysis
     */
    <R> AnalysisResult<R> compareAnalyze(IVector<T> signal1, IVector<T> signal2, AnalysisType analysisType, AnalysisParameters parameters) throws SignalProcessingException;
    
    /**
     * 获取支持的分析类型 / Get supported analysis types
     * <p>
     * 返回当前分析器支持的所有分析类型。
     * Return all analysis types supported by current analyzer.
     * </p>
     *
     * @return 支持的分析类型数组 / Supported analysis type array
     */
    AnalysisType[] getSupportedAnalysisTypes();
    
    /**
     * 验证分析参数 / Validate analysis parameters
     * <p>
     * 验证给定的分析参数是否有效。
     * Validate if given analysis parameters are valid.
     * </p>
     *
     * @param analysisType 分析类型 / Analysis type
     * @param parameters 分析参数 / Analysis parameters
     * @return 验证是否通过 / Whether validation passes
     */
    boolean validateParameters(AnalysisType analysisType, AnalysisParameters parameters);
    
    /**
     * 获取推荐参数 / Get recommended parameters
     * <p>
     * 根据信号特征推荐最优的分析参数。
     * Recommend optimal analysis parameters based on signal characteristics.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param analysisType 分析类型 / Analysis type
     * @return 推荐的分析参数 / Recommended analysis parameters
     */
    AnalysisParameters getRecommendedParameters(IVector<T> signal, AnalysisType analysisType);
    
    /**
     * 默认实现process方法 / Default implementation of process method
     */
    @Override
    default IVector<T> process(IVector<T> input) throws SignalProcessingException {
        // 默认返回统计特征向量 / Default return statistical feature vector
        AnalysisResult<IVector<T>> result = analyze(input, AnalysisType.STATISTICAL);
        return result.getResult();
    }
}