package com.yishape.lab.image.core;

import com.yishape.lab.image.ImageData;
import com.yishape.lab.math.linalg.IVector;

import java.util.Map;

/**
 * 图像分析器接口 / Image Analyzer Interface
 * <p>
 * 定义图像分析操作的统一接口，用于提取图像特征、计算统计信息等。
 * 支持多种分析类型和可配置的分析参数。
 * </p>
 * <p>
 * Defines unified interface for image analysis operations, for extracting image features,
 * calculating statistics, etc. Supports multiple analysis types and configurable parameters.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public interface IImageAnalyzer {
    
    /**
     * 分析结果接口 / Analysis Result Interface
     */
    public interface AnalysisResult {
        /**
         * 获取分析类型 / Get Analysis Type
         */
        String getAnalysisType();
        
        /**
         * 获取特征向量 / Get Feature Vector
         */
        IVector<Double> getFeatureVector();
        
        /**
         * 获取数值结果 / Get Numeric Results
         */
        Map<String, Double> getNumericResults();
        
        /**
         * 获取结果描述 / Get Result Description
         */
        String getDescription();
        
        /**
         * 获取置信度 / Get Confidence
         */
        double getConfidence();
        
        /**
         * 转换为JSON格式 / Convert to JSON
         */
        String toJSON();
    }
    
    /**
     * 分析图像 / Analyze Image
     * <p>
     * 对输入图像进行分析，返回分析结果。
     * Analyzes input image and returns analysis result.
     * </p>
     * 
     * @param input 输入图像 / Input image
     * @return 分析结果 / Analysis result
     * @throws ImageProcessingException 分析过程中发生错误 / Error occurred during analysis
     */
    AnalysisResult analyze(ImageData input) throws ImageProcessingException;
    
    /**
     * 使用参数分析图像 / Analyze Image with Parameters
     * <p>
     * 对输入图像进行分析，使用指定的参数。
     * Analyzes input image with specified parameters.
     * </p>
     * 
     * @param input 输入图像 / Input image
     * @param parameters 分析参数 / Analysis parameters
     * @return 分析结果 / Analysis result
     * @throws ImageProcessingException 分析过程中发生错误 / Error occurred during analysis
     */
    AnalysisResult analyze(ImageData input, Map<String, Object> parameters) throws ImageProcessingException;
    
    /**
     * 批量分析图像 / Batch Analyze Images
     * <p>
     * 对多个图像进行批量分析。
     * Performs batch analysis on multiple images.
     * </p>
     * 
     * @param inputs 输入图像列表 / Input image list
     * @return 分析结果列表 / Analysis result list
     * @throws ImageProcessingException 分析过程中发生错误 / Error occurred during analysis
     */
    java.util.List<AnalysisResult> analyzeBatch(java.util.List<ImageData> inputs) throws ImageProcessingException;
    
    /**
     * 获取分析器名称 / Get Analyzer Name
     * 
     * @return 分析器名称 / Analyzer name
     */
    String getName();
    
    /**
     * 获取分析器描述 / Get Analyzer Description
     * 
     * @return 分析器描述 / Analyzer description
     */
    String getDescription();
    
    /**
     * 获取分析类型 / Get Analysis Type
     * 
     * @return 分析类型 / Analysis type
     */
    String getAnalysisType();
    
    /**
     * 获取支持的参数 / Get Supported Parameters
     * 
     * @return 支持的参数名称列表 / List of supported parameter names
     */
    java.util.Set<String> getSupportedParameters();
    
    /**
     * 获取默认参数 / Get Default Parameters
     * 
     * @return 默认参数映射 / Default parameter mapping
     */
    Map<String, Object> getDefaultParameters();
    
    /**
     * 验证输入图像 / Validate Input Image
     * <p>
     * 验证输入图像是否符合分析器的要求。
     * Validates if input image meets analyzer requirements.
     * </p>
     * 
     * @param input 输入图像 / Input image
     * @return 验证结果 / Validation result
     */
    boolean validateInput(ImageData input);
    
    /**
     * 验证参数 / Validate Parameters
     * <p>
     * 验证参数是否有效。
     * Validates if parameters are valid.
     * </p>
     * 
     * @param parameters 参数映射 / Parameter mapping
     * @return 验证结果 / Validation result
     */
    boolean validateParameters(Map<String, Object> parameters);
    
    /**
     * 获取输出特征维度 / Get Output Feature Dimension
     * 
     * @return 特征向量维度 / Feature vector dimension
     */
    int getFeatureDimension();
    
    /**
     * 是否支持实时分析 / Supports Real-time Analysis
     * 
     * @return 是否支持实时分析 / Whether real-time analysis is supported
     */
    default boolean supportsRealTime() {
        return false;
    }
    
    /**
     * 是否支持增量分析 / Supports Incremental Analysis
     * 
     * @return 是否支持增量分析 / Whether incremental analysis is supported
     */
    default boolean supportsIncremental() {
        return false;
    }
    
    /**
     * 获取分析精度 / Get Analysis Precision
     * 
     * @return 分析精度描述 / Analysis precision description
     */
    default String getPrecision() {
        return "STANDARD";
    }
    
    /**
     * 克隆分析器 / Clone Analyzer
     * <p>
     * 创建分析器的副本，用于并行处理。
     * Creates a copy of analyzer for parallel processing.
     * </p>
     * 
     * @return 分析器副本 / Analyzer copy
     */
    IImageAnalyzer clone();
    
    /**
     * 获取版本信息 / Get Version
     * 
     * @return 版本字符串 / Version string
     */
    default String getVersion() {
        return "1.0";
    }
}