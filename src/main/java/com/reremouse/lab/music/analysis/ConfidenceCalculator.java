package com.reremouse.lab.music.analysis;

import java.util.Map;

/**
 * 标准化置信度计算器接口 / Standardized confidence calculator interface
 * <p>
 * 提供统一的置信度计算方法，确保不同分析器之间的一致性。
 * Provides unified confidence calculation methods to ensure consistency across different analyzers.
 * </p>
 */
public interface ConfidenceCalculator {
    
    /**
     * 计算基于特征质量的置信度 / Calculate confidence based on feature quality
     * 
     * @param features 特征数组 / Feature array
     * @param baseScore 基础匹配分数 / Base match score
     * @return 标准化置信度值 (0.0-1.0) / Standardized confidence value (0.0-1.0)
     */
    double calculateFeatureBasedConfidence(double[] features, double baseScore);
    
    /**
     * 计算基于统计特性的置信度 / Calculate confidence based on statistical properties
     * 
     * @param values 数值数组 / Value array
     * @return 标准化置信度值 (0.0-1.0) / Standardized confidence value (0.0-1.0)
     */
    double calculateStatisticalConfidence(double[] values);
    
    /**
     * 计算基于多因素的综合置信度 / Calculate comprehensive confidence based on multiple factors
     * 
     * @param factors 因素映射 (因子名称 -> 值) / Factor map (factor name -> value)
     * @param weights 权重映射 (因子名称 -> 权重) / Weight map (factor name -> weight)
     * @return 标准化置信度值 (0.0-1.0) / Standardized confidence value (0.0-1.0)
     */
    double calculateWeightedConfidence(Map<String, Double> factors, Map<String, Double> weights);
    
    /**
     * 计算基于概率分布的置信度 / Calculate confidence based on probability distribution
     * 
     * @param probabilities 概率分布映射 (类别 -> 概率) / Probability distribution map (category -> probability)
     * @return 标准化置信度值 (0.0-1.0) / Standardized confidence value (0.0-1.0)
     */
    double calculateDistributionConfidence(Map<String, Double> probabilities);
    
    /**
     * 计算最小置信度阈值 / Calculate minimum confidence threshold
     * 
     * @param featureCount 特征数量 / Number of features
     * @param qualityScore 质量评分 / Quality score
     * @return 最小置信度值 / Minimum confidence value
     */
    double calculateMinimumConfidence(int featureCount, double qualityScore);
}