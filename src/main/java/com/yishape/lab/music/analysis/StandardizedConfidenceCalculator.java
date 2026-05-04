package com.yishape.lab.music.analysis;

import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

/**
 * 标准化置信度计算器实现 / Standardized confidence calculator implementation
 * <p>
 * 提供统一的置信度计算方法，确保不同分析器之间的一致性。
 * Provides unified confidence calculation methods to ensure consistency across different analyzers.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class StandardizedConfidenceCalculator implements ConfidenceCalculator {
    
    // 默认最小置信度阈值 / Default minimum confidence threshold
    private static final double DEFAULT_MIN_CONFIDENCE = 0.1;
    
    // 默认最大置信度阈值 / Default maximum confidence threshold
    private static final double DEFAULT_MAX_CONFIDENCE = 0.95;
    
    @Override
    public double calculateFeatureBasedConfidence(double[] features, double baseScore) {
        // 添加保护性检查 / Add protective checks
        if (features == null || features.length == 0) {
            return DEFAULT_MIN_CONFIDENCE;
        }
        
        // 检查是否为零向量 / Check if it's a zero vector
        double totalEnergy = 0.0;
        double maxValue = 0.0;
        double minValue = Double.MAX_VALUE;
        
        for (double value : features) {
            totalEnergy += Math.abs(value);
            maxValue = Math.max(maxValue, Math.abs(value));
            minValue = Math.min(minValue, Math.abs(value));
        }
        
        // 如果总能量太低，返回最小置信度 / If total energy is too low, return minimum confidence
        if (totalEnergy < 1e-10) {
            return DEFAULT_MIN_CONFIDENCE;
        }
        
        // 计算方差来评估分布的非均匀性 / Calculate variance to assess distribution non-uniformity
        double average = totalEnergy / features.length;
        double variance = 0.0;
        
        for (double value : features) {
            variance += Math.pow(Math.abs(value) - average, 2);
        }
        variance /= features.length;
        
        // 计算动态范围因子 (0-1) / Calculate dynamic range factor (0-1)
        double dynamicRange = maxValue - minValue;
        double rangeFactor = Math.min(1.0, dynamicRange * 2.0); // 对于L2归一化，动态范围通常较小 / For L2 normalized, dynamic range is usually small
        
        // 计算方差因子 (0-1) / Calculate variance factor (0-1)
        double varianceFactor = Math.min(1.0, variance * 50.0); // 调整方差权重 / Adjust variance weight
        
        // 计算峰值突出度 / Calculate peak prominence
        double[] sortedValues = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            sortedValues[i] = Math.abs(features[i]);
        }
        java.util.Arrays.sort(sortedValues);
        
        int topCount = Math.min(3, features.length); // 前三大值 / Top three values
        int bottomCount = Math.max(0, features.length - topCount); // 其余值 / Remaining values
        
        double topSum = 0.0;
        double bottomSum = 0.0;
        
        for (int i = 0; i < features.length; i++) {
            if (i >= features.length - topCount) {
                topSum += sortedValues[i];
            } else {
                bottomSum += sortedValues[i];
            }
        }
        
        double peakProminence = bottomSum > 0 ? topSum / bottomSum : topSum;
        double prominenceFactor = Math.min(1.0, peakProminence / 3.0);
        
        // 综合置信度计算：基于模板匹配分数和特征质量 / Comprehensive confidence calculation: based on template match score and feature quality
        double featureQuality = (rangeFactor * 0.4 + varianceFactor * 0.4 + prominenceFactor * 0.2);
        double confidence = baseScore * (0.7 + featureQuality * 0.3); // 基础分数70%，特征质量30% / Base score 70%, feature quality 30%
        
        // 确保置信度在合理范围内 / Ensure confidence is within reasonable range
        return Math.max(DEFAULT_MIN_CONFIDENCE, Math.min(DEFAULT_MAX_CONFIDENCE, confidence));
    }
    
    @Override
    public double calculateStatisticalConfidence(double[] values) {
        // 添加保护性检查 / Add protective checks
        if (values == null || values.length == 0) {
            return DEFAULT_MIN_CONFIDENCE;
        }
        
        // 计算基本统计量 / Calculate basic statistics
        double sum = 0.0;
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        
        for (double value : values) {
            sum += value;
            max = Math.max(max, value);
            min = Math.min(min, value);
        }
        
        double mean = sum / values.length;
        
        // 计算方差 / Calculate variance
        double variance = 0.0;
        for (double value : values) {
            variance += Math.pow(value - mean, 2);
        }
        variance /= values.length;
        
        // 计算标准差 / Calculate standard deviation
        double stdDev = Math.sqrt(variance);
        
        // 计算变异系数 (如果均值不为零) / Calculate coefficient of variation (if mean is not zero)
        double coefficientOfVariation = Math.abs(mean) > 1e-10 ? stdDev / Math.abs(mean) : 0.0;
        
        // 基于统计特性的置信度计算 / Confidence calculation based on statistical properties
        // 变异系数越小，数据越稳定，置信度越高 / Lower coefficient of variation means more stable data, higher confidence
        double stabilityFactor = Math.max(0.0, 1.0 - coefficientOfVariation);
        
        // 数据范围越大，置信度越高 / Larger data range means higher confidence
        double rangeFactor = Math.min(1.0, (max - min) / 2.0);
        
        // 综合置信度 / Combined confidence
        double confidence = (stabilityFactor * 0.6 + rangeFactor * 0.4);
        
        // 确保置信度在合理范围内 / Ensure confidence is within reasonable range
        return Math.max(DEFAULT_MIN_CONFIDENCE, Math.min(DEFAULT_MAX_CONFIDENCE, confidence));
    }
    
    @Override
    public double calculateWeightedConfidence(Map<String, Double> factors, Map<String, Double> weights) {
        // 添加保护性检查 / Add protective checks
        if (factors == null || weights == null || factors.isEmpty() || weights.isEmpty()) {
            return DEFAULT_MIN_CONFIDENCE;
        }
        
        double totalWeightedValue = 0.0;
        double totalWeight = 0.0;
        
        // 计算加权平均 / Calculate weighted average
        for (Map.Entry<String, Double> entry : factors.entrySet()) {
            String factorName = entry.getKey();
            Double factorValue = entry.getValue();
            Double weight = weights.get(factorName);
            
            // 如果因子值或权重为空，跳过 / Skip if factor value or weight is null
            if (factorValue == null || weight == null) {
                continue;
            }
            
            // 确保因子值在0-1范围内 / Ensure factor value is within 0-1 range
            double normalizedFactorValue = Math.max(0.0, Math.min(1.0, factorValue));
            
            totalWeightedValue += normalizedFactorValue * weight;
            totalWeight += weight;
        }
        
        // 如果总权重为零，返回最小置信度 / If total weight is zero, return minimum confidence
        if (totalWeight <= 0) {
            return DEFAULT_MIN_CONFIDENCE;
        }
        
        double confidence = totalWeightedValue / totalWeight;
        
        // 确保置信度在合理范围内 / Ensure confidence is within reasonable range
        return Math.max(DEFAULT_MIN_CONFIDENCE, Math.min(DEFAULT_MAX_CONFIDENCE, confidence));
    }
    
    @Override
    public double calculateDistributionConfidence(Map<String, Double> probabilities) {
        // 添加保护性检查 / Add protective checks
        if (probabilities == null || probabilities.isEmpty()) {
            return DEFAULT_MIN_CONFIDENCE;
        }
        
        // 计算概率分布的熵值 / Calculate entropy of probability distribution
        double entropy = 0.0;
        double totalProbability = 0.0;
        
        // 首先计算总概率以进行归一化 / First calculate total probability for normalization
        for (Double prob : probabilities.values()) {
            if (prob != null && prob > 0) {
                totalProbability += prob;
            }
        }
        
        // 如果总概率为零，返回最小置信度 / If total probability is zero, return minimum confidence
        if (totalProbability <= 0) {
            return DEFAULT_MIN_CONFIDENCE;
        }
        
        // 计算归一化熵 / Calculate normalized entropy
        for (Double prob : probabilities.values()) {
            if (prob != null && prob > 0) {
                double normalizedProb = prob / totalProbability;
                entropy -= normalizedProb * Math.log(normalizedProb) / Math.log(2);
            }
        }
        
        // 归一化熵值到[0,1]范围 / Normalize entropy to [0,1] range
        double maxEntropy = Math.log(probabilities.size()) / Math.log(2);
        double normalizedEntropy = maxEntropy > 0 ? entropy / maxEntropy : 0.0;
        
        // 熵值越低（分布越集中），置信度越高 / Lower entropy (more concentrated distribution) means higher confidence
        double confidence = 1.0 - normalizedEntropy;
        
        // 确保置信度在合理范围内 / Ensure confidence is within reasonable range
        return Math.max(DEFAULT_MIN_CONFIDENCE, Math.min(DEFAULT_MAX_CONFIDENCE, confidence));
    }
    
    @Override
    public double calculateMinimumConfidence(int featureCount, double qualityScore) {
        // 基于特征数量和质量评分计算最小置信度 / Calculate minimum confidence based on feature count and quality score
        double featureFactor = Math.min(1.0, featureCount / 10.0); // 特征数量因子 / Feature count factor
        double qualityFactor = Math.max(0.0, Math.min(1.0, qualityScore)); // 质量因子 / Quality factor
        
        // 综合最小置信度 / Combined minimum confidence
        double minConfidence = DEFAULT_MIN_CONFIDENCE + (featureFactor * 0.2 + qualityFactor * 0.3);
        
        // 确保最小置信度在合理范围内 / Ensure minimum confidence is within reasonable range
        return Math.max(DEFAULT_MIN_CONFIDENCE, Math.min(DEFAULT_MAX_CONFIDENCE, minConfidence));
    }
    
    /**
     * 计算基于差距的置信度 / Calculate confidence based on gaps
     * 
     * @param topValues 顶部值数组 (按降序排列) / Top values array (in descending order)
     * @return 标准化置信度值 (0.0-1.0) / Standardized confidence value (0.0-1.0)
     */
    public double calculateGapBasedConfidence(double[] topValues) {
        if (topValues == null || topValues.length == 0) {
            return DEFAULT_MIN_CONFIDENCE;
        }
        
        // 如果只有一个值，基于其绝对值返回置信度 / If there's only one value, return confidence based on its absolute value
        if (topValues.length == 1) {
            return Math.max(DEFAULT_MIN_CONFIDENCE, Math.min(DEFAULT_MAX_CONFIDENCE, Math.abs(topValues[0])));
        }
        
        // 计算前两名之间的差距 / Calculate gap between top two
        double top1 = Math.abs(topValues[0]);
        double top2 = Math.abs(topValues.length > 1 ? topValues[1] : 0.0);
        double topGap = top1 - top2;
        
        // 根据差距调整置信度 / Adjust confidence based on gap
        double gapFactor = 1.0;
        if (topGap > 0.3) {
            gapFactor = 1.2; // 明显区分 / Clear distinction
        } else if (topGap > 0.15) {
            gapFactor = 1.0; // 中等区分 / Moderate distinction
        } else if (topGap > 0.05) {
            gapFactor = 0.8; // 轻微区分 / Slight distinction
        } else {
            gapFactor = 0.5; // 几乎无区分 / Almost no distinction
        }
        
        // 计算最终置信度 / Calculate final confidence
        double confidence = top1 * gapFactor;
        
        // 确保置信度在合理范围内 / Ensure confidence is within reasonable range
        return Math.max(DEFAULT_MIN_CONFIDENCE, Math.min(DEFAULT_MAX_CONFIDENCE, confidence));
    }
}