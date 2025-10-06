package com.yishape.lab.math.stats.bayes.inference;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

/**
 * 重要性采样
 * Importance Sampling
 * 
 * <p>重要性采样是一种蒙特卡洛方法，用于估计难以直接采样的分布的期望值。
 * 通过从一个容易采样的提议分布中采样，并使用重要性权重来校正偏差。</p>
 * 
 * <p>Importance sampling is a Monte Carlo method for estimating expectations 
 * of distributions that are difficult to sample from directly.
 * It samples from an easy-to-sample proposal distribution and uses importance weights to correct bias.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ImportanceSampling {
    
    private final Random random;
    
    /**
     * 构造函数
     */
    public ImportanceSampling() {
        this(new Random());
    }
    
    /**
     * 构造函数
     * 
     * @param random 随机数生成器
     */
    public ImportanceSampling(Random random) {
        this.random = random;
    }
    
    /**
     * 重要性采样结果
     * Importance sampling result
     */
    public static class ImportanceSamplingResult {
        private final double estimate;
        private final double variance;
        private final double effectiveSampleSize;
        private final IVector samples;
        private final IVector weights;
        
        public ImportanceSamplingResult(double estimate, double variance, 
                                      double effectiveSampleSize, 
                                      IVector samples, IVector weights) {
            this.estimate = estimate;
            this.variance = variance;
            this.effectiveSampleSize = effectiveSampleSize;
            this.samples = samples;
            this.weights = weights;
        }
        
        public double getEstimate() { return estimate; }
        public double getVariance() { return variance; }
        public double getEffectiveSampleSize() { return effectiveSampleSize; }
        public IVector getSamples() { return samples; }
        public IVector getWeights() { return weights; }
        
        public double getStandardError() {
            return Math.sqrt(variance);
        }
        
        public double[] getConfidenceInterval(double alpha) {
            double z = 1.96; // 95% confidence interval
            if (alpha == 0.01) z = 2.576;
            else if (alpha == 0.05) z = 1.96;
            else if (alpha == 0.1) z = 1.645;
            
            double se = getStandardError();
            return new double[]{estimate - z * se, estimate + z * se};
        }
    }
    
    /**
     * 目标分布接口
     * Target distribution interface
     */
    @FunctionalInterface
    public interface TargetDistribution {
        /**
         * 计算目标分布的未归一化密度
         * Calculate unnormalized density of target distribution
         * 
         * @param x 输入值
         * @return 未归一化密度值
         */
        double unnormalizedDensity(double x);
    }
    
    /**
     * 提议分布接口
     * Proposal distribution interface
     */
    public interface ProposalDistribution {
        /**
         * 从提议分布中采样
         * Sample from proposal distribution
         * 
         * @return 采样值
         */
        double sample();
        
        /**
         * 计算提议分布的密度
         * Calculate density of proposal distribution
         * 
         * @param x 输入值
         * @return 密度值
         */
        double density(double x);
    }
    
    /**
     * 执行重要性采样估计期望值
     * Perform importance sampling to estimate expectation
     * 
     * @param targetFunction 目标函数
     * @param targetDistribution 目标分布
     * @param proposalDistribution 提议分布
     * @param numSamples 采样数量
     * @return 重要性采样结果
     */
    public ImportanceSamplingResult estimateExpectation(
            Function<Double, Double> targetFunction,
            TargetDistribution targetDistribution,
            ProposalDistribution proposalDistribution,
            int numSamples) {
        
        List<Double> samplesList = new ArrayList<>();
        List<Double> weightsList = new ArrayList<>();
        List<Double> functionValuesList = new ArrayList<>();
        
        // 采样和计算权重
        for (int i = 0; i < numSamples; i++) {
            double sample = proposalDistribution.sample();
            double targetDensity = targetDistribution.unnormalizedDensity(sample);
            double proposalDensity = proposalDistribution.density(sample);
            
            if (proposalDensity > 0) {
                double weight = targetDensity / proposalDensity;
                double functionValue = targetFunction.apply(sample);
                
                samplesList.add(sample);
                weightsList.add(weight);
                functionValuesList.add(functionValue);
            }
        }
        
        // 转换为向量
        IVector samples = vectorFromList(samplesList);
        IVector weights = vectorFromList(weightsList);
        IVector functionValues = vectorFromList(functionValuesList);
        
        // 归一化权重
        double weightSum = sum(weights);
        IVector normalizedWeights = Linalg.vector(weights.size());
        for (int i = 0; i < weights.size(); i++) {
            normalizedWeights.set(i, weights.get(i).doubleValue() / weightSum);
        }
        
        // 计算估计值
        double estimate = 0.0;
        for (int i = 0; i < functionValues.size(); i++) {
            estimate += functionValues.get(i).doubleValue() * normalizedWeights.get(i).doubleValue();
        }
        
        // 计算方差
        double variance = calculateVariance(functionValues, normalizedWeights, estimate);
        
        // 计算有效样本大小
        double effectiveSampleSize = calculateEffectiveSampleSize(normalizedWeights);
        
        return new ImportanceSamplingResult(estimate, variance, effectiveSampleSize, 
                                          samples, normalizedWeights);
    }
    
    /**
     * 自适应重要性采样
     * Adaptive importance sampling
     * 
     * @param targetFunction 目标函数
     * @param targetDistribution 目标分布
     * @param initialProposal 初始提议分布
     * @param numSamples 采样数量
     * @param adaptationSteps 自适应步数
     * @return 重要性采样结果
     */
    public ImportanceSamplingResult adaptiveImportanceSampling(
            Function<Double, Double> targetFunction,
            TargetDistribution targetDistribution,
            ProposalDistribution initialProposal,
            int numSamples,
            int adaptationSteps) {
        
        ProposalDistribution currentProposal = initialProposal;
        ImportanceSamplingResult bestResult = null;
        double bestESS = 0.0;
        
        for (int step = 0; step < adaptationSteps; step++) {
            ImportanceSamplingResult result = estimateExpectation(
                targetFunction, targetDistribution, currentProposal, numSamples);
            
            if (result.getEffectiveSampleSize() > bestESS) {
                bestESS = result.getEffectiveSampleSize();
                bestResult = result;
            }
            
            // 简单的自适应策略：基于样本均值和方差调整提议分布
            if (step < adaptationSteps - 1) {
                currentProposal = adaptProposal(currentProposal, result);
            }
        }
        
        return bestResult;
    }
    
    /**
     * 序贯重要性采样
     * Sequential importance sampling
     * 
     * @param targetFunction 目标函数
     * @param targetDistribution 目标分布
     * @param proposalDistribution 提议分布
     * @param batchSize 批次大小
     * @param maxBatches 最大批次数
     * @param convergenceThreshold 收敛阈值
     * @return 重要性采样结果
     */
    public ImportanceSamplingResult sequentialImportanceSampling(
            Function<Double, Double> targetFunction,
            TargetDistribution targetDistribution,
            ProposalDistribution proposalDistribution,
            int batchSize,
            int maxBatches,
            double convergenceThreshold) {
        
        List<Double> allSamples = new ArrayList<>();
        List<Double> allWeights = new ArrayList<>();
        List<Double> estimates = new ArrayList<>();
        
        for (int batch = 0; batch < maxBatches; batch++) {
            ImportanceSamplingResult batchResult = estimateExpectation(
                targetFunction, targetDistribution, proposalDistribution, batchSize);
            
            // 累积样本和权重
            for (int i = 0; i < batchResult.getSamples().size(); i++) {
                allSamples.add(batchResult.getSamples().get(i).doubleValue());
                allWeights.add(batchResult.getWeights().get(i).doubleValue());
            }
            
            estimates.add(batchResult.getEstimate());
            
            // 检查收敛性
            if (batch >= 2 && hasConverged(estimates, convergenceThreshold)) {
                break;
            }
        }
        
        // 重新计算最终结果
        IVector samples = vectorFromList(allSamples);
        IVector weights = vectorFromList(allWeights);
        
        // 重新归一化权重
        double weightSum = sum(weights);
        IVector normalizedWeights = Linalg.vector(weights.size());
        for (int i = 0; i < weights.size(); i++) {
            normalizedWeights.set(i, weights.get(i).doubleValue() / weightSum);
        }
        
        // 重新计算估计值和方差
        double estimate = 0.0;
        for (int i = 0; i < allSamples.size(); i++) {
            double functionValue = targetFunction.apply(allSamples.get(i));
            estimate += functionValue * normalizedWeights.get(i).doubleValue();
        }
        
        List<Double> functionValues = new ArrayList<>();
        for (double sample : allSamples) {
            functionValues.add(targetFunction.apply(sample));
        }
        
        double variance = calculateVariance(vectorFromList(functionValues), 
                                          normalizedWeights, estimate);
        double effectiveSampleSize = calculateEffectiveSampleSize(normalizedWeights);
        
        return new ImportanceSamplingResult(estimate, variance, effectiveSampleSize, 
                                          samples, normalizedWeights);
    }
    
    /**
     * 计算方差
     * Calculate variance
     */
    private double calculateVariance(IVector functionValues, IVector weights, double mean) {
        double variance = 0.0;
        
        for (int i = 0; i < functionValues.size(); i++) {
            double diff = functionValues.get(i).doubleValue() - mean;
            variance += weights.get(i).doubleValue() * diff * diff;
        }
        
        return variance;
    }
    
    /**
     * 计算有效样本大小
     * Calculate effective sample size
     */
    private double calculateEffectiveSampleSize(IVector weights) {
        double sumWeights = 0.0;
        double sumSquaredWeights = 0.0;
        
        for (int i = 0; i < weights.size(); i++) {
            double w = weights.get(i).doubleValue();
            sumWeights += w;
            sumSquaredWeights += w * w;
        }
        
        return (sumWeights * sumWeights) / sumSquaredWeights;
    }
    
    /**
     * 自适应调整提议分布
     * Adapt proposal distribution
     */
    private ProposalDistribution adaptProposal(ProposalDistribution current, 
                                             ImportanceSamplingResult result) {
        // 简化的自适应策略：基于样本统计调整
        IVector samples = result.getSamples();
        IVector weights = result.getWeights();
        
        // 计算加权均值和方差
        double weightedMean = 0.0;
        double weightedVariance = 0.0;
        
        for (int i = 0; i < samples.size(); i++) {
            weightedMean += samples.get(i).doubleValue() * weights.get(i).doubleValue();
        }
        
        for (int i = 0; i < samples.size(); i++) {
            double diff = samples.get(i).doubleValue() - weightedMean;
            weightedVariance += weights.get(i).doubleValue() * diff * diff;
        }
        
        double weightedStd = Math.sqrt(weightedVariance);
        
        // 返回调整后的正态提议分布
        return new NormalProposal(weightedMean, weightedStd, random);
    }
    
    /**
     * 检查收敛性
     * Check convergence
     */
    private boolean hasConverged(List<Double> estimates, double threshold) {
        if (estimates.size() < 3) return false;
        
        int n = estimates.size();
        double recent = estimates.get(n - 1);
        double previous = estimates.get(n - 2);
        
        return Math.abs(recent - previous) / Math.abs(previous) < threshold;
    }
    
    /**
     * 从列表创建向量
     * Create vector from list
     */
    private IVector vectorFromList(List<Double> list) {
        IVector vector = Linalg.vector(list.size());
        for (int i = 0; i < list.size(); i++) {
            vector.set(i, list.get(i));
        }
        return vector;
    }
    
    /**
     * 计算向量元素和
     * Calculate sum of vector elements
     */
    private double sum(IVector vector) {
        double sum = 0.0;
        for (int i = 0; i < vector.size(); i++) {
            sum += vector.get(i).doubleValue();
        }
        return sum;
    }
    
    /**
     * 正态提议分布实现
     * Normal proposal distribution implementation
     */
    public static class NormalProposal implements ProposalDistribution {
        private final double mean;
        private final double std;
        private final Random random;
        
        public NormalProposal(double mean, double std, Random random) {
            this.mean = mean;
            this.std = std;
            this.random = random;
        }
        
        @Override
        public double sample() {
            return mean + std * random.nextGaussian();
        }
        
        @Override
        public double density(double x) {
            double z = (x - mean) / std;
            return Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI));
        }
    }
    
    /**
     * 均匀提议分布实现
     * Uniform proposal distribution implementation
     */
    public static class UniformProposal implements ProposalDistribution {
        private final double min;
        private final double max;
        private final Random random;
        
        public UniformProposal(double min, double max, Random random) {
            this.min = min;
            this.max = max;
            this.random = random;
        }
        
        @Override
        public double sample() {
            return min + (max - min) * random.nextDouble();
        }
        
        @Override
        public double density(double x) {
            if (x >= min && x <= max) {
                return 1.0 / (max - min);
            }
            return 0.0;
        }
    }
}