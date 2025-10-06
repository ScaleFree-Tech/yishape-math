package com.yishape.lab.math.stats.bayes.mcmc;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;

import java.util.List;

/**
 * MCMC采样器接口
 * MCMC Sampler Interface
 * 
 * <p>定义了马尔可夫链蒙特卡洛采样器的核心接口，支持单变量和多变量参数的采样。</p>
 * <p>Defines the core interface for Markov Chain Monte Carlo samplers, supporting 
 * both univariate and multivariate parameter sampling.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IMCMCSampler {
    
    /**
     * 目标分布接口
     * Target distribution interface
     */
    @FunctionalInterface
    interface TargetDistribution {
        /**
         * 计算对数概率密度
         * Calculate log probability density
         * 
         * @param parameters 参数向量 / Parameter vector
         * @return 对数概率密度 / Log probability density
         */
        double logPdf(IVector parameters);
    }
    
    /**
     * 提议分布接口
     * Proposal distribution interface
     */
    @FunctionalInterface
    interface ProposalDistribution {
        /**
         * 从当前状态生成新的提议状态
         * Generate new proposal state from current state
         * 
         * @param currentState 当前状态 / Current state
         * @return 提议状态 / Proposed state
         */
        IVector propose(IVector currentState);
    }
    
    /**
     * 采样结果类
     * Sampling result class
     */
    class SamplingResult {
        private final IMatrix samples;
        private final double[] logProbabilities;
        private final double acceptanceRate;
        private final int effectiveSampleSize;
        private final boolean converged;
        
        public SamplingResult(IMatrix samples, double[] logProbabilities, 
                            double acceptanceRate, int effectiveSampleSize, boolean converged) {
            this.samples = samples;
            this.logProbabilities = logProbabilities;
            this.acceptanceRate = acceptanceRate;
            this.effectiveSampleSize = effectiveSampleSize;
            this.converged = converged;
        }
        
        public IMatrix getSamples() { return samples; }
        public double[] getLogProbabilities() { return logProbabilities; }
        public double getAcceptanceRate() { return acceptanceRate; }
        public int getEffectiveSampleSize() { return effectiveSampleSize; }
        public boolean isConverged() { return converged; }
        
        /**
         * 获取指定参数的样本
         * Get samples for specified parameter
         */
        public IVector getParameterSamples(int parameterIndex) {
            return samples.getColumn(parameterIndex);
        }
        
        /**
         * 获取样本数量
         * Get number of samples
         */
        public int getNumSamples() {
            return samples.rows();
        }
        
        /**
         * 获取参数维度
         * Get parameter dimension
         */
        public int getParameterDimension() {
            return samples.cols();
        }
    }
    
    /**
     * 执行MCMC采样
     * Perform MCMC sampling
     * 
     * @param targetDistribution 目标分布 / Target distribution
     * @param initialState 初始状态 / Initial state
     * @param numSamples 采样数量 / Number of samples
     * @param burnIn 预热期样本数 / Number of burn-in samples
     * @return 采样结果 / Sampling result
     */
    SamplingResult sample(TargetDistribution targetDistribution, 
                         IVector initialState, 
                         int numSamples, 
                         int burnIn);
    
    /**
     * 执行MCMC采样（带提议分布）
     * Perform MCMC sampling with proposal distribution
     * 
     * @param targetDistribution 目标分布 / Target distribution
     * @param proposalDistribution 提议分布 / Proposal distribution
     * @param initialState 初始状态 / Initial state
     * @param numSamples 采样数量 / Number of samples
     * @param burnIn 预热期样本数 / Number of burn-in samples
     * @return 采样结果 / Sampling result
     */
    SamplingResult sample(TargetDistribution targetDistribution,
                         ProposalDistribution proposalDistribution,
                         IVector initialState,
                         int numSamples,
                         int burnIn);
    
    /**
     * 设置采样器参数
     * Set sampler parameters
     * 
     * @param stepSize 步长 / Step size
     * @param adaptationPeriod 自适应期 / Adaptation period
     */
    void setParameters(double stepSize, int adaptationPeriod);
    
    /**
     * 获取采样器名称
     * Get sampler name
     * 
     * @return 采样器名称 / Sampler name
     */
    String getSamplerName();
    
    /**
     * 是否支持自适应
     * Whether supports adaptation
     * 
     * @return 是否支持自适应 / Whether supports adaptation
     */
    boolean supportsAdaptation();
}