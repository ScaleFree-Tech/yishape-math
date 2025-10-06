package com.yishape.lab.math.stats.bayes.variational;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;

import java.util.Map;

/**
 * 变分推断接口
 * Variational Inference Interface
 * 
 * <p>定义了变分推断算法的通用接口，包括变分分布的优化和推断结果的获取。</p>
 * <p>Defines the common interface for variational inference algorithms, 
 * including optimization of variational distributions and retrieval of inference results.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IVariationalInference {
    
    /**
     * 变分分布接口
     * Variational distribution interface
     */
    interface VariationalDistribution {
        /**
         * 计算对数概率密度
         * Calculate log probability density
         * 
         * @param parameters 参数向量 / Parameter vector
         * @return 对数概率密度 / Log probability density
         */
        double logPdf(IVector parameters);
        
        /**
         * 从变分分布中采样
         * Sample from variational distribution
         * 
         * @param numSamples 采样数量 / Number of samples
         * @return 样本矩阵 / Sample matrix
         */
        IMatrix sample(int numSamples);
        
        /**
         * 获取变分参数
         * Get variational parameters
         * 
         * @return 变分参数 / Variational parameters
         */
        IVector getVariationalParameters();
        
        /**
         * 设置变分参数
         * Set variational parameters
         * 
         * @param parameters 变分参数 / Variational parameters
         */
        void setVariationalParameters(IVector parameters);
        
        /**
         * 计算熵
         * Calculate entropy
         * 
         * @return 熵值 / Entropy value
         */
        double entropy();
        
        /**
         * 获取均值
         * Get mean
         * 
         * @return 均值向量 / Mean vector
         */
        IVector mean();
        
        /**
         * 获取协方差矩阵
         * Get covariance matrix
         * 
         * @return 协方差矩阵 / Covariance matrix
         */
        IMatrix covariance();
    }
    
    /**
     * 目标分布接口
     * Target distribution interface
     */
    interface TargetDistribution {
        /**
         * 计算对数概率密度
         * Calculate log probability density
         * 
         * @param parameters 参数向量 / Parameter vector
         * @return 对数概率密度 / Log probability density
         */
        double logPdf(IVector parameters);
        
        /**
         * 计算对数概率密度的梯度
         * Calculate gradient of log probability density
         * 
         * @param parameters 参数向量 / Parameter vector
         * @return 梯度向量 / Gradient vector
         */
        IVector logPdfGradient(IVector parameters);
    }
    
    /**
     * 变分推断结果
     * Variational inference result
     */
    class VariationalResult {
        private final VariationalDistribution posteriorApproximation;
        private final double[] elbos;
        private final boolean converged;
        private final int iterations;
        private final Map<String, Object> diagnostics;
        
        public VariationalResult(VariationalDistribution posteriorApproximation,
                               double[] elbos,
                               boolean converged,
                               int iterations,
                               Map<String, Object> diagnostics) {
            this.posteriorApproximation = posteriorApproximation;
            this.elbos = elbos;
            this.converged = converged;
            this.iterations = iterations;
            this.diagnostics = diagnostics;
        }
        
        /**
         * 获取后验近似分布
         * Get posterior approximation
         */
        public VariationalDistribution getPosteriorApproximation() {
            return posteriorApproximation;
        }
        
        /**
         * 获取ELBO轨迹
         * Get ELBO trajectory
         */
        public double[] getElbos() {
            return elbos.clone();
        }
        
        /**
         * 获取最终ELBO值
         * Get final ELBO value
         */
        public double getFinalElbo() {
            return elbos.length > 0 ? elbos[elbos.length - 1] : Double.NaN;
        }
        
        /**
         * 是否收敛
         * Whether converged
         */
        public boolean isConverged() {
            return converged;
        }
        
        /**
         * 获取迭代次数
         * Get number of iterations
         */
        public int getIterations() {
            return iterations;
        }
        
        /**
         * 获取诊断信息
         * Get diagnostics
         */
        public Map<String, Object> getDiagnostics() {
            return diagnostics;
        }
        
        /**
         * 从后验近似分布中采样
         * Sample from posterior approximation
         * 
         * @param numSamples 采样数量 / Number of samples
         * @return 样本矩阵 / Sample matrix
         */
        public IMatrix sample(int numSamples) {
            return posteriorApproximation.sample(numSamples);
        }
        
        /**
         * 获取后验均值
         * Get posterior mean
         */
        public IVector getPosteriorMean() {
            return posteriorApproximation.mean();
        }
        
        /**
         * 获取后验协方差
         * Get posterior covariance
         */
        public IMatrix getPosteriorCovariance() {
            return posteriorApproximation.covariance();
        }
    }
    
    /**
     * 执行变分推断
     * Perform variational inference
     * 
     * @param targetDistribution 目标分布 / Target distribution
     * @param initialVariationalDist 初始变分分布 / Initial variational distribution
     * @param maxIterations 最大迭代次数 / Maximum iterations
     * @param tolerance 收敛容忍度 / Convergence tolerance
     * @return 变分推断结果 / Variational inference result
     */
    VariationalResult infer(TargetDistribution targetDistribution,
                          VariationalDistribution initialVariationalDist,
                          int maxIterations,
                          double tolerance);
    
    /**
     * 计算证据下界（ELBO）
     * Calculate Evidence Lower BOund (ELBO)
     * 
     * @param targetDistribution 目标分布 / Target distribution
     * @param variationalDist 变分分布 / Variational distribution
     * @param numSamples 蒙特卡洛样本数 / Number of Monte Carlo samples
     * @return ELBO值 / ELBO value
     */
    double calculateElbo(TargetDistribution targetDistribution,
                        VariationalDistribution variationalDist,
                        int numSamples);
    
    /**
     * 计算ELBO的梯度
     * Calculate gradient of ELBO
     * 
     * @param targetDistribution 目标分布 / Target distribution
     * @param variationalDist 变分分布 / Variational distribution
     * @param numSamples 蒙特卡洛样本数 / Number of Monte Carlo samples
     * @return ELBO梯度 / ELBO gradient
     */
    IVector calculateElboGradient(TargetDistribution targetDistribution,
                                 VariationalDistribution variationalDist,
                                 int numSamples);
    
    /**
     * 设置优化器参数
     * Set optimizer parameters
     * 
     * @param learningRate 学习率 / Learning rate
     * @param momentum 动量 / Momentum
     */
    void setOptimizerParameters(double learningRate, double momentum);
    
    /**
     * 获取算法名称
     * Get algorithm name
     * 
     * @return 算法名称 / Algorithm name
     */
    String getAlgorithmName();
    
    /**
     * 是否支持自动微分
     * Whether supports automatic differentiation
     * 
     * @return 是否支持 / Whether supported
     */
    boolean supportsAutomaticDifferentiation();
}