package com.yishape.lab.math.stats.bayes.diagnostics;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.List;

/**
 * MCMC收敛诊断工具
 * MCMC Convergence Diagnostics
 * 
 * <p>提供多种MCMC收敛诊断方法，包括Gelman-Rubin统计量、有效样本量、自相关函数等。</p>
 * <p>Provides various MCMC convergence diagnostic methods including 
 * Gelman-Rubin statistic, effective sample size, autocorrelation function, etc.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ConvergenceDiagnostics {
    
    /**
     * 收敛诊断结果
     * Convergence diagnostic result
     */
    public static class DiagnosticResult {
        private final double rHat;
        private final double effectiveSampleSize;
        private final IVector autocorrelation;
        private final double monteCarloStandardError;
        private final boolean converged;
        
        public DiagnosticResult(double rHat, double effectiveSampleSize, 
                              IVector autocorrelation, double monteCarloStandardError, 
                              boolean converged) {
            this.rHat = rHat;
            this.effectiveSampleSize = effectiveSampleSize;
            this.autocorrelation = autocorrelation;
            this.monteCarloStandardError = monteCarloStandardError;
            this.converged = converged;
        }
        
        public double getRHat() { return rHat; }
        public double getEffectiveSampleSize() { return effectiveSampleSize; }
        public IVector getAutocorrelation() { return autocorrelation; }
        public double getMonteCarloStandardError() { return monteCarloStandardError; }
        public boolean isConverged() { return converged; }
        
        @Override
        public String toString() {
            return String.format("DiagnosticResult{R-hat=%.4f, ESS=%.1f, MCSE=%.6f, converged=%s}", 
                               rHat, effectiveSampleSize, monteCarloStandardError, converged);
        }
    }
    
    /**
     * 计算Gelman-Rubin统计量（R-hat）
     * Calculate Gelman-Rubin statistic (R-hat)
     * 
     * @param chains 多条链的样本，每行是一个样本，每列是一个参数
     * @return R-hat统计量
     */
    public static double calculateRHat(List<IMatrix> chains) {
        if (chains.size() < 2) {
            throw new IllegalArgumentException("At least 2 chains are required for R-hat calculation");
        }
        
        int numChains = chains.size();
        int numSamples = chains.get(0).rows();
        int numParams = chains.get(0).cols();
        
        // 验证所有链的维度一致
        for (IMatrix chain : chains) {
            if (chain.rows() != numSamples || chain.cols() != numParams) {
                throw new IllegalArgumentException("All chains must have the same dimensions");
            }
        }
        
        double maxRHat = 0.0;
        
        // 对每个参数计算R-hat
        for (int p = 0; p < numParams; p++) {
            // 计算链内方差
            double withinChainVariance = 0.0;
            IVector chainMeans = Linalg.vector(numChains);
            
            for (int c = 0; c < numChains; c++) {
                IVector chainSamples = Linalg.vector(numSamples);
                for (int s = 0; s < numSamples; s++) {
                    chainSamples.set(s, chains.get(c).get(s, p));
                }
                
                double chainMean = calculateMean(chainSamples);
                chainMeans.set(c, chainMean);
                
                double chainVar = calculateVariance(chainSamples, chainMean);
                withinChainVariance += chainVar;
            }
            withinChainVariance /= numChains;
            
            // 计算链间方差
            double overallMean = calculateMean(chainMeans);
            double betweenChainVariance = 0.0;
            for (int c = 0; c < numChains; c++) {
                double diff = chainMeans.get(c).doubleValue() - overallMean;
                betweenChainVariance += diff * diff;
            }
            betweenChainVariance = betweenChainVariance * numSamples / (numChains - 1);
            
            // 计算R-hat
            double pooledVariance = ((numSamples - 1) * withinChainVariance + betweenChainVariance) / numSamples;
            double rHat = Math.sqrt(pooledVariance / withinChainVariance);
            
            maxRHat = Math.max(maxRHat, rHat);
        }
        
        return maxRHat;
    }
    
    /**
     * 计算有效样本量
     * Calculate effective sample size
     * 
     * @param samples 样本序列
     * @return 有效样本量
     */
    public static double calculateEffectiveSampleSize(IVector samples) {
        int n = samples.size();
        IVector autocorr = calculateAutocorrelation(samples, Math.min(n / 4, 100));
        
        // 找到第一个负的自相关值
        int cutoff = 1;
        for (int lag = 1; lag < autocorr.size(); lag++) {
            if (autocorr.get(lag).doubleValue() <= 0) {
                cutoff = lag;
                break;
            }
        }
        
        // 计算积分自相关时间
        double integratedAutocorrTime = 1.0;
        for (int lag = 1; lag < cutoff; lag++) {
            integratedAutocorrTime += 2 * autocorr.get(lag).doubleValue();
        }
        
        return n / integratedAutocorrTime;
    }
    
    /**
     * 计算自相关函数
     * Calculate autocorrelation function
     * 
     * @param samples 样本序列
     * @param maxLag 最大滞后
     * @return 自相关函数值
     */
    public static IVector calculateAutocorrelation(IVector samples, int maxLag) {
        int n = samples.size();
        maxLag = Math.min(maxLag, n - 1);
        
        double mean = calculateMean(samples);
        double variance = calculateVariance(samples, mean);
        
        IVector autocorr = Linalg.vector(maxLag + 1);
        autocorr.set(0, 1.0); // lag 0的自相关总是1
        
        for (int lag = 1; lag <= maxLag; lag++) {
            double covariance = 0.0;
            int count = n - lag;
            
            for (int i = 0; i < count; i++) {
                double x1 = samples.get(i).doubleValue() - mean;
                double x2 = samples.get(i + lag).doubleValue() - mean;
                covariance += x1 * x2;
            }
            
            covariance /= count;
            autocorr.set(lag, covariance / variance);
        }
        
        return autocorr;
    }
    
    /**
     * 计算蒙特卡洛标准误差
     * Calculate Monte Carlo standard error
     * 
     * @param samples 样本序列
     * @return 蒙特卡洛标准误差
     */
    public static double calculateMonteCarloStandardError(IVector samples) {
        double variance = calculateVariance(samples, calculateMean(samples));
        double ess = calculateEffectiveSampleSize(samples);
        return Math.sqrt(variance / ess);
    }
    
    /**
     * 综合收敛诊断
     * Comprehensive convergence diagnosis
     * 
     * @param chains 多条链的样本
     * @param rHatThreshold R-hat阈值（通常为1.1）
     * @param essThreshold 有效样本量阈值
     * @return 诊断结果
     */
    public static DiagnosticResult diagnoseConvergence(List<IMatrix> chains, 
                                                     double rHatThreshold, 
                                                     double essThreshold) {
        if (chains.isEmpty()) {
            throw new IllegalArgumentException("At least one chain is required");
        }
        
        // 合并所有链的样本
        IMatrix firstChain = chains.get(0);
        int totalSamples = firstChain.rows() * chains.size();
        int numParams = firstChain.cols();
        
        IMatrix allSamples = Linalg.zeros(totalSamples, numParams);
        int sampleIndex = 0;
        
        for (IMatrix chain : chains) {
            for (int i = 0; i < chain.rows(); i++) {
                for (int j = 0; j < chain.cols(); j++) {
                    allSamples.set(sampleIndex, j, chain.get(i, j));
                }
                sampleIndex++;
            }
        }
        
        // 计算第一个参数的诊断统计量（简化）
        IVector firstParamSamples = Linalg.vector(totalSamples);
        for (int i = 0; i < totalSamples; i++) {
            firstParamSamples.set(i, allSamples.get(i, 0));
        }
        
        double rHat = chains.size() > 1 ? calculateRHat(chains) : 1.0;
        double ess = calculateEffectiveSampleSize(firstParamSamples);
        IVector autocorr = calculateAutocorrelation(firstParamSamples, Math.min(50, totalSamples / 4));
        double mcse = calculateMonteCarloStandardError(firstParamSamples);
        
        boolean converged = rHat <= rHatThreshold && ess >= essThreshold;
        
        return new DiagnosticResult(rHat, ess, autocorr, mcse, converged);
    }
    
    /**
     * 计算向量均值
     */
    private static double calculateMean(IVector vector) {
        double sum = 0.0;
        for (int i = 0; i < vector.size(); i++) {
            sum += vector.get(i).doubleValue();
        }
        return sum / vector.size();
    }
    
    /**
     * 计算向量方差
     */
    private static double calculateVariance(IVector vector, double mean) {
        double sumSquaredDiff = 0.0;
        for (int i = 0; i < vector.size(); i++) {
            double diff = vector.get(i).doubleValue() - mean;
            sumSquaredDiff += diff * diff;
        }
        return sumSquaredDiff / (vector.size() - 1);
    }
}