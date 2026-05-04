package com.yishape.lab.math.stats.bayes;

import com.yishape.lab.math.linalg.IVector;

/**
 * 贝叶斯分析工具类
 * Utility class for Bayesian analysis
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class BayesUtils {
    
    /**
     * 计算贝叶斯定理的基本形式：P(A|B) = P(B|A) * P(A) / P(B)
     * Calculate basic Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)
     * 
     * @param likelihood P(B|A) - 似然 / Likelihood
     * @param prior P(A) - 先验概率 / Prior probability
     * @param evidence P(B) - 边际似然 / Marginal likelihood
     * @return P(A|B) - 后验概率 / Posterior probability
     */
    public static double bayesTheorem(double likelihood, double prior, double evidence) {
        if (evidence <= 0) {
            throw new BayesException("边际似然必须大于0 / Marginal likelihood must be greater than 0");
        }
        return (likelihood * prior) / evidence;
    }
    
    /**
     * 计算多个假设下的后验概率
     * Calculate posterior probabilities for multiple hypotheses
     * 
     * @param likelihoods 似然数组 P(B|A_i) / Array of likelihoods P(B|A_i)
     * @param priors 先验概率数组 P(A_i) / Array of prior probabilities P(A_i)
     * @return 后验概率数组 P(A_i|B) / Array of posterior probabilities P(A_i|B)
     */
    public static IVector bayesTheoremMultiple(IVector likelihoods, IVector priors) {
        if (likelihoods.length() != priors.length()) {
            throw new BayesException("似然和先验概率数组长度必须相同 / Likelihood and prior arrays must have the same length");
        }
        
        // 计算未归一化的后验概率
        IVector unnormalized = likelihoods.multiply(priors);
        
        // 计算边际似然（归一化常数）
        double evidence = unnormalized.sum().doubleValue();
        
        if (evidence <= 0) {
            throw new BayesException("边际似然必须大于0 / Marginal likelihood must be greater than 0");
        }
        
        // 归一化得到后验概率
        return unnormalized.divideByScalar(evidence);
    }
    
    /**
     * 计算高斯分布的共轭先验后验分布参数
     * Calculate posterior distribution parameters for Gaussian likelihood with conjugate prior
     * 
     * @param dataMean 数据均值 / Data mean
     * @param dataVariance 数据方差 / Data variance
     * @param dataCount 数据点数量 / Number of data points
     * @param priorMean 先验均值 / Prior mean
     * @param priorVariance 先验方差 / Prior variance
     * @return 包含后验均值和方差的数组 / Array containing posterior mean and variance
     */
    public static double[] gaussianConjugatePosterior(double dataMean, double dataVariance, int dataCount,
                                                     double priorMean, double priorVariance) {
        // 后验方差
        double posteriorVariance = 1.0 / (dataCount / dataVariance + 1.0 / priorVariance);
        
        // 后验均值
        double posteriorMean = posteriorVariance * (dataCount * dataMean / dataVariance + 
                                                   priorMean / priorVariance);
        
        return new double[]{posteriorMean, posteriorVariance};
    }
}