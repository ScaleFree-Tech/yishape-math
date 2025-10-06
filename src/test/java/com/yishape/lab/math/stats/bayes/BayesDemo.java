package com.yishape.lab.math.stats.bayes;
import java.util.Arrays;

/**
 * 简单的贝叶斯分析演示类
 * Simple Bayesian analysis demo class
 */
public class BayesDemo {
    
    public static void main(String[] args) {
        // 测试基本贝叶斯定理
        testBasicBayesTheorem();
    }
    
    /**
     * 测试基本贝叶斯定理
     */
    public static void testBasicBayesTheorem() {
        System.out.println("=== 测试基本贝叶斯定理 ===");
        
        // 假设有一种疾病，先验概率为1%
        double prior = 0.01;
        
        // 如果有疾病，检测呈阳性的概率为99%
        double likelihood = 0.99;
        
        // 检测呈阳性的边际概率（包括假阳性）
        double evidence = 0.01 * 0.99 + 0.99 * 0.05; // P(有病)*P(阳性|有病) + P(无病)*P(阳性|无病)
        
        // 计算后验概率
        double posterior = bayesTheorem(likelihood, prior, evidence);
        
        System.out.println("先验概率: " + String.format("%.2f%%", prior * 100));
        System.out.println("检测阳性时有病的后验概率: " + String.format("%.2f%%", posterior * 100));
        System.out.println();
    }
    
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
            throw new RuntimeException("边际似然必须大于0 / Marginal likelihood must be greater than 0");
        }
        return (likelihood * prior) / evidence;
    }
}