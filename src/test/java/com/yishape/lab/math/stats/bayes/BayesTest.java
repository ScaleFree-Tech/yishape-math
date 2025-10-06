package com.yishape.lab.math.stats.bayes;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;

/**
 * 贝叶斯分析测试类
 * Bayesian analysis test class
 */
public class BayesTest {
    
    public static void main(String[] args) {
        // 测试基本贝叶斯定理
        testBasicBayesTheorem();
        
        // 测试多个假设的贝叶斯定理
        testMultipleHypotheses();
        
        // 测试高斯共轭先验
        testGaussianConjugate();
        
        // 测试Beta-Binomial共轭先验
        testBetaBinomialConjugate();
        
        // 测试贝叶斯线性回归
        testBayesianLinearRegression();
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
        double posterior = BayesUtils.bayesTheorem(likelihood, prior, evidence);
        
        System.out.println("先验概率: " + String.format("%.2f%%", prior * 100));
        System.out.println("检测阳性时有病的后验概率: " + String.format("%.2f%%", posterior * 100));
        System.out.println();
    }
    
    /**
     * 测试多个假设的贝叶斯定理
     */
    public static void testMultipleHypotheses() {
        System.out.println("=== 测试多个假设的贝叶斯定理 ===");
        
        // 三个假设的先验概率
        IVector priors = Linalg.vector(new double[]{0.3, 0.5, 0.2});
        
        // 在每个假设下观察到数据的似然
        IVector likelihoods = Linalg.vector(new double[]{0.8, 0.6, 0.9});
        
        // 计算后验概率
        IVector posteriors = BayesUtils.bayesTheoremMultiple(likelihoods, priors);
        
        System.out.println("先验概率: " + priors);
        System.out.println("似然: " + likelihoods);
        System.out.println("后验概率: " + posteriors);
        System.out.println();
    }
    
    /**
     * 测试高斯共轭先验
     */
    public static void testGaussianConjugate() {
        System.out.println("=== 测试高斯共轭先验 ===");
        
        // 观察数据
        double dataMean = 5.0;
        double dataVariance = 1.0;
        int dataCount = 10;
        
        // 先验分布
        double priorMean = 4.0;
        double priorVariance = 2.0;
        
        // 计算后验分布参数
        double[] posteriorParams = BayesUtils.gaussianConjugatePosterior(
            dataMean, dataVariance, dataCount, priorMean, priorVariance);
        
        System.out.println("数据均值: " + dataMean);
        System.out.println("数据方差: " + dataVariance);
        System.out.println("数据点数: " + dataCount);
        System.out.println("先验均值: " + priorMean);
        System.out.println("先验方差: " + priorVariance);
        System.out.println("后验均值: " + String.format("%.2f", posteriorParams[0]));
        System.out.println("后验方差: " + String.format("%.2f", posteriorParams[1]));
        System.out.println();
    }
    
    /**
     * 测试Beta-Binomial共轭先验
     */
    public static void testBetaBinomialConjugate() {
        System.out.println("=== 测试Beta-Binomial共轭先验 ===");
        
        // Beta先验参数
        double alpha = 2.0;
        double beta = 2.0;
        
        // 观察数据：10次试验中成功7次
        int successes = 7;
        int trials = 10;
        
        // 计算后验参数
        double[] posteriorParams = BayesianInference.betaBinomialUpdate(alpha, beta, successes, trials);
        
        System.out.println("先验Beta参数: α=" + alpha + ", β=" + beta);
        System.out.println("观察数据: 成功" + successes + "次，总共" + trials + "次试验");
        System.out.println("后验Beta参数: α=" + String.format("%.2f", posteriorParams[0]) + 
                          ", β=" + String.format("%.2f", posteriorParams[1]));
        System.out.println();
    }
    
    /**
     * 测试贝叶斯线性回归
     */
    public static void testBayesianLinearRegression() {
        System.out.println("=== 测试贝叶斯线性回归 ===");
        
        try {
            // 创建一些示例数据
            double[][] xData = {
                {1.0, 2.0},
                {2.0, 3.0},
                {3.0, 4.0},
                {4.0, 5.0},
                {5.0, 6.0}
            };
            double[] yData = {3.0, 5.0, 7.0, 9.0, 11.0};
            
            IMatrix X = Linalg.matrix(xData);
            IVector y = Linalg.vector(yData);
            
            // 创建贝叶斯线性回归模型
            BayesianLinearRegression blr = new BayesianLinearRegression(2, 1.0);
            
            // 训练模型
            blr.fit(X, y);
            
            // 预测
            BayesianLinearRegression.PredictionResult result = blr.predict(X);
            
            System.out.println("特征矩阵:");
            System.out.println(X);
            System.out.println("标签向量:");
            System.out.println(y);
            System.out.println("后验均值:");
            System.out.println(blr.getPosteriorMean());
            System.out.println("预测结果:");
            for (int i = 0; i < X.rows(); i++) {
                System.out.println("样本" + i + ": 预测值=" + String.format("%.2f", result.getMean(i)) + 
                                 ", 方差=" + String.format("%.2f", result.getVariance(i)));
            }
        } catch (Exception e) {
            System.err.println("贝叶斯线性回归测试出错: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println();
    }
}