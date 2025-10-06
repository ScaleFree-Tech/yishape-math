package com.yishape.lab.math.stats.bayes;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;

/**
 * 贝叶斯线性回归实现
 * Bayesian Linear Regression Implementation
 */
public class BayesianLinearRegression {
    
    // 先验参数
    private IMatrix priorMean;         // 先验均值
    private IMatrix priorPrecision;    // 先验精度矩阵（协方差矩阵的逆）
    
    // 后验参数
    private IMatrix posteriorMean;     // 后验均值
    private IMatrix posteriorPrecision; // 后验精度矩阵
    
    // 噪声方差
    private double noiseVariance;
    
    // 是否已训练
    private boolean trained = false;
    
    /**
     * 构造函数
     * 
     * @param priorMean 先验均值 / Prior mean
     * @param priorPrecision 先验精度矩阵 / Prior precision matrix
     * @param noiseVariance 噪声方差 / Noise variance
     */
    public BayesianLinearRegression(IMatrix priorMean, IMatrix priorPrecision, double noiseVariance) {
        this.priorMean = priorMean;
        this.priorPrecision = priorPrecision;
        this.noiseVariance = noiseVariance;
    }
    
    /**
     * 构造函数（使用默认先验）
     * 
     * @param featureDim 特征维度 / Feature dimension
     * @param noiseVariance 噪声方差 / Noise variance
     */
    public BayesianLinearRegression(int featureDim, double noiseVariance) {
        // 默认零均值先验
        this.priorMean = Linalg.matrix(new double[featureDim][1]);
        
        // 默认单位精度矩阵（对应单位协方差矩阵）
        this.priorPrecision = Linalg.eye(featureDim).multiplyScalar(1.0);
        
        this.noiseVariance = noiseVariance;
    }
    
    /**
     * 训练贝叶斯线性回归模型
     * Train Bayesian linear regression model
     * 
     * @param X 特征矩阵 (样本数 × 特征数) / Feature matrix (samples × features)
     * @param y 标签向量 / Label vector
     */
    public void fit(IMatrix X, IVector y) {
        if (X.rows() != y.length()) {
            throw new BayesException("特征矩阵行数必须等于标签向量长度 / Number of rows in feature matrix must equal length of label vector");
        }
        
        // 计算后验精度矩阵: Λ_post = Λ_prior + X^T * X / σ²
        IMatrix XtX = X.transpose().mmul(X);
        IMatrix precisionUpdate = XtX.multiplyScalar(1.0 / noiseVariance);
        this.posteriorPrecision = priorPrecision.add(precisionUpdate);
        
        // 计算后验均值: μ_post = Λ_post^(-1) * (Λ_prior * μ_prior + X^T * y / σ²)
        IMatrix priorContribution = priorPrecision.mmul(priorMean);
        IVector weightedLabels = X.transpose().mmul(y).multiplyScalar(1.0 / noiseVariance);
        
        // Convert IVector to IMatrix for addition
        double[][] dataContributionArray = new double[weightedLabels.length()][1];
        for (int i = 0; i < weightedLabels.length(); i++) {
            dataContributionArray[i][0] = weightedLabels.get(i).doubleValue();
        }
        IMatrix dataContribution = Linalg.matrix(dataContributionArray);
        
        IMatrix posteriorMeanUnnormalized = priorContribution.add(dataContribution);
        
        // Since there's no direct inverse method, we'll use solve with identity matrix
        IMatrix identity = Linalg.eye(posteriorPrecision.rows());
        IMatrix posteriorCovariance = posteriorPrecision.solve(identity);
        this.posteriorMean = posteriorCovariance.mmul(posteriorMeanUnnormalized);
        
        this.trained = true;
    }
    
    /**
     * 预测新样本
     * Predict for new samples
     * 
     * @param X 新样本特征矩阵 / New sample feature matrix
     * @return 预测结果（均值和方差） / Prediction results (mean and variance)
     */
    public PredictionResult predict(IMatrix X) {
        if (!trained) {
            throw new BayesException("模型必须先训练 / Model must be trained first");
        }
        
        if (X.cols() != posteriorMean.rows()) {
            throw new BayesException("特征维度必须与训练时一致 / Feature dimension must match training dimension");
        }
        
        // 预测均值: μ_pred = X * μ_post
        IMatrix meanPredictions = X.mmul(posteriorMean);
        
        // 预测方差: σ²_pred = σ² + X * Λ_post^(-1) * X^T
        // Since there's no direct inverse method, we'll use solve with identity matrix
        IMatrix identity = Linalg.eye(posteriorPrecision.rows());
        IMatrix posteriorCovariance = posteriorPrecision.solve(identity);
        IMatrix varianceTerms = X.mmul(posteriorCovariance).mmul(X.transpose());
        double[] variances = new double[X.rows()];
        for (int i = 0; i < X.rows(); i++) {
            variances[i] = noiseVariance + varianceTerms.get(i, i).doubleValue();
        }
        
        return new PredictionResult(meanPredictions, variances);
    }
    
    /**
     * 获取后验均值
     * Get posterior mean
     * 
     * @return 后验均值 / Posterior mean
     */
    public IMatrix getPosteriorMean() {
        return posteriorMean;
    }
    
    /**
     * 获取后验精度矩阵
     * Get posterior precision matrix
     * 
     * @return 后验精度矩阵 / Posterior precision matrix
     */
    public IMatrix getPosteriorPrecision() {
        return posteriorPrecision;
    }
    
    /**
     * 预测结果类
     * Prediction result class
     */
    public static class PredictionResult {
        private final IMatrix means;
        private final double[] variances;
        
        public PredictionResult(IMatrix means, double[] variances) {
            this.means = means;
            this.variances = variances;
        }
        
        public IMatrix getMeans() {
            return means;
        }
        
        public double[] getVariances() {
            return variances;
        }
        
        public double getMean(int index) {
            return means.get(index, 0).doubleValue();
        }
        
        public double getVariance(int index) {
            return variances[index];
        }
    }
}