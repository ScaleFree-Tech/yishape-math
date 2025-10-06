package com.yishape.lab.math.stats.bayes.models;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.stats.bayes.mcmc.IMCMCSampler;
import com.yishape.lab.math.stats.bayes.mcmc.MetropolisHastingsSampler;
import com.yishape.lab.math.stats.bayes.variational.IVariationalInference;
import com.yishape.lab.math.stats.bayes.variational.MeanFieldVariationalInference;
import com.yishape.lab.math.stats.distribution.NormalDistribution;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.newton.RereLBFGS;

import java.util.Random;
import java.util.List;

/**
 * 贝叶斯逻辑回归
 * Bayesian Logistic Regression
 * 
 * <p>实现贝叶斯逻辑回归模型，支持多种推断方法包括MCMC和变分推断。
 * 使用正态分布作为回归系数的先验分布。</p>
 * <p>Implements Bayesian logistic regression model with support for multiple inference methods 
 * including MCMC and variational inference. Uses normal distribution as prior for regression coefficients.</p>
 * 
 * <p>模型：
 * β ~ MVN(μ₀, Σ₀)
 * y_i | x_i, β ~ Bernoulli(σ(x_i^T β))
 * 其中 σ(z) = 1/(1 + exp(-z)) 是sigmoid函数</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class BayesianLogisticRegression {
    
    private final int numFeatures;
    private final boolean includeIntercept;
    
    // 先验参数
    private IVector priorMean;
    private IMatrix priorCovariance;
    private IMatrix priorPrecision;
    
    // 数据
    private IMatrix X;  // 特征矩阵
    private IVector y;  // 标签向量
    private boolean dataSet;
    
    // 推断结果
    private IMatrix posteriorSamples;
    private IVector posteriorMean;
    private IMatrix posteriorCovariance;
    private boolean inferenceCompleted;
    
    // 推断方法
    private String inferenceMethod;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param numFeatures 特征数量 / Number of features
     * @param includeIntercept 是否包含截距项 / Whether to include intercept
     */
    public BayesianLogisticRegression(int numFeatures, boolean includeIntercept) {
        if (numFeatures <= 0) {
            throw new IllegalArgumentException("Number of features must be positive");
        }
        
        this.numFeatures = numFeatures;
        this.includeIntercept = includeIntercept;
        this.dataSet = false;
        this.inferenceCompleted = false;
        
        // 设置默认先验（弱信息先验）
        int totalParams = includeIntercept ? numFeatures + 1 : numFeatures;
        setDefaultPrior(totalParams);
    }
    
    /**
     * 设置默认先验
     * Set default prior
     */
    private void setDefaultPrior(int totalParams) {
        // 使用零均值、大方差的正态先验
        this.priorMean = Linalg.vector(totalParams);
        for (int i = 0; i < totalParams; i++) {
            priorMean.set(i, 0.0);
        }
        
        // 对角协方差矩阵，方差为10
        double[][] identityData = new double[totalParams][totalParams];
        for (int i = 0; i < totalParams; i++) {
            identityData[i][i] = 10.0;
        }
        this.priorCovariance = Linalg.matrix(identityData);
        for (int i = 0; i < totalParams; i++) {
            priorCovariance.set(i, i, 10.0);
        }
        
        this.priorPrecision = priorCovariance.inv();
    }
    
    /**
     * 设置先验分布
     * Set prior distribution
     * 
     * @param priorMean 先验均值 / Prior mean
     * @param priorCovariance 先验协方差矩阵 / Prior covariance matrix
     */
    public void setPrior(IVector priorMean, IMatrix priorCovariance) {
        if (priorMean == null || priorCovariance == null) {
            throw new IllegalArgumentException("Prior mean and covariance cannot be null");
        }
        
        int totalParams = includeIntercept ? numFeatures + 1 : numFeatures;
        if (priorMean.size() != totalParams || 
            priorCovariance.rows() != totalParams || 
            priorCovariance.cols() != totalParams) {
            throw new IllegalArgumentException("Prior dimensions do not match model parameters");
        }
        
        this.priorMean = priorMean.copy();
        this.priorCovariance = priorCovariance.copy();
        this.priorPrecision = priorCovariance.inv();
    }
    
    /**
     * 设置训练数据
     * Set training data
     * 
     * @param X 特征矩阵 / Feature matrix
     * @param y 标签向量 / Label vector
     */
    public void setData(IMatrix X, IVector y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Features and labels cannot be null");
        }
        
        if (X.rows() != y.size()) {
            throw new IllegalArgumentException("Number of samples must match between features and labels");
        }
        
        if (X.cols() != numFeatures) {
            throw new IllegalArgumentException("Number of features must match model specification");
        }
        
        // 检查标签是否为二进制
        for (int i = 0; i < y.size(); i++) {
            double label = y.get(i).doubleValue();
            if (label != 0.0 && label != 1.0) {
                throw new IllegalArgumentException("Labels must be binary (0 or 1)");
            }
        }
        
        // 如果包含截距项，添加常数列
        if (includeIntercept) {
            double[][] xWithInterceptData = new double[X.rows()][X.cols() + 1];
            IMatrix XWithIntercept = Linalg.matrix(xWithInterceptData);
            
            // 设置截距列
            for (int i = 0; i < X.rows(); i++) {
                XWithIntercept.set(i, 0, 1.0);
            }
            
            // 复制原始特征
            for (int i = 0; i < X.rows(); i++) {
                for (int j = 0; j < X.cols(); j++) {
                    XWithIntercept.set(i, j + 1, X.get(i, j));
                }
            }
            
            this.X = XWithIntercept;
        } else {
            this.X = X.copy();
        }
        
        this.y = y.copy();
        this.dataSet = true;
        this.inferenceCompleted = false;
    }
    
    /**
     * 使用MCMC进行推断
     * Perform inference using MCMC
     * 
     * @param numSamples 采样数量 / Number of samples
     * @param burnIn 预热期 / Burn-in period
     * @param random 随机数生成器 / Random number generator
     */
    public void inferMCMC(int numSamples, int burnIn, Random random) {
        if (!dataSet) {
            throw new IllegalStateException("Data must be set before inference");
        }
        
        // 创建目标分布（后验分布）
        IMCMCSampler.TargetDistribution targetDistribution = new LogisticRegressionPosterior();
        
        // 创建Metropolis-Hastings采样器
        MetropolisHastingsSampler sampler = new MetropolisHastingsSampler();
        
        // 执行采样
        IMCMCSampler.SamplingResult result = sampler.sample(targetDistribution, priorMean, numSamples, burnIn);
        
        // 转换为矩阵形式
        this.posteriorSamples = result.getSamples();
        
        // 计算后验统计量
        computePosteriorStatistics();
        
        this.inferenceMethod = "MCMC";
        this.inferenceCompleted = true;
    }
    
    /**
     * 使用变分推断
     * Perform variational inference
     * 
     * @param maxIterations 最大迭代次数 / Maximum iterations
     * @param tolerance 收敛容忍度 / Convergence tolerance
     */
    public void inferVariational(int maxIterations, double tolerance) {
        if (!dataSet) {
            throw new IllegalStateException("Data must be set before inference");
        }
        
        // 创建目标分布
        IVariationalInference.TargetDistribution targetDistribution = new LogisticRegressionTarget();
        
        // 创建变分推断器
        MeanFieldVariationalInference vi = new MeanFieldVariationalInference();
        
        // 创建初始变分分布（使用单位对数标准差）
        int dimension = priorMean.size();
        IVector initialLogStds = Linalg.vector(dimension);
        for (int i = 0; i < dimension; i++) {
            initialLogStds.set(i, 0.0); // log(1) = 0
        }
        IVariationalInference.VariationalDistribution initialDist = 
            MeanFieldVariationalInference.createMeanFieldNormal(priorMean, initialLogStds);
        
        // 执行变分推断
        IVariationalInference.VariationalResult result = vi.infer(
            targetDistribution, initialDist, maxIterations, tolerance);
        
        // 提取结果
        this.posteriorMean = result.getPosteriorMean();
        this.posteriorCovariance = result.getPosteriorCovariance();
        
        this.inferenceMethod = "Variational";
        this.inferenceCompleted = true;
    }
    
    /**
     * 使用最大后验估计
     * Perform Maximum A Posteriori (MAP) estimation
     */
    public void inferMAP() {
        if (!dataSet) {
            throw new IllegalStateException("Data must be set before inference");
        }
        
        // 创建优化器
        IOptimizer optimizer = new RereLBFGS();
        
        // 定义目标函数（负对数后验）
        IObjectiveFunction objective = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector params) {
                return -logPosterior(params);
            }
        };
        
        // 定义梯度函数
        IGradientFunction gradient = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector params) {
                return logPosteriorGradient(params).multiplyScalar(-1.0);
            }
        };
        
        // 执行优化
        OptResult result = optimizer.optimize(priorMean, objective, gradient);
        
        if (!result.isConverged()) {
            throw new RuntimeException("MAP optimization failed: " + result.getConvergenceReason());
        }
        
        // 设置MAP估计作为后验均值
        this.posteriorMean = result.getOptimalPoint();
        
        // 计算Hessian矩阵的逆作为后验协方差的近似
        this.posteriorCovariance = computeHessianInverse(posteriorMean);
        
        this.inferenceMethod = "MAP";
        this.inferenceCompleted = true;
    }
    
    /**
     * 预测新样本
     * Predict new samples
     * 
     * @param XTest 测试特征矩阵 / Test feature matrix
     * @return 预测概率 / Predicted probabilities
     */
    public IVector predict(IMatrix XTest) {
        if (!inferenceCompleted) {
            throw new IllegalStateException("Inference must be completed before prediction");
        }
        
        if (XTest.cols() != numFeatures) {
            throw new IllegalArgumentException("Test features dimension mismatch");
        }
        
        // 添加截距项（如果需要）
        IMatrix XTestProcessed;
        if (includeIntercept) {
            double[][] xTestData = new double[XTest.rows()][XTest.cols() + 1];
            XTestProcessed = Linalg.matrix(xTestData);
            for (int i = 0; i < XTest.rows(); i++) {
                XTestProcessed.set(i, 0, 1.0);
                for (int j = 0; j < XTest.cols(); j++) {
                    XTestProcessed.set(i, j + 1, XTest.get(i, j));
                }
            }
        } else {
            XTestProcessed = XTest.copy();
        }
        
        IVector predictions = Linalg.vector(XTest.rows());
        
        if ("MCMC".equals(inferenceMethod)) {
            // 使用后验样本进行预测
            for (int i = 0; i < XTest.rows(); i++) {
                IVector xTest = XTestProcessed.getRow(i);
                double avgProb = 0.0;
                
                for (int s = 0; s < posteriorSamples.rows(); s++) {
                    IVector beta = posteriorSamples.getRow(s);
                    double logit = xTest.dot(beta).doubleValue();
                    avgProb += sigmoid(logit);
                }
                
                predictions.set(i, avgProb / posteriorSamples.rows());
            }
        } else {
            // 使用后验均值进行预测
            for (int i = 0; i < XTest.rows(); i++) {
                IVector xTest = XTestProcessed.getRow(i);
                double logit = xTest.dot(posteriorMean).doubleValue();
                predictions.set(i, sigmoid(logit));
            }
        }
        
        return predictions;
    }
    
    /**
     * 预测新样本（包含不确定性）
     * Predict new samples with uncertainty
     * 
     * @param XTest 测试特征矩阵 / Test feature matrix
     * @param numSamples 预测样本数量 / Number of prediction samples
     * @param random 随机数生成器 / Random number generator
     * @return 预测样本矩阵 / Prediction sample matrix
     */
    public IMatrix predictWithUncertainty(IMatrix XTest, int numSamples, Random random) {
        if (!inferenceCompleted) {
            throw new IllegalStateException("Inference must be completed before prediction");
        }
        
        if (XTest.cols() != numFeatures) {
            throw new IllegalArgumentException("Test features dimension mismatch");
        }
        
        // 添加截距项（如果需要）
        IMatrix XTestProcessed;
        if (includeIntercept) {
            double[][] xTestData = new double[XTest.rows()][XTest.cols() + 1];
            XTestProcessed = Linalg.matrix(xTestData);
            for (int i = 0; i < XTest.rows(); i++) {
                XTestProcessed.set(i, 0, 1.0);
                for (int j = 0; j < XTest.cols(); j++) {
                    XTestProcessed.set(i, j + 1, XTest.get(i, j));
                }
            }
        } else {
            XTestProcessed = XTest.copy();
        }
        
        IMatrix predictionSamples = Linalg.zeros(numSamples, XTest.rows());
        
        if ("MCMC".equals(inferenceMethod) && posteriorSamples != null) {
            // 从后验样本中随机选择
            for (int s = 0; s < numSamples; s++) {
                int sampleIdx = random.nextInt(posteriorSamples.rows());
                IVector beta = posteriorSamples.getRow(sampleIdx);
                
                for (int i = 0; i < XTest.rows(); i++) {
                    IVector xTest = XTestProcessed.getRow(i);
                    double logit = xTest.dot(beta).doubleValue();
                    double prob = sigmoid(logit);
                    predictionSamples.set(s, i, prob);
                }
            }
        } else {
            // 从后验分布中采样
            for (int s = 0; s < numSamples; s++) {
                // 从后验正态分布中采样参数
                IVector beta = sampleFromMultivariateNormal(posteriorMean, posteriorCovariance, random);
                
                for (int i = 0; i < XTest.rows(); i++) {
                    IVector xTest = XTestProcessed.getRow(i);
                    double logit = xTest.dot(beta).doubleValue();
                    double prob = sigmoid(logit);
                    predictionSamples.set(s, i, prob);
                }
            }
        }
        
        return predictionSamples;
    }
    
    /**
     * 计算模型的对数边际似然
     * Calculate log marginal likelihood of the model
     * 
     * @return 对数边际似然 / Log marginal likelihood
     */
    public double calculateLogMarginalLikelihood() {
        if (!inferenceCompleted) {
            throw new IllegalStateException("Inference must be completed before calculating marginal likelihood");
        }
        
        // 使用Laplace近似
        double logPrior = logMultivariateNormalPdf(posteriorMean, priorMean, priorCovariance);
        double logLikelihood = 0.0;
        
        for (int i = 0; i < X.rows(); i++) {
            IVector xi = X.getRow(i);
            double yi = y.get(i).doubleValue();
            double logit = xi.dot(posteriorMean).doubleValue();
            double prob = sigmoid(logit);
            
            if (yi == 1.0) {
                logLikelihood += Math.log(prob + 1e-15);
            } else {
                logLikelihood += Math.log(1.0 - prob + 1e-15);
            }
        }
        
        // Laplace近似的修正项
        IMatrix hessian = computeHessian(posteriorMean);
        double logDetHessian = Math.log(Math.abs(hessian.det().doubleValue()));
        double dimension = posteriorMean.size();
        double laplaceCorrection = 0.5 * dimension * Math.log(2 * Math.PI) - 0.5 * logDetHessian;
        
        return logPrior + logLikelihood + laplaceCorrection;
    }
    
    /**
     * 获取后验统计量
     * Get posterior statistics
     * 
     * @return 后验统计量 / Posterior statistics
     */
    public PosteriorStatistics getPosteriorStatistics() {
        if (!inferenceCompleted) {
            throw new IllegalStateException("Inference must be completed before getting statistics");
        }
        
        return new PosteriorStatistics(posteriorMean, posteriorCovariance, inferenceMethod);
    }
    
    /**
     * 计算后验统计量
     * Compute posterior statistics
     */
    private void computePosteriorStatistics() {
        if (posteriorSamples == null) {
            return;
        }
        
        int numSamples = posteriorSamples.rows();
        int numParams = posteriorSamples.cols();
        
        // 计算后验均值
        this.posteriorMean = Linalg.vector(numParams);
        for (int j = 0; j < numParams; j++) {
            double sum = 0.0;
            for (int i = 0; i < numSamples; i++) {
                sum += posteriorSamples.get(i, j).doubleValue();
            }
            posteriorMean.set(j, sum / numSamples);
        }
        
        // 计算后验协方差
        this.posteriorCovariance = Linalg.zeros(numParams, numParams);
        for (int j1 = 0; j1 < numParams; j1++) {
            for (int j2 = 0; j2 < numParams; j2++) {
                double covariance = 0.0;
                for (int i = 0; i < numSamples; i++) {
                    double diff1 = posteriorSamples.get(i, j1).doubleValue() - posteriorMean.get(j1).doubleValue();
                    double diff2 = posteriorSamples.get(i, j2).doubleValue() - posteriorMean.get(j2).doubleValue();
                    covariance += diff1 * diff2;
                }
                posteriorCovariance.set(j1, j2, covariance / (numSamples - 1));
            }
        }
    }
    
    /**
     * 计算对数后验概率
     * Calculate log posterior probability
     */
    private double logPosterior(IVector beta) {
        double logPrior = logMultivariateNormalPdf(beta, priorMean, priorCovariance);
        double logLikelihood = 0.0;
        
        for (int i = 0; i < X.rows(); i++) {
            IVector xi = X.getRow(i);
            double yi = y.get(i).doubleValue();
            double logit = xi.dot(beta).doubleValue();
            
            if (yi == 1.0) {
                logLikelihood += logit - Math.log(1.0 + Math.exp(logit));
            } else {
                logLikelihood += -Math.log(1.0 + Math.exp(logit));
            }
        }
        
        return logPrior + logLikelihood;
    }
    
    /**
     * 计算对数后验概率的梯度
     * Calculate gradient of log posterior probability
     */
    private IVector logPosteriorGradient(IVector beta) {
        // 先验梯度
        IVector priorGrad = priorPrecision.mmul(priorMean.sub(beta));
        
        // 似然梯度
        IVector likelihoodGrad = Linalg.vector(beta.size());
        for (int i = 0; i < X.rows(); i++) {
            IVector xi = X.getRow(i);
            double yi = y.get(i).doubleValue();
            double logit = xi.dot(beta).doubleValue();
            double prob = sigmoid(logit);
            
            IVector contribution = xi.multiplyScalar(yi - prob);
            likelihoodGrad = likelihoodGrad.add(contribution);
        }
        
        return priorGrad.add(likelihoodGrad);
    }
    
    /**
     * 计算Hessian矩阵
     * Calculate Hessian matrix
     */
    private IMatrix computeHessian(IVector beta) {
        IMatrix hessian = priorPrecision.multiplyScalar(-1.0);
        
        for (int i = 0; i < X.rows(); i++) {
            IVector xi = X.getRow(i);
            double logit = xi.dot(beta).doubleValue();
            double prob = sigmoid(logit);
            double weight = prob * (1.0 - prob);
            
            IMatrix xiOuter = xi.outer(xi);
            hessian = hessian.sub(xiOuter.multiplyScalar(weight));
        }
        
        return hessian;
    }
    
    /**
     * 计算Hessian矩阵的逆
     * Calculate inverse of Hessian matrix
     */
    private IMatrix computeHessianInverse(IVector beta) {
        IMatrix hessian = computeHessian(beta);
        return hessian.inv().multiplyScalar(-1.0);
    }
    
    /**
     * Sigmoid函数
     * Sigmoid function
     */
    private double sigmoid(double x) {
        if (x > 500) return 1.0;
        if (x < -500) return 0.0;
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    /**
     * 计算多元正态分布的对数概率密度
     * Calculate log probability density of multivariate normal distribution
     */
    private double logMultivariateNormalPdf(IVector x, IVector mean, IMatrix covariance) {
        IVector diff = x.sub(mean);
        IMatrix precision = covariance.inv();
        double quadForm = diff.dot(precision.mmul(diff)).doubleValue();
        double logDet = Math.log(Math.abs(covariance.det().doubleValue()));
        double dimension = x.size();
        
        return -0.5 * (dimension * Math.log(2 * Math.PI) + logDet + quadForm);
    }
    
    /**
     * 从多元正态分布中采样
     * Sample from multivariate normal distribution
     */
    private IVector sampleFromMultivariateNormal(IVector mean, IMatrix covariance, Random random) {
        int dimension = mean.size();
        IVector standardNormal = Linalg.vector(dimension);
        
        for (int i = 0; i < dimension; i++) {
            NormalDistribution normal = new NormalDistribution(0, 1);
            standardNormal.set(i, normal.sample());
        }
        
        // Cholesky分解
        IMatrix L = choleskyDecomposition(covariance);
        
        return mean.add(L.mmul(standardNormal));
    }
    
    /**
     * Cholesky分解
     * Cholesky decomposition
     */
    private IMatrix choleskyDecomposition(IMatrix matrix) {
        int n = matrix.rows();
        IMatrix L = Linalg.zeros(n, n);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (i == j) {
                    double sum = 0;
                    for (int k = 0; k < j; k++) {
                        sum += L.get(j, k).doubleValue() * L.get(j, k).doubleValue();
                    }
                    L.set(j, j, Math.sqrt(matrix.get(j, j).doubleValue() - sum));
                } else {
                    double sum = 0;
                    for (int k = 0; k < j; k++) {
                        sum += L.get(i, k).doubleValue() * L.get(j, k).doubleValue();
                    }
                    L.set(i, j, (matrix.get(i, j).doubleValue() - sum) / L.get(j, j).doubleValue());
                }
            }
        }
        
        return L;
    }
    
    /**
     * 逻辑回归后验分布类
     * Logistic regression posterior distribution class
     */
    private class LogisticRegressionPosterior implements IMCMCSampler.TargetDistribution {
        @Override
        public double logPdf(IVector x) {
            return logPosterior(x);
        }
    }
    
    /**
     * 逻辑回归目标分布类（用于变分推断）
     * Logistic regression target distribution class (for variational inference)
     */
    private class LogisticRegressionTarget implements IVariationalInference.TargetDistribution {
        @Override
        public double logPdf(IVector x) {
            return logPosterior(x);
        }
        
        @Override
        public IVector logPdfGradient(IVector x) {
            return logPosteriorGradient(x);
        }
    }
    
    /**
     * 后验统计量类
     * Posterior statistics class
     */
    public static class PosteriorStatistics {
        private final IVector mean;
        private final IMatrix covariance;
        private final String method;
        
        public PosteriorStatistics(IVector mean, IMatrix covariance, String method) {
            this.mean = mean;
            this.covariance = covariance;
            this.method = method;
        }
        
        public IVector getMean() { return mean; }
        public IMatrix getCovariance() { return covariance; }
        public String getMethod() { return method; }
        
        public IVector getStandardDeviations() {
            IVector std = Linalg.vector(mean.size());
            for (int i = 0; i < mean.size(); i++) {
                std.set(i, Math.sqrt(covariance.get(i, i).doubleValue()));
            }
            return std;
        }
        
        public double[] getCredibleInterval(int paramIndex, double alpha) {
            if (paramIndex < 0 || paramIndex >= mean.size()) {
                throw new IllegalArgumentException("Invalid parameter index");
            }
            
            double paramMean = mean.get(paramIndex).doubleValue();
            double paramStd = Math.sqrt(covariance.get(paramIndex, paramIndex).doubleValue());
            
            // 使用正态近似
            NormalDistribution normal = new NormalDistribution(paramMean, paramStd);
            double lowerQuantile = alpha / 2;
            double upperQuantile = 1 - alpha / 2;
            
            return new double[] {
                normal.ppf(lowerQuantile),
                normal.ppf(upperQuantile)
            };
        }
    }
}