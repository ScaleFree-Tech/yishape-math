package com.yishape.lab.math.stats.bayes.models;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.stats.distribution.NormalDistribution;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.newton.RereLBFGS;

import java.util.Random;

/**
 * 高斯过程
 * Gaussian Process
 * 
 * <p>实现高斯过程回归和分类模型，支持多种核函数和超参数优化。
 * 高斯过程是一种非参数贝叶斯方法，可以用于回归和分类任务。</p>
 * <p>Implements Gaussian Process regression and classification models with support for 
 * multiple kernel functions and hyperparameter optimization. Gaussian Process is a 
 * non-parametric Bayesian method for regression and classification tasks.</p>
 * 
 * <p>模型：
 * f(x) ~ GP(m(x), k(x, x'))
 * y = f(x) + ε, ε ~ N(0, σ²) (回归)
 * y = sign(f(x)) (分类)</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class GaussianProcess {
    
    /**
     * 核函数接口
     * Kernel function interface
     */
    public interface KernelFunction {
        /**
         * 计算核函数值
         * Calculate kernel function value
         * 
         * @param x1 第一个输入 / First input
         * @param x2 第二个输入 / Second input
         * @param hyperparameters 超参数 / Hyperparameters
         * @return 核函数值 / Kernel function value
         */
        double evaluate(IVector x1, IVector x2, IVector hyperparameters);
        
        /**
         * 计算核函数关于超参数的梯度
         * Calculate gradient of kernel function with respect to hyperparameters
         * 
         * @param x1 第一个输入 / First input
         * @param x2 第二个输入 / Second input
         * @param hyperparameters 超参数 / Hyperparameters
         * @return 梯度向量 / Gradient vector
         */
        IVector gradient(IVector x1, IVector x2, IVector hyperparameters);
        
        /**
         * 获取超参数数量
         * Get number of hyperparameters
         * 
         * @return 超参数数量 / Number of hyperparameters
         */
        int getNumHyperparameters();
        
        /**
         * 获取核函数名称
         * Get kernel function name
         * 
         * @return 核函数名称 / Kernel function name
         */
        String getName();
    }
    
    /**
     * 均值函数接口
     * Mean function interface
     */
    public interface MeanFunction {
        /**
         * 计算均值函数值
         * Calculate mean function value
         * 
         * @param x 输入 / Input
         * @param parameters 参数 / Parameters
         * @return 均值函数值 / Mean function value
         */
        double evaluate(IVector x, IVector parameters);
        
        /**
         * 计算均值函数关于参数的梯度
         * Calculate gradient of mean function with respect to parameters
         * 
         * @param x 输入 / Input
         * @param parameters 参数 / Parameters
         * @return 梯度向量 / Gradient vector
         */
        IVector gradient(IVector x, IVector parameters);
        
        /**
         * 获取参数数量
         * Get number of parameters
         * 
         * @return 参数数量 / Number of parameters
         */
        int getNumParameters();
    }
    
    // 模型组件
    private KernelFunction kernelFunction;
    private MeanFunction meanFunction;
    private boolean isRegression;
    
    // 训练数据
    private IMatrix XTrain;
    private IVector yTrain;
    private int numTrainingPoints;
    private boolean dataSet;
    
    // 模型参数
    private IVector kernelHyperparameters;
    private IVector meanParameters;
    private double noiseVariance;  // 仅用于回归
    
    // 预计算的矩阵
    private IMatrix KTrain;        // 训练数据的核矩阵
    private IMatrix KTrainInv;     // 核矩阵的逆
    private IMatrix L;             // Cholesky分解
    private boolean matricesComputed;
    
    // 优化相关
    private boolean hyperparametersOptimized;
    
    /**
     * 构造函数（回归）
     * Constructor (regression)
     * 
     * @param kernelFunction 核函数 / Kernel function
     * @param meanFunction 均值函数 / Mean function
     */
    public GaussianProcess(KernelFunction kernelFunction, MeanFunction meanFunction) {
        this.kernelFunction = kernelFunction;
        this.meanFunction = meanFunction;
        this.isRegression = true;
        this.dataSet = false;
        this.matricesComputed = false;
        this.hyperparametersOptimized = false;
        this.noiseVariance = 1e-6;  // 默认噪声方差
        
        // 初始化参数
        initializeParameters();
    }
    
    /**
     * 构造函数（分类）
     * Constructor (classification)
     * 
     * @param kernelFunction 核函数 / Kernel function
     * @param meanFunction 均值函数 / Mean function
     * @param isRegression 是否为回归任务 / Whether it's a regression task
     */
    public GaussianProcess(KernelFunction kernelFunction, MeanFunction meanFunction, boolean isRegression) {
        this.kernelFunction = kernelFunction;
        this.meanFunction = meanFunction;
        this.isRegression = isRegression;
        this.dataSet = false;
        this.matricesComputed = false;
        this.hyperparametersOptimized = false;
        
        if (isRegression) {
            this.noiseVariance = 1e-6;
        }
        
        // 初始化参数
        initializeParameters();
    }
    
    /**
     * 初始化参数
     * Initialize parameters
     */
    private void initializeParameters() {
        // 初始化核函数超参数
        int numKernelParams = kernelFunction.getNumHyperparameters();
        this.kernelHyperparameters = Linalg.vector(numKernelParams);
        for (int i = 0; i < numKernelParams; i++) {
            kernelHyperparameters.set(i, 1.0);  // 默认值
        }
        
        // 初始化均值函数参数
        if (meanFunction != null) {
            int numMeanParams = meanFunction.getNumParameters();
            this.meanParameters = Linalg.vector(numMeanParams);
            for (int i = 0; i < numMeanParams; i++) {
                meanParameters.set(i, 0.0);  // 默认值
            }
        }
    }
    
    /**
     * 设置训练数据
     * Set training data
     * 
     * @param X 输入特征矩阵 / Input feature matrix
     * @param y 目标值向量 / Target value vector
     */
    public void setTrainingData(IMatrix X, IVector y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Training data cannot be null");
        }
        
        if (X.rows() != y.size()) {
            throw new IllegalArgumentException("Number of samples must match between X and y");
        }
        
        // 对于分类任务，检查标签
        if (!isRegression) {
            for (int i = 0; i < y.size(); i++) {
                double label = y.get(i).doubleValue();
                if (label != -1.0 && label != 1.0) {
                    throw new IllegalArgumentException("Classification labels must be -1 or 1");
                }
            }
        }
        
        this.XTrain = X.copy();
        this.yTrain = y.copy();
        this.numTrainingPoints = X.rows();
        this.dataSet = true;
        this.matricesComputed = false;
        this.hyperparametersOptimized = false;
    }
    
    /**
     * 设置核函数超参数
     * Set kernel hyperparameters
     * 
     * @param hyperparameters 超参数向量 / Hyperparameter vector
     */
    public void setKernelHyperparameters(IVector hyperparameters) {
        if (hyperparameters.size() != kernelFunction.getNumHyperparameters()) {
            throw new IllegalArgumentException("Hyperparameter dimension mismatch");
        }
        
        this.kernelHyperparameters = hyperparameters.copy();
        this.matricesComputed = false;
    }
    
    /**
     * 设置噪声方差（仅用于回归）
     * Set noise variance (regression only)
     * 
     * @param noiseVariance 噪声方差 / Noise variance
     */
    public void setNoiseVariance(double noiseVariance) {
        if (!isRegression) {
            throw new IllegalStateException("Noise variance only applicable for regression");
        }
        
        if (noiseVariance <= 0) {
            throw new IllegalArgumentException("Noise variance must be positive");
        }
        
        this.noiseVariance = noiseVariance;
        this.matricesComputed = false;
    }
    
    /**
     * 优化超参数
     * Optimize hyperparameters
     * 
     * @param maxIterations 最大迭代次数 / Maximum iterations
     */
    public void optimizeHyperparameters(int maxIterations) {
        if (!dataSet) {
            throw new IllegalStateException("Training data must be set before optimization");
        }
        
        // 创建优化器
        IOptimizer optimizer = new RereLBFGS();
        
        // 定义目标函数（负对数边际似然）
        IObjectiveFunction objective = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector params) {
                return -logMarginalLikelihood(params);
            }
        };
        
        // 定义梯度函数
        IGradientFunction gradient = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector params) {
                return logMarginalLikelihoodGradient(params).multiplyScalar(-1.0);
            }
        };
        
        // 构建初始参数向量
        IVector initialParams = buildParameterVector();
        
        // 执行优化
        OptResult result = optimizer.optimize(initialParams, objective, gradient);
        
        if (!result.isConverged()) {
            throw new RuntimeException("Hyperparameter optimization failed: " + result.getConvergenceReason());
        }
        
        // 更新参数
        updateParametersFromVector(result.getOptimalPoint());
        this.hyperparametersOptimized = true;
        this.matricesComputed = false;
    }
    
    /**
     * 预测（回归）
     * Predict (regression)
     * 
     * @param XTest 测试输入矩阵 / Test input matrix
     * @return 预测结果 / Prediction result
     */
    public GPRegressionResult predictRegression(IMatrix XTest) {
        if (!isRegression) {
            throw new IllegalStateException("This method is only for regression tasks");
        }
        
        if (!dataSet) {
            throw new IllegalStateException("Training data must be set before prediction");
        }
        
        // 确保矩阵已计算
        if (!matricesComputed) {
            computeMatrices();
        }
        
        int numTestPoints = XTest.rows();
        IVector meanPredictions = Linalg.vector(numTestPoints);
        IVector variancePredictions = Linalg.vector(numTestPoints);
        
        for (int i = 0; i < numTestPoints; i++) {
            IVector xTest = XTest.getRow(i);
            
            // 计算测试点与训练点的核向量
            IVector kStar = Linalg.vector(numTrainingPoints);
            for (int j = 0; j < numTrainingPoints; j++) {
                IVector xTrain = XTrain.getRow(j);
                kStar.set(j, kernelFunction.evaluate(xTest, xTrain, kernelHyperparameters));
            }
            
            // 计算测试点的核值
            double kStarStar = kernelFunction.evaluate(xTest, xTest, kernelHyperparameters);
            
            // 计算均值函数值
            double meanValue = 0.0;
            if (meanFunction != null) {
                meanValue = meanFunction.evaluate(xTest, meanParameters);
            }
            
            // 计算训练数据的均值函数值
            IVector meanTrain = Linalg.vector(numTrainingPoints);
            for (int j = 0; j < numTrainingPoints; j++) {
                if (meanFunction != null) {
                    meanTrain.set(j, meanFunction.evaluate(XTrain.getRow(j), meanParameters));
                } else {
                    meanTrain.set(j, 0.0);
                }
            }
            
            // 预测均值
            IVector yAdjusted = yTrain.sub(meanTrain);
            IVector alpha = KTrainInv.mmul(yAdjusted);
            double predictedMean = meanValue + kStar.dot(alpha).doubleValue();
            
            // 预测方差
            IVector v = KTrainInv.mmul(kStar);
            double predictedVariance = kStarStar - kStar.dot(v).doubleValue();
            
            meanPredictions.set(i, predictedMean);
            variancePredictions.set(i, Math.max(predictedVariance, 1e-12));  // 确保方差为正
        }
        
        return new GPRegressionResult(meanPredictions, variancePredictions);
    }
    
    /**
     * 预测（分类）
     * Predict (classification)
     * 
     * @param XTest 测试输入矩阵 / Test input matrix
     * @param numSamples 采样数量 / Number of samples
     * @param random 随机数生成器 / Random number generator
     * @return 预测结果 / Prediction result
     */
    public GPClassificationResult predictClassification(IMatrix XTest, int numSamples, Random random) {
        if (isRegression) {
            throw new IllegalStateException("This method is only for classification tasks");
        }
        
        if (!dataSet) {
            throw new IllegalStateException("Training data must be set before prediction");
        }
        
        // 对于分类，使用Laplace近似
        return predictClassificationLaplace(XTest);
    }
    
    /**
     * 使用Laplace近似进行分类预测
     * Classification prediction using Laplace approximation
     */
    private GPClassificationResult predictClassificationLaplace(IMatrix XTest) {
        // 计算后验模式（MAP估计）
        IVector fMap = findPosteriorMode();
        
        // 计算Hessian矩阵（对于分类，是负的二阶导数对角矩阵）
        IMatrix W = computeClassificationHessian(fMap);
        
        // 计算核矩阵
        IMatrix K = computeKernelMatrix(XTrain, XTrain);
        
        // 计算 (K^-1 + W)^-1
        IMatrix KInv = K.inv();
        IMatrix A = KInv.add(W);
        IMatrix AInv = A.inv();
        
        int numTestPoints = XTest.rows();
        IVector meanPredictions = Linalg.vector(numTestPoints);
        IVector variancePredictions = Linalg.vector(numTestPoints);
        
        for (int i = 0; i < numTestPoints; i++) {
            IVector xTest = XTest.getRow(i);
            
            // 计算测试点与训练点的核向量
            IVector kStar = Linalg.vector(numTrainingPoints);
            for (int j = 0; j < numTrainingPoints; j++) {
                IVector xTrain = XTrain.getRow(j);
                kStar.set(j, kernelFunction.evaluate(xTest, xTrain, kernelHyperparameters));
            }
            
            // 预测均值
            double predictedMean = kStar.dot(W.mmul(fMap)).doubleValue();
            
            // 预测方差
            double kStarStar = kernelFunction.evaluate(xTest, xTest, kernelHyperparameters);
            IVector v = AInv.mmul(kStar);
            double predictedVariance = kStarStar - kStar.dot(v).doubleValue();
            
            meanPredictions.set(i, predictedMean);
            variancePredictions.set(i, Math.max(predictedVariance, 1e-12));
        }
        
        // 计算概率
        IVector probabilities = Linalg.vector(numTestPoints);
        for (int i = 0; i < numTestPoints; i++) {
            double mean = meanPredictions.get(i).doubleValue();
            double variance = variancePredictions.get(i).doubleValue();
            
            // 使用probit近似
            double c = Math.PI / 8.0;
            double prob = sigmoid(mean / Math.sqrt(1 + c * variance));
            probabilities.set(i, prob);
        }
        
        return new GPClassificationResult(meanPredictions, variancePredictions, probabilities);
    }
    
    /**
     * 计算对数边际似然
     * Calculate log marginal likelihood
     */
    private double logMarginalLikelihood(IVector params) {
        updateParametersFromVector(params);
        
        if (isRegression) {
            return logMarginalLikelihoodRegression();
        } else {
            return logMarginalLikelihoodClassification();
        }
    }
    
    /**
     * 计算回归的对数边际似然
     * Calculate log marginal likelihood for regression
     */
    private double logMarginalLikelihoodRegression() {
        // 计算核矩阵
        IMatrix K = computeKernelMatrix(XTrain, XTrain);
        
        // 添加噪声项
        for (int i = 0; i < numTrainingPoints; i++) {
            K.set(i, i, K.get(i, i).doubleValue() + noiseVariance);
        }
        
        // Cholesky分解
        IMatrix L = choleskyDecomposition(K);
        
        // 计算均值函数值
        IVector meanTrain = Linalg.vector(numTrainingPoints);
        for (int i = 0; i < numTrainingPoints; i++) {
            if (meanFunction != null) {
                meanTrain.set(i, meanFunction.evaluate(XTrain.getRow(i), meanParameters));
            } else {
                meanTrain.set(i, 0.0);
            }
        }
        
        // 计算 y - m(X)
        IVector yAdjusted = yTrain.sub(meanTrain);
        
        // 解 L * alpha = y - m(X)
        IVector alpha = solveTriangular(L, yAdjusted, true);
        
        // 计算对数似然
        double dataFit = -0.5 * alpha.dot(alpha).doubleValue();
        double complexity = -sumLogDiagonal(L);
        double normalization = -0.5 * numTrainingPoints * Math.log(2 * Math.PI);
        
        return dataFit + complexity + normalization;
    }
    
    /**
     * 计算分类的对数边际似然
     * Calculate log marginal likelihood for classification
     */
    private double logMarginalLikelihoodClassification() {
        // 使用Laplace近似
        IVector fMap = findPosteriorMode();
        
        // 计算对数似然
        double logLikelihood = 0.0;
        for (int i = 0; i < numTrainingPoints; i++) {
            double fi = fMap.get(i).doubleValue();
            double yi = yTrain.get(i).doubleValue();
            logLikelihood += logSigmoid(yi * fi);
        }
        
        // 计算先验项
        IMatrix K = computeKernelMatrix(XTrain, XTrain);
        IMatrix KInv = K.inv();
        double logPrior = -0.5 * fMap.dot(KInv.mmul(fMap)).doubleValue();
        
        // 计算Hessian项
        IMatrix W = computeClassificationHessian(fMap);
        IMatrix A = KInv.add(W);
        double logDetA = Math.log(Math.abs(A.det().doubleValue()));
        double logDetK = Math.log(Math.abs(K.det().doubleValue()));
        
        return logLikelihood + logPrior - 0.5 * (logDetA - logDetK);
    }
    
    /**
     * 计算对数边际似然的梯度
     * Calculate gradient of log marginal likelihood
     */
    private IVector logMarginalLikelihoodGradient(IVector params) {
        updateParametersFromVector(params);
        
        if (isRegression) {
            return logMarginalLikelihoodGradientRegression();
        } else {
            return logMarginalLikelihoodGradientClassification();
        }
    }
    
    /**
     * 计算回归的对数边际似然梯度
     * Calculate gradient of log marginal likelihood for regression
     */
    private IVector logMarginalLikelihoodGradientRegression() {
        int numParams = kernelFunction.getNumHyperparameters();
        if (meanFunction != null) {
            numParams += meanFunction.getNumParameters();
        }
        if (isRegression) {
            numParams += 1;  // 噪声方差
        }
        
        IVector gradient = Linalg.vector(numParams);
        
        // 计算核矩阵和其逆
        IMatrix K = computeKernelMatrix(XTrain, XTrain);
        for (int i = 0; i < numTrainingPoints; i++) {
            K.set(i, i, K.get(i, i).doubleValue() + noiseVariance);
        }
        IMatrix KInv = K.inv();
        
        // 计算alpha
        IVector meanTrain = Linalg.vector(numTrainingPoints);
        for (int i = 0; i < numTrainingPoints; i++) {
            if (meanFunction != null) {
                meanTrain.set(i, meanFunction.evaluate(XTrain.getRow(i), meanParameters));
            } else {
                meanTrain.set(i, 0.0);
            }
        }
        IVector yAdjusted = yTrain.sub(meanTrain);
        IVector alpha = KInv.mmul(yAdjusted);
        
        // 核函数超参数的梯度
        int paramIndex = 0;
        for (int p = 0; p < kernelFunction.getNumHyperparameters(); p++) {
            IMatrix dKdTheta = computeKernelMatrixDerivative(p);
            double trace = KInv.multiply(dKdTheta).trace().doubleValue();
            double quadForm = alpha.dot(dKdTheta.mmul(alpha)).doubleValue();
            gradient.set(paramIndex++, 0.5 * (trace - quadForm));
        }
        
        // 均值函数参数的梯度（如果有）
        if (meanFunction != null) {
            for (int p = 0; p < meanFunction.getNumParameters(); p++) {
                IVector dMdTheta = Linalg.vector(numTrainingPoints);
                for (int i = 0; i < numTrainingPoints; i++) {
                    IVector gradMean = meanFunction.gradient(XTrain.getRow(i), meanParameters);
                    dMdTheta.set(i, gradMean.get(p));
                }
                double gradValue = alpha.dot(dMdTheta).doubleValue();
                gradient.set(paramIndex++, gradValue);
            }
        }
        
        // 噪声方差的梯度
        if (isRegression) {
            double trace = KInv.trace().doubleValue();
            double quadForm = alpha.dot(alpha).doubleValue();
            gradient.set(paramIndex, 0.5 * (trace - quadForm));
        }
        
        return gradient;
    }
    
    /**
     * 计算分类的对数边际似然梯度
     * Calculate gradient of log marginal likelihood for classification
     */
    private IVector logMarginalLikelihoodGradientClassification() {
        // 简化实现，返回零梯度
        int numParams = kernelFunction.getNumHyperparameters();
        if (meanFunction != null) {
            numParams += meanFunction.getNumParameters();
        }
        return Linalg.vector(numParams);
    }
    
    /**
     * 计算矩阵
     * Compute matrices
     */
    private void computeMatrices() {
        if (!dataSet) {
            throw new IllegalStateException("Training data must be set");
        }
        
        // 计算核矩阵
        this.KTrain = computeKernelMatrix(XTrain, XTrain);
        
        if (isRegression) {
            // 添加噪声项
            for (int i = 0; i < numTrainingPoints; i++) {
                KTrain.set(i, i, KTrain.get(i, i).doubleValue() + noiseVariance);
            }
        }
        
        // 计算逆矩阵
        this.KTrainInv = KTrain.inv();
        
        // Cholesky分解
        this.L = choleskyDecomposition(KTrain);
        
        this.matricesComputed = true;
    }
    
    /**
     * 计算核矩阵
     * Compute kernel matrix
     */
    private IMatrix computeKernelMatrix(IMatrix X1, IMatrix X2) {
        int n1 = X1.rows();
        int n2 = X2.rows();
        IMatrix K = Linalg.zeros(n1, n2);
        
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                double kernelValue = kernelFunction.evaluate(
                    X1.getRow(i), X2.getRow(j), kernelHyperparameters);
                K.set(i, j, kernelValue);
            }
        }
        
        return K;
    }
    
    /**
     * 计算核矩阵关于超参数的导数
     * Compute derivative of kernel matrix with respect to hyperparameter
     */
    private IMatrix computeKernelMatrixDerivative(int paramIndex) {
        IMatrix dK = Linalg.zeros(numTrainingPoints, numTrainingPoints);
        
        for (int i = 0; i < numTrainingPoints; i++) {
            for (int j = 0; j < numTrainingPoints; j++) {
                IVector grad = kernelFunction.gradient(
                    XTrain.getRow(i), XTrain.getRow(j), kernelHyperparameters);
                dK.set(i, j, grad.get(paramIndex));
            }
        }
        
        return dK;
    }
    
    /**
     * 寻找后验模式（用于分类）
     * Find posterior mode (for classification)
     */
    private IVector findPosteriorMode() {
        // 使用Newton-Raphson方法
        IVector f = Linalg.vector(numTrainingPoints);  // 初始化为零
        
        IMatrix K = computeKernelMatrix(XTrain, XTrain);
        IMatrix KInv = K.inv();
        
        for (int iter = 0; iter < 100; iter++) {
            // 计算梯度
            IVector grad = Linalg.vector(numTrainingPoints);
            for (int i = 0; i < numTrainingPoints; i++) {
                double fi = f.get(i).doubleValue();
                double yi = yTrain.get(i).doubleValue();
                double pi = sigmoid(fi);
                grad.set(i, yi * (1 - pi) - (1 - yi) * pi);
            }
            grad = grad.sub(KInv.mmul(f));
            
            // 计算Hessian
            IMatrix W = computeClassificationHessian(f);
            IMatrix H = KInv.add(W);
            
            // Newton更新
            IVector delta = H.inv().mmul(grad);
            f = f.add(delta);
            
            // 检查收敛
            if (delta.norm2().doubleValue() < 1e-6) {
                break;
            }
        }
        
        return f;
    }
    
    /**
     * 计算分类的Hessian矩阵
     * Compute Hessian matrix for classification
     */
    private IMatrix computeClassificationHessian(IVector f) {
        int n = f.size();
        IMatrix W = Linalg.zeros(n, n);
        
        for (int i = 0; i < n; i++) {
            double fi = f.get(i).doubleValue();
            double pi = sigmoid(fi);
            // 对于伯努利分布，Hessian是对角矩阵，对角元素为pi*(1-pi)
            double wii = pi * (1 - pi);
            W.set(i, i, wii);
        }
        
        return W;
    }
    

    
    /**
     * 构建参数向量
     * Build parameter vector
     */
    private IVector buildParameterVector() {
        int totalParams = kernelFunction.getNumHyperparameters();
        if (meanFunction != null) {
            totalParams += meanFunction.getNumParameters();
        }
        if (isRegression) {
            totalParams += 1;  // 噪声方差
        }
        
        IVector params = Linalg.vector(totalParams);
        int index = 0;
        
        // 核函数超参数
        for (int i = 0; i < kernelFunction.getNumHyperparameters(); i++) {
            params.set(index++, kernelHyperparameters.get(i));
        }
        
        // 均值函数参数
        if (meanFunction != null) {
            for (int i = 0; i < meanFunction.getNumParameters(); i++) {
                params.set(index++, meanParameters.get(i));
            }
        }
        
        // 噪声方差
        if (isRegression) {
            params.set(index, Math.log(noiseVariance));  // 对数空间优化
        }
        
        return params;
    }
    
    /**
     * 从参数向量更新参数
     * Update parameters from parameter vector
     */
    private void updateParametersFromVector(IVector params) {
        int index = 0;
        
        // 核函数超参数
        for (int i = 0; i < kernelFunction.getNumHyperparameters(); i++) {
            kernelHyperparameters.set(i, params.get(index++));
        }
        
        // 均值函数参数
        if (meanFunction != null) {
            for (int i = 0; i < meanFunction.getNumParameters(); i++) {
                meanParameters.set(i, params.get(index++));
            }
        }
        
        // 噪声方差
        if (isRegression) {
            this.noiseVariance = Math.exp(params.get(index).doubleValue());
        }
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
                    double value = matrix.get(j, j).doubleValue() - sum;
                    if (value <= 0) {
                        value = 1e-12;  // 数值稳定性
                    }
                    L.set(j, j, Math.sqrt(value));
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
     * 解三角线性系统
     * Solve triangular linear system
     */
    private IVector solveTriangular(IMatrix L, IVector b, boolean lower) {
        int n = L.rows();
        IVector x = Linalg.vector(n);
        
        if (lower) {
            // 前向替换
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int j = 0; j < i; j++) {
                    sum += L.get(i, j).doubleValue() * x.get(j).doubleValue();
                }
                x.set(i, (b.get(i).doubleValue() - sum) / L.get(i, i).doubleValue());
            }
        } else {
            // 后向替换
            for (int i = n - 1; i >= 0; i--) {
                double sum = 0;
                for (int j = i + 1; j < n; j++) {
                    sum += L.get(i, j).doubleValue() * x.get(j).doubleValue();
                }
                x.set(i, (b.get(i).doubleValue() - sum) / L.get(i, i).doubleValue());
            }
        }
        
        return x;
    }
    
    /**
     * 计算对角线元素的对数和
     * Calculate sum of logarithms of diagonal elements
     */
    private double sumLogDiagonal(IMatrix matrix) {
        double sum = 0;
        for (int i = 0; i < matrix.rows(); i++) {
            sum += Math.log(Math.abs(matrix.get(i, i).doubleValue()));
        }
        return sum;
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
     * 对数Sigmoid函数
     * Log sigmoid function
     */
    private double logSigmoid(double x) {
        if (x > 0) {
            return -Math.log(1.0 + Math.exp(-x));
        } else {
            return x - Math.log(1.0 + Math.exp(x));
        }
    }
    
    /**
     * 获取核函数
     * Get kernel function
     */
    public KernelFunction getKernelFunction() {
        return kernelFunction;
    }
    
    /**
     * 获取均值函数
     * Get mean function
     */
    public MeanFunction getMeanFunction() {
        return meanFunction;
    }
    
    /**
     * 获取核函数超参数
     * Get kernel hyperparameters
     */
    public IVector getKernelHyperparameters() {
        return kernelHyperparameters.copy();
    }
    
    /**
     * 获取噪声方差
     * Get noise variance
     */
    public double getNoiseVariance() {
        return noiseVariance;
    }
    
    /**
     * 是否为回归任务
     * Whether it's a regression task
     */
    public boolean isRegression() {
        return isRegression;
    }
    
    /**
     * 高斯过程回归结果类
     * Gaussian Process regression result class
     */
    public static class GPRegressionResult {
        private final IVector meanPredictions;
        private final IVector variancePredictions;
        
        public GPRegressionResult(IVector meanPredictions, IVector variancePredictions) {
            this.meanPredictions = meanPredictions;
            this.variancePredictions = variancePredictions;
        }
        
        public IVector getMeanPredictions() { return meanPredictions; }
        public IVector getVariancePredictions() { return variancePredictions; }
        
        public IVector getStandardDeviations() {
            IVector std = Linalg.vector(variancePredictions.size());
            for (int i = 0; i < variancePredictions.size(); i++) {
                std.set(i, Math.sqrt(variancePredictions.get(i).doubleValue()));
            }
            return std;
        }
        
        public IMatrix getCredibleIntervals(double alpha) {
            int n = meanPredictions.size();
            IMatrix intervals = Linalg.zeros(n, 2);
            
            double z = new NormalDistribution(0, 1).ppf(1 - alpha / 2);
            
            for (int i = 0; i < n; i++) {
                double mean = meanPredictions.get(i).doubleValue();
                double std = Math.sqrt(variancePredictions.get(i).doubleValue());
                
                intervals.set(i, 0, mean - z * std);
                intervals.set(i, 1, mean + z * std);
            }
            
            return intervals;
        }
    }
    
    /**
     * 高斯过程分类结果类
     * Gaussian Process classification result class
     */
    public static class GPClassificationResult {
        private final IVector meanPredictions;
        private final IVector variancePredictions;
        private final IVector probabilities;
        
        public GPClassificationResult(IVector meanPredictions, IVector variancePredictions, IVector probabilities) {
            this.meanPredictions = meanPredictions;
            this.variancePredictions = variancePredictions;
            this.probabilities = probabilities;
        }
        
        public IVector getMeanPredictions() { return meanPredictions; }
        public IVector getVariancePredictions() { return variancePredictions; }
        public IVector getProbabilities() { return probabilities; }
        
        public IVector getPredictedLabels() {
            IVector labels = Linalg.vector(probabilities.size());
            for (int i = 0; i < probabilities.size(); i++) {
                labels.set(i, probabilities.get(i).doubleValue() > 0.5 ? 1.0 : -1.0);
            }
            return labels;
        }
    }
}