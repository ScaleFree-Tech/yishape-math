package com.reremouse.lab.math.stats.bayes.variational;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.distribution.NormalDistribution;
import com.reremouse.lab.math.optimize.IOptimizer;
import com.reremouse.lab.math.optimize.newton.RereLBFGS;
import java.util.Random;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;
import java.util.List;

/**
 * 平均场变分推断
 * Mean Field Variational Inference
 * 
 * <p>实现平均场变分推断算法，假设后验分布可以分解为独立的因子分布。
 * 使用坐标上升算法优化变分参数。</p>
 * <p>Implements mean field variational inference algorithm, assuming the posterior 
 * distribution can be factorized into independent factor distributions. 
 * Uses coordinate ascent algorithm to optimize variational parameters.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class MeanFieldVariationalInference implements IVariationalInference {
    
    private double learningRate;
    private double momentum;
    private Random random;
    private IOptimizer optimizer;
    
    /**
     * 平均场变分分布（假设为多元正态分布的对角协方差形式）
     * Mean field variational distribution (diagonal covariance multivariate normal)
     */
    public static class MeanFieldNormalDistribution implements VariationalDistribution {
        private IVector means;
        private IVector logStds;
        private Random random;
        
        public MeanFieldNormalDistribution(IVector means, IVector logStds) {
            this.means = means.copy();
            this.logStds = logStds.copy();
            this.random = new Random();
        }
        
        public MeanFieldNormalDistribution(int dimension) {
            double[] meansData = new double[dimension];
            double[] logStdsData = new double[dimension];
            
            for (int i = 0; i < dimension; i++) {
                meansData[i] = 0.0;
                logStdsData[i] = 0.0; // log(1) = 0
            }
            
            this.means = Linalg.vector(meansData);
            this.logStds = Linalg.vector(logStdsData);
            this.random = new Random();
        }
        
        @Override
        public double logPdf(IVector parameters) {
            if (parameters.size() != means.size()) {
                throw new IllegalArgumentException("Parameter dimension mismatch");
            }
            
            double logProb = 0.0;
            
            for (int i = 0; i < parameters.size(); i++) {
                double x = parameters.get(i).doubleValue();
                double mu = means.get(i).doubleValue();
                double logSigma = logStds.get(i).doubleValue();
                double sigma = Math.exp(logSigma);
                
                // 正态分布的对数概率密度
                logProb += -0.5 * Math.log(2 * Math.PI) - logSigma - 
                          0.5 * Math.pow((x - mu) / sigma, 2);
            }
            
            return logProb;
        }
        
        @Override
        public IMatrix sample(int numSamples) {
            int dimension = means.size();
            double[][] samples = new double[numSamples][dimension];
            
            for (int i = 0; i < numSamples; i++) {
                for (int j = 0; j < dimension; j++) {
                    double mu = means.get(j).doubleValue();
                    double sigma = Math.exp(logStds.get(j).doubleValue());
                    
                    NormalDistribution normal = new NormalDistribution(mu, sigma);
                    samples[i][j] = normal.sample(1)[0];
                }
            }
            
            return Linalg.matrix(samples);
        }
        
        @Override
        public IVector getVariationalParameters() {
            // 将均值和对数标准差连接为一个向量
            int dimension = means.size();
            double[] params = new double[2 * dimension];
            
            for (int i = 0; i < dimension; i++) {
                params[i] = means.get(i).doubleValue();
                params[i + dimension] = logStds.get(i).doubleValue();
            }
            
            return Linalg.vector(params);
        }
        
        @Override
        public void setVariationalParameters(IVector parameters) {
            int dimension = means.size();
            if (parameters.size() != 2 * dimension) {
                throw new IllegalArgumentException("Parameter dimension mismatch");
            }
            
            for (int i = 0; i < dimension; i++) {
                means.set(i, parameters.get(i).doubleValue());
                logStds.set(i, parameters.get(i + dimension).doubleValue());
            }
        }
        
        @Override
        public double entropy() {
            double entropy = 0.0;
            
            for (int i = 0; i < means.size(); i++) {
                double logSigma = logStds.get(i).doubleValue();
                // 正态分布的熵：0.5 * log(2πe) + log(σ)
                entropy += 0.5 * Math.log(2 * Math.PI * Math.E) + logSigma;
            }
            
            return entropy;
        }
        
        @Override
        public IVector mean() {
            return means.copy();
        }
        
        @Override
        public IMatrix covariance() {
            int dimension = means.size();
            double[][] covData = new double[dimension][dimension];
            
            // 对角协方差矩阵
            for (int i = 0; i < dimension; i++) {
                double variance = Math.exp(2 * logStds.get(i).doubleValue());
                covData[i][i] = variance;
            }
            
            return Linalg.matrix(covData);
        }
        
        /**
         * 获取均值向量
         * Get mean vector
         */
        public IVector getMeans() {
            return means.copy();
        }
        
        /**
         * 获取对数标准差向量
         * Get log standard deviation vector
         */
        public IVector getLogStds() {
            return logStds.copy();
        }
        
        /**
         * 设置均值向量
         * Set mean vector
         */
        public void setMeans(IVector means) {
            this.means = means.copy();
        }
        
        /**
         * 设置对数标准差向量
         * Set log standard deviation vector
         */
        public void setLogStds(IVector logStds) {
            this.logStds = logStds.copy();
        }
    }
    
    /**
     * 默认构造函数
     * Default constructor
     */
    public MeanFieldVariationalInference() {
        this(0.01, 0.9, new Random());
    }
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param learningRate 学习率 / Learning rate
     * @param momentum 动量 / Momentum
     * @param random 随机数生成器 / Random number generator
     */
    public MeanFieldVariationalInference(double learningRate, double momentum, Random random) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.random = random;
        this.optimizer = new RereLBFGS();
    }
    
    @Override
    public VariationalResult infer(TargetDistribution targetDistribution,
                                 VariationalDistribution initialVariationalDist,
                                 int maxIterations,
                                 double tolerance) {
        
        if (!(initialVariationalDist instanceof MeanFieldNormalDistribution)) {
            throw new IllegalArgumentException(
                "MeanFieldVariationalInference requires MeanFieldNormalDistribution");
        }
        
        MeanFieldNormalDistribution variationalDist = 
            (MeanFieldNormalDistribution) initialVariationalDist;
        
        List<Double> elboHistory = new ArrayList<>();
        Map<String, Object> diagnostics = new HashMap<>();
        
        double previousElbo = Double.NEGATIVE_INFINITY;
        boolean converged = false;
        
        for (int iter = 0; iter < maxIterations; iter++) {
            // 计算当前ELBO
            double currentElbo = calculateElbo(targetDistribution, variationalDist, 1000);
            elboHistory.add(currentElbo);
            
            // 检查收敛性
            if (iter > 0 && Math.abs(currentElbo - previousElbo) < tolerance) {
                converged = true;
                diagnostics.put("convergence_iteration", iter);
                break;
            }
            
            // 计算ELBO梯度
            IVector gradient = calculateElboGradient(targetDistribution, variationalDist, 1000);
            
            // 更新变分参数
            IVector currentParams = variationalDist.getVariationalParameters();
            IVector updatedParams = updateParameters(currentParams, gradient);
            variationalDist.setVariationalParameters(updatedParams);
            
            previousElbo = currentElbo;
            
            // 记录诊断信息
            if (iter % 100 == 0) {
                diagnostics.put("elbo_at_" + iter, currentElbo);
                diagnostics.put("gradient_norm_at_" + iter, gradient.norm2().doubleValue());
            }
        }
        
        // 转换ELBO历史为数组
        double[] elbos = elboHistory.stream().mapToDouble(Double::doubleValue).toArray();
        
        diagnostics.put("final_elbo", elbos[elbos.length - 1]);
        diagnostics.put("total_iterations", elboHistory.size());
        
        return new VariationalResult(variationalDist, elbos, converged, 
                                   elboHistory.size(), diagnostics);
    }
    
    @Override
    public double calculateElbo(TargetDistribution targetDistribution,
                              VariationalDistribution variationalDist,
                              int numSamples) {
        
        // ELBO = E_q[log p(x)] - E_q[log q(x)]
        // 其中 q 是变分分布，p 是目标分布
        
        IMatrix samples = variationalDist.sample(numSamples);
        double expectedLogTarget = 0.0;
        double expectedLogVariational = 0.0;
        
        for (int i = 0; i < numSamples; i++) {
            IVector sample = samples.getRow(i);
            
            expectedLogTarget += targetDistribution.logPdf(sample);
            expectedLogVariational += variationalDist.logPdf(sample);
        }
        
        expectedLogTarget /= numSamples;
        expectedLogVariational /= numSamples;
        
        return expectedLogTarget - expectedLogVariational;
    }
    
    @Override
    public IVector calculateElboGradient(TargetDistribution targetDistribution,
                                       VariationalDistribution variationalDist,
                                       int numSamples) {
        
        if (!(variationalDist instanceof MeanFieldNormalDistribution)) {
            throw new IllegalArgumentException("Unsupported variational distribution type");
        }
        
        MeanFieldNormalDistribution meanFieldDist = (MeanFieldNormalDistribution) variationalDist;
        int dimension = meanFieldDist.getMeans().size();
        
        // 使用重参数化技巧计算梯度
        IMatrix samples = meanFieldDist.sample(numSamples);
        
        double[] meanGradients = new double[dimension];
        double[] logStdGradients = new double[dimension];
        
        for (int i = 0; i < numSamples; i++) {
            IVector sample = samples.getRow(i);
            
            // 计算目标分布的梯度
            IVector targetGradient = targetDistribution.logPdfGradient(sample);
            
            // 计算变分分布的梯度
            for (int j = 0; j < dimension; j++) {
                double x = sample.get(j).doubleValue();
                double mu = meanFieldDist.getMeans().get(j).doubleValue();
                double logSigma = meanFieldDist.getLogStds().get(j).doubleValue();
                double sigma = Math.exp(logSigma);
                
                // 对均值的梯度
                double meanGrad = targetGradient.get(j).doubleValue() + (x - mu) / (sigma * sigma);
                meanGradients[j] += meanGrad;
                
                // 对对数标准差的梯度
                double logStdGrad = targetGradient.get(j).doubleValue() * (x - mu) / sigma +
                                   (Math.pow((x - mu) / sigma, 2) - 1);
                logStdGradients[j] += logStdGrad;
            }
        }
        
        // 平均梯度
        for (int j = 0; j < dimension; j++) {
            meanGradients[j] /= numSamples;
            logStdGradients[j] /= numSamples;
        }
        
        // 组合梯度
        double[] combinedGradient = new double[2 * dimension];
        System.arraycopy(meanGradients, 0, combinedGradient, 0, dimension);
        System.arraycopy(logStdGradients, 0, combinedGradient, dimension, dimension);
        
        return Linalg.vector(combinedGradient);
    }
    
    /**
     * 更新参数
     * Update parameters
     */
    private IVector updateParameters(IVector currentParams, IVector gradient) {
        // 简单的梯度上升更新
        double[] updatedParams = new double[currentParams.size()];
        
        for (int i = 0; i < currentParams.size(); i++) {
            updatedParams[i] = currentParams.get(i).doubleValue() + 
                              learningRate * gradient.get(i).doubleValue();
        }
        
        return Linalg.vector(updatedParams);
    }
    
    @Override
    public void setOptimizerParameters(double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
    }
    
    @Override
    public String getAlgorithmName() {
        return "Mean Field Variational Inference";
    }
    
    @Override
    public boolean supportsAutomaticDifferentiation() {
        return false; // 当前实现使用手动梯度计算
    }
    
    /**
     * 创建平均场正态变分分布
     * Create mean field normal variational distribution
     * 
     * @param dimension 维度 / Dimension
     * @return 平均场正态分布 / Mean field normal distribution
     */
    public static MeanFieldNormalDistribution createMeanFieldNormal(int dimension) {
        return new MeanFieldNormalDistribution(dimension);
    }
    
    /**
     * 创建平均场正态变分分布（指定初始参数）
     * Create mean field normal variational distribution with initial parameters
     * 
     * @param initialMeans 初始均值 / Initial means
     * @param initialLogStds 初始对数标准差 / Initial log standard deviations
     * @return 平均场正态分布 / Mean field normal distribution
     */
    public static MeanFieldNormalDistribution createMeanFieldNormal(IVector initialMeans, 
                                                                   IVector initialLogStds) {
        return new MeanFieldNormalDistribution(initialMeans, initialLogStds);
    }
}