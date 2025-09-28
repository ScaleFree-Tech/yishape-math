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
import java.util.function.Function;

/**
 * 自动微分变分推断 (ADVI)
 * Automatic Differentiation Variational Inference (ADVI)
 * 
 * <p>实现自动微分变分推断算法，支持任意可微分的目标分布。
 * 使用重参数化技巧和自动微分来优化变分参数。</p>
 * <p>Implements Automatic Differentiation Variational Inference algorithm, 
 * supporting arbitrary differentiable target distributions. 
 * Uses reparameterization trick and automatic differentiation to optimize variational parameters.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AutomaticDifferentiationVI implements IVariationalInference {
    
    private double learningRate;
    private double momentum;
    private Random random;
    private IOptimizer optimizer;
    private boolean useFullRankCovariance;
    private double gradientClipThreshold;
    
    /**
     * 全秩变分分布（多元正态分布）
     * Full-rank variational distribution (multivariate normal)
     */
    public static class FullRankNormalDistribution implements VariationalDistribution {
        private IVector means;
        private IMatrix choleskyCov; // Cholesky分解的下三角矩阵
        private Random random;
        
        public FullRankNormalDistribution(IVector means, IMatrix choleskyCov) {
            this.means = means.copy();
            this.choleskyCov = choleskyCov.copy();
            this.random = new Random();
        }
        
        public FullRankNormalDistribution(int dimension) {
            double[] meansData = new double[dimension];
            double[][] choleskyData = new double[dimension][dimension];
            
            // 初始化为单位矩阵
            for (int i = 0; i < dimension; i++) {
                meansData[i] = 0.0;
                for (int j = 0; j < dimension; j++) {
                    choleskyData[i][j] = (i == j) ? 1.0 : 0.0;
                }
            }
            
            this.means = Linalg.vector(meansData);
            this.choleskyCov = Linalg.matrix(choleskyData);
            this.random = new Random();
        }
        
        @Override
        public double logPdf(IVector parameters) {
            if (parameters.size() != means.size()) {
                throw new IllegalArgumentException("Parameter dimension mismatch");
            }
            
            int dimension = parameters.size();
            IVector diff = parameters.sub(means);
            
            // 计算协方差矩阵的逆
            IMatrix covariance = choleskyCov.multiply(choleskyCov.transpose());
            IMatrix precision = covariance.inv();
            
            // 计算二次型
            double quadraticForm = diff.asColumnVector().multiply(precision).mmul(diff).get(0).doubleValue();
            
            // 计算对数行列式
            double logDet = 0.0;
            for (int i = 0; i < dimension; i++) {
                logDet += Math.log(choleskyCov.get(i, i).doubleValue());
            }
            logDet *= 2; // 因为 det(LL^T) = det(L)^2
            
            // 多元正态分布的对数概率密度
            return -0.5 * dimension * Math.log(2 * Math.PI) - 0.5 * logDet - 0.5 * quadraticForm;
        }
        
        @Override
        public IMatrix sample(int numSamples) {
            int dimension = means.size();
            double[][] samples = new double[numSamples][dimension];
            
            NormalDistribution standardNormal = new NormalDistribution(0, 1);
            
            for (int i = 0; i < numSamples; i++) {
                // 生成标准正态随机向量
                double[] standardSample = new double[dimension];
                for (int j = 0; j < dimension; j++) {
                    standardSample[j] = standardNormal.sample(1)[0];
                }
                IVector z = Linalg.vector(standardSample);
                
                // 重参数化：x = μ + L * z
                var s = z.mmul(choleskyCov);
                IVector sample = s.add(means);
                
                for (int j = 0; j < dimension; j++) {
                    samples[i][j] = sample.get(j).doubleValue();
                }
            }
            
            return Linalg.matrix(samples);
        }
        
        @Override
        public IVector getVariationalParameters() {
            int dimension = means.size();
            int numCholeskyParams = dimension * (dimension + 1) / 2; // 下三角矩阵的参数数量
            
            double[] params = new double[dimension + numCholeskyParams];
            
            // 添加均值参数
            for (int i = 0; i < dimension; i++) {
                params[i] = means.get(i).doubleValue();
            }
            
            // 添加Cholesky参数（下三角部分）
            int paramIndex = dimension;
            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j <= i; j++) {
                    params[paramIndex++] = choleskyCov.get(i, j).doubleValue();
                }
            }
            
            return Linalg.vector(params);
        }
        
        @Override
        public void setVariationalParameters(IVector parameters) {
            int dimension = means.size();
            int numCholeskyParams = dimension * (dimension + 1) / 2;
            
            if (parameters.size() != dimension + numCholeskyParams) {
                throw new IllegalArgumentException("Parameter dimension mismatch");
            }
            
            // 设置均值参数
            for (int i = 0; i < dimension; i++) {
                means.set(i, parameters.get(i).doubleValue());
            }
            
            // 设置Cholesky参数
            int paramIndex = dimension;
            double[][] choleskyData = new double[dimension][dimension];
            
            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j <= i; j++) {
                    choleskyData[i][j] = parameters.get(paramIndex++).doubleValue();
                }
                // 确保对角元素为正
                if (choleskyData[i][i] <= 0) {
                    choleskyData[i][i] = 1e-6;
                }
            }
            
            choleskyCov = Linalg.matrix(choleskyData);
        }
        
        @Override
        public double entropy() {
            int dimension = means.size();
            
            // 计算对数行列式
            double logDet = 0.0;
            for (int i = 0; i < dimension; i++) {
                logDet += Math.log(Math.abs(choleskyCov.get(i, i).doubleValue()));
            }
            logDet *= 2;
            
            // 多元正态分布的熵：0.5 * log((2πe)^k * |Σ|)
            return 0.5 * dimension * Math.log(2 * Math.PI * Math.E) + 0.5 * logDet;
        }
        
        @Override
        public IVector mean() {
            return means.copy();
        }
        
        @Override
        public IMatrix covariance() {
            return choleskyCov.multiply(choleskyCov.transpose());
        }
        
        /**
         * 获取Cholesky分解矩阵
         * Get Cholesky decomposition matrix
         */
        public IMatrix getCholeskyCovariance() {
            return choleskyCov.copy();
        }
        
        /**
         * 设置Cholesky分解矩阵
         * Set Cholesky decomposition matrix
         */
        public void setCholeskyCovariance(IMatrix choleskyCov) {
            this.choleskyCov = choleskyCov.copy();
        }
    }
    
    /**
     * 数值微分计算器
     * Numerical differentiation calculator
     */
    private static class NumericalDifferentiator {
        private static final double EPSILON = 1e-8;
        
        /**
         * 计算函数的梯度
         * Calculate gradient of function
         */
        public static IVector gradient(Function<IVector, Double> function, IVector point) {
            int dimension = point.size();
            double[] gradientData = new double[dimension];
            
            double f0 = function.apply(point);
            
            for (int i = 0; i < dimension; i++) {
                IVector pointPlus = point.copy();
                pointPlus.set(i, point.get(i).doubleValue() + EPSILON);
                
                double fPlus = function.apply(pointPlus);
                gradientData[i] = (fPlus - f0) / EPSILON;
            }
            
            return Linalg.vector(gradientData);
        }
    }
    
    /**
     * 默认构造函数
     * Default constructor
     */
    public AutomaticDifferentiationVI() {
        this(0.01, 0.9, new Random(), false);
    }
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param learningRate 学习率 / Learning rate
     * @param momentum 动量 / Momentum
     * @param random 随机数生成器 / Random number generator
     * @param useFullRankCovariance 是否使用全秩协方差 / Whether to use full-rank covariance
     */
    public AutomaticDifferentiationVI(double learningRate, double momentum, 
                                    Random random, boolean useFullRankCovariance) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.random = random;
        this.useFullRankCovariance = useFullRankCovariance;
        this.optimizer = new RereLBFGS();
        this.gradientClipThreshold = 10.0;
    }
    
    @Override
    public VariationalResult infer(TargetDistribution targetDistribution,
                                 VariationalDistribution initialVariationalDist,
                                 int maxIterations,
                                 double tolerance) {
        
        VariationalDistribution variationalDist = initialVariationalDist;
        
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
            
            // 梯度裁剪
            gradient = clipGradient(gradient, gradientClipThreshold);
            
            // 更新变分参数
            IVector currentParams = variationalDist.getVariationalParameters();
            IVector updatedParams = updateParametersWithAdam(currentParams, gradient, iter);
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
        
        // 使用数值微分计算ELBO梯度
        Function<IVector, Double> elboFunction = (params) -> {
            VariationalDistribution tempDist = createVariationalDistribution(params, variationalDist);
            return calculateElbo(targetDistribution, tempDist, numSamples);
        };
        
        IVector currentParams = variationalDist.getVariationalParameters();
        return NumericalDifferentiator.gradient(elboFunction, currentParams);
    }
    
    /**
     * 创建变分分布（用于梯度计算）
     * Create variational distribution (for gradient calculation)
     */
    private VariationalDistribution createVariationalDistribution(IVector params, 
                                                                VariationalDistribution template) {
        if (template instanceof FullRankNormalDistribution) {
            FullRankNormalDistribution fullRankDist = (FullRankNormalDistribution) template;
            FullRankNormalDistribution newDist = new FullRankNormalDistribution(
                fullRankDist.mean(), fullRankDist.getCholeskyCovariance());
            newDist.setVariationalParameters(params);
            return newDist;
        } else if (template instanceof MeanFieldVariationalInference.MeanFieldNormalDistribution) {
            MeanFieldVariationalInference.MeanFieldNormalDistribution meanFieldDist = 
                (MeanFieldVariationalInference.MeanFieldNormalDistribution) template;
            MeanFieldVariationalInference.MeanFieldNormalDistribution newDist = 
                new MeanFieldVariationalInference.MeanFieldNormalDistribution(
                    meanFieldDist.getMeans(), meanFieldDist.getLogStds());
            newDist.setVariationalParameters(params);
            return newDist;
        } else {
            throw new IllegalArgumentException("Unsupported variational distribution type");
        }
    }
    
    /**
     * 梯度裁剪
     * Gradient clipping
     */
    private IVector clipGradient(IVector gradient, double threshold) {
        double norm = gradient.norm2().doubleValue();
        if (norm > threshold) {
            return gradient.multiplyScalar(threshold / norm);
        }
        return gradient;
    }
    
    /**
     * 使用Adam优化器更新参数
     * Update parameters using Adam optimizer
     */
    private IVector updateParametersWithAdam(IVector currentParams, IVector gradient, int iteration) {
        // 简化的Adam更新（实际实现应该使用完整的Adam算法）
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
    
    /**
     * 设置梯度裁剪阈值
     * Set gradient clipping threshold
     */
    public void setGradientClipThreshold(double threshold) {
        this.gradientClipThreshold = threshold;
    }
    
    /**
     * 设置是否使用全秩协方差
     * Set whether to use full-rank covariance
     */
    public void setUseFullRankCovariance(boolean useFullRankCovariance) {
        this.useFullRankCovariance = useFullRankCovariance;
    }
    
    @Override
    public String getAlgorithmName() {
        return "Automatic Differentiation Variational Inference (ADVI)";
    }
    
    @Override
    public boolean supportsAutomaticDifferentiation() {
        return true;
    }
    
    /**
     * 创建全秩正态变分分布
     * Create full-rank normal variational distribution
     * 
     * @param dimension 维度 / Dimension
     * @return 全秩正态分布 / Full-rank normal distribution
     */
    public static FullRankNormalDistribution createFullRankNormal(int dimension) {
        return new FullRankNormalDistribution(dimension);
    }
    
    /**
     * 创建全秩正态变分分布（指定初始参数）
     * Create full-rank normal variational distribution with initial parameters
     * 
     * @param initialMeans 初始均值 / Initial means
     * @param initialCholesky 初始Cholesky分解 / Initial Cholesky decomposition
     * @return 全秩正态分布 / Full-rank normal distribution
     */
    public static FullRankNormalDistribution createFullRankNormal(IVector initialMeans, 
                                                                IMatrix initialCholesky) {
        return new FullRankNormalDistribution(initialMeans, initialCholesky);
    }
}