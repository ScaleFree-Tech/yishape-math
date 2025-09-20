package com.reremouse.lab.math.stats.distribution.multiv;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;

/**
 * 多元指数分布实现 / Multivariate Exponential Distribution Implementation
 * 
 * <p>多元指数分布是指数分布在多维空间的推广，常用于可靠性分析和生存分析。
 * 这里实现的是独立多元指数分布，其概率密度函数为：</p>
 * <p>f(x) = ∏λᵢ * exp(-∑λᵢxᵢ) 对于 x ≥ 0</p>
 * <p>其中 λᵢ 是第i个维度的率参数。</p>
 * 
 * <p>Multivariate exponential distribution is a generalization of the exponential distribution
 * to multiple dimensions, commonly used in reliability analysis and survival analysis.
 * This implementation is for independent multivariate exponential distribution with PDF:</p>
 * <p>f(x) = ∏λᵢ * exp(-∑λᵢxᵢ) for x ≥ 0</p>
 * <p>where λᵢ is the rate parameter for the i-th dimension.</p>
 * 
 * <h3>主要特性 / Key Properties:</h3>
 * <ul>
 *   <li>无记忆性 / Memoryless property</li>
 *   <li>非负支撑 / Non-negative support</li>
 *   <li>独立的边际分布 / Independent marginal distributions</li>
 *   <li>单调递减的概率密度 / Monotonically decreasing probability density</li>
 *   <li>常用于建模等待时间 / Commonly used for modeling waiting times</li>
 * </ul>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class MultivariateExponentialDistribution implements IMultivariateDistribution<Double> {
    
    private static final long serialVersionUID = 1L;
    
    /** 率参数向量 / Rate parameters vector */
    private final IVector<Double> rates;
    
    /** 维度 / Dimensionality */
    private final int dimension;
    
    /** 对数率参数乘积 / Log product of rate parameters */
    private final double logRateProduct;
    
    /** 随机数生成器 / Random number generator */
    private final Random random;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param rates 率参数向量 / Rate parameters vector
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public MultivariateExponentialDistribution(IVector<Double> rates) {
        this(rates, new Random());
    }
    
    /**
     * 构造函数（带随机种子）
     * Constructor with random seed
     * 
     * @param rates 率参数向量 / Rate parameters vector
     * @param random 随机数生成器 / Random number generator
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public MultivariateExponentialDistribution(IVector<Double> rates, Random random) {
        validateParameters(rates);
        
        this.dimension = rates.length();
        this.rates = rates.copy();
        this.random = random;
        
        // 计算对数率参数乘积
        this.logRateProduct = computeLogRateProduct();
    }
    
    /**
     * 验证参数有效性
     * Validate parameter validity
     */
    private void validateParameters(IVector<Double> rates) {
        if (rates == null) {
            throw new IllegalArgumentException("率参数向量不能为null");
        }
        
        // 验证所有率参数都为正
        for (int i = 0; i < rates.length(); i++) {
            if (rates.get(i) <= 0) {
                throw new IllegalArgumentException(
                    String.format("率参数 %d 必须大于0，实际值: %.6f", i, rates.get(i)));
            }
        }
    }
    
    /**
     * 计算对数率参数乘积
     * Compute log product of rate parameters
     */
    private double computeLogRateProduct() {
        double logProduct = 0.0;
        for (int i = 0; i < dimension; i++) {
            logProduct += Math.log(rates.get(i));
        }
        return logProduct;
    }
    
    /**
     * 检查点是否在支撑区域内（所有分量非负）
     * Check if point is within support region (all components non-negative)
     */
    private boolean isInSupport(IVector<Double> x) {
        for (int i = 0; i < dimension; i++) {
            if (x.get(i) < 0) {
                return false;
            }
        }
        return true;
    }
    
    // ==================== IMultivariateDistribution 接口实现 ====================
    
    @Override
    public int getDimension() {
        return dimension;
    }
    
    @Override
    public String getDistributionName() {
        return "Multivariate Exponential Distribution";
    }
    
    @Override
    public String getParameterInfo() {
        return String.format("Dimension: %d, Rates: %s", dimension, rates.toString());
    }
    
    @Override
    public double pdf(IVector<Double> x) {
        validateDimension(x);
        if (!isInSupport(x)) {
            return 0.0;
        }
        return Math.exp(logPdf(x));
    }
    
    @Override
    public double logPdf(IVector<Double> x) {
        validateDimension(x);
        if (!isInSupport(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        
        // log f(x) = log(∏λᵢ) - ∑λᵢxᵢ
        double logDensity = logRateProduct;
        for (int i = 0; i < dimension; i++) {
            logDensity -= rates.get(i) * x.get(i);
        }
        
        return logDensity;
    }
    
    @Override
    public double[] pdf(List<IVector<Double>> samples) {
        double[] results = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            results[i] = pdf(samples.get(i));
        }
        return results;
    }
    
    @Override
    public double[] logPdf(List<IVector<Double>> samples) {
        double[] results = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            results[i] = logPdf(samples.get(i));
        }
        return results;
    }
    
    @Override
    public IVector<Double> getMean() {
        // 均值是 1/λᵢ
        double[] meanArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            meanArray[i] = 1.0 / rates.get(i);
        }
        return Linalg.vector(meanArray);
    }
    
    @Override
    public IMatrix<Double> getCovariance() {
        // 协方差矩阵是对角矩阵，对角元素为 1/λᵢ²
        double[][] covArray = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            double rate = rates.get(i);
            covArray[i][i] = 1.0 / (rate * rate);
        }
        return Linalg.matrix(covArray);
    }
    
    @Override
    public IMatrix<Double> getCorrelation() {
        // 由于各维度独立，相关矩阵是单位矩阵
        return Linalg.eye(dimension);
    }
    
    @Override
    public IMatrix<Double> getPrecision() {
        return getCovariance().inv();
    }
    
    @Override
    public IVector<Double> getStandardDeviation() {
        // 标准差是 1/λᵢ
        double[] stdDevArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            stdDevArray[i] = 1.0 / rates.get(i);
        }
        return Linalg.vector(stdDevArray);
    }
    
    @Override
    public double mahalanobisDistance(IVector<Double> x) {
        return Math.sqrt(squaredMahalanobisDistance(x));
    }
    
    @Override
    public double squaredMahalanobisDistance(IVector<Double> x) {
        validateDimension(x);
        
        IVector<Double> mean = getMean();
        IVector<Double> diff = x.sub(mean);
        IMatrix<Double> precision = getPrecision();
        IVector<Double> temp = precision.mmul(diff);
        return diff.dot(temp);
    }
    
    @Override
    public IVector<Double> sample() {
        double[] sampleArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            // 使用逆变换方法：X = -ln(U)/λ，其中U~Uniform(0,1)
            double u = random.nextDouble();
            sampleArray[i] = -Math.log(u) / rates.get(i);
        }
        return Linalg.vector(sampleArray);
    }
    
    @Override
    public List<IVector<Double>> sample(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("样本数量必须大于0");
        }
        
        List<IVector<Double>> samples = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            samples.add(sample());
        }
        return samples;
    }
    
    @Override
    public IMatrix<Double> sampleMatrix(int n) {
        List<IVector<Double>> samples = sample(n);
        double[][] sampleArray = new double[n][dimension];
        
        for (int i = 0; i < n; i++) {
            IVector<Double> sample = samples.get(i);
            for (int j = 0; j < dimension; j++) {
                sampleArray[i][j] = sample.get(j);
            }
        }
        
        return Linalg.matrix(sampleArray);
    }
    
    @Override
    public IMultivariateDistribution<Double> getMarginal(int... indices) {
        if (indices == null || indices.length == 0) {
            throw new IllegalArgumentException("索引不能为空");
        }
        
        // 验证索引有效性
        for (int index : indices) {
            if (index < 0 || index >= dimension) {
                throw new IllegalArgumentException("索引超出范围: " + index);
            }
        }
        
        // 提取边际率参数
        double[] marginalRatesArray = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            marginalRatesArray[i] = rates.get(indices[i]);
        }
        
        IVector<Double> marginalRates = Linalg.vector(marginalRatesArray);
        return new MultivariateExponentialDistribution(marginalRates, random);
    }
    
    @Override
    public IMultivariateDistribution<Double> getConditional(int[] conditionIndices, IVector<Double> conditionValues) {
        // 对于独立的指数分布，条件分布就是边际分布
        if (conditionIndices == null || conditionValues == null) {
            throw new IllegalArgumentException("条件索引和条件值不能为null");
        }
        if (conditionIndices.length != conditionValues.length()) {
            throw new IllegalArgumentException("条件索引和条件值的长度必须相同");
        }
        
        // 验证条件值在支撑区域内
        for (int i = 0; i < conditionIndices.length; i++) {
            double value = conditionValues.get(i);
            if (value < 0) {
                throw new IllegalArgumentException(
                    String.format("条件值 %.6f 在维度 %d 上必须非负", value, conditionIndices[i]));
            }
        }
        
        // 构建剩余维度的边际分布
        List<Integer> remainingIndices = new ArrayList<>();
        for (int i = 0; i < dimension; i++) {
            boolean isConditioned = false;
            for (int condIndex : conditionIndices) {
                if (i == condIndex) {
                    isConditioned = true;
                    break;
                }
            }
            if (!isConditioned) {
                remainingIndices.add(i);
            }
        }
        
        if (remainingIndices.isEmpty()) {
            throw new IllegalArgumentException("条件化后没有剩余维度");
        }
        
        int[] remainingArray = remainingIndices.stream().mapToInt(Integer::intValue).toArray();
        return getMarginal(remainingArray);
    }
    
    @Override
    public IMultivariateDistribution<Double> linearTransform(IMatrix<Double> A, IVector<Double> b) {
        // 线性变换后的指数分布一般不再是指数分布
        throw new UnsupportedOperationException("指数分布的线性变换结果一般不再是指数分布");
    }
    
    @Override
    public IMultivariateDistribution<Double> affineTransform(IMatrix<Double> A) {
        return linearTransform(A, Linalg.zeros(A.rows()));
    }
    
    @Override
    public double klDivergence(IMultivariateDistribution<Double> other) {
        if (!(other instanceof MultivariateExponentialDistribution)) {
            throw new IllegalArgumentException("只支持与其他多元指数分布计算KL散度");
        }
        
        MultivariateExponentialDistribution otherExp = (MultivariateExponentialDistribution) other;
        if (otherExp.dimension != this.dimension) {
            throw new IllegalArgumentException("分布维度必须相同");
        }
        
        // KL散度公式：KL(P||Q) = ∑[log(λᵢᵖ/λᵢᵠ) + λᵢᵠ/λᵢᵖ - 1]
        double klDiv = 0.0;
        for (int i = 0; i < dimension; i++) {
            double lambdaP = this.rates.get(i);
            double lambdaQ = otherExp.rates.get(i);
            klDiv += Math.log(lambdaP / lambdaQ) + lambdaQ / lambdaP - 1.0;
        }
        
        return klDiv;
    }
    
    @Override
    public double wassersteinDistance(IMultivariateDistribution<Double> other) {
        if (!(other instanceof MultivariateExponentialDistribution)) {
            throw new IllegalArgumentException("只支持与其他多元指数分布计算Wasserstein距离");
        }
        
        MultivariateExponentialDistribution otherExp = (MultivariateExponentialDistribution) other;
        if (otherExp.dimension != this.dimension) {
            throw new IllegalArgumentException("分布维度必须相同");
        }
        
        // 对于独立的指数分布，Wasserstein距离是各维度距离的和
        double distance = 0.0;
        for (int i = 0; i < dimension; i++) {
            double mean1 = 1.0 / this.rates.get(i);
            double mean2 = 1.0 / otherExp.rates.get(i);
            distance += Math.abs(mean1 - mean2);
        }
        
        return distance;
    }
    
    @Override
    public IMultivariateDistribution<Double> fit(List<IVector<Double>> samples) {
        return fitFromSamples(samples);
    }
    
    @Override
    public IMultivariateDistribution<Double> fit(List<IVector<Double>> samples, List<Double> weights) {
        return fitFromWeightedSamples(samples, weights);
    }
    
    @Override
    public boolean isElliptical() {
        return false; // 指数分布不是椭圆分布
    }
    
    @Override
    public boolean isSymmetric() {
        return false; // 指数分布不是对称分布
    }
    
    @Override
    public boolean isPositiveDefinite() {
        return true; // 协方差矩阵是对角正定的
    }
    
    @Override
    public void validateDimension(IVector<Double> x) {
        if (x == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        if (x.length() != dimension) {
            throw new IllegalArgumentException(
                String.format("输入向量维度不匹配：期望 %d，实际 %d", dimension, x.length()));
        }
    }
    
    @Override
    public double entropy() {
        // 熵公式：H = ∑[1 - log(λᵢ)]
        double entropy = 0.0;
        for (int i = 0; i < dimension; i++) {
            entropy += 1.0 - Math.log(rates.get(i));
        }
        return entropy;
    }
    
    @Override
    public IMatrix<Double> informationMatrix() {
        // 信息矩阵是协方差矩阵的逆
        return getPrecision();
    }
    
    @Override
    public ConfidenceEllipse getConfidenceEllipse(double confidence) {
        if (dimension != 2) {
            throw new UnsupportedOperationException("置信椭圆只支持二维分布");
        }
        if (confidence <= 0 || confidence >= 1) {
            throw new IllegalArgumentException("置信水平必须在(0,1)范围内");
        }
        
        // 对于二维指数分布，置信区域不是椭圆，这里用椭圆近似
        IVector<Double> mean = getMean();
        IVector<Double> stdDev = getStandardDeviation();
        
        // 使用标准差和置信水平计算椭圆参数
        double quantile = -Math.log(1 - confidence);
        double majorAxis = stdDev.get(0) * quantile;
        double minorAxis = stdDev.get(1) * quantile;
        
        return new ConfidenceEllipse(mean, majorAxis, minorAxis, 0.0);
    }
    
    // ==================== 静态工厂方法 ====================
    
    /**
     * 从样本数据估计多元指数分布参数
     * Estimate multivariate exponential distribution parameters from sample data
     */
    public static MultivariateExponentialDistribution fitFromSamples(List<IVector<Double>> samples) {
        if (samples == null || samples.isEmpty()) {
            throw new IllegalArgumentException("样本数据不能为空");
        }
        
        int n = samples.size();
        int dimension = samples.get(0).length();
        
        // 验证所有样本维度一致
        for (IVector<Double> sample : samples) {
            if (sample.length() != dimension) {
                throw new IllegalArgumentException("所有样本维度必须一致");
            }
        }
        
        // 验证所有样本值非负
        for (IVector<Double> sample : samples) {
            for (int i = 0; i < dimension; i++) {
                if (sample.get(i) < 0) {
                    throw new IllegalArgumentException("指数分布的样本值必须非负");
                }
            }
        }
        
        // 计算每个维度的样本均值
        double[] meanArray = new double[dimension];
        for (IVector<Double> sample : samples) {
            for (int i = 0; i < dimension; i++) {
                meanArray[i] += sample.get(i);
            }
        }
        for (int i = 0; i < dimension; i++) {
            meanArray[i] /= n;
        }
        
        // 率参数的最大似然估计是样本均值的倒数
        double[] ratesArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            if (meanArray[i] <= 0) {
                throw new IllegalArgumentException("样本均值必须大于0");
            }
            ratesArray[i] = 1.0 / meanArray[i];
        }
        
        IVector<Double> rates = Linalg.vector(ratesArray);
        return new MultivariateExponentialDistribution(rates);
    }
    
    /**
     * 从加权样本数据估计多元指数分布参数
     * Estimate multivariate exponential distribution parameters from weighted sample data
     */
    public static MultivariateExponentialDistribution fitFromWeightedSamples(List<IVector<Double>> samples, 
                                                                           List<Double> weights) {
        if (samples == null || samples.isEmpty() || weights == null || weights.isEmpty()) {
            throw new IllegalArgumentException("样本数据和权重不能为空");
        }
        if (samples.size() != weights.size()) {
            throw new IllegalArgumentException("样本数量和权重数量必须相等");
        }
        
        int n = samples.size();
        int dimension = samples.get(0).length();
        
        // 计算权重总和
        double weightSum = weights.stream().mapToDouble(Double::doubleValue).sum();
        
        // 计算加权均值
        double[] meanArray = new double[dimension];
        for (int i = 0; i < n; i++) {
            IVector<Double> sample = samples.get(i);
            double weight = weights.get(i);
            for (int j = 0; j < dimension; j++) {
                meanArray[j] += weight * sample.get(j);
            }
        }
        for (int i = 0; i < dimension; i++) {
            meanArray[i] /= weightSum;
        }
        
        // 率参数的加权最大似然估计
        double[] ratesArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            if (meanArray[i] <= 0) {
                throw new IllegalArgumentException("加权样本均值必须大于0");
            }
            ratesArray[i] = 1.0 / meanArray[i];
        }
        
        IVector<Double> rates = Linalg.vector(ratesArray);
        return new MultivariateExponentialDistribution(rates);
    }
    
    /**
     * 创建标准多元指数分布（所有维度的率参数都为1）
     * Create standard multivariate exponential distribution (rate parameter 1 for all dimensions)
     */
    public static MultivariateExponentialDistribution standard(int dimension) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("维度必须大于0");
        }
        
        double[] ratesArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            ratesArray[i] = 1.0;
        }
        
        IVector<Double> rates = Linalg.vector(ratesArray);
        return new MultivariateExponentialDistribution(rates);
    }
    
    /**
     * 创建具有相同率参数的多元指数分布
     * Create multivariate exponential distribution with same rate parameter for all dimensions
     */
    public static MultivariateExponentialDistribution uniform(int dimension, double rate) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("维度必须大于0");
        }
        if (rate <= 0) {
            throw new IllegalArgumentException("率参数必须大于0");
        }
        
        double[] ratesArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            ratesArray[i] = rate;
        }
        
        IVector<Double> rates = Linalg.vector(ratesArray);
        return new MultivariateExponentialDistribution(rates);
    }
    
    /**
     * 获取率参数向量
     * Get rate parameters vector
     */
    public IVector<Double> getRates() {
        return rates.copy();
    }
    
    /**
     * 计算生存函数（可靠性函数）
     * Compute survival function (reliability function)
     */
    public double survivalFunction(IVector<Double> x) {
        validateDimension(x);
        if (!isInSupport(x)) {
            return 0.0;
        }
        
        // S(x) = ∏exp(-λᵢxᵢ) = exp(-∑λᵢxᵢ)
        double exponent = 0.0;
        for (int i = 0; i < dimension; i++) {
            exponent -= rates.get(i) * x.get(i);
        }
        
        return Math.exp(exponent);
    }
    
    /**
     * 计算累积分布函数
     * Compute cumulative distribution function
     */
    public double cdf(IVector<Double> x) {
        validateDimension(x);
        if (!isInSupport(x)) {
            return 0.0;
        }
        
        // F(x) = ∏(1 - exp(-λᵢxᵢ))
        double cdf = 1.0;
        for (int i = 0; i < dimension; i++) {
            cdf *= (1.0 - Math.exp(-rates.get(i) * x.get(i)));
        }
        
        return cdf;
    }
    
    /**
     * 计算风险函数（故障率函数）
     * Compute hazard function (failure rate function)
     */
    public double hazardFunction(IVector<Double> x) {
        validateDimension(x);
        if (!isInSupport(x)) {
            return 0.0;
        }
        
        // h(x) = f(x) / S(x) = ∑λᵢ（对于指数分布，风险函数是常数）
        double hazard = 0.0;
        for (int i = 0; i < dimension; i++) {
            hazard += rates.get(i);
        }
        
        return hazard;
    }
    
    @Override
    public String toString() {
        return String.format("MultivariateExponentialDistribution(dimension=%d, rates=%s)", 
                           dimension, rates.toString());
    }
}