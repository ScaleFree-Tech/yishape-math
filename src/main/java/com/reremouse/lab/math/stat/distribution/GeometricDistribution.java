package com.reremouse.lab.math.stat.distribution;

import com.reremouse.lab.math.RereMathUtil;

/**
 * 几何分布 (Geometric Distribution)
 * 
 * 几何分布是在一系列独立伯努利试验中，第一次成功所需的试验次数的概率分布。
 * 其概率质量函数为：P(X=k) = (1-p)^(k-1) * p，k = 1, 2, 3, ...
 * 其中p是每次试验成功的概率，k是第一次成功所需的试验次数。
 * 
 * 几何分布具有无记忆性，是唯一具有无记忆性的离散分布。
 * 
 * Geometric distribution is the probability distribution of the number of trials
 * needed to get the first success in a sequence of independent Bernoulli trials.
 * Its probability mass function is: P(X=k) = (1-p)^(k-1) * p, k = 1, 2, 3, ...
 * where p is the probability of success in each trial, and k is the number of trials needed for the first success.
 * 
 * Geometric distribution has the memoryless property and is the only discrete distribution with this property.
 * 
 * @author lteb2
 */
public class GeometricDistribution implements IDiscreteDistribution {
    
    private static final long serialVersionUID = 1L;
    
    /** 成功概率 / Probability of success */
    private final float p;
    
    /** 失败概率 / Probability of failure */
    private final float q;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param p 成功概率，范围(0,1] / Probability of success, range (0,1]
     * @throws IllegalArgumentException 如果概率不在(0,1]范围内 / If probability is not in range (0,1]
     */
    public GeometricDistribution(float p) {
        if (p <= 0.0f || p > 1.0f) {
            throw new IllegalArgumentException("成功概率必须在(0,1]范围内 / Probability must be in range (0,1]");
        }
        this.p = p;
        this.q = 1.0f - p;
    }
    
    /**
     * 获取成功概率
     * Get probability of success
     * 
     * @return 成功概率 / Probability of success
     */
    public float getProbability() {
        return p;
    }
    
    // ==================== 基本统计量 / Basic Statistics ====================
    
    @Override
    public float mean() {
        return 1.0f / p;
    }
    
    @Override
    public float var() {
        return q / (p * p);
    }
    
    @Override
    public float std() {
        return (float) Math.sqrt(q) / p;
    }
    
    @Override
    public float median() {
        return (float) Math.ceil(-Math.log(2.0) / Math.log(q));
    }
    
    @Override
    public float mode() {
        return 1.0f; // 几何分布的众数总是1
    }
    
    @Override
    public float q1() {
        return ppf(0.25f);
    }
    
    @Override
    public float q3() {
        return ppf(0.75f);
    }
    
    @Override
    public float skewness() {
        return (2.0f - p) / (float) Math.sqrt(q);
    }
    
    @Override
    public float kurtosis() {
        return 6.0f + (p * p) / q;
    }
    
    // ==================== 概率计算 / Probability Calculations ====================
    
    @Override
    public float pmf(int x) {
        if (x < 1) return 0.0f;
        return (float) Math.pow(q, x - 1) * p;
    }
    
    @Override
    public float cdf(int x) {
        if (x < 1) return 0.0f;
        return 1.0f - (float) Math.pow(q, x);
    }
    
    @Override
    public int ppf(float prob) {
        if (prob < 0.0f || prob > 1.0f) {
            throw new IllegalArgumentException("概率值必须在[0,1]范围内 / Probability must be in range [0,1]");
        }
        
        if (prob == 0.0f) return 1;
        if (prob == 1.0f) return Integer.MAX_VALUE;
        
        return (int) Math.ceil(Math.log(1.0 - prob) / Math.log(q));
    }
    
    @Override
    public float sf(int x) {
        return 1.0f - cdf(x);
    }
    
    @Override
    public int isf(float prob) {
        if (prob < 0.0f || prob > 1.0f) {
            throw new IllegalArgumentException("概率值必须在[0,1]范围内 / Probability must be in range [0,1]");
        }
        return ppf(1.0f - prob);
    }
    
    // ==================== 随机采样 / Random Sampling ====================
    
    @Override
    public int sample() {
        // 使用逆变换采样
        double u = Math.random();
        return (int) Math.ceil(Math.log(1.0 - u) / Math.log(q));
    }
    
    @Override
    public int[] sample(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("样本数量必须大于0 / Sample size must be greater than 0");
        }
        
        int[] samples = new int[n];
        for (int i = 0; i < n; i++) {
            samples[i] = sample();
        }
        return samples;
    }
    
    // ==================== 支持区间和验证 / Support and Validation ====================
    
    @Override
    public int getMinSupport() {
        return 1;
    }
    
    @Override
    public int getMaxSupport() {
        return Integer.MAX_VALUE; // 几何分布是无界的
    }
    
    @Override
    public boolean isInSupport(int x) {
        return x >= 1;
    }
    
    @Override
    public boolean isBounded() {
        return false; // 几何分布是无界的
    }
    
    // ==================== 高级统计方法 / Advanced Statistical Methods ====================
    
    @Override
    public float moment(int k) {
        if (k < 0) {
            throw new IllegalArgumentException("矩的阶数必须非负 / Moment order must be non-negative");
        }
        if (k == 0) return 1.0f;
        if (k == 1) return mean();
        
        // 对于几何分布，k阶矩 = k! * (q/p)^k * Li_{-k}(q)
        // 其中Li_{-k}(q)是负k阶多对数函数
        float result = 0.0f;
        for (int i = 1; i <= 100; i++) { // 使用截断级数近似
            result += (float) Math.pow(i, k) * pmf(i);
        }
        return result;
    }
    
    @Override
    public float centralMoment(int k) {
        if (k < 0) {
            throw new IllegalArgumentException("矩的阶数必须非负 / Moment order must be non-negative");
        }
        if (k == 0) return 1.0f;
        if (k == 1) return 0.0f;
        if (k == 2) return var();
        
        // 使用递推公式计算中心矩
        float result = 0.0f;
        for (int i = 0; i <= k; i++) {
            float term = RereMathUtil.combination(k, i) * (float) Math.pow(-mean(), k - i) * moment(i);
            result += term;
        }
        return result;
    }
    
    @Override
    public float standardizedMoment(int k) {
        if (k < 0) {
            throw new IllegalArgumentException("矩的阶数必须非负 / Moment order must be non-negative");
        }
        if (std() == 0.0f) {
            throw new IllegalArgumentException("标准差为0时无法计算标准化矩 / Cannot calculate standardized moment when standard deviation is 0");
        }
        return centralMoment(k) / (float) Math.pow(std(), k);
    }
    
    @Override
    public float entropy() {
        float entropy = 0.0f;
        for (int k = 1; k <= 1000; k++) { // 使用截断级数
            float prob = pmf(k);
            if (prob > 0.0f) {
                entropy -= prob * (float) Math.log(prob);
            }
        }
        return entropy;
    }
    
    @Override
    public float cgf(float t) {
        if (t >= Math.log(1.0 / q)) {
            return Float.POSITIVE_INFINITY;
        }
        return (float) Math.log(p * Math.exp(t) / (1.0 - q * Math.exp(t)));
    }
    
    // ==================== 分位数和区间估计 / Quantiles and Interval Estimation ====================
    
    @Override
    public int quantile(float prob) {
        return ppf(prob);
    }
    
    @Override
    public int[] confidenceInterval(float confidence) {
        if (confidence < 0.0f || confidence > 1.0f) {
            throw new IllegalArgumentException("置信水平必须在[0,1]范围内 / Confidence level must be in range [0,1]");
        }
        
        float alpha = 1.0f - confidence;
        int lower = ppf(alpha / 2.0f);
        int upper = ppf(1.0f - alpha / 2.0f);
        
        return new int[]{lower, upper};
    }
    
    // ==================== 分布比较和距离 / Distribution Comparison and Distance ====================
    
    @Override
    public float klDivergence(IDiscreteDistribution other) {
        if (!(other instanceof GeometricDistribution)) {
            throw new IllegalArgumentException("只能与几何分布计算KL散度 / Can only calculate KL divergence with Geometric distribution");
        }
        
        GeometricDistribution otherGeometric = (GeometricDistribution) other;
        
        float kl = 0.0f;
        for (int k = 1; k <= 1000; k++) { // 使用截断级数
            float p1 = pmf(k);
            float p2 = otherGeometric.pmf(k);
            if (p1 > 0.0f && p2 > 0.0f) {
                kl += p1 * (float) Math.log(p1 / p2);
            } else if (p1 > 0.0f) {
                return Float.POSITIVE_INFINITY;
            }
        }
        
        return kl;
    }
    
    @Override
    public float jsDivergence(IDiscreteDistribution other) {
        if (!(other instanceof GeometricDistribution)) {
            throw new IllegalArgumentException("只能与几何分布计算JS散度 / Can only calculate JS divergence with Geometric distribution");
        }
        
        GeometricDistribution otherGeometric = (GeometricDistribution) other;
        float mP = (p + otherGeometric.getProbability()) / 2.0f;
        
        // 创建混合分布
        GeometricDistribution mixed = new GeometricDistribution(mP);
        
        float kl1 = klDivergence(mixed);
        float kl2 = otherGeometric.klDivergence(mixed);
        
        return 0.5f * kl1 + 0.5f * kl2;
    }
    
    @Override
    public float wassersteinDistance(IDiscreteDistribution other) {
        if (!(other instanceof GeometricDistribution)) {
            throw new IllegalArgumentException("只能与几何分布计算Wasserstein距离 / Can only calculate Wasserstein distance with Geometric distribution");
        }
        
        GeometricDistribution otherGeometric = (GeometricDistribution) other;
        return Math.abs(mean() - otherGeometric.mean());
    }
    
    // ==================== 分布信息 / Distribution Information ====================
    
    @Override
    public String getDistributionName() {
        return "Geometric";
    }
    
    @Override
    public String getParameterInfo() {
        return String.format("p=%.4f", p);
    }
    
    @Override
    public boolean isSymmetric() {
        return false; // 几何分布不是对称的
    }
    
    @Override
    public boolean isMemoryless() {
        return true; // 几何分布具有无记忆性
    }
    
    @Override
    public String toString() {
        return String.format("GeometricDistribution(p=%.4f)", p);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        GeometricDistribution that = (GeometricDistribution) obj;
        return Float.compare(that.p, p) == 0;
    }
    
    @Override
    public int hashCode() {
        return Float.hashCode(p);
    }
}
