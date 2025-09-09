package com.reremouse.lab.math.stat.distribution;

import com.reremouse.lab.math.RereMathUtil;

/**
 * 离散均匀分布 (Discrete Uniform Distribution)
 * 
 * 离散均匀分布是在有限个离散值上等概率的分布。
 * 其概率质量函数为：P(X=k) = 1/n，k = a, a+1, ..., b
 * 其中a是下界，b是上界，n = b - a + 1是可能值的个数。
 * 
 * 离散均匀分布是连续均匀分布的离散版本。
 * 
 * Discrete uniform distribution is a probability distribution where each value
 * in a finite set of discrete values has equal probability.
 * Its probability mass function is: P(X=k) = 1/n, k = a, a+1, ..., b
 * where a is the lower bound, b is the upper bound, and n = b - a + 1 is the number of possible values.
 * 
 * Discrete uniform distribution is the discrete version of continuous uniform distribution.
 * 
 * @author lteb2
 */
public class DiscreteUniformDistribution implements IDiscreteDistribution {
    
    private static final long serialVersionUID = 1L;
    
    /** 下界 / Lower bound */
    private final int a;
    
    /** 上界 / Upper bound */
    private final int b;
    
    /** 可能值的个数 / Number of possible values */
    private final int n;
    
    /** 概率质量 / Probability mass */
    private final float probMass;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param a 下界 / Lower bound
     * @param b 上界 / Upper bound
     * @throws IllegalArgumentException 如果上界小于下界 / If upper bound is less than lower bound
     */
    public DiscreteUniformDistribution(int a, int b) {
        if (b < a) {
            throw new IllegalArgumentException("上界必须大于等于下界 / Upper bound must be greater than or equal to lower bound");
        }
        this.a = a;
        this.b = b;
        this.n = b - a + 1;
        this.probMass = 1.0f / n;
    }
    
    /**
     * 构造函数，创建[0, n-1]上的离散均匀分布
     * Constructor for discrete uniform distribution on [0, n-1]
     * 
     * @param n 上界（下界为0）/ Upper bound (lower bound is 0)
     * @throws IllegalArgumentException 如果n小于等于0 / If n is less than or equal to 0
     */
    public DiscreteUniformDistribution(int n) {
        this(0, n - 1);
    }
    
    /**
     * 获取下界
     * Get lower bound
     * 
     * @return 下界 / Lower bound
     */
    public int getA() {
        return a;
    }
    
    /**
     * 获取上界
     * Get upper bound
     * 
     * @return 上界 / Upper bound
     */
    public int getB() {
        return b;
    }
    
    /**
     * 获取可能值的个数
     * Get number of possible values
     * 
     * @return 可能值的个数 / Number of possible values
     */
    public int getN() {
        return n;
    }
    
    // ==================== 基本统计量 / Basic Statistics ====================
    
    @Override
    public float mean() {
        return (a + b) / 2.0f;
    }
    
    @Override
    public float var() {
        return (n * n - 1) / 12.0f;
    }
    
    @Override
    public float std() {
        return (float) Math.sqrt((n * n - 1) / 12.0f);
    }
    
    @Override
    public float median() {
        return (a + b) / 2.0f;
    }
    
    @Override
    public float mode() {
        return Float.NaN; // 离散均匀分布没有唯一的众数
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
        return 0.0f; // 离散均匀分布是对称的
    }
    
    @Override
    public float kurtosis() {
        return -6.0f * (n * n + 1) / (5.0f * (n * n - 1));
    }
    
    // ==================== 概率计算 / Probability Calculations ====================
    
    @Override
    public float pmf(int x) {
        if (x < a || x > b) return 0.0f;
        return probMass;
    }
    
    @Override
    public float cdf(int x) {
        if (x < a) return 0.0f;
        if (x >= b) return 1.0f;
        return (x - a + 1) * probMass;
    }
    
    @Override
    public int ppf(float prob) {
        if (prob < 0.0f || prob > 1.0f) {
            throw new IllegalArgumentException("概率值必须在[0,1]范围内 / Probability must be in range [0,1]");
        }
        
        if (prob == 0.0f) return a;
        if (prob == 1.0f) return b;
        
        return a + (int) Math.floor(prob * n);
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
        return a + (int) (Math.random() * n);
    }
    
    @Override
    public int[] sample(int sampleSize) {
        if (sampleSize <= 0) {
            throw new IllegalArgumentException("样本数量必须大于0 / Sample size must be greater than 0");
        }
        
        int[] samples = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            samples[i] = sample();
        }
        return samples;
    }
    
    // ==================== 支持区间和验证 / Support and Validation ====================
    
    @Override
    public int getMinSupport() {
        return a;
    }
    
    @Override
    public int getMaxSupport() {
        return b;
    }
    
    @Override
    public boolean isInSupport(int x) {
        return x >= a && x <= b;
    }
    
    @Override
    public boolean isBounded() {
        return true; // 离散均匀分布是有界的
    }
    
    // ==================== 高级统计方法 / Advanced Statistical Methods ====================
    
    @Override
    public float moment(int k) {
        if (k < 0) {
            throw new IllegalArgumentException("矩的阶数必须非负 / Moment order must be non-negative");
        }
        if (k == 0) return 1.0f;
        if (k == 1) return mean();
        
        // 对于离散均匀分布，k阶矩 = (1/n) * Σ(i=a to b) i^k
        float result = 0.0f;
        for (int i = a; i <= b; i++) {
            result += (float) Math.pow(i, k);
        }
        return result / n;
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
        return (float) Math.log(n);
    }
    
    @Override
    public float cgf(float t) {
        if (t == 0.0f) return 0.0f;
        
        float sum = 0.0f;
        for (int i = a; i <= b; i++) {
            sum += (float) Math.exp(t * i);
        }
        return (float) Math.log(sum / n);
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
        if (!(other instanceof DiscreteUniformDistribution)) {
            throw new IllegalArgumentException("只能与离散均匀分布计算KL散度 / Can only calculate KL divergence with Discrete Uniform distribution");
        }
        
        DiscreteUniformDistribution otherUniform = (DiscreteUniformDistribution) other;
        int otherA = otherUniform.getA();
        int otherB = otherUniform.getB();
        
        // 检查支持区间是否相同
        if (a != otherA || b != otherB) {
            return Float.POSITIVE_INFINITY;
        }
        
        // 如果支持区间相同，KL散度为0
        return 0.0f;
    }
    
    @Override
    public float jsDivergence(IDiscreteDistribution other) {
        if (!(other instanceof DiscreteUniformDistribution)) {
            throw new IllegalArgumentException("只能与离散均匀分布计算JS散度 / Can only calculate JS divergence with Discrete Uniform distribution");
        }
        
        DiscreteUniformDistribution otherUniform = (DiscreteUniformDistribution) other;
        int otherA = otherUniform.getA();
        int otherB = otherUniform.getB();
        
        // 检查支持区间是否相同
        if (a != otherA || b != otherB) {
            return Float.POSITIVE_INFINITY;
        }
        
        // 如果支持区间相同，JS散度为0
        return 0.0f;
    }
    
    @Override
    public float wassersteinDistance(IDiscreteDistribution other) {
        if (!(other instanceof DiscreteUniformDistribution)) {
            throw new IllegalArgumentException("只能与离散均匀分布计算Wasserstein距离 / Can only calculate Wasserstein distance with Discrete Uniform distribution");
        }
        
        DiscreteUniformDistribution otherUniform = (DiscreteUniformDistribution) other;
        return Math.abs(mean() - otherUniform.mean());
    }
    
    // ==================== 分布信息 / Distribution Information ====================
    
    @Override
    public String getDistributionName() {
        return "Discrete Uniform";
    }
    
    @Override
    public String getParameterInfo() {
        return String.format("a=%d, b=%d", a, b);
    }
    
    @Override
    public boolean isSymmetric() {
        return true; // 离散均匀分布是对称的
    }
    
    @Override
    public boolean isMemoryless() {
        return false; // 离散均匀分布不是无记忆的
    }
    
    @Override
    public String toString() {
        return String.format("DiscreteUniformDistribution([%d, %d])", a, b);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        DiscreteUniformDistribution that = (DiscreteUniformDistribution) obj;
        return a == that.a && b == that.b;
    }
    
    @Override
    public int hashCode() {
        return 31 * a + b;
    }
}
