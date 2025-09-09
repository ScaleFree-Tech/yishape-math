package com.reremouse.lab.math.stat.distribution;

import com.reremouse.lab.math.RereMathUtil;

/**
 * 伯努利分布 (Bernoulli Distribution)
 * 
 * 伯努利分布是只有两个可能结果的离散概率分布：成功（1）和失败（0）。
 * 其概率质量函数为：P(X=1) = p, P(X=0) = 1-p
 * 其中p是成功的概率。
 * 
 * 伯努利分布是二项分布的特殊情况（n=1）。
 * 
 * Bernoulli distribution is a discrete probability distribution with only two possible outcomes:
 * success (1) and failure (0). Its probability mass function is: P(X=1) = p, P(X=0) = 1-p
 * where p is the probability of success.
 * 
 * Bernoulli distribution is a special case of binomial distribution (n=1).
 * 
 * @author lteb2
 */
public class BernoulliDistribution implements IDiscreteDistribution {
    
    private static final long serialVersionUID = 1L;
    
    /** 成功概率 / Probability of success */
    private final float p;
    
    /** 失败概率 / Probability of failure */
    private final float q;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param p 成功概率，范围[0,1] / Probability of success, range [0,1]
     * @throws IllegalArgumentException 如果概率不在[0,1]范围内 / If probability is not in range [0,1]
     */
    public BernoulliDistribution(float p) {
        if (p < 0.0f || p > 1.0f) {
            throw new IllegalArgumentException("成功概率必须在[0,1]范围内 / Probability must be in range [0,1]");
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
        return p;
    }
    
    @Override
    public float var() {
        return p * q;
    }
    
    @Override
    public float std() {
        return (float) Math.sqrt(p * q);
    }
    
    @Override
    public float median() {
        if (p < 0.5f) return 0.0f;
        if (p > 0.5f) return 1.0f;
        return 0.5f; // 当p=0.5时，中位数可以是0或1，这里返回0.5
    }
    
    @Override
    public float mode() {
        if (p > 0.5f) return 1.0f;
        if (p < 0.5f) return 0.0f;
        return Float.NaN; // 当p=0.5时，没有唯一的众数
    }
    
    @Override
    public float q1() {
        if (p < 0.25f) return 0.0f;
        if (p > 0.25f) return 1.0f;
        return 0.0f; // 当p=0.25时
    }
    
    @Override
    public float q3() {
        if (p < 0.75f) return 0.0f;
        if (p > 0.75f) return 1.0f;
        return 1.0f; // 当p=0.75时
    }
    
    @Override
    public float skewness() {
        if (p == 0.0f || p == 1.0f) return 0.0f;
        return (1.0f - 2.0f * p) / (float) Math.sqrt(p * q);
    }
    
    @Override
    public float kurtosis() {
        if (p == 0.0f || p == 1.0f) return 0.0f;
        return (1.0f - 6.0f * p * q) / (p * q);
    }
    
    // ==================== 概率计算 / Probability Calculations ====================
    
    @Override
    public float pmf(int x) {
        if (x == 1) return p;
        if (x == 0) return q;
        return 0.0f;
    }
    
    @Override
    public float cdf(int x) {
        if (x < 0) return 0.0f;
        if (x >= 1) return 1.0f;
        return q; // x = 0时
    }
    
    @Override
    public int ppf(float prob) {
        if (prob < 0.0f || prob > 1.0f) {
            throw new IllegalArgumentException("概率值必须在[0,1]范围内 / Probability must be in range [0,1]");
        }
        if (prob <= q) return 0;
        return 1;
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
        return Math.random() < p ? 1 : 0;
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
        return 0;
    }
    
    @Override
    public int getMaxSupport() {
        return 1;
    }
    
    @Override
    public boolean isInSupport(int x) {
        return x == 0 || x == 1;
    }
    
    @Override
    public boolean isBounded() {
        return true;
    }
    
    // ==================== 高级统计方法 / Advanced Statistical Methods ====================
    
    @Override
    public float moment(int k) {
        if (k < 0) {
            throw new IllegalArgumentException("矩的阶数必须非负 / Moment order must be non-negative");
        }
        if (k == 0) return 1.0f;
        return p; // 对于伯努利分布，所有非零阶矩都等于p
    }
    
    @Override
    public float centralMoment(int k) {
        if (k < 0) {
            throw new IllegalArgumentException("矩的阶数必须非负 / Moment order must be non-negative");
        }
        if (k == 0) return 1.0f;
        if (k == 1) return 0.0f;
        if (k == 2) return p * q;
        if (k == 3) return p * q * (q - p);
        if (k == 4) return p * q * (1.0f - 6.0f * p * q);
        
        // 对于高阶矩，使用递推公式
        float result = 0.0f;
        for (int i = 0; i <= k; i++) {
            float term = RereMathUtil.combination(k, i) * (float) Math.pow(-p, k - i) * p;
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
        if (p == 0.0f || p == 1.0f) return 0.0f;
        return -(p * (float) Math.log(p) + q * (float) Math.log(q));
    }
    
    @Override
    public float cgf(float t) {
        return (float) Math.log(q + p * Math.exp(t));
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
        
        // 对于伯努利分布，置信区间就是[0,1]
        return new int[]{0, 1};
    }
    
    // ==================== 分布比较和距离 / Distribution Comparison and Distance ====================
    
    @Override
    public float klDivergence(IDiscreteDistribution other) {
        if (!(other instanceof BernoulliDistribution)) {
            throw new IllegalArgumentException("只能与伯努利分布计算KL散度 / Can only calculate KL divergence with Bernoulli distribution");
        }
        
        BernoulliDistribution otherBernoulli = (BernoulliDistribution) other;
        float otherP = otherBernoulli.getProbability();
        
        if (p == 0.0f && otherP > 0.0f) return Float.POSITIVE_INFINITY;
        if (p == 1.0f && otherP < 1.0f) return Float.POSITIVE_INFINITY;
        if (otherP == 0.0f && p > 0.0f) return Float.POSITIVE_INFINITY;
        if (otherP == 1.0f && p < 1.0f) return Float.POSITIVE_INFINITY;
        
        float kl = 0.0f;
        if (p > 0.0f) kl += p * (float) Math.log(p / otherP);
        if (q > 0.0f) kl += q * (float) Math.log(q / (1.0f - otherP));
        
        return kl;
    }
    
    @Override
    public float jsDivergence(IDiscreteDistribution other) {
        if (!(other instanceof BernoulliDistribution)) {
            throw new IllegalArgumentException("只能与伯努利分布计算JS散度 / Can only calculate JS divergence with Bernoulli distribution");
        }
        
        // JS散度 = 0.5 * KL(P||M) + 0.5 * KL(Q||M)，其中M = 0.5 * (P + Q)
        BernoulliDistribution otherBernoulli = (BernoulliDistribution) other;
        float otherP = otherBernoulli.getProbability();
        float mP = (p + otherP) / 2.0f;
        float mQ = 1.0f - mP;
        
        float kl1 = 0.0f;
        if (p > 0.0f) kl1 += p * (float) Math.log(p / mP);
        if (q > 0.0f) kl1 += q * (float) Math.log(q / mQ);
        
        float kl2 = 0.0f;
        if (otherP > 0.0f) kl2 += otherP * (float) Math.log(otherP / mP);
        if ((1.0f - otherP) > 0.0f) kl2 += (1.0f - otherP) * (float) Math.log((1.0f - otherP) / mQ);
        
        return 0.5f * kl1 + 0.5f * kl2;
    }
    
    @Override
    public float wassersteinDistance(IDiscreteDistribution other) {
        if (!(other instanceof BernoulliDistribution)) {
            throw new IllegalArgumentException("只能与伯努利分布计算Wasserstein距离 / Can only calculate Wasserstein distance with Bernoulli distribution");
        }
        
        BernoulliDistribution otherBernoulli = (BernoulliDistribution) other;
        return Math.abs(p - otherBernoulli.getProbability());
    }
    
    // ==================== 分布信息 / Distribution Information ====================
    
    @Override
    public String getDistributionName() {
        return "Bernoulli";
    }
    
    @Override
    public String getParameterInfo() {
        return String.format("p=%.4f", p);
    }
    
    @Override
    public boolean isSymmetric() {
        return Math.abs(p - 0.5f) < 1e-6f;
    }
    
    @Override
    public boolean isMemoryless() {
        return false; // 伯努利分布不是无记忆的
    }
    
    @Override
    public String toString() {
        return String.format("BernoulliDistribution(p=%.4f)", p);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        BernoulliDistribution that = (BernoulliDistribution) obj;
        return Float.compare(that.p, p) == 0;
    }
    
    @Override
    public int hashCode() {
        return Float.hashCode(p);
    }
}
