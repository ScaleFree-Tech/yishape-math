package com.reremouse.lab.math.stat.distribution;

import com.reremouse.lab.math.RereMathUtil;

/**
 * 泊松分布 (Poisson Distribution)
 * 
 * 泊松分布是描述在固定时间间隔内发生随机事件次数的离散概率分布。
 * 其概率质量函数为：P(X=k) = (λ^k * e^(-λ)) / k!，k = 0, 1, 2, ...
 * 其中λ是平均发生率（期望值），k是事件发生的次数。
 * 
 * 泊松分布是二项分布在n很大、p很小时的极限情况。
 * 
 * Poisson distribution is a discrete probability distribution that expresses
 * the probability of a given number of events occurring in a fixed interval of time.
 * Its probability mass function is: P(X=k) = (λ^k * e^(-λ)) / k!, k = 0, 1, 2, ...
 * where λ is the average rate of occurrence (expected value), and k is the number of events.
 * 
 * Poisson distribution is the limiting case of binomial distribution when n is large and p is small.
 * 
 * @author lteb2
 */
public class PoissonDistribution implements IDiscreteDistribution {
    
    private static final long serialVersionUID = 1L;
    
    /** 平均发生率（期望值）/ Average rate of occurrence (expected value) */
    private final float lambda;
    
    /** e^(-λ)的预计算值 / Precomputed value of e^(-λ) */
    private final float expNegLambda;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param lambda 平均发生率，必须大于0 / Average rate of occurrence, must be greater than 0
     * @throws IllegalArgumentException 如果λ小于等于0 / If lambda is less than or equal to 0
     */
    public PoissonDistribution(float lambda) {
        if (lambda <= 0.0f) {
            throw new IllegalArgumentException("平均发生率必须大于0 / Lambda must be greater than 0");
        }
        this.lambda = lambda;
        this.expNegLambda = (float) Math.exp(-lambda);
    }
    
    /**
     * 获取平均发生率
     * Get average rate of occurrence
     * 
     * @return 平均发生率 / Average rate of occurrence
     */
    public float getLambda() {
        return lambda;
    }
    
    // ==================== 基本统计量 / Basic Statistics ====================
    
    @Override
    public float mean() {
        return lambda;
    }
    
    @Override
    public float var() {
        return lambda;
    }
    
    @Override
    public float std() {
        return (float) Math.sqrt(lambda);
    }
    
    @Override
    public float median() {
        // 泊松分布的中位数近似为 floor(λ + 1/3 - 0.02/λ)
        return (float) Math.floor(lambda + 1.0f / 3.0f - 0.02f / lambda);
    }
    
    @Override
    public float mode() {
        return (float) Math.floor(lambda);
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
        return 1.0f / (float) Math.sqrt(lambda);
    }
    
    @Override
    public float kurtosis() {
        return 1.0f / lambda;
    }
    
    // ==================== 概率计算 / Probability Calculations ====================
    
    @Override
    public float pmf(int x) {
        if (x < 0) return 0.0f;
        
        // 使用对数避免数值溢出
        double logPmf = x * Math.log(lambda) - lambda - RereMathUtil.logFactorial(x);
        return (float) Math.exp(logPmf);
    }
    
    @Override
    public float cdf(int x) {
        if (x < 0) return 0.0f;
        
        // 使用正则化不完全Gamma函数计算CDF
        return (float) RereMathUtil.regularizedIncompleteGamma(x + 1, lambda);
    }
    
    @Override
    public int ppf(float prob) {
        if (prob < 0.0f || prob > 1.0f) {
            throw new IllegalArgumentException("概率值必须在[0,1]范围内 / Probability must be in range [0,1]");
        }
        
        if (prob == 0.0f) return 0;
        if (prob == 1.0f) return Integer.MAX_VALUE;
        
        // 使用二分搜索查找分位数
        int left = 0, right = Math.max(10, (int) (lambda + 10 * Math.sqrt(lambda)));
        while (left < right) {
            int mid = (left + right) / 2;
            if (cdf(mid) < prob) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
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
        if (lambda < 10.0f) {
            // 对于小的lambda，使用Knuth算法
            return sampleKnuth();
        } else {
            // 对于大的lambda，使用正态近似
            return Math.max(0, (int) Math.round(RereMathUtil.normalSample(lambda, (float) Math.sqrt(lambda))));
        }
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
        return Integer.MAX_VALUE; // 泊松分布是无界的
    }
    
    @Override
    public boolean isInSupport(int x) {
        return x >= 0;
    }
    
    @Override
    public boolean isBounded() {
        return false; // 泊松分布是无界的
    }
    
    // ==================== 高级统计方法 / Advanced Statistical Methods ====================
    
    @Override
    public float moment(int k) {
        if (k < 0) {
            throw new IllegalArgumentException("矩的阶数必须非负 / Moment order must be non-negative");
        }
        if (k == 0) return 1.0f;
        if (k == 1) return lambda;
        
        // 对于泊松分布，k阶矩 = λ * (k-1阶矩 + λ^(k-1))
        float result = lambda;
        for (int i = 2; i <= k; i++) {
            result = lambda * (result + (float) Math.pow(lambda, i - 1));
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
        if (k == 2) return lambda;
        
        // 对于泊松分布，中心矩有特殊性质
        if (k == 3) return lambda;
        if (k == 4) return lambda + 3 * lambda * lambda;
        
        // 对于高阶矩，使用递推公式
        float result = 0.0f;
        for (int i = 0; i <= k; i++) {
            float term = RereMathUtil.combination(k, i) * (float) Math.pow(-lambda, k - i) * moment(i);
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
        for (int k = 0; k <= (int) (lambda + 10 * Math.sqrt(lambda)); k++) {
            float prob = pmf(k);
            if (prob > 0.0f) {
                entropy -= prob * (float) Math.log(prob);
            }
        }
        return entropy;
    }
    
    @Override
    public float cgf(float t) {
        return lambda * ((float) Math.exp(t) - 1.0f);
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
        if (!(other instanceof PoissonDistribution)) {
            throw new IllegalArgumentException("只能与泊松分布计算KL散度 / Can only calculate KL divergence with Poisson distribution");
        }
        
        PoissonDistribution otherPoisson = (PoissonDistribution) other;
        float otherLambda = otherPoisson.getLambda();
        
        float kl = lambda * (float) Math.log(lambda / otherLambda) - lambda + otherLambda;
        return kl;
    }
    
    @Override
    public float jsDivergence(IDiscreteDistribution other) {
        if (!(other instanceof PoissonDistribution)) {
            throw new IllegalArgumentException("只能与泊松分布计算JS散度 / Can only calculate JS divergence with Poisson distribution");
        }
        
        PoissonDistribution otherPoisson = (PoissonDistribution) other;
        float otherLambda = otherPoisson.getLambda();
        float mLambda = (lambda + otherLambda) / 2.0f;
        
        // 创建混合分布
        PoissonDistribution mixed = new PoissonDistribution(mLambda);
        
        float kl1 = klDivergence(mixed);
        float kl2 = otherPoisson.klDivergence(mixed);
        
        return 0.5f * kl1 + 0.5f * kl2;
    }
    
    @Override
    public float wassersteinDistance(IDiscreteDistribution other) {
        if (!(other instanceof PoissonDistribution)) {
            throw new IllegalArgumentException("只能与泊松分布计算Wasserstein距离 / Can only calculate Wasserstein distance with Poisson distribution");
        }
        
        PoissonDistribution otherPoisson = (PoissonDistribution) other;
        return Math.abs(lambda - otherPoisson.getLambda());
    }
    
    // ==================== 分布信息 / Distribution Information ====================
    
    @Override
    public String getDistributionName() {
        return "Poisson";
    }
    
    @Override
    public String getParameterInfo() {
        return String.format("λ=%.4f", lambda);
    }
    
    @Override
    public boolean isSymmetric() {
        return false; // 泊松分布不是对称的
    }
    
    @Override
    public boolean isMemoryless() {
        return false; // 泊松分布不是无记忆的
    }
    
    @Override
    public String toString() {
        return String.format("PoissonDistribution(λ=%.4f)", lambda);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        PoissonDistribution that = (PoissonDistribution) obj;
        return Float.compare(that.lambda, lambda) == 0;
    }
    
    @Override
    public int hashCode() {
        return Float.hashCode(lambda);
    }
    
    // ==================== 辅助方法 / Helper Methods ====================
    
    /**
     * 使用Knuth算法生成泊松分布样本（适用于小的lambda）
     * Generate Poisson distribution sample using Knuth's algorithm (for small lambda)
     * 
     * @return 泊松分布样本 / Poisson distribution sample
     */
    private int sampleKnuth() {
        double p = expNegLambda;
        int k = 0;
        double u = Math.random();
        while (u > p) {
            u *= Math.random();
            k++;
        }
        return k;
    }
}
