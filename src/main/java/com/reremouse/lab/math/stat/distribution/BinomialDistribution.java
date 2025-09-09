package com.reremouse.lab.math.stat.distribution;

import com.reremouse.lab.math.RereMathUtil;

/**
 * 二项分布 (Binomial Distribution)
 * 
 * 二项分布是n次独立伯努利试验中成功次数的离散概率分布。
 * 其概率质量函数为：P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
 * 其中n是试验次数，p是每次试验成功的概率，k是成功次数。
 * 
 * 二项分布是伯努利分布的推广，当n=1时退化为伯努利分布。
 * 
 * Binomial distribution is the discrete probability distribution of the number of successes
 * in a sequence of n independent Bernoulli trials. Its probability mass function is:
 * P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
 * where n is the number of trials, p is the probability of success in each trial, and k is the number of successes.
 * 
 * Binomial distribution is a generalization of Bernoulli distribution, and reduces to Bernoulli when n=1.
 * 
 * @author lteb2
 */
public class BinomialDistribution implements IDiscreteDistribution {
    
    private static final long serialVersionUID = 1L;
    
    /** 试验次数 / Number of trials */
    private final int n;
    
    /** 成功概率 / Probability of success */
    private final float p;
    
    /** 失败概率 / Probability of failure */
    private final float q;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param n 试验次数，必须大于0 / Number of trials, must be greater than 0
     * @param p 成功概率，范围[0,1] / Probability of success, range [0,1]
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public BinomialDistribution(int n, float p) {
        if (n <= 0) {
            throw new IllegalArgumentException("试验次数必须大于0 / Number of trials must be greater than 0");
        }
        if (p < 0.0f || p > 1.0f) {
            throw new IllegalArgumentException("成功概率必须在[0,1]范围内 / Probability must be in range [0,1]");
        }
        this.n = n;
        this.p = p;
        this.q = 1.0f - p;
    }
    
    /**
     * 获取试验次数
     * Get number of trials
     * 
     * @return 试验次数 / Number of trials
     */
    public int getN() {
        return n;
    }
    
    /**
     * 获取成功概率
     * Get probability of success
     * 
     * @return 成功概率 / Probability of success
     */
    public float getP() {
        return p;
    }
    
    // ==================== 基本统计量 / Basic Statistics ====================
    
    @Override
    public float mean() {
        return n * p;
    }
    
    @Override
    public float var() {
        return n * p * q;
    }
    
    @Override
    public float std() {
        return (float) Math.sqrt(n * p * q);
    }
    
    @Override
    public float median() {
        // 二项分布的中位数近似为 floor(n*p)
        return (float) Math.floor(n * p);
    }
    
    @Override
    public float mode() {
        if (p == 0.0f) return 0.0f;
        if (p == 1.0f) return (float) n;
        return (float) Math.floor((n + 1) * p);
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
        if (p == 0.0f || p == 1.0f) return 0.0f;
        return (1.0f - 2.0f * p) / (float) Math.sqrt(n * p * q);
    }
    
    @Override
    public float kurtosis() {
        if (p == 0.0f || p == 1.0f) return 0.0f;
        return (1.0f - 6.0f * p * q) / (n * p * q);
    }
    
    // ==================== 概率计算 / Probability Calculations ====================
    
    @Override
    public float pmf(int x) {
        if (x < 0 || x > n) return 0.0f;
        if (p == 0.0f) return (x == 0) ? 1.0f : 0.0f;
        if (p == 1.0f) return (x == n) ? 1.0f : 0.0f;
        
        // 使用对数避免数值溢出
        double logPmf = RereMathUtil.logCombination(n, x) + x * Math.log(p) + (n - x) * Math.log(q);
        return (float) Math.exp(logPmf);
    }
    
    @Override
    public float cdf(int x) {
        if (x < 0) return 0.0f;
        if (x >= n) return 1.0f;
        
        // 使用正则化不完全Beta函数计算CDF
        return 1.0f - (float) RereMathUtil.regularizedIncompleteBeta(x + 1, n - x, p);
    }
    
    @Override
    public int ppf(float prob) {
        if (prob < 0.0f || prob > 1.0f) {
            throw new IllegalArgumentException("概率值必须在[0,1]范围内 / Probability must be in range [0,1]");
        }
        
        if (prob == 0.0f) return 0;
        if (prob == 1.0f) return n;
        
        // 使用二分搜索查找分位数
        int left = 0, right = n;
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
        // 使用逆变换采样
        double u = Math.random();
        float cdf = 0.0f;
        
        for (int k = 0; k <= n; k++) {
            cdf += pmf(k);
            if (u <= cdf) {
                return k;
            }
        }
        return n; // 理论上不应该到达这里
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
        return 0;
    }
    
    @Override
    public int getMaxSupport() {
        return n;
    }
    
    @Override
    public boolean isInSupport(int x) {
        return x >= 0 && x <= n;
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
        if (k == 1) return mean();
        
        // 使用递推公式计算高阶矩
        float result = 0.0f;
        for (int i = 0; i <= k; i++) {
            float stirling = (float) RereMathUtil.stirlingNumber2(k, i);
            float factorial = (float) RereMathUtil.factorial(i);
            result += stirling * factorial * (float) Math.pow(n, i) * (float) Math.pow(p, i);
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
        
        // 对于二项分布，中心矩有递推公式
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
        for (int k = 0; k <= n; k++) {
            float prob = pmf(k);
            if (prob > 0.0f) {
                entropy -= prob * (float) Math.log(prob);
            }
        }
        return entropy;
    }
    
    @Override
    public float cgf(float t) {
        return n * (float) Math.log(q + p * Math.exp(t));
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
        if (!(other instanceof BinomialDistribution)) {
            throw new IllegalArgumentException("只能与二项分布计算KL散度 / Can only calculate KL divergence with Binomial distribution");
        }
        
        BinomialDistribution otherBinomial = (BinomialDistribution) other;
        if (otherBinomial.getN() != n) {
            throw new IllegalArgumentException("两个二项分布的试验次数必须相同 / Both binomial distributions must have the same number of trials");
        }
        
        float kl = 0.0f;
        
        for (int k = 0; k <= n; k++) {
            float p1 = pmf(k);
            float p2 = otherBinomial.pmf(k);
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
        if (!(other instanceof BinomialDistribution)) {
            throw new IllegalArgumentException("只能与二项分布计算JS散度 / Can only calculate JS divergence with Binomial distribution");
        }
        
        BinomialDistribution otherBinomial = (BinomialDistribution) other;
        if (otherBinomial.getN() != n) {
            throw new IllegalArgumentException("两个二项分布的试验次数必须相同 / Both binomial distributions must have the same number of trials");
        }
        
        float otherP = otherBinomial.getP();
        float mP = (p + otherP) / 2.0f;
        
        // 创建混合分布
        BinomialDistribution mixed = new BinomialDistribution(n, mP);
        
        float kl1 = klDivergence(mixed);
        float kl2 = otherBinomial.klDivergence(mixed);
        
        return 0.5f * kl1 + 0.5f * kl2;
    }
    
    @Override
    public float wassersteinDistance(IDiscreteDistribution other) {
        if (!(other instanceof BinomialDistribution)) {
            throw new IllegalArgumentException("只能与二项分布计算Wasserstein距离 / Can only calculate Wasserstein distance with Binomial distribution");
        }
        
        BinomialDistribution otherBinomial = (BinomialDistribution) other;
        if (otherBinomial.getN() != n) {
            throw new IllegalArgumentException("两个二项分布的试验次数必须相同 / Both binomial distributions must have the same number of trials");
        }
        
        return Math.abs(p - otherBinomial.getP()) * n;
    }
    
    // ==================== 分布信息 / Distribution Information ====================
    
    @Override
    public String getDistributionName() {
        return "Binomial";
    }
    
    @Override
    public String getParameterInfo() {
        return String.format("n=%d, p=%.4f", n, p);
    }
    
    @Override
    public boolean isSymmetric() {
        return Math.abs(p - 0.5f) < 1e-6f;
    }
    
    @Override
    public boolean isMemoryless() {
        return false; // 二项分布不是无记忆的
    }
    
    @Override
    public String toString() {
        return String.format("BinomialDistribution(n=%d, p=%.4f)", n, p);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        BinomialDistribution that = (BinomialDistribution) obj;
        return n == that.n && Float.compare(that.p, p) == 0;
    }
    
    @Override
    public int hashCode() {
        return 31 * n + Float.hashCode(p);
    }
}