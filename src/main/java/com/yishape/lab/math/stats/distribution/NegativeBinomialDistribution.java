package com.yishape.lab.math.stats.distribution;

import com.yishape.lab.math.RereMathUtil;

/**
 * 负二项分布 (Negative Binomial Distribution)
 * 
 * 负二项分布是在一系列独立伯努利试验中，第r次成功所需的试验次数的概率分布。
 * 其概率质量函数为：P(X=k) = C(k-1, r-1) * p^r * (1-p)^(k-r)，k = r, r+1, r+2, ...
 * 其中r是成功次数，p是每次试验成功的概率，k是第r次成功所需的试验次数。
 * 
 * 当r=1时，负二项分布退化为几何分布。
 * 
 * Negative binomial distribution is the probability distribution of the number of trials
 * needed to get the r-th success in a sequence of independent Bernoulli trials.
 * Its probability mass function is: P(X=k) = C(k-1, r-1) * p^r * (1-p)^(k-r), k = r, r+1, r+2, ...
 * where r is the number of successes, p is the probability of success in each trial, and k is the number of trials needed for the r-th success.
 * 
 * When r=1, negative binomial distribution reduces to geometric distribution.
 * 
 * @author lteb2
 */
public class NegativeBinomialDistribution implements IDiscreteDistribution {
    
    private static final long serialVersionUID = 1L;
    
    /** 成功次数 / Number of successes */
    private final int r;
    
    /** 成功概率 / Probability of success */
    private final double p;
    
    /** 失败概率 / Probability of failure */
    private final double q;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param r 成功次数，必须大于0 / Number of successes, must be greater than 0
     * @param p 成功概率，范围(0,1) / Probability of success, range (0,1)
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public NegativeBinomialDistribution(int r, double p) {
        if (r <= 0) {
            throw new IllegalArgumentException("成功次数必须大于0 / Number of successes must be greater than 0");
        }
        if (p <= 0.0f || p >= 1.0f) {
            throw new IllegalArgumentException("成功概率必须在(0,1)范围内 / Probability must be in range (0,1)");
        }
        this.r = r;
        this.p = p;
        this.q = 1.0f - p;
    }
    
    /**
     * 获取成功次数
     * Get number of successes
     * 
     * @return 成功次数 / Number of successes
     */
    public int getR() {
        return r;
    }
    
    /**
     * 获取成功概率
     * Get probability of success
     * 
     * @return 成功概率 / Probability of success
     */
    public double getP() {
        return p;
    }
    
    // ==================== 基本统计量 / Basic Statistics ====================
    
    @Override
    public double mean() {
        return r / p;
    }
    
    @Override
    public double var() {
        return r * q / (p * p);
    }
    
    @Override
    public double std() {
        return (float) Math.sqrt(r * q) / p;
    }
    
    @Override
    public double median() {
        // 负二项分布的中位数近似为 floor(r/p)
        return (float) Math.floor(r / p);
    }
    
    @Override
    public double mode() {
        if (r == 1) return 1.0f; // 几何分布的情况
        return (float) Math.floor((r - 1) * q / p);
    }
    
    @Override
    public double q1() {
        return ppf(0.25f);
    }
    
    @Override
    public double q3() {
        return ppf(0.75f);
    }
    
    @Override
    public double skewness() {
        return (2.0f - p) / (float) Math.sqrt(r * q);
    }
    
    @Override
    public double kurtosis() {
        return 6.0f / r + (p * p) / (r * q);
    }
    
    // ==================== 概率计算 / Probability Calculations ====================
    
    @Override
    public double pmf(int x) {
        if (x < r) return 0.0f;
        
        // 使用对数避免数值溢出
        double logPmf = RereMathUtil.logCombination(x - 1, r - 1) + r * Math.log(p) + (x - r) * Math.log(q);
        return (float) Math.exp(logPmf);
    }
    
    @Override
    public double cdf(int x) {
        if (x < r) return 0.0f;
        
        // 使用正则化不完全Beta函数计算CDF
        return 1.0 - RereMathUtil.regularizedIncompleteBeta(x - r + 1, r, p);
    }
    
    @Override
    public int ppf(double prob) {
        if (prob < 0.0f || prob > 1.0f) {
            throw new IllegalArgumentException("概率值必须在[0,1]范围内 / Probability must be in range [0,1]");
        }
        
        if (prob == 0.0f) return r;
        if (prob == 1.0f) return Integer.MAX_VALUE;
        
        // 使用二分搜索查找分位数
        int left = r, right = r + 1000; // 设置一个合理的上界
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
    public double sf(int x) {
        return 1.0f - cdf(x);
    }
    
    @Override
    public int isf(double prob) {
        if (prob < 0.0f || prob > 1.0f) {
            throw new IllegalArgumentException("概率值必须在[0,1]范围内 / Probability must be in range [0,1]");
        }
        return ppf(1.0f - prob);
    }
    
    // ==================== 随机采样 / Random Sampling ====================
    
    @Override
    public int sample() {
        // 使用Gamma-Poisson混合方法进行采样
        // 首先生成Gamma分布的样本，然后生成泊松分布样本
        double gammaSample = sampleGamma(r, 1.0f);
        return samplePoisson((float) (gammaSample * p / q)) + r;
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
        return r;
    }
    
    @Override
    public int getMaxSupport() {
        return Integer.MAX_VALUE; // 负二项分布是无界的
    }
    
    @Override
    public boolean isInSupport(int x) {
        return x >= r;
    }
    
    @Override
    public boolean isBounded() {
        return false; // 负二项分布是无界的
    }
    
    // ==================== 高级统计方法 / Advanced Statistical Methods ====================
    
    @Override
    public double moment(int k) {
        if (k < 0) {
            throw new IllegalArgumentException("矩的阶数必须非负 / Moment order must be non-negative");
        }
        if (k == 0) return 1.0f;
        if (k == 1) return mean();
        
        // 对于负二项分布，使用递推公式计算高阶矩
        double result = 0.0f;
        for (int i = 1; i <= 1000; i++) { // 使用截断级数
            result += (float) Math.pow(i, k) * pmf(i);
        }
        return result;
    }
    
    @Override
    public double centralMoment(int k) {
        if (k < 0) {
            throw new IllegalArgumentException("矩的阶数必须非负 / Moment order must be non-negative");
        }
        if (k == 0) return 1.0f;
        if (k == 1) return 0.0f;
        if (k == 2) return var();
        
        // 使用递推公式计算中心矩
        double result = 0.0f;
        for (int i = 0; i <= k; i++) {
            double term = RereMathUtil.combination(k, i) * (float) Math.pow(-mean(), k - i) * moment(i);
            result += term;
        }
        return result;
    }
    
    @Override
    public double standardizedMoment(int k) {
        if (k < 0) {
            throw new IllegalArgumentException("矩的阶数必须非负 / Moment order must be non-negative");
        }
        if (std() == 0.0f) {
            throw new IllegalArgumentException("标准差为0时无法计算标准化矩 / Cannot calculate standardized moment when standard deviation is 0");
        }
        return centralMoment(k) / (float) Math.pow(std(), k);
    }
    
    @Override
    public double entropy() {
        double entropy = 0.0f;
        for (int k = r; k <= r + 1000; k++) { // 使用截断级数
            double prob = pmf(k);
            if (prob > 0.0f) {
                entropy -= prob * (float) Math.log(prob);
            }
        }
        return entropy;
    }
    
    @Override
    public double cgf(double t) {
        if (t >= Math.log(1.0 / q)) {
            return Double.POSITIVE_INFINITY;
        }
        return r * (float) Math.log(p / (1.0 - q * Math.exp(t)));
    }
    
    // ==================== 分位数和区间估计 / Quantiles and Interval Estimation ====================
    
    @Override
    public int quantile(double prob) {
        return ppf(prob);
    }
    
    @Override
    public int[] confidenceInterval(double confidence) {
        if (confidence < 0.0f || confidence > 1.0f) {
            throw new IllegalArgumentException("置信水平必须在[0,1]范围内 / Confidence level must be in range [0,1]");
        }
        
        double alpha = 1.0f - confidence;
        int lower = ppf(alpha / 2.0f);
        int upper = ppf(1.0f - alpha / 2.0f);
        
        return new int[]{lower, upper};
    }
    
    // ==================== 分布比较和距离 / Distribution Comparison and Distance ====================
    
    @Override
    public double klDivergence(IDiscreteDistribution other) {
        if (!(other instanceof NegativeBinomialDistribution)) {
            throw new IllegalArgumentException("只能与负二项分布计算KL散度 / Can only calculate KL divergence with Negative Binomial distribution");
        }
        
        NegativeBinomialDistribution otherNB = (NegativeBinomialDistribution) other;
        if (otherNB.getR() != r) {
            throw new IllegalArgumentException("两个负二项分布的成功次数必须相同 / Both negative binomial distributions must have the same number of successes");
        }
        
        double kl = 0.0f;
        
        for (int k = r; k <= r + 1000; k++) { // 使用截断级数
            double p1 = pmf(k);
            double p2 = otherNB.pmf(k);
            if (p1 > 0.0f && p2 > 0.0f) {
                kl += p1 * (float) Math.log(p1 / p2);
            } else if (p1 > 0.0f) {
                return Double.POSITIVE_INFINITY;
            }
        }
        
        return kl;
    }
    
    @Override
    public double jsDivergence(IDiscreteDistribution other) {
        if (!(other instanceof NegativeBinomialDistribution)) {
            throw new IllegalArgumentException("只能与负二项分布计算JS散度 / Can only calculate JS divergence with Negative Binomial distribution");
        }
        
        NegativeBinomialDistribution otherNB = (NegativeBinomialDistribution) other;
        if (otherNB.getR() != r) {
            throw new IllegalArgumentException("两个负二项分布的成功次数必须相同 / Both negative binomial distributions must have the same number of successes");
        }
        
        double mP = (p + otherNB.getP()) / 2.0f;
        
        // 创建混合分布
        NegativeBinomialDistribution mixed = new NegativeBinomialDistribution(r, mP);
        
        double kl1 = klDivergence(mixed);
        double kl2 = otherNB.klDivergence(mixed);
        
        return 0.5f * kl1 + 0.5f * kl2;
    }
    
    @Override
    public double wassersteinDistance(IDiscreteDistribution other) {
        if (!(other instanceof NegativeBinomialDistribution)) {
            throw new IllegalArgumentException("只能与负二项分布计算Wasserstein距离 / Can only calculate Wasserstein distance with Negative Binomial distribution");
        }
        
        NegativeBinomialDistribution otherNB = (NegativeBinomialDistribution) other;
        if (otherNB.getR() != r) {
            throw new IllegalArgumentException("两个负二项分布的成功次数必须相同 / Both negative binomial distributions must have the same number of successes");
        }
        
        return Math.abs(mean() - otherNB.mean());
    }
    
    // ==================== 分布信息 / Distribution Information ====================
    
    @Override
    public String getDistributionName() {
        return "Negative Binomial";
    }
    
    @Override
    public String getParameterInfo() {
        return String.format("r=%d, p=%.4f", r, p);
    }
    
    @Override
    public boolean isSymmetric() {
        return false; // 负二项分布不是对称的
    }
    
    @Override
    public boolean isMemoryless() {
        return r == 1; // 只有当r=1（几何分布）时才具有无记忆性
    }
    
    @Override
    public String toString() {
        return String.format("NegativeBinomialDistribution(r=%d, p=%.4f)", r, p);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        NegativeBinomialDistribution that = (NegativeBinomialDistribution) obj;
        return r == that.r && Double.compare(that.p, p) == 0;
    }
    
    @Override
    public int hashCode() {
        return 31 * r + Double.hashCode(p);
    }
    
    // ==================== 辅助方法 / Helper Methods ====================
    
    /**
     * 生成Gamma分布的样本
     * Generate Gamma distribution sample
     * 
     * @param shape 形状参数 / Shape parameter
     * @param scale 尺度参数 / Scale parameter
     * @return Gamma分布样本 / Gamma distribution sample
     */
    private double sampleGamma(double shape, double scale) {
        // 使用Marsaglia和Tsang的方法生成Gamma分布样本
        if (shape < 1.0f) {
            return sampleGamma(shape + 1.0f, scale) * (float) Math.pow(Math.random(), 1.0 / shape);
        }
        
        double d = shape - 1.0f / 3.0f;
        double c = 1.0f / (float) Math.sqrt(9.0 * d);
        
        while (true) {
            double x, v;
            do {
                x = (float) RereMathUtil.normalSample(0, 1);
                v = 1.0f + c * x;
            } while (v <= 0.0f);
            
            v = v * v * v;
            double u = (float) Math.random();
            
            if (u < 1.0f - 0.0331f * (x * x * x * x)) {
                return d * v * scale;
            }
            
            if (Math.log(u) < 0.5f * x * x + d * (1.0f - v + Math.log(v))) {
                return d * v * scale;
            }
        }
    }
    
    /**
     * 生成泊松分布的样本
     * Generate Poisson distribution sample
     * 
     * @param lambda 泊松参数 / Poisson parameter
     * @return 泊松分布样本 / Poisson distribution sample
     */
    private int samplePoisson(double lambda) {
        if (lambda < 10.0f) {
            // 对于小的lambda，使用直接方法
            double p = Math.exp(-lambda);
            int k = 0;
            double u = Math.random();
            while (u > p) {
                u *= Math.random();
                k++;
            }
            return k;
        } else {
            // 对于大的lambda，使用正态近似
            return Math.max(0, (int) Math.round(RereMathUtil.normalSample(lambda, (float) Math.sqrt(lambda))));
        }
    }
}
