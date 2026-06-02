package com.yishape.lab.math.stats.distribution;

import com.yishape.lab.math.RereMathUtil;
import java.io.Serializable;
import com.yishape.lab.math.linalg.IDoubleVector;

/**
 * t分布 (Student's t-Distribution)
 *
 * t分布是统计学中用于小样本推断的重要分布，特别是在总体方差未知时。 其概率密度函数为：f(x) = Γ((ν+1)/2) / (√(νπ) *
 * Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2) 其中ν是自由度。
 *
 * Student's t-distribution is an important distribution in statistics for small
 * sample inference, especially when the population variance is unknown. Its
 * probability density function is: f(x) = Γ((ν+1)/2) / (√(νπ) * Γ(ν/2)) * (1 +
 * x²/ν)^(-(ν+1)/2) where ν is the degrees of freedom.
 *
 * @author lteb2
 */
public class StudentDistribution implements IContinuousDistribution, Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 自由度 / Degrees of freedom
     */
    private final double degreesOfFreedom;

    /**
     * 位置参数 / Location parameter
     */
    private final double location;

    /**
     * 尺度参数 / Scale parameter
     */
    private final double scale;

    /**
     * 预计算的常数 / Precomputed constants
     */
    private final double normalizationConstant;
    private final double halfDof;
    private final double halfDofPlusHalf;

    /**
     * 构造函数 Constructor
     *
     * @param degreesOfFreedom 自由度，必须大于0 / Degrees of freedom, must be greater
     * than 0
     * @throws IllegalArgumentException 如果自由度小于等于0 / If degrees of freedom is
     * less than or equal to 0
     */
    public StudentDistribution(double degreesOfFreedom) {
        this(degreesOfFreedom, 0.0, 1.0);
    }

    /**
     * 构造函数，创建位置-尺度t分布 Constructor for location-scale t-distribution
     *
     * @param dof 自由度，必须大于0 / Degrees of freedom, must be greater than 0
     * @param location 位置参数 / Location parameter
     * @param scale 尺度参数，必须大于0 / Scale parameter, must be greater than 0
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public StudentDistribution(double dof, double location, double scale) {
        if (dof <= 0) {
            throw new IllegalArgumentException("自由度必须大于0 / Degrees of freedom must be greater than 0");
        }
        if (scale <= 0) {
            throw new IllegalArgumentException("尺度参数必须大于0 / Scale parameter must be greater than 0");
        }

        this.degreesOfFreedom = dof;
        this.location = location;
        this.scale = scale;
        this.halfDof = dof / 2.0;
        this.halfDofPlusHalf = (dof + 1.0) / 2.0;

        // 计算归一化常数
        // Calculate normalization constant
        double x1 = RereMathUtil.gamma(halfDofPlusHalf);
        double x2 = Math.sqrt(dof * Math.PI) * RereMathUtil.gamma(halfDof);

        this.normalizationConstant = x1 / x2;
    }

    /**
     * 计算概率密度函数值 Calculate probability density function value
     *
     * @param x 输入值 / Input value
     * @return 概率密度函数值 / PDF value
     */
    @Override
    public double pdf(double x) {
        // 对于位置-尺度变换: 如果Z~t(ν), 则X = μ + σ*Z 的PDF为 (1/σ) * f_Z((x-μ)/σ)
        double standardized = (x - location) / scale;
        double power = -(halfDofPlusHalf);
        double base = 1.0 + (standardized * standardized) / degreesOfFreedom;
        return (normalizationConstant / scale) * Math.pow(base, power);
    }

    /**
     * 计算累积分布函数值（使用近似方法） Calculate cumulative distribution function value (using
     * approximation)
     *
     * @param x 输入值 / Input value
     * @return 累积分布函数值 / CDF value
     */
    @Override
    public double cdf(double x) {
        // 对于位置-尺度变换: 如果Z~t(ν), 则X = μ + σ*Z 的CDF为 F_Z((x-μ)/σ)
        double standardized = (x - location) / scale;

        if (degreesOfFreedom >= 30) {
            // 对于大自由度，使用正态分布近似
            // For large degrees of freedom, use normal distribution approximation
            return 0.5 * (1.0 + RereMathUtil.erf(standardized / Math.sqrt(2.0)));
        }

        // 使用不完全贝塔函数
        // Using incomplete beta function
        double t = standardized / Math.sqrt(degreesOfFreedom + standardized * standardized);
        return 0.5 + 0.5 * sign(standardized) * RereMathUtil.incompleteBeta(halfDof, 0.5, t * t);
    }

    /**
     * 计算百分点函数值（分位数函数） Calculate percent point function value (quantile
     * function)
     *
     * @param p 概率值，范围[0,1] / Probability value, range [0,1]
     * @return 百分点函数值 / PPF value
     */
    @Override
    public double ppf(double p) {
        if (p < 0.0 || p > 1.0) {
            throw new IllegalArgumentException("概率值必须在[0,1]范围内 / Probability must be in range [0,1]");
        }

        if (p == 0.0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (p == 1.0) {
            return Double.POSITIVE_INFINITY;
        }

        // 对于位置-尺度变换: 如果Z~t(ν), 则X = μ + σ*Z 的PPF为 μ + σ*F_Z^(-1)(p)
        double standardPpf;
        if (degreesOfFreedom >= 30) {
            // 对于大自由度，使用正态分布近似
            // For large degrees of freedom, use normal distribution approximation
            standardPpf = RereMathUtil.inverseNormalCDF(p);
        } else {
            // 使用数值方法求解标准t分布的分位数
            // Using numerical method to solve for standard t-distribution quantile
            standardPpf = inverseTCDF(p);
        }

        return location + scale * standardPpf;
    }

    /**
     * 计算生存函数值（1 - CDF） Calculate survival function value (1 - CDF)
     *
     * @param x 输入值 / Input value
     * @return 生存函数值 / Survival function value
     */
    @Override
    public double sf(double x) {
        return 1.0f - cdf(x);
    }

    /**
     * 计算逆生存函数值 Calculate inverse survival function value
     *
     * @param p 概率值，范围[0,1] / Probability value, range [0,1]
     * @return 逆生存函数值 / Inverse survival function value
     */
    @Override
    public double isf(double p) {
        return ppf(1.0f - p);
    }

    /**
     * 获取自由度 Get degrees of freedom
     *
     * @return 自由度 / Degrees of freedom
     */
    public double getDegreesOfFreedom() {
        return degreesOfFreedom;
    }

    // 使用RereMathUtil中的gamma函数
    // Using gamma function from RereMathUtil
    // 使用RereMathUtil中的incompleteBeta函数
    // Using incompleteBeta function from RereMathUtil
    // 使用RereMathUtil中的betaCF函数
    // Using betaCF function from RereMathUtil
    // 使用RereMathUtil中的erf和inverseNormalCDF函数
    // Using erf and inverseNormalCDF functions from RereMathUtil
    /**
     * 逆t分布累积分布函数的数值求解 Numerical solution for inverse t-distribution CDF
     */
    private double inverseTCDF(double p) {
        // 使用改进的二分法求解
        // Using improved bisection method to solve
        double left = -10.0;
        double right = 10.0;
        double tolerance = 1e-8;
        int maxIter = 200;

        // 调整边界以确保包含解
        // Adjust boundaries to ensure solution is included
        while (standardCDF(left) > p && left > -1000.0) {
            left *= 2.0;
        }
        while (standardCDF(right) < p && right < 1000.0) {
            right *= 2.0;
        }

        for (int i = 0; i < maxIter; i++) {
            double mid = (left + right) / 2.0;
            double cdfMid = standardCDF(mid);

            if (Math.abs(cdfMid - p) < tolerance) {
                return mid;
            }

            if (cdfMid < p) {
                left = mid;
            } else {
                right = mid;
            }

            // 检查收敛
            if (right - left < tolerance) {
                break;
            }
        }

        return (left + right) / 2.0;
    }

    /**
     * 标准t分布的累积分布函数 CDF of standard t-distribution
     *
     * @param t 输入值 / Input value
     * @return 累积分布函数值 / CDF value
     */
    private double standardCDF(double t) {
        if (degreesOfFreedom >= 30) {
            // 对于大自由度，使用正态分布近似
            // For large degrees of freedom, use normal distribution approximation
            return 0.5 * (1.0 + RereMathUtil.erf(t / Math.sqrt(2.0)));
        }

        // 使用正则化的不完全贝塔函数
        // Using incomplete beta function
        double x = degreesOfFreedom / (degreesOfFreedom + t * t);
        double betaValue = RereMathUtil.regularizedIncompleteBeta(halfDof, 0.5, x);
        if (t >= 0) {
            return 1.0 - 0.5 *betaValue;
        } else {
            return 0.5 *betaValue;
        }
    }

    /**
     * 符号函数 Sign function
     */
    private double sign(double x) {
        return x >= 0 ? 1.0f : -1.0f;
    }

    /**
     * 获取均值 Get mean
     *
     * @return 均值 / Mean
     */
    @Override
    public double mean() {
        if (degreesOfFreedom > 1) {
            return location; // 位置-尺度t分布的均值为location参数
        }
        return Double.NaN; // 当自由度 <= 1 时均值不存在
    }

    /**
     * 获取方差 Get variance
     *
     * @return 方差 / Variance
     */
    @Override
    public double var() {
        if (degreesOfFreedom > 2) {
            // 对于位置-尺度变换: 如果Z~t(ν), 则Var(X) = σ² * Var(Z)
            double standardVar = degreesOfFreedom / (degreesOfFreedom - 2.0);
            return scale * scale * standardVar;
        }
        return Double.NaN; // 当自由度 <= 2 时方差不存在
    }

    /**
     * 获取标准差 Get standard deviation
     *
     * @return 标准差 / Standard deviation
     */
    @Override
    public double std() {
        double variance = var();
        if (Double.isNaN(variance)) {
            return Double.NaN;
        }
        return Math.sqrt(variance);
    }

    /**
     * 获取中位数 Get median
     *
     * @return 中位数 / Median
     */
    @Override
    public double median() {
        return location; // 位置-尺度t分布的中位数为location参数
    }

    /**
     * 获取众数 Get mode
     *
     * @return 众数 / Mode
     */
    @Override
    public double mode() {
        return location; // 位置-尺度t分布的众数为location参数
    }

    /**
     * 获取第一四分位数（Q1） Get first quartile (Q1)
     *
     * @return 第一四分位数 / First quartile
     */
    @Override
    public double q1() {
        return ppf(0.25f);
    }

    /**
     * 获取第三四分位数（Q3） Get third quartile (Q3)
     *
     * @return 第三四分位数 / Third quartile
     */
    @Override
    public double q3() {
        return ppf(0.75f);
    }

    /**
     * 获取偏度 Get skewness
     *
     * @return 偏度 / Skewness
     */
    @Override
    public double skewness() {
        return 0.0; // 位置-尺度t分布仍是对称的，偏度为0
    }

    /**
     * 获取峰度 Get kurtosis
     *
     * @return 峰度 / Kurtosis
     */
    @Override
    public double kurtosis() {
        if (degreesOfFreedom > 4) {
            // 位置-尺度t分布的峰度与标准t分布相同（尺度变换不影响峰度）
            return 6.0 / (degreesOfFreedom - 4.0);
        }
        return Double.NaN; // 当自由度 <= 4 时峰度不存在
    }

    // 缓存的分布对象，避免重复创建
    // Cached distribution objects to avoid repeated creation
    private NormalDistribution normal;
    private Chi2Distribution chi2;

    /**
     * 生成一个随机样本 Generate a random sample
     *
     * @return 随机样本 / Random sample
     */
    @Override
    public double sample() {
        // 使用正态分布和卡方分布生成t分布随机数
        // Using normal and chi-squared distributions to generate t-distribution random numbers
        if (normal == null) {
            normal = new NormalDistribution(0.0, 1.0);
        }
        if (chi2 == null) {
            chi2 = new Chi2Distribution(degreesOfFreedom);
        }

        double z = normal.sample();
        double chi2Sample = chi2.sample();

        // 对于位置-尺度变换: 如果Z~t(ν), 则X = μ + σ*Z
        double standardT = z / Math.sqrt(chi2Sample / degreesOfFreedom);
        return location + scale * standardT;
    }

    /**
     * 生成n个随机样本 Generate n random samples
     *
     * @param n 样本数量 / Number of samples
     * @return 随机样本数组 / Array of random samples
     */
    @Override
    public double[] sample(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("样本数量必须大于0 / Sample size must be greater than 0");
        }

        // 使用IVector进行数组操作
        // Using IDoubleVector for array operations
        IDoubleVector samples = IDoubleVector.zeros(n);
        for (int i = 0; i < n; i++) {
            samples.set(i, sample());
        }
        return samples.getData();
    }

    @Override
    public String toString() {
        return String.format("StudentDistribution(degreesOfFreedom=%.3f, location=%.3f, scale=%.3f)",
                degreesOfFreedom, location, scale);
    }
}
