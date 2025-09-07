package com.reremouse.lab.math.stat;

import com.reremouse.lab.math.stat.distribution.NormalDistribution;
import com.reremouse.lab.math.stat.distribution.StudentDistribution;
import com.reremouse.lab.math.stat.distribution.UniformDistribution;
import com.reremouse.lab.math.stat.distribution.ExponentialDistribution;
import com.reremouse.lab.math.stat.distribution.Chi2Distribution;
import com.reremouse.lab.math.stat.distribution.FDistribution;

/**
 * 统计分布工厂类 / Statistical Distribution Factory Class
 * 
 * <p>Stat类提供了创建各种常用概率分布对象的静态工厂方法。该类是统计学计算的核心入口点，
 * 支持正态分布、t分布、均匀分布、指数分布、卡方分布和F分布等六种重要的概率分布。</p>
 * 
 * <p>Stat class provides static factory methods for creating various common probability distribution objects.
 * This class is the core entry point for statistical calculations, supporting six important probability
 * distributions: normal, t-distribution, uniform, exponential, chi-squared, and F-distribution.</p>
 * 
 * <h3>支持的分布类型 / Supported Distribution Types:</h3>
 * <ul>
 *   <li><strong>正态分布 (Normal Distribution)</strong> - 用于连续随机变量的建模</li>
 *   <li><strong>t分布 (Student's t-Distribution)</strong> - 用于小样本统计推断</li>
 *   <li><strong>均匀分布 (Uniform Distribution)</strong> - 用于等概率事件的建模</li>
 *   <li><strong>指数分布 (Exponential Distribution)</strong> - 用于时间间隔的建模</li>
 *   <li><strong>卡方分布 (Chi-Squared Distribution)</strong> - 用于假设检验</li>
 *   <li><strong>F分布 (F-Distribution)</strong> - 用于方差分析</li>
 * </ul>
 * 
 * <h3>使用示例 / Usage Examples:</h3>
 * <pre>{@code
 * // 创建标准正态分布 (均值为0，标准差为1)
 * // Create standard normal distribution (mean=0, stdDev=1)
 * NormalDistribution stdNormal = Stat.norm();
 * 
 * // 创建自定义正态分布
 * // Create custom normal distribution
 * NormalDistribution normal = Stat.norm(5.0f, 2.0f);
 * 
 * // 创建t分布
 * // Create t-distribution
 * StudentDistribution tDist = Stat.t(10.0f);
 * 
 * // 创建均匀分布
 * // Create uniform distribution
 * UniformDistribution uniform = Stat.uniform(0.0f, 10.0f);
 * 
 * // 创建指数分布
 * // Create exponential distribution
 * ExponentialDistribution exp = Stat.exponential(1.5f);
 * 
 * // 创建卡方分布
 * // Create chi-squared distribution
 * Chi2Distribution chi2 = Stat.chi2(5.0f);
 * 
 * // 创建F分布
 * // Create F-distribution
 * FDistribution fDist = Stat.f(3.0f, 10.0f);
 * }</pre>
 * 
 * <h3>分布对象功能 / Distribution Object Features:</h3>
 * <p>每个分布对象都实现了IContinuousDistribution接口，提供以下功能：</p>
 * <p>Each distribution object implements the IContinuousDistribution interface, providing the following features:</p>
 * <ul>
 *   <li><strong>概率密度函数 (PDF)</strong> - pdf(x) 方法</li>
 *   <li><strong>累积分布函数 (CDF)</strong> - cdf(x) 方法</li>
 *   <li><strong>百分点函数 (PPF)</strong> - ppf(p) 方法</li>
 *   <li><strong>生存函数 (SF)</strong> - sf(x) 方法</li>
 *   <li><strong>逆生存函数 (ISF)</strong> - isf(p) 方法</li>
 *   <li><strong>统计量计算</strong> - 均值、方差、标准差、中位数、众数、四分位数、偏度、峰度</li>
 *   <li><strong>随机采样</strong> - sample() 和 sample(n) 方法</li>
 * </ul>
 * 
 * <h3>应用场景 / Application Scenarios:</h3>
 * <ul>
 *   <li><strong>假设检验</strong> - 使用t分布、卡方分布、F分布进行统计检验</li>
 *   <li><strong>置信区间估计</strong> - 使用正态分布和t分布计算置信区间</li>
 *   <li><strong>蒙特卡洛模拟</strong> - 使用各种分布生成随机样本</li>
 *   <li><strong>风险评估</strong> - 使用正态分布和指数分布建模风险</li>
 *   <li><strong>质量控制</strong> - 使用正态分布进行过程控制</li>
 *   <li><strong>可靠性分析</strong> - 使用指数分布建模故障时间</li>
 * </ul>
 * 
 * <h3>注意事项 / Notes:</h3>
 * <ul>
 *   <li>所有分布参数都使用float类型，确保计算效率</li>
 *   <li>分布对象是不可变的，创建后参数不能修改</li>
 *   <li>随机数生成使用Java内置的Math.random()方法</li>
 *   <li>某些统计量在特定条件下可能不存在（返回Float.NaN）</li>
 *   <li>数值计算使用近似方法，精度可能有限</li>
 * </ul>
 * 
 * <p><strong>Author:</strong> lteb2</p>
 * <p><strong>Version:</strong> 1.0</p>
 * <p><strong>Since:</strong> 1.0</p>
 * 
 * @see com.reremouse.lab.math.stat.distribution.NormalDistribution
 * @see com.reremouse.lab.math.stat.distribution.StudentDistribution
 * @see com.reremouse.lab.math.stat.distribution.UniformDistribution
 * @see com.reremouse.lab.math.stat.distribution.ExponentialDistribution
 * @see com.reremouse.lab.math.stat.distribution.Chi2Distribution
 * @see com.reremouse.lab.math.stat.distribution.FDistribution
 * @see com.reremouse.lab.math.stat.distribution.IContinuousDistribution
 */
public class Stat {


    /**
     * 创建正态分布对象
     * Create a normal distribution object
     * 
     * <p>正态分布（也称为高斯分布）是统计学中最重要的连续概率分布之一。
     * 其概率密度函数为：f(x) = (1/σ√(2π)) * e^(-(x-μ)²/(2σ²))</p>
     * 
     * <p>Normal distribution (also known as Gaussian distribution) is one of the most important
     * continuous probability distributions in statistics. Its probability density function is:
     * f(x) = (1/σ√(2π)) * e^(-(x-μ)²/(2σ²))</p>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>中心极限定理的应用 / Applications of central limit theorem</li>
     *   <li>假设检验 / Hypothesis testing</li>
     *   <li>置信区间估计 / Confidence interval estimation</li>
     *   <li>质量控制 / Quality control</li>
     *   <li>风险评估 / Risk assessment</li>
     * </ul>
     * 
     * @param mean 均值，分布的中心位置 / Mean, the center of the distribution
     * @param stdDev 标准差，必须大于0 / Standard deviation, must be greater than 0
     * @return 正态分布对象 / Normal distribution object
     * @throws IllegalArgumentException 如果标准差小于等于0 / If standard deviation is less than or equal to 0
     * 
     * @see NormalDistribution
     * @since 1.0
     */
    public static NormalDistribution norm(float mean, float stdDev) {
        return new NormalDistribution(mean, stdDev);
    }
    
    /**
     * 创建标准正态分布对象（均值为0，标准差为1）
     * Create a standard normal distribution object (mean=0, stdDev=1)
     * 
     * <p>标准正态分布是均值为0、标准差为1的正态分布，也称为Z分布。
     * 它是统计学中的基准分布，许多统计方法都基于标准正态分布。</p>
     * 
     * <p>Standard normal distribution is a normal distribution with mean=0 and stdDev=1,
     * also known as Z-distribution. It is the reference distribution in statistics,
     * and many statistical methods are based on the standard normal distribution.</p>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>Z检验 / Z-tests</li>
     *   <li>标准化数据 / Data standardization</li>
     *   <li>概率计算 / Probability calculations</li>
     *   <li>统计推断 / Statistical inference</li>
     * </ul>
     * 
     * @return 标准正态分布对象 / Standard normal distribution object
     * 
     * @see NormalDistribution
     * @since 1.0
     */
    public static NormalDistribution norm() {
        return new NormalDistribution(0,1);
    }

    /**
     * 创建t分布对象
     * Create a t-distribution object
     * 
     * <p>t分布（学生t分布）是统计学中用于小样本推断的重要分布，特别是在总体方差未知时。
     * 其概率密度函数为：f(x) = Γ((ν+1)/2) / (√(νπ) * Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)</p>
     * 
     * <p>Student's t-distribution is an important distribution in statistics for small sample inference,
     * especially when the population variance is unknown. Its probability density function is:
     * f(x) = Γ((ν+1)/2) / (√(νπ) * Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)</p>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>小样本t检验 / Small sample t-tests</li>
     *   <li>置信区间估计 / Confidence interval estimation</li>
     *   <li>回归分析 / Regression analysis</li>
     *   <li>方差分析 / Analysis of variance</li>
     * </ul>
     * 
     * @param degreesOfFreedom 自由度，必须大于0 / Degrees of freedom, must be greater than 0
     * @return t分布对象 / t-distribution object
     * @throws IllegalArgumentException 如果自由度小于等于0 / If degrees of freedom is less than or equal to 0
     * 
     * @see StudentDistribution
     * @since 1.0
     */
    public static StudentDistribution t(float degreesOfFreedom) {
        return new StudentDistribution(degreesOfFreedom);
    }

    /**
     * 创建均匀分布对象
     * Create a uniform distribution object
     * 
     * <p>均匀分布是概率论中最简单的连续概率分布之一，在区间[a,b]内所有值出现的概率相等。
     * 其概率密度函数为：f(x) = 1/(b-a) for a ≤ x ≤ b, 0 otherwise</p>
     * 
     * <p>Uniform distribution is one of the simplest continuous probability distributions in probability theory,
     * where all values in the interval [a,b] have equal probability. Its probability density function is:
     * f(x) = 1/(b-a) for a ≤ x ≤ b, 0 otherwise</p>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>随机数生成 / Random number generation</li>
     *   <li>蒙特卡洛模拟 / Monte Carlo simulation</li>
     *   <li>等概率事件建模 / Equal probability event modeling</li>
     *   <li>数值积分 / Numerical integration</li>
     * </ul>
     * 
     * @param lowerBound 下界 / Lower bound
     * @param upperBound 上界，必须大于下界 / Upper bound, must be greater than lower bound
     * @return 均匀分布对象 / Uniform distribution object
     * @throws IllegalArgumentException 如果上界小于等于下界 / If upper bound is less than or equal to lower bound
     * 
     * @see UniformDistribution
     * @since 1.0
     */
    public static UniformDistribution uniform(float lowerBound, float upperBound) {
        return new UniformDistribution(lowerBound, upperBound);
    }

    /**
     * 创建指数分布对象
     * Create an exponential distribution object
     * 
     * <p>指数分布是描述泊松过程中事件间隔时间的连续概率分布，具有无记忆性。
     * 其概率密度函数为：f(x) = λe^(-λx) for x ≥ 0, 0 otherwise</p>
     * 
     * <p>Exponential distribution is a continuous probability distribution that describes the time between events
     * in a Poisson process, and has the memoryless property. Its probability density function is:
     * f(x) = λe^(-λx) for x ≥ 0, 0 otherwise</p>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>可靠性分析 / Reliability analysis</li>
     *   <li>排队论 / Queueing theory</li>
     *   <li>生存分析 / Survival analysis</li>
     *   <li>故障时间建模 / Failure time modeling</li>
     *   <li>放射性衰变 / Radioactive decay</li>
     * </ul>
     * 
     * @param rate 速率参数（λ），必须大于0 / Rate parameter (λ), must be greater than 0
     * @return 指数分布对象 / Exponential distribution object
     * @throws IllegalArgumentException 如果速率参数小于等于0 / If rate parameter is less than or equal to 0
     * 
     * @see ExponentialDistribution
     * @since 1.0
     */
    public static ExponentialDistribution exponential(float rate) {
        return new ExponentialDistribution(rate);
    }

    /**
     * 创建卡方分布对象
     * Create a chi-squared distribution object
     * 
     * <p>卡方分布是统计学中重要的连续概率分布，常用于假设检验和置信区间估计。
     * 其概率密度函数为：f(x) = (1/(2^(k/2) * Γ(k/2))) * x^(k/2-1) * e^(-x/2) for x > 0, 0 otherwise</p>
     * 
     * <p>Chi-squared distribution is an important continuous probability distribution in statistics,
     * commonly used in hypothesis testing and confidence interval estimation. Its probability density function is:
     * f(x) = (1/(2^(k/2) * Γ(k/2))) * x^(k/2-1) * e^(-x/2) for x > 0, 0 otherwise</p>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>卡方检验 / Chi-squared tests</li>
     *   <li>拟合优度检验 / Goodness of fit tests</li>
     *   <li>独立性检验 / Independence tests</li>
     *   <li>方差分析 / Analysis of variance</li>
     *   <li>置信区间估计 / Confidence interval estimation</li>
     * </ul>
     * 
     * @param degreesOfFreedom 自由度，必须大于0 / Degrees of freedom, must be greater than 0
     * @return 卡方分布对象 / Chi-squared distribution object
     * @throws IllegalArgumentException 如果自由度小于等于0 / If degrees of freedom is less than or equal to 0
     * 
     * @see Chi2Distribution
     * @since 1.0
     */
    public static Chi2Distribution chi2(float degreesOfFreedom) {
        return new Chi2Distribution(degreesOfFreedom);
    }

    /**
     * 创建F分布对象
     * Create an F-distribution object
     * 
     * <p>F分布是统计学中重要的连续概率分布，常用于方差分析和回归分析中的假设检验。
     * 其概率密度函数为：f(x) = (Γ((d1+d2)/2) / (Γ(d1/2) * Γ(d2/2))) * (d1/d2)^(d1/2) * x^(d1/2-1) * (1 + d1*x/d2)^(-(d1+d2)/2)</p>
     * 
     * <p>F-distribution is an important continuous probability distribution in statistics,
     * commonly used in hypothesis testing in analysis of variance and regression analysis.
     * Its probability density function is:
     * f(x) = (Γ((d1+d2)/2) / (Γ(d1/2) * Γ(d2/2))) * (d1/d2)^(d1/2) * x^(d1/2-1) * (1 + d1*x/d2)^(-(d1+d2)/2)</p>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>方差分析 / Analysis of variance (ANOVA)</li>
     *   <li>F检验 / F-tests</li>
     *   <li>回归分析 / Regression analysis</li>
     *   <li>方差齐性检验 / Homogeneity of variance tests</li>
     *   <li>模型比较 / Model comparison</li>
     * </ul>
     * 
     * @param degreesOfFreedom1 分子自由度，必须大于0 / Numerator degrees of freedom, must be greater than 0
     * @param degreesOfFreedom2 分母自由度，必须大于0 / Denominator degrees of freedom, must be greater than 0
     * @return F分布对象 / F-distribution object
     * @throws IllegalArgumentException 如果任一自由度小于等于0 / If any degrees of freedom is less than or equal to 0
     * 
     * @see FDistribution
     * @since 1.0
     */
    public static FDistribution f(float degreesOfFreedom1, float degreesOfFreedom2) {
        return new FDistribution(degreesOfFreedom1, degreesOfFreedom2);
    }
}
