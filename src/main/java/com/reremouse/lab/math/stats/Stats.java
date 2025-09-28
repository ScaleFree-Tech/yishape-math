package com.reremouse.lab.math.stats;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.stats.anova.ANOVA;
import com.reremouse.lab.math.stats.testing.ParameterEstimation;
import com.reremouse.lab.math.stats.testing.HypothesisTesting;
import com.reremouse.lab.math.stats.distribution.NormalDistribution;
import com.reremouse.lab.math.stats.distribution.StudentDistribution;
import com.reremouse.lab.math.stats.distribution.UniformDistribution;
import com.reremouse.lab.math.stats.distribution.ExponentialDistribution;
import com.reremouse.lab.math.stats.distribution.Chi2Distribution;
import com.reremouse.lab.math.stats.distribution.FDistribution;
import com.reremouse.lab.math.stats.distribution.BetaDistribution;
import com.reremouse.lab.math.stats.distribution.GammaDistribution;
import com.reremouse.lab.math.stats.distribution.BernoulliDistribution;
import com.reremouse.lab.math.stats.distribution.BinomialDistribution;
import com.reremouse.lab.math.stats.distribution.DiscreteUniformDistribution;
import com.reremouse.lab.math.stats.distribution.GeometricDistribution;
import com.reremouse.lab.math.stats.distribution.NegativeBinomialDistribution;
import com.reremouse.lab.math.stats.distribution.PoissonDistribution;

/**
 * 统计工厂类 / Statistical Factory Class
 * 
 * <p>Stat类提供了创建各种常用概率分布对象的静态工厂方法。该类是统计学计算的核心入口点，
 * 支持连续型分布和离散型分布两大类，包括正态分布、t分布、均匀分布、指数分布、卡方分布、F分布、
 * Beta分布、Gamma分布、伯努利分布、二项分布、离散均匀分布、几何分布、负二项分布和泊松分布等十四种重要的概率分布。</p>
 * 
 * <p>Stats class provides static factory methods for creating various common probability distribution objects.
 This class is the core entry point for statistical calculations, supporting continuous and discrete distributions 
 including normal, t-distribution, uniform, exponential, chi-squared, F-distribution, Beta, Gamma, Bernoulli, 
 binomial, discrete uniform, geometric, negative binomial, and Poisson distributions.</p>
 * 
 * <h3>支持的分布类型 / Supported Distribution Types:</h3>
 * 
 * <h4>连续型分布 / Continuous Distributions:</h4>
 * <ul>
 *   <li><strong>正态分布 (Normal Distribution)</strong> - 用于连续随机变量的建模</li>
 *   <li><strong>t分布 (Student's t-Distribution)</strong> - 用于小样本统计推断</li>
 *   <li><strong>均匀分布 (Uniform Distribution)</strong> - 用于等概率事件的建模</li>
 *   <li><strong>指数分布 (Exponential Distribution)</strong> - 用于时间间隔的建模</li>
 *   <li><strong>卡方分布 (Chi-Squared Distribution)</strong> - 用于假设检验</li>
 *   <li><strong>F分布 (F-Distribution)</strong> - 用于方差分析</li>
 *   <li><strong>Beta分布 (Beta Distribution)</strong> - 用于比例建模和贝叶斯统计</li>
 *   <li><strong>Gamma分布 (Gamma Distribution)</strong> - 用于等待时间建模和可靠性分析</li>
 * </ul>
 * 
 * <h4>离散型分布 / Discrete Distributions:</h4>
 * <ul>
 *   <li><strong>伯努利分布 (Bernoulli Distribution)</strong> - 用于二分类问题建模</li>
 *   <li><strong>二项分布 (Binomial Distribution)</strong> - 用于重复试验建模</li>
 *   <li><strong>离散均匀分布 (Discrete Uniform Distribution)</strong> - 用于等概率离散选择</li>
 *   <li><strong>几何分布 (Geometric Distribution)</strong> - 用于首次成功时间建模</li>
 *   <li><strong>负二项分布 (Negative Binomial Distribution)</strong> - 用于过度分散计数数据建模</li>
 *   <li><strong>泊松分布 (Poisson Distribution)</strong> - 用于事件计数建模</li>
 * </ul>
 * 
 * <h3>使用示例 / Usage Examples:</h3>
 * <pre>{@code
 // 连续型分布示例 / Continuous Distribution Examples
 // 创建标准正态分布 (均值为0，标准差为1)
 // Create standard normal distribution (mean=0, stdDev=1)
 NormalDistribution stdNormal = Stats.norm();
 
 // 创建自定义正态分布
 // Create custom normal distribution
 NormalDistribution normal = Stats.norm(5.0f, 2.0f);
 
 // 创建t分布
 // Create t-distribution
 StudentDistribution tDist = Stats.t(10.0f);
 
 // 创建均匀分布
 // Create uniform distribution
 UniformDistribution uniform = Stats.uniform(0.0f, 10.0f);
 
 // 创建指数分布
 // Create exponential distribution
 ExponentialDistribution exp = Stats.exponential(1.5f);
 
 // 创建卡方分布
 // Create chi-squared distribution
 Chi2Distribution chi2 = Stats.chi2(5.0f);
 
 // 创建F分布
 // Create F-distribution
 FDistribution fDist = Stats.f(3.0f, 10.0f);
 
 // 创建Beta分布
 // Create Beta distribution
 BetaDistribution beta = Stats.beta(2.0f, 3.0f);
 
 // 创建Gamma分布
 // Create Gamma distribution
 GammaDistribution gamma = Stats.gamma(2.0f, 1.0f);
 
 // 离散型分布示例 / Discrete Distribution Examples
 // 创建伯努利分布
 // Create Bernoulli distribution
 BernoulliDistribution bernoulli = Stats.bernoulli(0.3f);
 
 // 创建二项分布
 // Create binomial distribution
 BinomialDistribution binomial = Stats.binomial(10, 0.5f);
 
 // 创建离散均匀分布
 // Create discrete uniform distribution
 DiscreteUniformDistribution discreteUniform = Stats.discreteUniform(1, 6);
 
 // 创建几何分布
 // Create geometric distribution
 GeometricDistribution geometric = Stats.geometric(0.2f);
 
 // 创建负二项分布
 // Create negative binomial distribution
 NegativeBinomialDistribution negBin = Stats.negativeBinomial(3, 0.4f);
 
 // 创建泊松分布
 // Create Poisson distribution
 PoissonDistribution poisson = Stats.poisson(2.5f);
 }</pre>
 * 
 * <h3>分布对象功能 / Distribution Object Features:</h3>
 * <p>连续型分布对象实现了IContinuousDistribution接口，离散型分布对象实现了IDiscreteDistribution接口，提供以下功能：</p>
 * <p>Continuous distribution objects implement the IContinuousDistribution interface, discrete distribution objects implement the IDiscreteDistribution interface, providing the following features:</p>
 * <ul>
 *   <li><strong>概率函数</strong> - 连续型分布：pdf(x) 方法；离散型分布：pmf(x) 方法</li>
 *   <li><strong>Probability Functions</strong> - Continuous: pdf(x) method; Discrete: pmf(x) method</li>
 *   <li><strong>累积分布函数 (CDF)</strong> - cdf(x) 方法</li>
 *   <li><strong>百分点函数 (PPF)</strong> - ppf(p) 方法</li>
 *   <li><strong>生存函数 (SF)</strong> - sf(x) 方法</li>
 *   <li><strong>逆生存函数 (ISF)</strong> - isf(p) 方法</li>
 *   <li><strong>统计量计算</strong> - 均值、方差、标准差、中位数、众数、四分位数、偏度、峰度</li>
 *   <li><strong>随机采样</strong> - sample() 和 sample(n) 方法</li>
 *   <li><strong>高级统计方法</strong> - 矩计算、熵计算、分布比较等</li>
 * </ul>
 * 
 * <h3>应用场景 / Application Scenarios:</h3>
 * <ul>
 *   <li><strong>假设检验</strong> - 使用t分布、卡方分布、F分布进行统计检验</li>
 *   <li><strong>置信区间估计</strong> - 使用正态分布和t分布计算置信区间</li>
 *   <li><strong>蒙特卡洛模拟</strong> - 使用各种分布生成随机样本</li>
 *   <li><strong>风险评估</strong> - 使用正态分布和指数分布建模风险</li>
 *   <li><strong>质量控制</strong> - 使用正态分布进行过程控制</li>
 *   <li><strong>可靠性分析</strong> - 使用指数分布、Gamma分布建模故障时间</li>
 *   <li><strong>贝叶斯统计</strong> - 使用Beta分布、Gamma分布作为先验分布</li>
 *   <li><strong>机器学习</strong> - 使用各种分布进行概率建模和采样</li>
 *   <li><strong>计数数据建模</strong> - 使用泊松分布、负二项分布建模计数数据</li>
 *   <li><strong>二分类问题</strong> - 使用伯努利分布、二项分布建模二分类结果</li>
 *   <li><strong>等待时间建模</strong> - 使用几何分布、指数分布建模等待时间</li>
 *   <li><strong>比例建模</strong> - 使用Beta分布建模比例数据</li>
 * </ul>
 * 
 * <h3>注意事项 / Notes:</h3>
 * <ul>
 *   <li>所有分布参数都使用double类型，确保计算效率</li>
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
 * @see com.reremouse.lab.math.stats.distribution.NormalDistribution
 * @see com.reremouse.lab.math.stats.distribution.StudentDistribution
 * @see com.reremouse.lab.math.stats.distribution.UniformDistribution
 * @see com.reremouse.lab.math.stats.distribution.ExponentialDistribution
 * @see com.reremouse.lab.math.stats.distribution.Chi2Distribution
 * @see com.reremouse.lab.math.stats.distribution.FDistribution
 * @see com.reremouse.lab.math.stats.distribution.BetaDistribution
 * @see com.reremouse.lab.math.stats.distribution.GammaDistribution
 * @see com.reremouse.lab.math.stats.distribution.BernoulliDistribution
 * @see com.reremouse.lab.math.stats.distribution.BinomialDistribution
 * @see com.reremouse.lab.math.stats.distribution.DiscreteUniformDistribution
 * @see com.reremouse.lab.math.stats.distribution.GeometricDistribution
 * @see com.reremouse.lab.math.stats.distribution.NegativeBinomialDistribution
 * @see com.reremouse.lab.math.stats.distribution.PoissonDistribution
 * @see com.reremouse.lab.math.stats.distribution.IContinuousDistribution
 * @see com.reremouse.lab.math.stats.distribution.IDiscreteDistribution
 */
public class Stats {

    
    /**
     * 参数估计 / Parameter estimation
     * Parameter estimation
     */
    public static ParameterEstimation estimator = new ParameterEstimation();//参数估计
    
    /**
     * 假设检验 / Hypothesis testing
     * Hypothesis testing
     */
    public static HypothesisTesting tester = new HypothesisTesting();//假设检验
    
    /**
     * 方差分析/ANAVA
     */
    public static ANOVA anova = new ANOVA();
    

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
    public static NormalDistribution norm(double mean, double stdDev) {
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
    public static StudentDistribution t(double degreesOfFreedom) {
        return new StudentDistribution(degreesOfFreedom);
    }
    
    /**
     * 
     * @param dof 自由度
     * @param location 均值
     * @param scale 标准差
     * @return 
     */
    public static StudentDistribution t(double dof, double location, double scale) {
        return new StudentDistribution(dof,location,scale);
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
    public static UniformDistribution uniform(double lowerBound, double upperBound) {
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
    public static ExponentialDistribution exponential(double rate) {
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
    public static Chi2Distribution chi2(double degreesOfFreedom) {
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
    public static FDistribution f(double degreesOfFreedom1, double degreesOfFreedom2) {
        return new FDistribution(degreesOfFreedom1, degreesOfFreedom2);
    }

    // ==================== 离散型分布工厂方法 / Discrete Distribution Factory Methods ====================

    /**
     * 创建伯努利分布对象
     * Create a Bernoulli distribution object
     * 
     * <p>伯努利分布是只有两个可能结果的离散概率分布：成功（1）和失败（0）。
     * 其概率质量函数为：P(X=1) = p, P(X=0) = 1-p</p>
     * 
     * <p>Bernoulli distribution is a discrete probability distribution with only two possible outcomes:
     * success (1) and failure (0). Its probability mass function is: P(X=1) = p, P(X=0) = 1-p</p>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>二分类问题 / Binary classification problems</li>
     *   <li>成功/失败事件建模 / Success/failure event modeling</li>
     *   <li>随机试验 / Random experiments</li>
     *   <li>质量控制 / Quality control</li>
     * </ul>
     * 
     * @param p 成功概率，范围[0,1] / Probability of success, range [0,1]
     * @return 伯努利分布对象 / Bernoulli distribution object
     * @throws IllegalArgumentException 如果概率不在[0,1]范围内 / If probability is not in range [0,1]
     * 
     * @see BernoulliDistribution
     * @since 1.0
     */
    public static BernoulliDistribution bernoulli(double p) {
        return new BernoulliDistribution(p);
    }

    /**
     * 创建二项分布对象
     * Create a binomial distribution object
     * 
     * <p>二项分布是n次独立伯努利试验中成功次数的离散概率分布。
     * 其概率质量函数为：P(X=k) = C(n,k) * p^k * (1-p)^(n-k)</p>
     * 
     * <p>Binomial distribution is the discrete probability distribution of the number of successes
     * in a sequence of n independent Bernoulli trials. Its probability mass function is:
     * P(X=k) = C(n,k) * p^k * (1-p)^(n-k)</p>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>重复试验建模 / Repeated trial modeling</li>
     *   <li>质量控制 / Quality control</li>
     *   <li>市场调研 / Market research</li>
     *   <li>医学试验 / Medical trials</li>
     * </ul>
     * 
     * @param n 试验次数，必须大于0 / Number of trials, must be greater than 0
     * @param p 成功概率，范围[0,1] / Probability of success, range [0,1]
     * @return 二项分布对象 / Binomial distribution object
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     * 
     * @see BinomialDistribution
     * @since 1.0
     */
    public static BinomialDistribution binomial(int n, double p) {
        return new BinomialDistribution(n, p);
    }

    /**
     * 创建离散均匀分布对象
     * Create a discrete uniform distribution object
     * 
     * <p>离散均匀分布是在有限个离散值上等概率的分布。
     * 其概率质量函数为：P(X=k) = 1/n，k = a, a+1, ..., b</p>
     * 
     * <p>Discrete uniform distribution is a probability distribution where each value
     * in a finite set of discrete values has equal probability. Its probability mass function is:
     * P(X=k) = 1/n, k = a, a+1, ..., b</p>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>随机数生成 / Random number generation</li>
     *   <li>等概率选择 / Equal probability selection</li>
     *   <li>模拟实验 / Simulation experiments</li>
     *   <li>游戏设计 / Game design</li>
     * </ul>
     * 
     * @param a 下界 / Lower bound
     * @param b 上界，必须大于等于下界 / Upper bound, must be greater than or equal to lower bound
     * @return 离散均匀分布对象 / Discrete uniform distribution object
     * @throws IllegalArgumentException 如果上界小于下界 / If upper bound is less than lower bound
     * 
     * @see DiscreteUniformDistribution
     * @since 1.0
     */
    public static DiscreteUniformDistribution discreteUniform(int a, int b) {
        return new DiscreteUniformDistribution(a, b);
    }

    /**
     * 创建离散均匀分布对象（在[0, n-1]上）
     * Create a discrete uniform distribution object on [0, n-1]
     * 
     * @param n 上界（下界为0）/ Upper bound (lower bound is 0)
     * @return 离散均匀分布对象 / Discrete uniform distribution object
     * @throws IllegalArgumentException 如果n小于等于0 / If n is less than or equal to 0
     * 
     * @see DiscreteUniformDistribution
     * @since 1.0
     */
    public static DiscreteUniformDistribution discreteUniform(int n) {
        return new DiscreteUniformDistribution(n);
    }

    /**
     * 创建几何分布对象
     * Create a geometric distribution object
     * 
     * <p>几何分布是在一系列独立伯努利试验中，第一次成功所需的试验次数的概率分布。
     * 其概率质量函数为：P(X=k) = (1-p)^(k-1) * p，k = 1, 2, 3, ...</p>
     * 
     * <p>Geometric distribution is the probability distribution of the number of trials
     * needed to get the first success in a sequence of independent Bernoulli trials.
     * Its probability mass function is: P(X=k) = (1-p)^(k-1) * p, k = 1, 2, 3, ...</p>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>等待时间建模 / Waiting time modeling</li>
     *   <li>可靠性分析 / Reliability analysis</li>
     *   <li>首次成功时间 / Time to first success</li>
     *   <li>排队论 / Queueing theory</li>
     * </ul>
     * 
     * @param p 成功概率，范围(0,1] / Probability of success, range (0,1]
     * @return 几何分布对象 / Geometric distribution object
     * @throws IllegalArgumentException 如果概率不在(0,1]范围内 / If probability is not in range (0,1]
     * 
     * @see GeometricDistribution
     * @since 1.0
     */
    public static GeometricDistribution geometric(double p) {
        return new GeometricDistribution(p);
    }

    /**
     * 创建负二项分布对象
     * Create a negative binomial distribution object
     * 
     * <p>负二项分布是在一系列独立伯努利试验中，第r次成功所需的试验次数的概率分布。
     * 其概率质量函数为：P(X=k) = C(k-1, r-1) * p^r * (1-p)^(k-r)，k = r, r+1, r+2, ...</p>
     * 
     * <p>Negative binomial distribution is the probability distribution of the number of trials
     * needed to get the r-th success in a sequence of independent Bernoulli trials.
     * Its probability mass function is: P(X=k) = C(k-1, r-1) * p^r * (1-p)^(k-r), k = r, r+1, r+2, ...</p>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>重复试验建模 / Repeated trial modeling</li>
     *   <li>可靠性分析 / Reliability analysis</li>
     *   <li>计数数据建模 / Count data modeling</li>
     *   <li>过度分散数据 / Overdispersed data</li>
     * </ul>
     * 
     * @param r 成功次数，必须大于0 / Number of successes, must be greater than 0
     * @param p 成功概率，范围(0,1) / Probability of success, range (0,1)
     * @return 负二项分布对象 / Negative binomial distribution object
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     * 
     * @see NegativeBinomialDistribution
     * @since 1.0
     */
    public static NegativeBinomialDistribution negativeBinomial(int r, double p) {
        return new NegativeBinomialDistribution(r, p);
    }

    /**
     * 创建泊松分布对象
     * Create a Poisson distribution object
     * 
     * <p>泊松分布是描述在固定时间间隔内发生随机事件次数的离散概率分布。
     * 其概率质量函数为：P(X=k) = (λ^k * e^(-λ)) / k!，k = 0, 1, 2, ...</p>
     * 
     * <p>Poisson distribution is a discrete probability distribution that expresses
     * the probability of a given number of events occurring in a fixed interval of time.
     * Its probability mass function is: P(X=k) = (λ^k * e^(-λ)) / k!, k = 0, 1, 2, ...</p>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>事件计数建模 / Event count modeling</li>
     *   <li>排队论 / Queueing theory</li>
     *   <li>可靠性分析 / Reliability analysis</li>
     *   <li>生物学建模 / Biological modeling</li>
     * </ul>
     * 
     * @param lambda 平均发生率（期望值），必须大于0 / Average rate of occurrence (expected value), must be greater than 0
     * @return 泊松分布对象 / Poisson distribution object
     * @throws IllegalArgumentException 如果λ小于等于0 / If lambda is less than or equal to 0
     * 
     * @see PoissonDistribution
     * @since 1.0
     */
    public static PoissonDistribution poisson(double lambda) {
        return new PoissonDistribution(lambda);
    }

    // ==================== 连续型分布工厂方法（补充）/ Continuous Distribution Factory Methods (Additional) ====================

    /**
     * 创建Beta分布对象
     * Create a Beta distribution object
     * 
     * <p>Beta分布是定义在区间[0,1]上的连续概率分布，由两个形状参数α和β控制。
     * 其概率密度函数为：f(x) = (1/B(α,β)) * x^(α-1) * (1-x)^(β-1)</p>
     * 
     * <p>Beta distribution is a continuous probability distribution defined on the interval [0,1],
     * controlled by two shape parameters α and β. Its probability density function is:
     * f(x) = (1/B(α,β)) * x^(α-1) * (1-x)^(β-1)</p>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>贝叶斯统计 / Bayesian statistics</li>
     *   <li>比例建模 / Proportion modeling</li>
     *   <li>先验分布 / Prior distributions</li>
     *   <li>机器学习 / Machine learning</li>
     * </ul>
     * 
     * @param alpha 形状参数α，必须大于0 / Shape parameter α, must be greater than 0
     * @param beta 形状参数β，必须大于0 / Shape parameter β, must be greater than 0
     * @return Beta分布对象 / Beta distribution object
     * @throws IllegalArgumentException 如果参数小于等于0 / If parameters are less than or equal to 0
     * 
     * @see BetaDistribution
     * @since 1.0
     */
    public static BetaDistribution beta(double alpha, double beta) {
        return new BetaDistribution(alpha, beta);
    }

    /**
     * 创建Gamma分布对象
     * Create a Gamma distribution object
     * 
     * <p>Gamma分布是连续概率分布，由形状参数α和尺度参数β控制。
     * 其概率密度函数为：f(x) = (β^α / Γ(α)) * x^(α-1) * e^(-βx)</p>
     * 
     * <p>Gamma distribution is a continuous probability distribution controlled by 
     * shape parameter α and scale parameter β. Its probability density function is:
     * f(x) = (β^α / Γ(α)) * x^(α-1) * e^(-βx)</p>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>等待时间建模 / Waiting time modeling</li>
     *   <li>可靠性分析 / Reliability analysis</li>
     *   <li>贝叶斯统计 / Bayesian statistics</li>
     *   <li>机器学习 / Machine learning</li>
     * </ul>
     * 
     * @param alpha 形状参数α，必须大于0 / Shape parameter α, must be greater than 0
     * @param beta 尺度参数β，必须大于0 / Scale parameter β, must be greater than 0
     * @return Gamma分布对象 / Gamma distribution object
     * @throws IllegalArgumentException 如果参数小于等于0 / If parameters are less than or equal to 0
     * 
     * @see GammaDistribution
     * @since 1.0
     */
    public static GammaDistribution gamma(double alpha, double beta) {
        return new GammaDistribution(alpha, beta);
    }
    
    /**
     * 计算两个向量的皮尔逊相关系数 / Calculate Pearson correlation coefficient between two vectors
     * 
     * <p>皮尔逊相关系数衡量两个向量之间的线性相关性，取值范围为[-1, 1]。
     * 其中1表示完全正相关，-1表示完全负相关，0表示无线性相关。</p>
     * 
     * <p>Pearson correlation coefficient measures the linear correlation between two vectors,
     * with values in the range [-1, 1]. 1 indicates perfect positive correlation,
     * -1 indicates perfect negative correlation, and 0 indicates no linear correlation.</p>
     * 
     * <h4>数学公式 / Mathematical Formula:</h4>
     * <p>r = cov(X,Y) / (σX × σY)</p>
     * <ul>
     *   <li>cov(X,Y) 是两个向量的协方差 / cov(X,Y) is the covariance of the two vectors</li>
     *   <li>σX 和 σY 分别是两个向量的标准差 / σX and σY are the standard deviations of the two vectors</li>
     * </ul>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>特征相关性分析 / Feature correlation analysis</li>
     *   <li>数据预处理 / Data preprocessing</li>
     *   <li>降维分析 / Dimensionality reduction</li>
     *   <li>统计建模 / Statistical modeling</li>
     * </ul>
     * 
     * @param x1 第一个向量，不能为null / First vector, cannot be null
     * @param x2 第二个向量，不能为null，长度必须与x1相同 / Second vector, cannot be null, must have same length as x1
     * @return 皮尔逊相关系数，取值范围[-1, 1] / Pearson correlation coefficient, range [-1, 1]
     * @throws IllegalArgumentException 如果输入向量为null或长度不匹配 / if input vectors are null or lengths don't match
     * @throws ArithmeticException 如果任一向量的标准差为0（无法计算相关系数）/ if any vector's standard deviation is 0 (cannot calculate correlation)
     * 
     * @see IVector#corr(IVector) 向量实例方法 / Vector instance method
     * @since 1.0
     */
    public static <T extends Number> double corr(IVector<T> x1, IVector<T> x2) {
        // 参数验证 / Parameter validation
        if (x1 == null) {
            throw new IllegalArgumentException("第一个向量不能为null / First vector cannot be null");
        }
        if (x2 == null) {
            throw new IllegalArgumentException("第二个向量不能为null / Second vector cannot be null");
        }
        if (x1.length() != x2.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + x1.length() + " != " + x2.length()
                    + " / Vector lengths don't match: " + x1.length() + " != " + x2.length());
        }
        
        // 调用向量实例的corr方法 / Call vector instance's corr method
        return x1.corr(x2).doubleValue();
    }
    
    /**
     * 计算两个向量的协方差 / Calculate covariance between two vectors
     * 
     * <p>协方差衡量两个向量之间的线性相关性。协方差的计算公式为：
     * cov(X,Y) = E[(X-μX)(Y-μY)] = E[XY] - μXμY</p>
     * 
     * <p>Covariance measures the linear relationship between two vectors. The formula is:
     * cov(X,Y) = E[(X-μX)(Y-μY)] = E[XY] - μXμY</p>
     * 
     * <h4>协方差的性质 / Properties of Covariance:</h4>
     * <ul>
     *   <li>cov(X,Y) > 0: 正相关，X增大时Y倾向于增大 / Positive correlation, Y tends to increase when X increases</li>
     *   <li>cov(X,Y) < 0: 负相关，X增大时Y倾向于减小 / Negative correlation, Y tends to decrease when X increases</li>
     *   <li>cov(X,Y) = 0: 无线性相关 / No linear correlation</li>
     *   <li>cov(X,X) = var(X): 自协方差等于方差 / Auto-covariance equals variance</li>
     * </ul>
     * 
     * <h4>应用场景 / Applications:</h4>
     * <ul>
     *   <li>特征相关性分析 / Feature correlation analysis</li>
     *   <li>投资组合分析 / Portfolio analysis</li>
     *   <li>风险分析 / Risk analysis</li>
     *   <li>统计建模 / Statistical modeling</li>
     * </ul>
     * 
     * @param x1 第一个向量，不能为null / First vector, cannot be null
     * @param x2 第二个向量，不能为null，长度必须与x1相同 / Second vector, cannot be null, must have same length as x1
     * @return 协方差值 / Covariance value
     * @throws IllegalArgumentException 如果输入向量为null或长度不匹配 / if input vectors are null or lengths don't match
     * @throws ArithmeticException 如果向量长度为0（无法计算协方差）/ if vector length is 0 (cannot calculate covariance)
     * 
     * @see IVector#cov(IVector) 向量实例方法 / Vector instance method
     * @since 1.0
     */
    public static <T extends Number> double cov(IVector<T> x1, IVector<T> x2) {
        // 参数验证 / Parameter validation
        if (x1 == null) {
            throw new IllegalArgumentException("第一个向量不能为null / First vector cannot be null");
        }
        if (x2 == null) {
            throw new IllegalArgumentException("第二个向量不能为null / Second vector cannot be null");
        }
        if (x1.length() != x2.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + x1.length() + " != " + x2.length()
                    + " / Vector lengths don't match: " + x1.length() + " != " + x2.length());
        }
        
        // 调用向量实例的cov方法 / Call vector instance's cov method
        return x1.cov(x2).doubleValue();
    }
    
}
