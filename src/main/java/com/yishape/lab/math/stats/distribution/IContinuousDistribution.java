package com.yishape.lab.math.stats.distribution;

/**
 * 连续型分布接口 / Continuous Distribution Interface
 * <p>
 * 定义连续型概率分布的基本操作，包括统计量计算、采样、概率密度函数和累积分布函数等。
 * 所有连续型概率分布实现类都应实现此接口。
 * </p>
 * <p>
 * Defines basic operations for continuous probability distributions, including
 * statistical calculations, sampling, probability density functions, and
 * cumulative distribution functions. All continuous probability distribution
 * implementation classes should implement this interface.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IContinuousDistribution {

    /**
     * 计算均值 / Calculate mean
     *
     * @return 均值 / Mean value
     */
    double mean();

    /**
     * 计算方差 / Calculate variance
     *
     * @return 方差 / Variance value
     */
    double var();

    /**
     * 计算标准差 / Calculate standard deviation
     *
     * @return 标准差 / Standard deviation value
     */
    double std();

    /**
     * 计算中位数 / Calculate median
     *
     * @return 中位数 / Median value
     */
    double median();

    /**
     * 计算众数 / Calculate mode
     *
     * @return 众数 / Mode value
     */
    double mode();

    /**
     * 计算第一四分位数 / Calculate first quartile
     *
     * @return 第一四分位数 / First quartile value
     */
    double q1();

    /**
     * 计算第三四分位数 / Calculate third quartile
     *
     * @return 第三四分位数 / Third quartile value
     */
    double q3();

    /**
     * 计算偏度 / Calculate skewness
     *
     * @return 偏度值 / Skewness value
     */
    double skewness();

    /**
     * 计算峰度 / Calculate kurtosis
     *
     * @return 峰度值 / Kurtosis value
     */
    double kurtosis();

    /**
     * 采样一个值 / Sample one value
     *
     * @return 采样值 / Sampled value
     */
    double sample();

    /**
     * 采样N个值 / Sample N values
     *
     * @param n 采样数量 / Number of samples
     * @return 采样值数组 / Array of sampled values
     */
    double[] sample(int n);

    /**
     * 计算概率密度函数值 / Evaluate probability density function
     *
     * @param x 随机变量值 / Random variable value
     * @return 概率密度值 / Probability density value
     */
    double pdf(double x);

    /**
     * 计算累积分布函数值 / Evaluate cumulative distribution function
     *
     * @param x 随机变量值 / Random variable value
     * @return 累积分布概率值 / Cumulative distribution probability value
     */
    double cdf(double x);

    /**
     * 计算百分点函数值（逆累积分布函数）/ Evaluate percentile point function (inverse CDF)
     *
     * @param prob 概率值 / Probability value
     * @return 对应的随机变量值 / Corresponding random variable value
     */
    double ppf(double prob);

    /**
     * 计算生存函数值 / Evaluate survival function
     *
     * @param x 随机变量值 / Random variable value
     * @return 生存函数值 / Survival function value
     */
    double sf(double x);

    /**
     * 计算逆生存函数值 / Evaluate inverse survival function
     *
     * @param prob 概率值 / Probability value
     * @return 对应的随机变量值 / Corresponding random variable value
     */
    double isf(double prob);
}
