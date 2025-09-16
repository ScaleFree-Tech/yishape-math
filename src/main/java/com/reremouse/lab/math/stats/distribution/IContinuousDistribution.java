package com.reremouse.lab.math.stats.distribution;

/**
 *
 * @author lteb2
 */
public interface IContinuousDistribution {

    /**
     * 均值
     * @return 
     */
    public double mean();

    /**
     * 方差
     * @return 
     */
    public double var();

    /**
     * 标准差
     * @return 
     */
    public double std();
    
    /**
     * 中位数
     * @return 
     */
    public double median();
    
    /**
     * 众数
     * @return 
     */
    public double mode();
    
    /**
     * 四分位数
     * @return 
     */
    public double q1();
    
    /**
     * 四分之三分位数
     * @return 
     */    
    public double q3();
    
    /**
     * 偏度
     * @return 
     */
    public double skewness();
    
    /**
     * 峰度
     * @return 
     */
    public double kurtosis();
    
    /**
     * 采样
     * @return 
     */
    public double sample();
    
    /**
     * 采样N个
     * @param n
     * @return 
     */
    public double[] sample(int n);

    /**
     * 概率密度函数
     *
     * @param x
     * @return
     */
    public double pdf(double x);

    /**
     * 累积分布
     *
     * @param x
     * @return
     */
    public double cdf(double x);

    /**
     * 百分点函数
     *
     * @param prob
     * @return
     */
    public double ppf(double prob);

    /**
     * 生存函数
     *
     * @param x
     * @return
     */
    public double sf(double x);

    /**
     * 逆生存函数
     *
     * @param prob
     * @return
     */
    public double isf(double prob);
}
