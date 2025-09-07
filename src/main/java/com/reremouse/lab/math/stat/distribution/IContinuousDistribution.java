package com.reremouse.lab.math.stat.distribution;

/**
 *
 * @author lteb2
 */
public interface IContinuousDistribution {

    /**
     * 均值
     * @return 
     */
    public float mean();

    /**
     * 方差
     * @return 
     */
    public float var();

    /**
     * 标准差
     * @return 
     */
    public float std();
    
    /**
     * 中位数
     * @return 
     */
    public float median();
    
    /**
     * 众数
     * @return 
     */
    public float mode();
    
    /**
     * 四分位数
     * @return 
     */
    public float q1();
    
    /**
     * 四分之三分位数
     * @return 
     */    
    public float q3();
    
    /**
     * 偏度
     * @return 
     */
    public float skewness();
    
    /**
     * 峰度
     * @return 
     */
    public float kurtosis();
    
    /**
     * 采样
     * @return 
     */
    public float sample();
    
    /**
     * 采样N个
     * @param n
     * @return 
     */
    public float[] sample(int n);

    /**
     * 概率密度函数
     *
     * @param x
     * @return
     */
    public float pdf(float x);

    /**
     * 累积分布
     *
     * @param x
     * @return
     */
    public float cdf(float x);

    /**
     * 百分点函数
     *
     * @param prob
     * @return
     */
    public float ppf(float prob);

    /**
     * 生存函数
     *
     * @param x
     * @return
     */
    public float sf(float x);

    /**
     * 逆生存函数
     *
     * @param prob
     * @return
     */
    public float isf(float prob);
}
