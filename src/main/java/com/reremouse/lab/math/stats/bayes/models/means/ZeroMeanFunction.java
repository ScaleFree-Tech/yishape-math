package com.reremouse.lab.math.stats.bayes.models.means;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.bayes.models.GaussianProcess;

/**
 * 零均值函数
 * Zero Mean Function
 * 
 * <p>最简单的均值函数，总是返回0。这是高斯过程中最常用的均值函数。</p>
 * <p>The simplest mean function that always returns 0. 
 * This is the most commonly used mean function in Gaussian processes.</p>
 * 
 * <p>均值函数形式：m(x) = 0</p>
 * <p>无超参数</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ZeroMeanFunction implements GaussianProcess.MeanFunction {
    
    @Override
    public double evaluate(IVector x, IVector parameters) {
        // 零均值函数不需要参数
        return 0.0;
    }
    
    @Override
    public IVector gradient(IVector x, IVector parameters) {
        // 零均值函数没有参数，返回空向量
        return Linalg.vector(0);
    }
    
    @Override
    public int getNumParameters() {
        return 0;
    }
    
    public String getName() {
        return "Zero";
    }
}