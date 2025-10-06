package com.yishape.lab.math.stats.bayes.models.kernels;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.stats.bayes.models.GaussianProcess;

/**
 * 周期核函数
 * Periodic Kernel Function
 * 
 * <p>周期核函数适用于建模具有周期性模式的数据，如季节性时间序列。</p>
 * <p>Periodic kernel is suitable for modeling data with periodic patterns, 
 * such as seasonal time series.</p>
 * 
 * <p>核函数形式：k(x, x') = σ² * exp(-2 * sin²(π * ||x - x'|| / p) / l²)</p>
 * <p>超参数：[σ² (signal variance), l (length scale), p (period)]</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class PeriodicKernel implements GaussianProcess.KernelFunction {
    
    @Override
    public double evaluate(IVector x1, IVector x2, IVector hyperparameters) {
        if (hyperparameters.size() != 3) {
            throw new IllegalArgumentException("Periodic kernel requires 3 hyperparameters: [signal_variance, length_scale, period]");
        }
        
        double signalVariance = hyperparameters.get(0).doubleValue();
        double lengthScale = hyperparameters.get(1).doubleValue();
        double period = hyperparameters.get(2).doubleValue();
        
        if (signalVariance <= 0 || lengthScale <= 0 || period <= 0) {
            throw new IllegalArgumentException("Hyperparameters must be positive");
        }
        
        // 计算欧几里得距离
        double distance = 0.0;
        for (int i = 0; i < x1.size(); i++) {
            double diff = x1.get(i).doubleValue() - x2.get(i).doubleValue();
            distance += diff * diff;
        }
        distance = Math.sqrt(distance);
        
        // 计算周期核值
        double sinTerm = Math.sin(Math.PI * distance / period);
        double exponent = -2.0 * sinTerm * sinTerm / (lengthScale * lengthScale);
        
        return signalVariance * Math.exp(exponent);
    }
    
    @Override
    public IVector gradient(IVector x1, IVector x2, IVector hyperparameters) {
        if (hyperparameters.size() != 3) {
            throw new IllegalArgumentException("Periodic kernel requires 3 hyperparameters: [signal_variance, length_scale, period]");
        }
        
        double signalVariance = hyperparameters.get(0).doubleValue();
        double lengthScale = hyperparameters.get(1).doubleValue();
        double period = hyperparameters.get(2).doubleValue();
        
        // 计算欧几里得距离
        double distance = 0.0;
        for (int i = 0; i < x1.size(); i++) {
            double diff = x1.get(i).doubleValue() - x2.get(i).doubleValue();
            distance += diff * diff;
        }
        distance = Math.sqrt(distance);
        
        double sinTerm = Math.sin(Math.PI * distance / period);
        double cosTerm = Math.cos(Math.PI * distance / period);
        double exponent = -2.0 * sinTerm * sinTerm / (lengthScale * lengthScale);
        double kernelValue = signalVariance * Math.exp(exponent);
        
        IVector gradient = Linalg.vector(3);
        
        // 关于信号方差的梯度
        gradient.set(0, kernelValue / signalVariance);
        
        // 关于长度尺度的梯度
        double dKdL = kernelValue * 4.0 * sinTerm * sinTerm / (lengthScale * lengthScale * lengthScale);
        gradient.set(1, dKdL);
        
        // 关于周期的梯度
        if (distance > 1e-12) {
            double dKdP = kernelValue * 4.0 * sinTerm * cosTerm * Math.PI * distance / 
                         (period * period * lengthScale * lengthScale);
            gradient.set(2, dKdP);
        } else {
            gradient.set(2, 0.0);
        }
        
        return gradient;
    }
    
    @Override
    public int getNumHyperparameters() {
        return 3;
    }
    
    @Override
    public String getName() {
        return "Periodic";
    }
}