package com.reremouse.lab.math.stats.bayes.models.kernels;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.bayes.models.GaussianProcess;

/**
 * RBF（径向基函数）核
 * RBF (Radial Basis Function) Kernel
 * 
 * <p>也称为高斯核或平方指数核，是最常用的核函数之一。</p>
 * <p>Also known as Gaussian kernel or squared exponential kernel, 
 * one of the most commonly used kernel functions.</p>
 * 
 * <p>核函数形式：k(x, x') = σ² * exp(-||x - x'||² / (2 * l²))</p>
 * <p>超参数：[σ² (signal variance), l (length scale)]</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RBFKernel implements GaussianProcess.KernelFunction {
    
    @Override
    public double evaluate(IVector x1, IVector x2, IVector hyperparameters) {
        if (hyperparameters.size() != 2) {
            throw new IllegalArgumentException("RBF kernel requires 2 hyperparameters: [signal_variance, length_scale]");
        }
        
        double signalVariance = hyperparameters.get(0).doubleValue();
        double lengthScale = hyperparameters.get(1).doubleValue();
        
        if (signalVariance <= 0 || lengthScale <= 0) {
            throw new IllegalArgumentException("Hyperparameters must be positive");
        }
        
        // 计算欧几里得距离的平方
        double squaredDistance = 0.0;
        for (int i = 0; i < x1.size(); i++) {
            double diff = x1.get(i).doubleValue() - x2.get(i).doubleValue();
            squaredDistance += diff * diff;
        }
        
        // 计算RBF核值
        return signalVariance * Math.exp(-squaredDistance / (2.0 * lengthScale * lengthScale));
    }
    
    @Override
    public IVector gradient(IVector x1, IVector x2, IVector hyperparameters) {
        if (hyperparameters.size() != 2) {
            throw new IllegalArgumentException("RBF kernel requires 2 hyperparameters: [signal_variance, length_scale]");
        }
        
        double signalVariance = hyperparameters.get(0).doubleValue();
        double lengthScale = hyperparameters.get(1).doubleValue();
        
        // 计算欧几里得距离的平方
        double squaredDistance = 0.0;
        for (int i = 0; i < x1.size(); i++) {
            double diff = x1.get(i).doubleValue() - x2.get(i).doubleValue();
            squaredDistance += diff * diff;
        }
        
        // 计算核值
        double kernelValue = signalVariance * Math.exp(-squaredDistance / (2.0 * lengthScale * lengthScale));
        
        IVector gradient = Linalg.vector(2);
        
        // 关于信号方差的梯度
        gradient.set(0, kernelValue / signalVariance);
        
        // 关于长度尺度的梯度
        gradient.set(1, kernelValue * squaredDistance / (lengthScale * lengthScale * lengthScale));
        
        return gradient;
    }
    
    @Override
    public int getNumHyperparameters() {
        return 2;
    }
    
    @Override
    public String getName() {
        return "RBF";
    }
}