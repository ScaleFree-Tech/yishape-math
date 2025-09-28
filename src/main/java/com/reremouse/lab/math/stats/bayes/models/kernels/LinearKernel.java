package com.reremouse.lab.math.stats.bayes.models.kernels;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.bayes.models.GaussianProcess;

/**
 * 线性核函数
 * Linear Kernel Function
 * 
 * <p>线性核函数适用于线性关系的建模，是最简单的核函数之一。</p>
 * <p>Linear kernel is suitable for modeling linear relationships 
 * and is one of the simplest kernel functions.</p>
 * 
 * <p>核函数形式：k(x, x') = σ² + (x - c)ᵀ(x' - c)</p>
 * <p>超参数：[σ² (signal variance), c (offset)]</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class LinearKernel implements GaussianProcess.KernelFunction {
    
    @Override
    public double evaluate(IVector x1, IVector x2, IVector hyperparameters) {
        if (hyperparameters.size() != x1.size() + 1) {
            throw new IllegalArgumentException("Linear kernel requires " + (x1.size() + 1) + 
                " hyperparameters: [signal_variance, offset_1, ..., offset_d]");
        }
        
        double signalVariance = hyperparameters.get(0).doubleValue();
        
        if (signalVariance < 0) {
            throw new IllegalArgumentException("Signal variance must be non-negative");
        }
        
        // 计算内积 (x - c)ᵀ(x' - c)
        double dotProduct = 0.0;
        for (int i = 0; i < x1.size(); i++) {
            double offset = hyperparameters.get(i + 1).doubleValue();
            double x1_centered = x1.get(i).doubleValue() - offset;
            double x2_centered = x2.get(i).doubleValue() - offset;
            dotProduct += x1_centered * x2_centered;
        }
        
        return signalVariance + dotProduct;
    }
    
    @Override
    public IVector gradient(IVector x1, IVector x2, IVector hyperparameters) {
        if (hyperparameters.size() != x1.size() + 1) {
            throw new IllegalArgumentException("Linear kernel requires " + (x1.size() + 1) + 
                " hyperparameters: [signal_variance, offset_1, ..., offset_d]");
        }
        
        IVector gradient = Linalg.vector(hyperparameters.size());
        
        // 关于信号方差的梯度
        gradient.set(0, 1.0);
        
        // 关于偏移量的梯度
        for (int i = 0; i < x1.size(); i++) {
            double offset = hyperparameters.get(i + 1).doubleValue();
            double x1_val = x1.get(i).doubleValue();
            double x2_val = x2.get(i).doubleValue();
            
            // ∂k/∂c_i = -(x1_i - c_i) - (x2_i - c_i) = -x1_i - x2_i + 2*c_i
            double grad = -x1_val - x2_val + 2 * offset;
            gradient.set(i + 1, grad);
        }
        
        return gradient;
    }
    
    @Override
    public int getNumHyperparameters() {
        return -1; // 依赖于输入维度
    }
    
    @Override
    public String getName() {
        return "Linear";
    }
}