package com.yishape.lab.math.stats.bayes.models.means;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.stats.bayes.models.GaussianProcess;

/**
 * 线性均值函数
 * Linear Mean Function
 * 
 * <p>线性均值函数适用于数据具有线性趋势的情况。</p>
 * <p>Linear mean function is suitable for data with linear trends.</p>
 * 
 * <p>均值函数形式：m(x) = a₀ + a₁x₁ + a₂x₂ + ... + aₐxₐ</p>
 * <p>超参数：[a₀ (intercept), a₁, a₂, ..., aₐ (coefficients)]</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class LinearMeanFunction implements GaussianProcess.MeanFunction {
    
    @Override
    public double evaluate(IVector x, IVector parameters) {
        if (parameters.size() != x.size() + 1) {
            throw new IllegalArgumentException("Linear mean function requires " + (x.size() + 1) + 
                " parameters: [intercept, coeff_1, ..., coeff_d]");
        }
        
        double result = parameters.get(0).doubleValue(); // 截距
        
        // 计算线性组合
        for (int i = 0; i < x.size(); i++) {
            result += parameters.get(i + 1).doubleValue() * x.get(i).doubleValue();
        }
        
        return result;
    }
    
    @Override
    public IVector gradient(IVector x, IVector parameters) {
        if (parameters.size() != x.size() + 1) {
            throw new IllegalArgumentException("Linear mean function requires " + (x.size() + 1) + 
                " parameters: [intercept, coeff_1, ..., coeff_d]");
        }
        
        IVector gradient = Linalg.vector(parameters.size());
        
        // 关于截距的梯度
        gradient.set(0, 1.0);
        
        // 关于系数的梯度
        for (int i = 0; i < x.size(); i++) {
            gradient.set(i + 1, x.get(i).doubleValue());
        }
        
        return gradient;
    }
    
    @Override
    public int getNumParameters() {
        return -1; // 依赖于输入维度
    }
    /**
     * 获取均值函数名称
     * Get mean function name
     *
     * @return 均值函数名称 / Mean function name
     */
    public String getName() {
        return "Linear";
    }
}