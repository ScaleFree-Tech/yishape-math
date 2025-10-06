package com.yishape.lab.math.stats.bayes.models.means;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.stats.bayes.models.GaussianProcess;

/**
 * 常数均值函数
 * Constant Mean Function
 * 
 * <p>返回常数值的均值函数，适用于数据具有非零均值的情况。</p>
 * <p>Mean function that returns a constant value, 
 * suitable for data with non-zero mean.</p>
 * 
 * <p>均值函数形式：m(x) = c</p>
 * <p>超参数：[c (constant)]</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ConstantMeanFunction implements GaussianProcess.MeanFunction {
    
    @Override
    public double evaluate(IVector x, IVector parameters) {
        if (parameters.size() != 1) {
            throw new IllegalArgumentException("Constant mean function requires 1 parameter: [constant]");
        }
        
        return parameters.get(0).doubleValue();
    }
    
    @Override
    public IVector gradient(IVector x, IVector parameters) {
        if (parameters.size() != 1) {
            throw new IllegalArgumentException("Constant mean function requires 1 parameter: [constant]");
        }
        
        IVector gradient = Linalg.vector(1);
        gradient.set(0, 1.0);  // ∂m/∂c = 1
        
        return gradient;
    }
    
    @Override
    public int getNumParameters() {
        return 1;
    }
    
    public String getName() {
        return "Constant";
    }
}