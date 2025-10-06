package com.yishape.lab.math.optimize;

import com.yishape.lab.math.linalg.IVector;

/**
 *
 * @author lteb2
 */
public interface IGradientFunction {
    
    /**
     * 计算梯度
     * @param x 变量值（向量）
     * @return 
     */
    public IVector computeGradient(IVector x);
    
}
