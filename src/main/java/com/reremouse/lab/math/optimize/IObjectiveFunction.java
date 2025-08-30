package com.reremouse.lab.math.optimize;

import com.reremouse.lab.math.IVector;

/**
 *
 * @author lteb2
 */
public interface IObjectiveFunction {
    
    /**
     * 计算梯度
     * @param x 变量值（向量）
     * @return 目标函数值
     */
    public float computeObjective(IVector x);
    
}
