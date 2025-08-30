package com.reremouse.lab.math.optimize;

import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.math.IVector;

/**
 * 各种最优化求解器的通用接口
 * @author lteb2
 */
public interface IOptimizer {
    
    /**
     * 根据提供的初始点、目标函数计算方法、梯度计算方法，求解数学最优化问题
     * @param initX 初始点
     * @param objFun 目标函数计算法
     * @param grdFun 梯度计算法
     * @return 返回最优值及最优点的变量值（向量）
     */
    public Tuple2<Float,IVector> optimize(IVector initX, IObjectiveFunction objFun,IGradientFunction grdFun);
    
}
