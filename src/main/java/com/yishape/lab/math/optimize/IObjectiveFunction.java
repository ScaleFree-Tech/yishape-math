package com.yishape.lab.math.optimize;

import com.yishape.lab.math.linalg.IVector;
import java.io.Serializable;

/**
 * 目标函数接口 / Objective Function Interface
 * <p>
 * 定义目标函数的标准接口，用于优化问题的目标函数计算。
 * Defines the standard interface for objective functions in optimization problems.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IObjectiveFunction extends Serializable {

    /**
     * 计算目标函数值 / Compute Objective Function Value
     *
     * @param x 变量值（向量）/ Variable values (vector)
     * @return 目标函数值 / Objective function value
     */
    public double computeObjective(IVector x);

}
