package com.yishape.lab.math.optimize;

import com.yishape.lab.math.linalg.IVector;
import java.io.Serializable;

/**
 * 梯度函数接口 / Gradient Function Interface
 * <p>
 * 定义梯度函数的标准接口，用于优化问题的梯度计算。
 * Defines the standard interface for gradient functions in optimization problems.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IGradientFunction extends Serializable {

    /**
     * 计算梯度 / Compute Gradient
     *
     * @param x 变量值（向量）/ Variable values (vector)
     * @return 梯度向量 / Gradient vector
     */
    public IVector computeGradient(IVector x);

}
