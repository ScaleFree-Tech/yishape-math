package com.yishape.lab.math.optimize;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.autodiff.AD;
import java.io.Serializable;
import java.util.function.Function;
import com.yishape.lab.math.autodiff.IDiffVector;

/**
 * 优化器接口 / Optimizer Interface
 * <p>
 * 定义最优化求解器的标准接口，所有优化器实现都应遵循此接口。 Defines the standard interface for optimization
 * solvers.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IOptimizer extends Serializable {

    /**
     * 执行优化计算 / Perform Optimization
     * <p>
     * 根据提供的初始点、目标函数计算方法、梯度计算方法，求解数学最优化问题。 Solves mathematical optimization
     * problems using the provided initial point, objective function, and
     * gradient function.
     * </p>
     *
     * @param initX 初始点 / Initial point
     * @param objFun 目标函数计算器 / Objective function
     * @param grdFun 梯度计算器 / Gradient function
     * @return 最优结果 / Optimization result
     */
    public OptResult optimize(IVector initX, IObjectiveFunction objFun, IGradientFunction grdFun);

    
    /**
     * 
     * @param initX 初始点
     * @param lossFun  
     * @return 
     */
    public default OptResult optimize(IVector initX, Function<IDiffVector, IDiffVector> lossFun) {
        var res = AD.optimize(
                initX,
                lossFun,
                Opts.lbfgs());

        return res;
    }

}
