package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 线性规划求解器通用接口
 *
 * @author lteb2
 */
public interface ILinProgSolver {

    /**
     * 求解线性规划问题（处理不等式和等式约束）
     *
     * @param c 目标函数系数（最小化问题）
     * @param A_ub 小于等于约束矩阵系数
     * @param b_ub 小于等于约束值（不等式右方）
     * @param A_eq 等式约束矩阵系数
     * @param b_eq 等式约束值（等式右方）
     * @param initX
     * @return 优化结果
     */
    public OptResult solve(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq, IVector initX);

    /**
     * 求解线性规划问题（处理不等式和等式约束）
     *
     * @param c 目标函数系数（最小化问题）
     * @param A_ub 小于等于约束矩阵系数
     * @param b_ub 小于等于约束值（不等式右方）
     * @param A_eq 等式约束矩阵系数
     * @param b_eq 等式约束值（等式右方）
     * @return 优化结果
     */
    public default OptResult solve(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq) {
        IVector initX = Linalg.ones(c.length());
        return this.solve(c, A_ub, b_ub, A_eq, b_eq, initX);
    }

    /**
     * 求解线性规划问题 本方法的存在是必要的，因为有些求解器（如内点法）能够在不添加松驰变量的情况下求解，大大节约变量，提高速度。因此提供专门的<=入口
     *
     * @param c 目标函数系数（最小化问题）
     * @param A_ub 小于等于约束矩阵系数
     * @param b_ub 小于等于约束值（不等式右方）
     * @return 优化结果
     */
    public default OptResult solve(IVector c, IMatrix A_ub, IVector b_ub) {
        IVector initX = Linalg.ones(c.length());
        // 直接调用完整的solve方法，传入null表示没有等式约束
        return this.solve(c, A_ub, b_ub, null, null, initX);
    }

    /**
     * 基于等式约束求解（solveWithNonNegativeEqualConstraints的快捷别名方法）
     *
     * @param c 目标函数系数（最小化问题）
     * @param A_eq 等式约束矩阵系数
     * @param b_eq 等式约束值（等式右方）
     * @return 优化结果
     */
    public default OptResult solveEq(IVector c, IMatrix A_eq, IVector b_eq) {
        return this.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
    }

    /**
     * 基于等式约束求解（核心求解方法，需要实现类提供具体实现）
     *
     * @param c 目标函数系数（最小化问题）
     * @param A_eq 等式约束矩阵系数
     * @param b_eq 等式约束值（等式右方）
     * @return 优化结果
     */
    public default OptResult solveWithNonNegativeEqualConstraints(IVector c, IMatrix A_eq, IVector b_eq) {
        IVector initX = Linalg.ones(c.length());
        return this.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, initX);
    }

    /**
     * 基于等式约束求解（带初始值的核心求解方法）
     *
     * @param c 目标函数系数（最小化问题）
     * @param A_eq 等式约束矩阵系数
     * @param b_eq 等式约束值（等式右方）
     * @param initX 初始值（主要用于某些问题的热启动）
     * @return 优化结果
     */
    public default OptResult solveWithNonNegativeEqualConstraints(IVector c, IMatrix A_eq, IVector b_eq, IVector initX) {
        return this.solve(c, null, null, A_eq, b_eq, initX);
    }

}
