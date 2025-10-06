package com.yishape.lab.math.optimize.linpg.simplex;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;

/**
 * 专门为单纯形法设计的接口，防止在实现中最小化、最大化转换混淆
 *
 * @author lteb2
 */
public interface ISimplexLinProgSolver extends ILinProgSolver {

    /**
     *
     * @param c 目标函数系数（最小化问题）
     * @param A_ub 小于等于约束矩阵系数
     * @param b_ub 小于等于约束值（不等式右方）
     * @param A_eq 等式约束矩阵系数
     * @param b_eq 等式约束值（等式右方）
     * @param initX 初始点（热启动点）
     * @return 优化结果
     */
    @Override
    public default OptResult solve(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq, IVector initX) {
        var result = this.maximize(c.multiplyScalar(-1.0), A_ub, b_ub, A_eq, b_eq, initX);
        result.setOptimalValue(-result.getOptimalValue());
        return result;
    }

    /**
     * 按单纯形法常用的最大化来处理问题，防止在程序中来回转换目标函数出现最大错误
     *
     * @param c 目标函数系数（最大化问题）
     * @param A_ub 小于等于约束矩阵系数
     * @param b_ub 小于等于约束值（不等式右方）
     * @param A_eq 等式约束矩阵系数
     * @param b_eq 等式约束值（等式右方）
     * @param initX 初始点（热启动点）
     * @return 优化结果
     */
    public OptResult maximize(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq, IVector initX);



}
