package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.util.Tuple2;

/**
 * 线性规划求解器通用接口
 *
 * @author lteb2
 */
public interface ILinProgSolver {

    /**
     * 求解线性规划问题
     *
     * @param c 目标函数系数（最小化问题）
     * @param A_ub 小于等于约束矩阵系数
     * @param b_ub 小于等于约束值（不等式右方）
     * @param A_eq 等式约束矩阵系数
     * @param b_eq 等式约束值（不等式右方）
     * @return
     */
    public default Tuple2<Double, IVector> solve(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq) {
        Tuple2<IMatrix, IVector> ss = LinProgUtil.convertUbEqToEqConstraits(A_ub, b_ub, A_eq, b_eq);
        return this.solveWithNonNegativeEqualConstraints(c, ss._1, ss._2);
    }

    

    /**
     * 基于等式约束求解
     *
     * @param c 目标函数系数（最小化问题）
     * @param A_eq 等式约束矩阵系数
     * @param b_eq 等式约束值（不等式右方）
     * @return
     */
    public Tuple2<Double, IVector> solveWithNonNegativeEqualConstraints(IVector c, IMatrix A_eq, IVector b_eq);

}