package com.yishape.lab.math.optimize;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

import java.io.Serializable;

/**
 * 二次规划求解器接口
 * Quadratic Programming Solver Interface
 *
 * <p>定义了二次规划（QP）求解器的标准行为。
 * 标准形式：
 * minimize 1/2 * x^T * Q * x + c^T * x
 * subject to {@code A_ub * x <= b_ub}
 *            A_eq * x = b_eq
 *            x >= 0 (optional)</p>
 *
 * @author lteb2
 */
public interface IQpSolver extends Serializable {

    /**
     * 求解二次规划问题（完整参数版）
     *
     * @param Q 二次项系数矩阵（对称正定或半正定）
     * @param c 线性项系数向量
     * @param A_ub 不等式约束矩阵
     * @param b_ub 不等式约束右端向量
     * @param A_eq 等式约束矩阵
     * @param b_eq 等式约束右端向量
     * @param initX 初始点（可选，可为null）
     * @return 优化结果
     */
    OptResult solve(IMatrix Q, IVector c, IMatrix A_ub, IVector b_ub,
                    IMatrix A_eq, IVector b_eq, IVector initX);

    /**
     * 求解二次规划问题（不等式约束版）
     *
     * @param Q 二次项系数矩阵
     * @param c 线性项系数向量
     * @param A_ub 不等式约束矩阵
     * @param b_ub 不等式约束右端向量
     * @return 优化结果
     */
    default OptResult solve(IMatrix Q, IVector c, IMatrix A_ub, IVector b_ub) {
        return solve(Q, c, A_ub, b_ub, null, null, null);
    }

    /**
     * 工厂方法：创建默认求解器
     *
     * @return 默认求解器实例
     */
    static IQpSolver of() {
        return new InteriorPointQpSolver();
    }
}
