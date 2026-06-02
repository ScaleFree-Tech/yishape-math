package com.yishape.lab.math.optimize.mcqp;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;

import java.io.Serializable;

/**
 * 多目标/多准则二次规划求解器通用接口
 * Multi-Criteria Quadratic Programming Solver Interface
 *
 * <p>该接口定义了多目标二次规划（MCQP）求解器的标准行为。
 * 支持多种求解方法：加权求和法、字典序法、目标规划法、Pareto最优解等。
 * This interface defines the standard behavior for multi-criteria quadratic programming
 * solvers, supporting various methods: weighted sum, lexicographic, goal programming,
 * Pareto optimality, etc.</p>
 *
 * <p>二次规划问题标准形式：
 * minimize 1/2 * x^T * Q_i * x + c_i^T * x
 * subject to {@code A_ub * x <= b_ub}
 *            A_eq * x = b_eq
 *            x >= 0 (可选)</p>
 *
 * @author lteb2
 * @see <a href="https://en.wikipedia.org/wiki/Multi-objective_optimization">Multi-objective Optimization</a>
 */
public interface IMcqpSolver extends Serializable {

    /**
     * 求解多目标二次规划问题（完整参数版）
     *
     * @param Q 多个目标函数的二次项系数矩阵数组，每行对应一个目标函数
     * @param c 多个目标函数的线性项系数向量数组
     * @param A_ub 不等式约束矩阵
     * @param b_ub 不等式约束右端向量
     * @param A_eq 等式约束矩阵
     * @param b_eq 等式约束右端向量
     * @param initX 初始点（可选，可为null）
     * @return 多目标优化结果
     */
    McqpResult solve(IMatrix[] Q, IVector[] c, IMatrix A_ub, IVector b_ub,
                     IMatrix A_eq, IVector b_eq, IVector initX);

    /**
     * 求解多目标二次规划问题（不等式约束版）
     *
     * @param Q 多个目标函数的二次项系数矩阵数组
     * @param c 多个目标函数的线性项系数向量数组
     * @param A_ub 不等式约束矩阵
     * @param b_ub 不等式约束右端向量
     * @return 多目标优化结果
     */
    default McqpResult solve(IMatrix[] Q, IVector[] c, IMatrix A_ub, IVector b_ub) {
        return solve(Q, c, A_ub, b_ub, null, null, null);
    }

    /**
     * 求解多目标二次规划问题（仅等式约束版）
     *
     * @param Q 多个目标函数的二次项系数矩阵数组
     * @param c 多个目标函数的线性项系数向量数组
     * @param A_eq 等式约束矩阵
     * @param b_eq 等式约束右端向量
     * @return 多目标优化结果
     */
    default McqpResult solveEq(IMatrix[] Q, IVector[] c, IMatrix A_eq, IVector b_eq) {
        return solve(Q, c, null, null, A_eq, b_eq, null);
    }

    /**
     * 获取求解器名称
     * @return 求解器名称
     */
    String getName();

    /**
     * 获取求解器描述
     * @return 求解器描述
     */
    default String getDescription() {
        return "";
    }

    /**
     * 获取求解器类型
     * @return 求解器类型
     */
    McqpSolverType getSolverType();

    /**
     * 工厂方法：创建默认求解器（加权求和法）
     *
     * @return 默认求解器实例
     */
    static IMcqpSolver of() {
        return of(McqpSolverType.WeightedSum);
    }

    /**
     * 工厂方法：根据类型创建求解器
     *
     * @param solverType 求解器类型
     * @return 求解器实例
     */
    static IMcqpSolver of(McqpSolverType solverType) {
        if (solverType == null) {
            return of(McqpSolverType.WeightedSum);
        }
        return switch (solverType) {
            case WeightedSum -> new RereWeightedSumQp();
            case Lexicographic -> new RereLexicographicQp();
            case GoalProgramming -> new RereGoalProgrammingQp();
            case Pareto -> new RereParetoQp();
            case Ahp -> new RereAhpQp();
            case Topsis -> new RereTopsisQp();
            case Interactive -> new RereInteractiveQp();
        };
    }
}
