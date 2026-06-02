package com.yishape.lab.math.optimize.mclp;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;

import java.io.Serializable;

/**
 * 多目标/多准则线性规划求解器通用接口
 * Multi-Criteria Linear Programming Solver Interface
 *
 * <p>该接口定义了多目标线性规划（MCLP）求解器的标准行为。
 * 支持多种求解方法：加权求和法、字典序法、目标规划法、Pareto最优解等。
 * This interface defines the standard behavior for multi-criteria linear programming
 * solvers, supporting various methods: weighted sum, lexicographic, goal programming,
 * Pareto optimality, etc.</p>
 *
 * @author lteb2
 * @see <a href="https://en.wikipedia.org/wiki/Multi-objective_optimization">Multi-objective Optimization</a>
 */
public interface IMclpSolver extends Serializable {

    /**
     * 求解多目标线性规划问题（完整参数版）
     *
     * @param c 多个目标函数的系数矩阵，每行对应一个目标函数
     * @param A_ub 不等式约束矩阵
     * @param b_ub 不等式约束右端向量
     * @param A_eq 等式约束矩阵
     * @param b_eq 等式约束右端向量
     * @param initX 初始点（可选，可为null）
     * @return 多目标优化结果
     */
    MclpResult solve(IVector[] c, IMatrix A_ub, IVector b_ub,
                     IMatrix A_eq, IVector b_eq, IVector initX);

    /**
     * 求解多目标线性规划问题（不等式约束版）
     *
     * @param c 多个目标函数的系数矩阵
     * @param A_ub 不等式约束矩阵
     * @param b_ub 不等式约束右端向量
     * @return 多目标优化结果
     */
    default MclpResult solve(IVector[] c, IMatrix A_ub, IVector b_ub) {
        return solve(c, A_ub, b_ub, null, null, null);
    }

    /**
     * 求解多目标线性规划问题（仅等式约束版）
     *
     * @param c 多个目标函数的系数矩阵
     * @param A_eq 等式约束矩阵
     * @param b_eq 等式约束右端向量
     * @return 多目标优化结果
     */
    default MclpResult solveEq(IVector[] c, IMatrix A_eq, IVector b_eq) {
        return solve(c, null, null, A_eq, b_eq, null);
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
    MclpSolverType getSolverType();

    /**
     * 工厂方法：创建默认求解器（加权求和法）
     *
     * @return 默认求解器实例
     */
    static IMclpSolver of() {
        return of(MclpSolverType.WeightedSum);
    }

    /**
     * 工厂方法：根据类型创建求解器
     *
     * @param solverType 求解器类型
     * @return 求解器实例
     */
    static IMclpSolver of(MclpSolverType solverType) {
        if (solverType == null) {
            return of(MclpSolverType.WeightedSum);
        }
        return switch (solverType) {
            case WeightedSum -> new RereWeightedSum();
            case Lexicographic -> new RereLexicographic();
            case GoalProgramming -> new RereGoalProgramming();
            case Pareto -> new RereParetoOptimal();
            case Ahp -> new RereAHP();
            case Topsis -> new RereTopsis();
            case Interactive -> new RereInteractive();
        };
    }
}
