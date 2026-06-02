package com.yishape.lab.math.optimize;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.mcqp.IMcqpSolver;
import com.yishape.lab.math.optimize.mcqp.McqpSolverType;

import java.io.Serializable;

/**
 * 多目标二次规划门面类
 * Multi-Criteria Quadratic Programming Facade
 *
 * <p>提供访问MCQP各种求解方法的统一入口。
 * This class provides a unified entry point for various MCQP solving methods.</p>
 *
 * @author lteb2
 */
public class McqpWrapper implements Serializable {

    /**
     * 创建加权求和法求解器
     *
     * @param weights 权重向量
     * @return 加权求和法求解器
     */
    public static IMcqpSolver weightedSumQp(double[] weights) {
        return new com.yishape.lab.math.optimize.mcqp.RereWeightedSumQp(weights);
    }

    /**
     * 创建字典序法求解器
     *
     * @param priorityOrder 优先级顺序数组
     * @return 字典序法求解器
     */
    public static IMcqpSolver lexicographicQp(int[] priorityOrder) {
        return new com.yishape.lab.math.optimize.mcqp.RereLexicographicQp(priorityOrder);
    }

    /**
     * 创建目标规划法求解器
     *
     * @param goals 目标值数组
     * @param weights 权重数组
     * @return 目标规划法求解器
     */
    public static IMcqpSolver goalProgrammingQp(double[] goals, double[] weights) {
        return new com.yishape.lab.math.optimize.mcqp.RereGoalProgrammingQp(goals, weights);
    }

    /**
     * 创建Pareto最优解法求解器
     *
     * @param numSamples 采样点数量
     * @return Pareto最优解法求解器
     */
    public static IMcqpSolver paretoQp(int numSamples) {
        return new com.yishape.lab.math.optimize.mcqp.RereParetoQp(numSamples);
    }

    /**
     * 创建AHP求解器
     *
     * @param comparisonMatrix 成对比较矩阵
     * @return AHP求解器
     */
    public static IMcqpSolver ahpQp(IMatrix comparisonMatrix) {
        return new com.yishape.lab.math.optimize.mcqp.RereAhpQp(comparisonMatrix);
    }

    /**
     * 创建TOPSIS求解器
     *
     * @param weights 权重向量
     * @return TOPSIS求解器
     */
    public static IMcqpSolver topsisQp(double[] weights) {
        return new com.yishape.lab.math.optimize.mcqp.RereTopsisQp(weights);
    }

    /**
     * 创建交互式STEM求解器
     *
     * @param maxIterations 最大迭代次数
     * @return 交互式STEM求解器
     */
    public static IMcqpSolver interactiveQp(int maxIterations) {
        return new com.yishape.lab.math.optimize.mcqp.RereInteractiveQp(maxIterations);
    }

    /**
     * 求解双目标二次规划问题（加权求和法）
     *
     * @param Q1 第一个目标的二次项系数矩阵
     * @param c1 第一个目标的线性项系数向量
     * @param Q2 第二个目标的二次项系数矩阵
     * @param c2 第二个目标的线性项系数向量
     * @param A_ub 不等式约束矩阵
     * @param b_ub 不等式约束右端向量
     * @param weight 权重（0到1之间）
     * @return 优化结果
     */
    public static com.yishape.lab.math.optimize.mcqp.McqpResult solveBiObjective(
            IMatrix Q1, IVector c1, IMatrix Q2, IVector c2,
            IMatrix A_ub, IVector b_ub, double weight) {
        IMatrix[] Q = new IMatrix[]{Q1, Q2};
        IVector[] c = new IVector[]{c1, c2};
        IMcqpSolver solver = new com.yishape.lab.math.optimize.mcqp.RereWeightedSumQp(new double[]{weight, 1 - weight});
        return solver.solve(Q, c, A_ub, b_ub);
    }
}
