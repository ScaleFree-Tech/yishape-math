package com.yishape.lab.math.optimize;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.optimize.mclp.IMclpSolver;
import com.yishape.lab.math.optimize.mclp.RereAHP;
import com.yishape.lab.math.optimize.mclp.RereGoalProgramming;
import com.yishape.lab.math.optimize.mclp.RereInteractive;
import com.yishape.lab.math.optimize.mclp.RereLexicographic;
import com.yishape.lab.math.optimize.mclp.RereParetoOptimal;
import com.yishape.lab.math.optimize.mclp.RereTopsis;
import com.yishape.lab.math.optimize.mclp.RereWeightedSum;
import com.yishape.lab.math.optimize.mclp.MclpSolverType;

/**
 * 多目标/多准则线性规划求解器包装类
 * @author lteb2
 */
public class MclpWrapper {

    // ==================== 多目标线性规划（MCLP）求解器 ====================
    /**
     * Create a Weighted Sum Multi-Criteria Linear Programming solver.
     * 创建加权求和多目标线性规划求解器。
     *
     * <p>
     * The weighted sum method combines multiple objectives into a single
     * objective by using weights to reflect the relative importance of each
     * objective. Works well for convex Pareto fronts but cannot find non-convex
     * solutions.</p>
     *
     * <p>
     * 加权求和方法通过权重反映各目标相对重要性，将多目标合并为单目标。 适用于凸Pareto前沿，但无法找到非凸解。</p>
     *
     * @return Weighted sum MCLP solver instance / 加权求和MCLP求解器实例
     */
    public static IMclpSolver weightedSumMclp() {
        return new RereWeightedSum();
    }

    /**
     * Create a Weighted Sum MCLP solver with specified weights.
     * 创建带指定权重的加权求和MCLP求解器。
     *
     * @param weights 权重向量 / Weight vector
     * @return Weighted sum MCLP solver instance / 加权求和MCLP求解器实例
     */
    public static IMclpSolver weightedSumMclp(double[] weights) {
        return new RereWeightedSum(weights);
    }

    /**
     * Create a Lexicographic Multi-Criteria Linear Programming solver.
     * 创建字典序多目标线性规划求解器。
     *
     * <p>
     * The lexicographic method optimizes objectives in order of priority. First
     * optimizes the most important objective, then optimizes the second most
     * important subject to the first being optimal, and so on.</p>
     *
     * <p>
     * 字典序法按优先级逐个优化目标。先优化最重要的目标， 然后在前一个目标最优的条件下优化次重要的目标，以此类推。</p>
     *
     * @return Lexicographic MCLP solver instance / 字典序MCLP求解器实例
     */
    public static IMclpSolver lexicographicMclp() {
        return new RereLexicographic();
    }

    /**
     * Create a Lexicographic MCLP solver with specified priority order.
     * 创建带指定优先级顺序的字典序MCLP求解器。
     *
     * @param priorityOrder 优先级顺序数组，例如[0,2,1]表示先优化目标0，再优化目标2，最后目标1
     * @return Lexicographic MCLP solver instance / 字典序MCLP求解器实例
     */
    public static IMclpSolver lexicographicMclp(int[] priorityOrder) {
        return new RereLexicographic(priorityOrder);
    }

    /**
     * Create a Goal Programming Multi-Criteria Linear Programming solver.
     * 创建目标规划多目标线性规划求解器。
     *
     * <p>
     * Goal programming allows decision makers to set target values for each
     * objective and then minimizes deviations from these targets.</p>
     *
     * <p>
     * 目标规划允许决策者为每个目标设定目标值，然后最小化与目标值的偏差。</p>
     *
     * @return Goal programming MCLP solver instance / 目标规划MCLP求解器实例
     */
    public static IMclpSolver goalProgrammingMclp() {
        return new RereGoalProgramming();
    }

    /**
     * Create a Goal Programming MCLP solver with specified goals.
     * 创建带指定目标值的目标规划MCLP求解器。
     *
     * @param goals 目标值数组 / Goal values array
     * @return Goal programming MCLP solver instance / 目标规划MCLP求解器实例
     */
    public static IMclpSolver goalProgrammingMclp(double[] goals) {
        return new RereGoalProgramming(goals);
    }

    /**
     * Create a Goal Programming MCLP solver with specified goals and weights.
     * 创建带指定目标值和权重的主题规划MCLP求解器。
     *
     * @param goals 目标值数组 / Goal values array
     * @param weights 权重向量 / Weight vector
     * @return Goal programming MCLP solver instance / 目标规划MCLP求解器实例
     */
    public static IMclpSolver goalProgrammingMclp(double[] goals, double[] weights) {
        return new RereGoalProgramming(goals, weights);
    }

    /**
     * Create a Pareto Optimal Multi-Criteria Linear Programming solver.
     * 创建Pareto最优多目标线性规划求解器。
     *
     * <p>
     * This method systematically explores the weight space to generate the
     * complete Pareto frontier, helping decision makers understand trade-offs
     * between objectives.</p>
     *
     * <p>
     * 该方法系统地探索权重空间，生成完整的Pareto前沿， 帮助决策者了解各目标之间的权衡关系。</p>
     *
     * @return Pareto optimal MCLP solver instance / Pareto最优MCLP求解器实例
     */
    public static IMclpSolver paretoMclp() {
        return new RereParetoOptimal();
    }

    /**
     * Create a Pareto Optimal MCLP solver with specified number of samples.
     * 创建带指定采样点数量的Pareto最优MCLP求解器。
     *
     * @param numSamples 采样点数量 / Number of sample points
     * @return Pareto optimal MCLP solver instance / Pareto最优MCLP求解器实例
     */
    public static IMclpSolver paretoMclp(int numSamples) {
        return new RereParetoOptimal(numSamples);
    }

    /**
     * Create an AHP (Analytic Hierarchy Process) Multi-Criteria Linear
     * Programming solver. 创建层次分析法（AHP）多目标线性规划求解器。
     *
     * <p>
     * AHP is a multi-criteria decision analysis method that calculates
     * objective weights through a pairwise comparison matrix.</p>
     *
     * <p>
     * AHP是一种多准则决策分析方法，通过成对比较矩阵计算目标权重。</p>
     *
     * @return AHP MCLP solver instance / AHP MCLP求解器实例
     */
    public static IMclpSolver ahpMclp() {
        return new RereAHP();
    }

    /**
     * Create an AHP MCLP solver with specified comparison matrix. 创建带指定比较矩阵的AHP
     * MCLP求解器。
     *
     * @param comparisonMatrix 成对比较矩阵（n×n）/ Pairwise comparison matrix (n×n)
     * @return AHP MCLP solver instance / AHP MCLP求解器实例
     */
    public static IMclpSolver ahpMclp(IMatrix comparisonMatrix) {
        return new RereAHP(comparisonMatrix);
    }

    /**
     * Create a TOPSIS Multi-Criteria Linear Programming solver.
     * 创建TOPSIS多目标线性规划求解器。
     *
     * <p>
     * TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
     * selects the alternative that is closest to the ideal solution and
     * farthest from the negative ideal solution.</p>
     *
     * <p>
     * TOPSIS选择距离理想解最近且距离负理想解最远的方案。</p>
     *
     * @return TOPSIS MCLP solver instance / TOPSIS MCLP求解器实例
     */
    public static IMclpSolver topsisMclp() {
        return new RereTopsis();
    }

    /**
     * Create a TOPSIS MCLP solver with specified weights. 创建带指定权重的TOPSIS
     * MCLP求解器。
     *
     * @param weights 权重向量 / Weight vector
     * @return TOPSIS MCLP solver instance / TOPSIS MCLP求解器实例
     */
    public static IMclpSolver topsisMclp(double[] weights) {
        return new RereTopsis(weights);
    }

    /**
     * Create an Interactive STEM Multi-Criteria Linear Programming solver.
     * 创建交互式STEM多目标线性规划求解器。
     *
     * <p>
     * STEM (STEp Method) is an interactive multi-objective optimization method
     * that progressively guides the search toward a satisfactory solution
     * through decision maker feedback.</p>
     *
     * <p>
     * STEM是一种交互式多目标优化方法，通过决策者反馈逐步引导搜索到满意解。</p>
     *
     * @return Interactive STEM MCLP solver instance / 交互式STEM MCLP求解器实例
     */
    public static IMclpSolver interactiveMclp() {
        return new RereInteractive();
    }

    /**
     * Create an Interactive STEM MCLP solver with decision maker callback.
     * 创建带决策者回调的交互式STEM MCLP求解器。
     *
     * @param callback 决策者回调接口 / Decision maker callback interface
     * @return Interactive STEM MCLP solver instance / 交互式STEM MCLP求解器实例
     */
    public static IMclpSolver interactiveMclp(RereInteractive.DecisionMakerCallback callback) {
        return new RereInteractive(callback);
    }

    /**
     * Create a MCLP solver by solver type. 根据求解器类型创建MCLP求解器。
     *
     * @param solverType 求解器类型 / Solver type
     * @return MCLP solver instance / MCLP求解器实例
     */
    public static IMclpSolver mclp(MclpSolverType solverType) {
        return IMclpSolver.of(solverType);
    }

    /**
     * Create a MCLP solver by solver type name. 根据求解器类型名称创建MCLP求解器。
     *
     * @param solverTypeName 求解器类型名称（WeightedSum, Lexicographic,
     * GoalProgramming, Pareto, Ahp, Topsis, Interactive）
     * @return MCLP solver instance / MCLP求解器实例
     */
    public static IMclpSolver mclp(String solverTypeName) {
        MclpSolverType solverType = MclpSolverType.valueOf(solverTypeName);
        return IMclpSolver.of(solverType);
    }

}
