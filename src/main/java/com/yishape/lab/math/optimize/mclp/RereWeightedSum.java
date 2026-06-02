package com.yishape.lab.math.optimize.mclp;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * 加权求和法求解多目标线性规划
 * Weighted Sum Method for Multi-Objective Linear Programming
 *
 * <p>该方法通过为每个目标函数分配权重，将多目标优化问题转化为单目标优化问题。
 * minimize Σ(w_i * c_i^T * x)
 * subject to constraints</p>
 *
 * <p>特点：
 * - 简单直观，易于实现
 * - 只能找到凸Pareto前沿上的点
 * - 无法处理非凸Pareto前沿
 * - 对权重敏感</p>
 *
 * @author lteb2
 * @see IMclpSolver
 */
public class RereWeightedSum implements IMclpSolver {

    /** 默认权重（等权重）/ Default weights (equal weights) */
    private static final double DEFAULT_WEIGHT = 1.0;

    /** 权重向量 / Weight vector */
    private double[] weights;

    /** 底层单目标线性规划求解器 / Underlying single-objective LP solver */
    private ILinProgSolver baseSolver;

    /** 权重归一化标志 / Weight normalization flag */
    private boolean normalizeWeights = true;

    /** 求解器类型 / Solver type */
    private final MclpSolverType solverType = MclpSolverType.WeightedSum;

    /** 求解器名称 / Solver name */
    private final String solverName = "加权求和法";

    /**
     * 默认构造函数，使用等权重
     */
    public RereWeightedSum() {
        this.baseSolver = ILinProgSolver.of();
    }

    /**
     * 使用指定权重的构造函数
     *
     * @param weights 权重向量（长度应等于目标函数数量）
     */
    public RereWeightedSum(double[] weights) {
        this.weights = weights.clone();
        this.baseSolver = ILinProgSolver.of();
    }

    /**
     * 使用指定权重和底层求解器的构造函数
     *
     * @param weights 权重向量
     * @param baseSolver 底层求解器
     */
    public RereWeightedSum(double[] weights, ILinProgSolver baseSolver) {
        this.weights = weights.clone();
        this.baseSolver = baseSolver;
    }

    @Override
    public MclpResult solve(IVector[] c, IMatrix A_ub, IVector b_ub,
                            IMatrix A_eq, IVector b_eq, IVector initX) {
        long startTime = System.currentTimeMillis();

        // 参数验证
        if (c == null || c.length == 0) {
            throw new IllegalArgumentException("目标函数不能为空 / Objectives cannot be empty");
        }
        for (int i = 0; i < c.length; i++) {
            if (c[i] == null) throw new IllegalArgumentException("目标函数向量 c[" + i + "] 不能为 null");
        }
        if (baseSolver == null) throw new IllegalStateException("baseSolver 未设置");

        int numObjectives = c.length;
        int numVariables = c[0].length();

        // 初始化权重（如果未设置）
        if (weights == null || weights.length != numObjectives) {
            initializeWeights(numObjectives);
        }

        // 归一化权重
        double[] normalizedWeights = normalizeWeights ? normalize(weights) : weights.clone();

        // 构造加权目标函数系数
        IVector weightedC = computeWeightedObjective(c, normalizedWeights);

        // 调用底层求解器
        OptResult result = baseSolver.solve(weightedC, A_ub, b_ub, A_eq, b_eq, initX);

        // 构建结果
        List<IVector> solutions = new ArrayList<>();
        List<double[]> objectiveValues = new ArrayList<>();
        List<OptResult> individualResults = new ArrayList<>();

        IVector optimalX = result.getOptimalPoint();
        solutions.add(optimalX);

        // 计算各目标的实际函数值
        double[] objVals = new double[numObjectives];
        for (int i = 0; i < numObjectives; i++) {
            objVals[i] = c[i].innerProductValue(optimalX);
        }
        objectiveValues.add(objVals);
        individualResults.add(result);

        // 计算理想点和反理想点
        double[] idealPoint = computeIdealPoint(c, A_ub, b_ub, A_eq, b_eq);
        double[] nadirPoint = computeNadirPoint(c, A_ub, b_ub, A_eq, b_eq);

        return new MclpResult.Builder()
            .solutions(solutions)
            .objectiveValues(objectiveValues)
            .individualResults(individualResults)
            .numObjectives(numObjectives)
            .numConstraints((A_ub != null ? A_ub.rows() : 0) + (A_eq != null ? A_eq.rows() : 0))
            .numVariables(numVariables)
            .weights(normalizedWeights)
            .solverType(solverType)
            .solverName(solverName)
            .converged(result.isConverged())
            .convergenceReason(result.getConvergenceReason())
            .totalIterations(result.getIterations())
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .idealPoint(idealPoint)
            .nadirPoint(nadirPoint)
            .diversityMetric(0.0)
            .build();
    }

    /**
     * 初始化等权重
     */
    private void initializeWeights(int numObjectives) {
        this.weights = new double[numObjectives];
        for (int i = 0; i < numObjectives; i++) {
            weights[i] = DEFAULT_WEIGHT;
        }
    }

    /**
     * 归一化权重向量
     */
    private double[] normalize(double[] weights) {
        double sum = 0.0;
        for (double w : weights) {
            sum += Math.abs(w);
        }
        if (sum < 1e-12) {
            double[] equalWeights = new double[weights.length];
            for (int i = 0; i < equalWeights.length; i++) {
                equalWeights[i] = 1.0 / weights.length;
            }
            return equalWeights;
        }
        double[] normalized = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            normalized[i] = Math.abs(weights[i]) / sum;
        }
        return normalized;
    }

    /**
     * 计算加权目标函数系数向量
     */
    private IVector computeWeightedObjective(IVector[] c, double[] normalizedWeights) {
        IVector weightedC = Linalg.zeros(c[0].length());
        for (int i = 0; i < c.length; i++) {
            IVector weighted = c[i].multiplyByScalar(normalizedWeights[i]);
            weightedC = weightedC.add(weighted);
        }
        return weightedC;
    }

    /**
     * 计算理想点（各目标的单目标最优值）
     */
    private double[] computeIdealPoint(IVector[] c, IMatrix A_ub, IVector b_ub,
                                         IMatrix A_eq, IVector b_eq) {
        double[] ideal = new double[c.length];
        for (int i = 0; i < c.length; i++) {
            OptResult result = baseSolver.solve(c[i], A_ub, b_ub, A_eq, b_eq);
            ideal[i] = result.getOptimalValue();
        }
        return ideal;
    }

    /**
     * 计算反理想点（各目标的单目标最差值）
     */
    private double[] computeNadirPoint(IVector[] c, IMatrix A_ub, IVector b_ub,
                                        IMatrix A_eq, IVector b_eq) {
        double[] nadir = new double[c.length];
        for (int i = 0; i < c.length; i++) {
            // 对于最小化问题，反理想点可以通过最大化该目标得到
            IVector negC = c[i].multiplyByScalar(-1.0);
            OptResult result = baseSolver.solve(negC, A_ub, b_ub, A_eq, b_eq);
            nadir[i] = -result.getOptimalValue();
        }
        return nadir;
    }

    // ==================== 配置方法 ====================

    /**
     * 设置权重向量
     *
     * @param weights 权重向量
     * @return this
     */
    public RereWeightedSum setWeights(double[] weights) {
        Objects.requireNonNull(weights, "参数不能为 null");
        this.weights = weights.clone();
        return this;
    }

    /**
     * 设置是否归一化权重
     *
     * @param normalize 是否归一化
     * @return this
     */
    public RereWeightedSum setNormalizeWeights(boolean normalize) {
        this.normalizeWeights = normalize;
        return this;
    }

    /**
     * 设置底层求解器
     *
     * @param baseSolver 底层求解器
     * @return this
     */
    public RereWeightedSum setBaseSolver(ILinProgSolver baseSolver) {
        Objects.requireNonNull(baseSolver, "baseSolver 不能为 null");
        this.baseSolver = baseSolver;
        return this;
    }

    // ==================== IMclpSolver 接口实现 ====================

    @Override
    public String getName() {
        return solverName;
    }

    @Override
    public String getDescription() {
        return "加权求和法通过线性组合多个目标函数转化为单目标优化问题。";
    }

    @Override
    public MclpSolverType getSolverType() {
        return solverType;
    }

    @Override
    public String toString() {
        return "RereWeightedSum{" +
                "weights=" + (weights != null ? java.util.Arrays.toString(weights) : "null") +
                ", normalizeWeights=" + normalizeWeights +
                '}';
    }
}
