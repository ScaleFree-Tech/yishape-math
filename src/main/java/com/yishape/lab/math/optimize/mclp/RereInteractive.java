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
 * 交互式STEM方法求解多目标线性规划
 * STEM (STEp Method) Interactive Method for Multi-Objective Linear Programming
 *
 * <p>STEM是一种交互式多目标优化方法，通过决策者逐步提供偏好信息来引导搜索满意解。
 * 每一步，求解器会展示当前Pareto前沿的一部分，决策者选择最期望的解或提供新的偏好信息。
 *
 * <p>算法步骤：
 * 1. 计算理想点（各目标的最优值）
 * 2. 使用参考点方法求解，获得当前最优折中解
 * 3. 显示解给决策者
 * 4. 决策者提供反馈（期望值、满意程度等）
 * 5. 根据反馈调整参考点，返回步骤2
 * 6. 直到决策者满意或达到最大迭代次数
 *
 * <p>参考点方法：
 * minimize max_i (w_i * (f_i(x) - f_i*))  subject to constraints
 * 其中 f_i* 是理想点，w_i 是权重
 *
 * <p>特点：
 * - 交互式，不需要预先给定完整偏好
 * - 逐步收敛到决策者满意的解
 * - 适合决策者难以一次表达完整偏好的场景
 * - 避免非凸问题中加权求和法的缺陷
 *
 * @author lteb2
 * @see IMclpSolver
 */
public class RereInteractive implements IMclpSolver {

    /** 默认最大迭代次数 / Default maximum iterations */
    private static final int DEFAULT_MAX_ITERATIONS = 10;

    /** 默认收敛容差 / Default convergence tolerance */
    private static final double DEFAULT_TOLERANCE = 1e-6;

    /** 底层单目标线性规划求解器 / Underlying single-objective LP solver */
    private ILinProgSolver baseSolver;

    /** 最大迭代次数 / Maximum iterations */
    private int maxIterations;

    /** 收敛容差 / Convergence tolerance */
    private double tolerance;

    /** 初始参考点 / Initial reference point */
    private double[] initialReferencePoint;

    /** 决策者输入的期望值 / Decision maker's aspiration levels */
    private double[] aspirationLevels;

    /** 权重向量 / Weight vector */
    private double[] weights;

    /** 交互回调接口 / Interactive callback interface */
    private transient DecisionMakerCallback callback;

    /** 求解器类型 / Solver type */
    private final MclpSolverType solverType = MclpSolverType.Interactive;

    /** 求解器名称 / Solver name */
    private final String solverName = "交互式STEM法";

    /**
     * 决策者回调接口
     */
    public interface DecisionMakerCallback {
        /**
         * 显示当前解供决策者评估
         *
         * @param currentSolution 当前解
         * @param objectiveValues 当前解的各目标函数值
         * @param iteration 当前迭代次数
         * @return 决策者的响应
         */
        DecisionMakerResponse provideFeedback(IVector currentSolution, double[] objectiveValues, int iteration);

        /**
         * 获取决策者的期望值（目标函数值）
         *
         * @return 期望值数组
         */
        double[] getAspirationLevels();
    }

    /**
     * 决策者响应
     */
    public static class DecisionMakerResponse {
        /** 是否满意当前解 / Whether satisfied with current solution */
        public final boolean satisfied;

        /** 是否调整期望值 / Whether to adjust aspiration levels */
        public final boolean adjustAspiration;

        /** 新的期望值 / New aspiration levels */
        public final double[] newAspirationLevels;

        /** 调整因子（0-1之间）/ Adjustment factor (between 0-1) */
        public final double adjustmentFactor;

        /** 是否调整权重 / Whether to adjust weights */
        public final boolean adjustWeights;

        /** 新的权重 / New weights */
        public final double[] newWeights;

        public DecisionMakerResponse(boolean satisfied, boolean adjustAspiration, double[] newAspirationLevels,
                                    double adjustmentFactor, boolean adjustWeights, double[] newWeights) {
            this.satisfied = satisfied;
            this.adjustAspiration = adjustAspiration;
            this.newAspirationLevels = newAspirationLevels;
            this.adjustmentFactor = adjustmentFactor;
            this.adjustWeights = adjustWeights;
            this.newWeights = newWeights;
        }

        public static DecisionMakerResponse satisfied() {
            return new DecisionMakerResponse(true, false, null, 0.0, false, null);
        }

        public static DecisionMakerResponse notSatisfiedAdjustAspiration(double[] newAspiration, double factor) {
            return new DecisionMakerResponse(false, true, newAspiration, factor, false, null);
        }

        public static DecisionMakerResponse notSatisfiedAdjustWeights(double[] newWeights) {
            return new DecisionMakerResponse(false, false, null, 0.0, true, newWeights);
        }
    }

    /**
     * 默认构造函数（无交互，需要手动设置回调）
     */
    public RereInteractive() {
        this.baseSolver = ILinProgSolver.of();
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
        this.tolerance = DEFAULT_TOLERANCE;
    }

    /**
     * 使用回调的构造函数
     *
     * @param callback 决策者回调
     */
    public RereInteractive(DecisionMakerCallback callback) {
        this.callback = callback;
        this.baseSolver = ILinProgSolver.of();
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
        this.tolerance = DEFAULT_TOLERANCE;
    }

    /**
     * 使用回调和底层求解器的构造函数
     *
     * @param callback 决策者回调
     * @param baseSolver 底层求解器
     */
    public RereInteractive(DecisionMakerCallback callback, ILinProgSolver baseSolver) {
        this.callback = callback;
        this.baseSolver = baseSolver;
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
        this.tolerance = DEFAULT_TOLERANCE;
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
        int numOriginalConstraints = (A_ub != null ? A_ub.rows() : 0) + (A_eq != null ? A_eq.rows() : 0);

        // 步骤1：计算理想点
        double[] idealPoint = computeIdealPoint(c, A_ub, b_ub, A_eq, b_eq);
        double[] nadirPoint = computeNadirPoint(c, A_ub, b_ub, A_eq, b_eq);

        // 初始化参考点（如果未设置，使用理想点）
        if (aspirationLevels == null) {
            aspirationLevels = idealPoint.clone();
        }

        // 初始化权重（如果未设置，使用等权重）
        if (weights == null) {
            weights = new double[numObjectives];
            for (int i = 0; i < numObjectives; i++) {
                weights[i] = 1.0 / numObjectives;
            }
        }

        // 记录历史
        List<IVector> solutionHistory = new ArrayList<>();
        List<double[]> objectiveValueHistory = new ArrayList<>();
        List<OptResult> resultHistory = new ArrayList<>();

        IVector currentX = initX != null ? initX.copy() : Linalg.ones(numVariables);
        IVector bestX = currentX;
        double[] bestObjVals = new double[numObjectives];

        // 步骤2：主交互循环
        int iteration = 0;
        boolean converged = false;

        while (iteration < maxIterations && !converged) {
            // 使用参考点方法求解
            IVector weightedC = computeWeightedObjective(c, weights);
            OptResult result = baseSolver.solve(weightedC, A_ub, b_ub, A_eq, b_eq, currentX);

            if (!result.isConverged() || result.getOptimalPoint() == null) {
                break;
            }

            IVector optimalX = result.getOptimalPoint();
            double[] objVals = new double[numObjectives];
            for (int i = 0; i < numObjectives; i++) {
                objVals[i] = c[i].innerProductValue(optimalX);
            }

            solutionHistory.add(optimalX.copy());
            objectiveValueHistory.add(objVals.clone());
            resultHistory.add(result);

            // 更新最佳解
            if (iteration == 0 || isBetter(objVals, bestObjVals, weights)) {
                bestX = optimalX.copy();
                bestObjVals = objVals.clone();
            }

            // 检查收敛
            if (checkConvergence(objVals, aspirationLevels, tolerance)) {
                converged = true;
                break;
            }

            // 如果有回调，让决策者提供反馈
            if (callback != null) {
                DecisionMakerResponse response = callback.provideFeedback(optimalX, objVals, iteration);

                if (response.satisfied) {
                    // 决策者满意，结束
                    converged = true;
                    break;
                }

                if (response.adjustAspiration && response.newAspirationLevels != null) {
                    // 调整期望值
                    aspirationLevels = response.newAspirationLevels.clone();
                    // 更新权重以反映新的期望
                    updateWeightsFromAspiration(idealPoint, aspirationLevels, weights);
                }

                if (response.adjustWeights && response.newWeights != null) {
                    // 调整权重
                    weights = response.newWeights.clone();
                }
            } else {
                // 无回调，使用自动调整策略
                adjustAutomatically(objVals, idealPoint, nadirPoint);
            }

            // 使用当前解作为下一个迭代的初始点
            currentX = optimalX;
            iteration++;
        }

        // 计算总迭代次数
        int totalIterations = resultHistory.stream()
            .mapToInt(OptResult::getIterations)
            .sum();

        // 构建结果
        List<IVector> solutions = new ArrayList<>();
        List<double[]> objectiveValues = new ArrayList<>();
        List<OptResult> individualResults = new ArrayList<>();

        solutions.add(bestX);
        objectiveValues.add(bestObjVals);
        if (!resultHistory.isEmpty()) {
            individualResults.add(resultHistory.get(resultHistory.size() - 1));
        }

        String convergenceReason = converged ?
            "决策者满意或达到收敛条件" :
            "达到最大迭代次数";

        return new MclpResult.Builder()
            .solutions(solutions)
            .objectiveValues(objectiveValues)
            .individualResults(individualResults)
            .numObjectives(numObjectives)
            .numConstraints(numOriginalConstraints)
            .numVariables(numVariables)
            .weights(weights)
            .solverType(solverType)
            .solverName(solverName)
            .converged(converged)
            .convergenceReason(convergenceReason)
            .totalIterations(totalIterations)
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .idealPoint(idealPoint)
            .nadirPoint(nadirPoint)
            .diversityMetric(computeSpreadMetric(objectiveValueHistory))
            .build();
    }

    /**
     * 计算加权目标函数
     */
    private IVector computeWeightedObjective(IVector[] c, double[] weights) {
        IVector weightedC = Linalg.zeros(c[0].length());
        for (int i = 0; i < c.length; i++) {
            IVector scaled = c[i].multiplyByScalar(weights[i]);
            weightedC = weightedC.add(scaled);
        }
        return weightedC;
    }

    /**
     * 检查收敛
     */
    private boolean checkConvergence(double[] objVals, double[] aspirations, double tolerance) {
        for (int i = 0; i < objVals.length; i++) {
            double diff = Math.abs(objVals[i] - aspirations[i]);
            double scale = Math.abs(aspirations[i]) > 1e-12 ? Math.abs(aspirations[i]) : 1.0;
            if (diff / scale > tolerance) {
                return false;
            }
        }
        return true;
    }

    /**
     * 判断是否更好
     */
    private boolean isBetter(double[] objVals1, double[] objVals2, double[] weights) {
        double score1 = 0.0, score2 = 0.0;
        for (int i = 0; i < objVals1.length; i++) {
            score1 += weights[i] * objVals1[i];
            score2 += weights[i] * objVals2[i];
        }
        return score1 < score2; // 最小化问题
    }

    /**
     * 从期望值更新权重
     */
    private void updateWeightsFromAspiration(double[] idealPoint, double[] aspirations, double[] weights) {
        for (int i = 0; i < idealPoint.length; i++) {
            double gap = Math.abs(aspirations[i] - idealPoint[i]);
            if (gap > 1e-12) {
                weights[i] = 1.0 / gap;
            } else {
                weights[i] = 1.0;
            }
        }
        // 归一化
        double sum = 0.0;
        for (double w : weights) sum += w;
        if (sum > 1e-12) {
            for (int i = 0; i < weights.length; i++) weights[i] /= sum;
        }
    }

    /**
     * 自动调整（当无回调时使用）
     */
    private void adjustAutomatically(double[] objVals, double[] idealPoint, double[] nadirPoint) {
        int numObjectives = objVals.length;

        // 计算到理想点的归一化距离
        double[] normalizedGap = new double[numObjectives];
        for (int i = 0; i < numObjectives; i++) {
            double range = nadirPoint[i] - idealPoint[i];
            if (range > 1e-12) {
                normalizedGap[i] = (objVals[i] - idealPoint[i]) / range;
            } else {
                normalizedGap[i] = 0.0;
            }
        }

        // 增加表现较差目标的权重
        for (int i = 0; i < numObjectives; i++) {
            weights[i] = 1.0 + normalizedGap[i];
        }

        // 归一化
        double sum = 0.0;
        for (double w : weights) sum += w;
        if (sum > 1e-12) {
            for (int i = 0; i < weights.length; i++) weights[i] /= sum;
        }
    }

    /**
     * 计算分布度量
     */
    private double computeSpreadMetric(List<double[]> objectiveValues) {
        if (objectiveValues.size() < 2) {
            return 0.0;
        }

        double totalSpread = 0.0;
        for (int i = 0; i < objectiveValues.size() - 1; i++) {
            double dist = euclideanDistance(objectiveValues.get(i), objectiveValues.get(i + 1));
            totalSpread += dist;
        }

        return totalSpread;
    }

    /**
     * 计算欧氏距离
     */
    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * 计算理想点
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
     * 计算反理想点
     */
    private double[] computeNadirPoint(IVector[] c, IMatrix A_ub, IVector b_ub,
                                     IMatrix A_eq, IVector b_eq) {
        double[] nadir = new double[c.length];
        for (int i = 0; i < c.length; i++) {
            IVector negC = c[i].multiplyByScalar(-1.0);
            OptResult result = baseSolver.solve(negC, A_ub, b_ub, A_eq, b_eq);
            nadir[i] = -result.getOptimalValue();
        }
        return nadir;
    }

    // ==================== 配置方法 ====================

    /**
     * 设置回调
     *
     * @param callback 决策者回调
     * @return this
     */
    public RereInteractive setCallback(DecisionMakerCallback callback) {
        this.callback = callback;
        return this;
    }

    /**
     * 设置最大迭代次数
     *
     * @param maxIterations 最大迭代次数
     * @return this
     */
    public RereInteractive setMaxIterations(int maxIterations) {
        if (maxIterations <= 0) {
            throw new IllegalArgumentException("maxIterations 必须为正数");
        }
        this.maxIterations = maxIterations;
        return this;
    }

    /**
     * 设置收敛容差
     *
     * @param tolerance 收敛容差
     * @return this
     */
    public RereInteractive setTolerance(double tolerance) {
        this.tolerance = tolerance;
        return this;
    }

    /**
     * 设置初始期望值
     *
     * @param aspirationLevels 初始期望值
     * @return this
     */
    public RereInteractive setAspirationLevels(double[] aspirationLevels) {
        Objects.requireNonNull(aspirationLevels, "参数不能为 null");
        this.aspirationLevels = aspirationLevels.clone();
        return this;
    }

    /**
     * 设置权重
     *
     * @param weights 权重向量
     * @return this
     */
    public RereInteractive setWeights(double[] weights) {
        Objects.requireNonNull(weights, "参数不能为 null");
        this.weights = weights.clone();
        return this;
    }

    /**
     * 设置底层求解器
     *
     * @param baseSolver 底层求解器
     * @return this
     */
    public RereInteractive setBaseSolver(ILinProgSolver baseSolver) {
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
        return "交互式STEM法通过决策者逐步提供偏好信息引导搜索满意解，适合偏好难以一次明确的场景。";
    }

    @Override
    public MclpSolverType getSolverType() {
        return solverType;
    }

    @Override
    public String toString() {
        return "RereInteractive{" +
                "maxIterations=" + maxIterations +
                ", tolerance=" + tolerance +
                '}';
    }
}
