package com.yishape.lab.math.optimize.mcqp;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IQpSolver;
import com.yishape.lab.math.optimize.OptResult;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * 目标规划法求解多目标二次规划
 * Goal Programming Method for Multi-Criteria Quadratic Programming
 *
 * <p>目标规划法允许决策者为每个目标设定一个目标值，
 * 然后最小化实际目标值与目标值之间的偏差。
 *
 * <p>数学形式：
 * minimize Σ(w_i * (d_i^+ + d_i^-))
 * subject to A * x + s = b (原约束)
 *       1/2*x^T*Q_i*x + c_i^T*x + d_i^+ - d_i^- = g_i (目标约束)
 *       x >= 0, d_i^+ >= 0, d_i^- >= 0</p>
 *
 * @author lteb2
 * @see IMcqpSolver
 */
public class RereGoalProgrammingQp implements IMcqpSolver {

    /** 目标值向量 / Goal values */
    private double[] goals;

    /** 各目标的权重 / Weights for each objective */
    private double[] weights;

    /** 底层单目标二次规划求解器 / Underlying single-objective QP solver */
    private IQpSolver baseSolver;

    /** 目标规划方法类型 / Goal programming method type */
    private GoalProgrammingType methodType = GoalProgrammingType.Weighted;

    /** 求解器类型 / Solver type */
    private final McqpSolverType solverType = McqpSolverType.GoalProgramming;

    /** 求解器名称 / Solver name */
    private final String solverName = "目标规划法(QP)";

    /**
     * 目标规划方法类型枚举
     */
    public enum GoalProgrammingType {
        /** 加权目标规划 / Weighted Goal Programming */
        Weighted("加权目标规划"),
        /** 字典序目标规划 / Lexicographic Goal Programming */
        Lexicographic("字典序目标规划"),
        /** Chebyshev（最小最大）目标规划 / Chebyshev (Minimax) Goal Programming */
        Chebyshev("Chebyshev目标规划");

        private final String chineseName;

        GoalProgrammingType(String chineseName) {
            this.chineseName = chineseName;
        }
    }

    /**
     * 默认构造函数
     */
    public RereGoalProgrammingQp() {
        this.baseSolver = IQpSolver.of();
    }

    /**
     * 使用指定目标值的构造函数
     *
     * @param goals 目标值数组
     */
    public RereGoalProgrammingQp(double[] goals) {
        this.goals = goals.clone();
        this.baseSolver = IQpSolver.of();
    }

    /**
     * 使用指定目标值和权重的构造函数
     *
     * @param goals 目标值数组
     * @param weights 权重数组
     */
    public RereGoalProgrammingQp(double[] goals, double[] weights) {
        this.goals = goals.clone();
        this.weights = weights.clone();
        this.baseSolver = IQpSolver.of();
    }

    @Override
    public McqpResult solve(IMatrix[] Q, IVector[] c, IMatrix A_ub, IVector b_ub,
                            IMatrix A_eq, IVector b_eq, IVector initX) {
        long startTime = System.currentTimeMillis();

        // 参数验证
        if (Q == null || Q.length == 0) {
            throw new IllegalArgumentException("二次项系数矩阵不能为空 / Q matrices cannot be empty");
        }
        if (c == null || c.length == 0) {
            throw new IllegalArgumentException("线性项系数向量不能为空 / c vectors cannot be empty");
        }
        if (baseSolver == null) throw new IllegalStateException("baseSolver 未设置");

        int numObjectives = Q.length;
        int numVariables = c[0].length();
        int numOriginalConstraints = (A_ub != null ? A_ub.rows() : 0) + (A_eq != null ? A_eq.rows() : 0);

        // 初始化目标值（如果未设置）
        if (goals == null || goals.length != numObjectives) {
            goals = computeDefaultGoals(Q, c, A_ub, b_ub, A_eq, b_eq);
        }

        // 初始化权重（如果未设置）
        if (weights == null || weights.length != numObjectives) {
            weights = initializeDefaultWeights(numObjectives);
        }

        // 使用加权目标规划（简化实现）
        return solveWeighted(Q, c, A_ub, b_ub, A_eq, b_eq, initX,
                numObjectives, numVariables, numOriginalConstraints, startTime);
    }

    /**
     * 加权目标规划：最小化加权偏差和。
     */
    private McqpResult solveWeighted(IMatrix[] Q, IVector[] c, IMatrix A_ub, IVector b_ub,
                                    IMatrix A_eq, IVector b_eq, IVector initX,
                                    int numObjectives, int numVariables,
                                    int numOriginalConstraints, long startTime) {
        int numDeviationVariables = 2 * numObjectives;

        // 构造扩展目标函数：minimize Σ(w_i * (d_i^+ + d_i^-))
        // 注意：QP的二次项在目标规划中被线性化处理，这里简化为线性目标

        // 扩展变量：[x; d^+; d^-]
        int totalVars = numVariables + numDeviationVariables;

        // 扩展Q矩阵：加权组合各目标的二次项
        double[][] qExtData = new double[totalVars][totalVars];
        for (int objIdx = 0; objIdx < numObjectives; objIdx++) {
            for (int i = 0; i < numVariables; i++) {
                for (int j = 0; j < numVariables; j++) {
                    qExtData[i][j] += weights[objIdx] * ((Number) Q[objIdx].get(i, j)).doubleValue();
                }
            }
        }
        IMatrix extendedQ = Linalg.matrix(qExtData);

        // 扩展c向量：仅偏差变量有系数
        double[] cExtData = new double[totalVars];
        for (int i = 0; i < numObjectives; i++) {
            cExtData[numVariables + 2 * i] = weights[i];     // d_i^+
            cExtData[numVariables + 2 * i + 1] = weights[i]; // d_i^-
        }
        IVector extendedC = Linalg.vector(cExtData);

        // 扩展不等式约束
        IMatrix extendedAub = buildExtendedInequalityConstraints(A_ub, numVariables, numObjectives);
        IVector extendedBub = b_ub != null ? b_ub.copy() : null;

        // 扩展等式约束 + 目标约束
        List<double[]> aeqRows = new ArrayList<>();
        List<Double> beqValues = new ArrayList<>();

        // 原等式约束
        if (A_eq != null && b_eq != null) {
            for (int i = 0; i < A_eq.rows(); i++) {
                double[] row = new double[totalVars];
                for (int j = 0; j < numVariables; j++) {
                    row[j] = ((Number) A_eq.get(i, j)).doubleValue();
                }
                aeqRows.add(row);
                beqValues.add(((Number) b_eq.get(i)).doubleValue());
            }
        }

        // 目标约束：1/2*x^T*Q_i*x + c_i^T*x + d_i^+ - d_i^- = g_i
        // 线性化：近似为 c_i^T*x + d_i^+ - d_i^- = g_i（忽略二次项）
        for (int i = 0; i < numObjectives; i++) {
            double[] row = new double[totalVars];
            for (int j = 0; j < numVariables; j++) {
                row[j] = ((Number) c[i].get(j)).doubleValue();
            }
            row[numVariables + 2 * i] = 1.0;     // d_i^+
            row[numVariables + 2 * i + 1] = -1.0; // d_i^-
            aeqRows.add(row);
            beqValues.add(goals[i]);
        }

        IMatrix extendedAeq = aeqRows.isEmpty() ? null : Linalg.matrix(aeqRows.toArray(new double[0][]));
        IVector extendedBeq = beqValues.isEmpty() ? null : Linalg.vector(beqValues.stream().mapToDouble(Double::doubleValue).toArray());

        // 创建扩展初始点
        IVector extendedInitX = createExtendedInitX(initX, numVariables, numObjectives);

        // 求解
        OptResult result = baseSolver.solve(extendedQ, extendedC, extendedAub, extendedBub, extendedAeq, extendedBeq, extendedInitX);

        // 提取原变量
        IVector optimalX = extractOriginalVariables(result.getOptimalPoint(), numVariables);

        // 计算各目标的实际函数值
        double[] actualObjVals = new double[numObjectives];
        for (int i = 0; i < numObjectives; i++) {
            actualObjVals[i] = computeQuadraticObjective(Q[i], c[i], optimalX);
        }

        List<IVector> solutions = new ArrayList<>();
        List<double[]> objectiveValues = new ArrayList<>();
        List<OptResult> individualResults = new ArrayList<>();
        solutions.add(optimalX);
        objectiveValues.add(actualObjVals);
        individualResults.add(result);

        double[] idealPoint = computeIdealPoint(Q, c, A_ub, b_ub, A_eq, b_eq);
        double[] nadirPoint = computeNadirPoint(Q, c, A_ub, b_ub, A_eq, b_eq);
        double achievementDegree = computeAchievementDegree(actualObjVals, goals, weights);

        return new McqpResult.Builder()
            .solutions(solutions).objectiveValues(objectiveValues).individualResults(individualResults)
            .numObjectives(numObjectives).numVariables(numVariables)
            .numConstraints(numOriginalConstraints + numObjectives)
            .goals(goals).weights(weights).solverType(solverType).solverName(solverName)
            .converged(result.isConverged()).convergenceReason(result.getConvergenceReason())
            .totalIterations(result.getIterations())
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .idealPoint(idealPoint).nadirPoint(nadirPoint).diversityMetric(achievementDegree)
            .build();
    }

    /**
     * 构建扩展不等式约束
     */
    private IMatrix buildExtendedInequalityConstraints(IMatrix A_ub, int numVariables, int numObjectives) {
        if (A_ub == null) {
            return null;
        }
        int numRows = A_ub.rows();
        int totalCols = numVariables + 2 * numObjectives;
        double[][] extendedData = new double[numRows][totalCols];

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numVariables; j++) {
                extendedData[i][j] = ((Number) A_ub.get(i, j)).doubleValue();
            }
            // 偏差变量系数为0
        }
        return Linalg.matrix(extendedData);
    }

    /**
     * 创建扩展初始点
     */
    private IVector createExtendedInitX(IVector initX, int numVariables, int numObjectives) {
        double[] initData = new double[numVariables + 2 * numObjectives];
        if (initX != null) {
            for (int i = 0; i < numVariables && i < initX.length(); i++) {
                initData[i] = ((Number) initX.get(i)).doubleValue();
            }
        } else {
            for (int i = 0; i < numVariables; i++) {
                initData[i] = 1.0;
            }
        }
        for (int i = 0; i < numObjectives; i++) {
            initData[numVariables + 2 * i] = 0.0;
            initData[numVariables + 2 * i + 1] = 0.0;
        }
        return Linalg.vector(initData);
    }

    /**
     * 从扩展解中提取原变量
     */
    private IVector extractOriginalVariables(IVector extendedX, int numVariables) {
        double[] original = new double[numVariables];
        for (int i = 0; i < numVariables; i++) {
            original[i] = ((Number) extendedX.get(i)).doubleValue();
        }
        return Linalg.vector(original);
    }

    /**
     * 计算二次目标函数值
     */
    private double computeQuadraticObjective(IMatrix Q, IVector c, IVector x) {
        IVector qx = Q.mmul(x);
        double xqx = x.dotValue(qx);
        double cx = c.dotValue(x);
        return 0.5 * xqx + cx;
    }

    /**
     * 计算默认目标值（各目标的单目标最优值）
     */
    private double[] computeDefaultGoals(IMatrix[] Q, IVector[] c, IMatrix A_ub, IVector b_ub,
                                        IMatrix A_eq, IVector b_eq) {
        double[] defaultGoals = new double[Q.length];
        for (int i = 0; i < Q.length; i++) {
            OptResult result = baseSolver.solve(Q[i], c[i], A_ub, b_ub, A_eq, b_eq, null);
            defaultGoals[i] = result.getOptimalValue();
        }
        return defaultGoals;
    }

    /**
     * 初始化默认权重（等权重）
     */
    private double[] initializeDefaultWeights(int numObjectives) {
        double[] defaultWeights = new double[numObjectives];
        for (int i = 0; i < numObjectives; i++) {
            defaultWeights[i] = 1.0 / numObjectives;
        }
        return defaultWeights;
    }

    /**
     * 计算达成度
     */
    private double computeAchievementDegree(double[] actualVals, double[] goals, double[] weights) {
        double weightedDeviation = 0.0;
        double totalWeight = 0.0;

        for (int i = 0; i < actualVals.length; i++) {
            double deviation = Math.abs(actualVals[i] - goals[i]);
            double normalizedDeviation = Math.abs(goals[i]) > 1e-12 ?
                deviation / Math.abs(goals[i]) : deviation;
            weightedDeviation += weights[i] * normalizedDeviation;
            totalWeight += weights[i];
        }

        return totalWeight > 0 ? 1.0 / (1.0 + weightedDeviation / totalWeight) : 0.0;
    }

    /**
     * 计算理想点
     */
    private double[] computeIdealPoint(IMatrix[] Q, IVector[] c, IMatrix A_ub, IVector b_ub,
                                     IMatrix A_eq, IVector b_eq) {
        double[] ideal = new double[Q.length];
        for (int i = 0; i < Q.length; i++) {
            OptResult result = baseSolver.solve(Q[i], c[i], A_ub, b_ub, A_eq, b_eq, null);
            ideal[i] = result.getOptimalValue();
        }
        return ideal;
    }

    /**
     * 计算反理想点
     */
    private double[] computeNadirPoint(IMatrix[] Q, IVector[] c, IMatrix A_ub, IVector b_ub,
                                     IMatrix A_eq, IVector b_eq) {
        double[] nadir = new double[Q.length];
        for (int i = 0; i < Q.length; i++) {
            IMatrix negQ = Q[i].multiplyByScalar(-1.0);
            OptResult result = baseSolver.solve(negQ, c[i].multiplyByScalar(-1.0), A_ub, b_ub, A_eq, b_eq, null);
            nadir[i] = -result.getOptimalValue();
        }
        return nadir;
    }

    // 配置方法

    public RereGoalProgrammingQp setGoals(double[] goals) {
        Objects.requireNonNull(goals, "参数不能为 null");
        this.goals = goals.clone();
        return this;
    }

    public RereGoalProgrammingQp setWeights(double[] weights) {
        Objects.requireNonNull(weights, "参数不能为 null");
        this.weights = weights.clone();
        return this;
    }

    public RereGoalProgrammingQp setMethodType(GoalProgrammingType methodType) {
        this.methodType = methodType;
        return this;
    }

    public RereGoalProgrammingQp setBaseSolver(IQpSolver baseSolver) {
        Objects.requireNonNull(baseSolver, "baseSolver 不能为 null");
        this.baseSolver = baseSolver;
        return this;
    }

    @Override
    public String getName() { return solverName; }

    @Override
    public String getDescription() {
        return "目标规划法通过设定目标值并最小化实际值与目标值的偏差来求解多目标问题。";
    }

    @Override
    public McqpSolverType getSolverType() { return solverType; }

    @Override
    public String toString() {
        return "RereGoalProgrammingQp{" +
                "goals=" + (goals != null ? java.util.Arrays.toString(goals) : "null") +
                ", weights=" + (weights != null ? java.util.Arrays.toString(weights) : "null") +
                ", methodType=" + methodType +
                '}';
    }
}
