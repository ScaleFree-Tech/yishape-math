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
 * 目标规划法求解多目标线性规划
 * Goal Programming Method for Multi-Objective Linear Programming
 *
 * <p>目标规划法（Goal Programming）允许决策者为每个目标设定一个目标值，
 * 然后最小化实际目标值与目标值之间的偏差。
 *
 * <p>数学形式：
 * minimize Σ(w_i * (d_i^+ + d_i^-))
 * subject to A * x + s = b (原约束)
 *       c_i^T * x + d_i^+ - d_i^- = g_i (目标约束)
 *       x >= 0, d_i^+ >= 0, d_i^- >= 0
 *
 * <p>其中：
 * - g_i 是第i个目标的目标值
 * - d_i^+ 是正偏差（超过目标的量）
 * - d_i^- 是负偏差（低于目标的量）
 * - w_i 是第i个目标的权重
 *
 * <p>目标规划的三种主要形式：
 * 1. 加权目标规划（Weighted Goal Programming）：最小化加权偏差和
 * 2. 字典序目标规划（Lexicographic/Preemptive GP）：按优先级优化目标
 * 3. Chebyshev目标规划：最小化最大偏差
 *
 * @author lteb2
 * @see IMclpSolver
 */
public class RereGoalProgramming implements IMclpSolver {

    /** 目标值向量 / Goal values */
    private double[] goals;

    /** 各目标的权重 / Weights for each objective */
    private double[] weights;

    /** 底层单目标线性规划求解器 / Underlying single-objective LP solver */
    private ILinProgSolver baseSolver;

    /** 目标规划方法类型 / Goal programming method type */
    private GoalProgrammingType methodType = GoalProgrammingType.Weighted;

    /** 求解器类型 / Solver type */
    private final MclpSolverType solverType = MclpSolverType.GoalProgramming;

    /** 求解器名称 / Solver name */
    private final String solverName = "目标规划法";

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

        public String getChineseName() {
            return chineseName;
        }
    }

    /**
     * 默认构造函数
     */
    public RereGoalProgramming() {
        this.baseSolver = ILinProgSolver.of();
    }

    /**
     * 使用指定目标值的构造函数
     *
     * @param goals 目标值数组
     */
    public RereGoalProgramming(double[] goals) {
        this.goals = goals.clone();
        this.baseSolver = ILinProgSolver.of();
    }

    /**
     * 使用指定目标值和权重的构造函数
     *
     * @param goals 目标值数组
     * @param weights 权重数组
     */
    public RereGoalProgramming(double[] goals, double[] weights) {
        this.goals = goals.clone();
        this.weights = weights.clone();
        this.baseSolver = ILinProgSolver.of();
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

        // 初始化目标值（如果未设置）
        if (goals == null || goals.length != numObjectives) {
            goals = computeDefaultGoals(c, A_ub, b_ub, A_eq, b_eq);
        }

        // 初始化权重（如果未设置）
        if (weights == null || weights.length != numObjectives) {
            weights = initializeDefaultWeights(numObjectives);
        }

        // 根据方法类型分派
        switch (methodType) {
            case Weighted:
                return solveWeighted(c, A_ub, b_ub, A_eq, b_eq, initX,
                        numObjectives, numVariables, numOriginalConstraints, startTime);
            case Lexicographic:
                return solveLexicographic(c, A_ub, b_ub, A_eq, b_eq, initX,
                        numObjectives, numVariables, numOriginalConstraints, startTime);
            case Chebyshev:
                return solveChebyshev(c, A_ub, b_ub, A_eq, b_eq, initX,
                        numObjectives, numVariables, numOriginalConstraints, startTime);
            default:
                return solveWeighted(c, A_ub, b_ub, A_eq, b_eq, initX,
                        numObjectives, numVariables, numOriginalConstraints, startTime);
        }
    }

    /**
     * 加权目标规划：最小化加权偏差和。
     */
    private MclpResult solveWeighted(IVector[] c, IMatrix A_ub, IVector b_ub,
                                     IMatrix A_eq, IVector b_eq, IVector initX,
                                     int numObjectives, int numVariables,
                                     int numOriginalConstraints, long startTime) {
        int numDeviationVariables = 2 * numObjectives;

        // 构造扩展目标函数：minimize Σ(w_i * (d_i^+ + d_i^-))
        IVector extendedC = constructGoalProgrammingObjective(numVariables, numObjectives);

        IMatrix extendedAub = constructExtendedInequalityConstraints(A_ub, numVariables, numObjectives);
        IMatrix extendedAeq = constructExtendedEqualityConstraints(A_eq, numVariables, numObjectives);
        IMatrix goalConstraints = constructGoalConstraints(c, numVariables, numObjectives);
        IMatrix combinedAeq = combineEqualityConstraints(extendedAeq, goalConstraints);
        IVector combinedBeq = combineEqualityBounds(b_eq, goals);

        IVector extendedInitX = createExtendedInitX(initX, numVariables, numObjectives);
        OptResult result = baseSolver.solve(extendedC, extendedAub, b_ub, combinedAeq, combinedBeq, extendedInitX);

        IVector optimalX = extractOriginalVariables(result.getOptimalPoint(), numVariables);
        double[] actualObjVals = new double[numObjectives];
        for (int i = 0; i < numObjectives; i++) {
            actualObjVals[i] = c[i].innerProductValue(optimalX);
        }

        List<IVector> solutions = new ArrayList<>();
        List<double[]> objectiveValues = new ArrayList<>();
        List<OptResult> individualResults = new ArrayList<>();
        solutions.add(optimalX);
        objectiveValues.add(actualObjVals);
        individualResults.add(result);

        double[] idealPoint = computeIdealPoint(c, A_ub, b_ub, A_eq, b_eq);
        double[] nadirPoint = computeNadirPoint(c, A_ub, b_ub, A_eq, b_eq);
        double achievementDegree = computeAchievementDegree(actualObjVals, goals, weights);

        return new MclpResult.Builder()
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
     * 字典序目标规划：按优先级顺序逐层最小化偏差。
     * 权重作为优先级（权重越大优先级越高）。
     */
    private MclpResult solveLexicographic(IVector[] c, IMatrix A_ub, IVector b_ub,
                                          IMatrix A_eq, IVector b_eq, IVector initX,
                                          int numObjectives, int numVariables,
                                          int numOriginalConstraints, long startTime) {
        int numDeviationVariables = 2 * numObjectives;
        int totalVars = numVariables + numDeviationVariables;

        // 按权重降序确定优先级
        Integer[] priorityIndices = new Integer[numObjectives];
        for (int i = 0; i < numObjectives; i++) priorityIndices[i] = i;
        java.util.Arrays.sort(priorityIndices, (a, b) -> Double.compare(weights[b], weights[a]));

        IMatrix currentAeq = constructExtendedEqualityConstraints(A_eq, numVariables, numObjectives);
        IMatrix goalConstraints = constructGoalConstraints(c, numVariables, numObjectives);
        currentAeq = combineEqualityConstraints(currentAeq, goalConstraints);
        IVector currentBeq = combineEqualityBounds(b_eq, goals);

        IVector currentX = createExtendedInitX(initX, numVariables, numObjectives);
        int totalIterations = 0;

        List<IVector> solutions = new ArrayList<>();
        List<double[]> objectiveValues = new ArrayList<>();
        List<OptResult> individualResults = new ArrayList<>();

        for (int p = 0; p < numObjectives; p++) {
            int idx = priorityIndices[p];

            // 仅优化第 idx 个目标的偏差
            double[] objCoeffs = new double[totalVars];
            objCoeffs[numVariables + 2 * idx] = 1.0;     // d_i^+
            objCoeffs[numVariables + 2 * idx + 1] = 1.0; // d_i^-
            IVector priorityObj = Linalg.vector(objCoeffs);

            IMatrix extendedAub = constructExtendedInequalityConstraints(A_ub, numVariables, numObjectives);
            OptResult result = baseSolver.solve(priorityObj, extendedAub, b_ub, currentAeq, currentBeq, currentX);

            if (result.isConverged() && result.getOptimalPoint() != null) {
                currentX = result.getOptimalPoint();
                totalIterations += result.getIterations();

                // 锁定当前目标的偏差为已达成值
                double devPlus = (Double) currentX.get(numVariables + 2 * idx);
                double devMinus = (Double) currentX.get(numVariables + 2 * idx + 1);

                double[] lockRow = new double[totalVars];
                lockRow[numVariables + 2 * idx] = 1.0;
                double[] lockRow2 = new double[totalVars];
                lockRow2[numVariables + 2 * idx + 1] = 1.0;

                IMatrix lockAeq1 = Linalg.matrix(new double[][]{lockRow});
                IMatrix lockAeq2 = Linalg.matrix(new double[][]{lockRow2});
                currentAeq = combineEqualityConstraints(currentAeq, lockAeq1);
                currentAeq = combineEqualityConstraints(currentAeq, lockAeq2);
                currentBeq = combineEqualityBounds(currentBeq, new double[]{devPlus});
                currentBeq = combineEqualityBounds(currentBeq, new double[]{devMinus});
            }
        }

        IVector optimalX = extractOriginalVariables(currentX, numVariables);
        double[] actualObjVals = new double[numObjectives];
        for (int i = 0; i < numObjectives; i++) {
            actualObjVals[i] = c[i].innerProductValue(optimalX);
        }

        solutions.add(optimalX);
        objectiveValues.add(actualObjVals);
        individualResults.add(null); // 多次求解，无单一结果

        double[] idealPoint = computeIdealPoint(c, A_ub, b_ub, A_eq, b_eq);
        double[] nadirPoint = computeNadirPoint(c, A_ub, b_ub, A_eq, b_eq);
        double achievementDegree = computeAchievementDegree(actualObjVals, goals, weights);

        return new MclpResult.Builder()
            .solutions(solutions).objectiveValues(objectiveValues).individualResults(individualResults)
            .numObjectives(numObjectives).numVariables(numVariables)
            .numConstraints(numOriginalConstraints + numObjectives)
            .goals(goals).weights(weights).solverType(solverType).solverName(solverName)
            .converged(true).convergenceReason("字典序目标规划完成")
            .totalIterations(totalIterations)
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .idealPoint(idealPoint).nadirPoint(nadirPoint).diversityMetric(achievementDegree)
            .build();
    }

    /**
     * Chebyshev目标规划：最小化最大加权偏差。
     * 引入变量 D，目标 min D，约束 w_i * d_i^+ ≤ D，w_i * d_i^- ≤ D。
     */
    private MclpResult solveChebyshev(IVector[] c, IMatrix A_ub, IVector b_ub,
                                      IMatrix A_eq, IVector b_eq, IVector initX,
                                      int numObjectives, int numVariables,
                                      int numOriginalConstraints, long startTime) {
        int numDeviationVariables = 2 * numObjectives;
        int totalVars = numVariables + numDeviationVariables + 1; // +1 for D
        int dIndex = numVariables + numDeviationVariables; // index of D

        // 目标函数：minimize D
        double[] objCoeffs = new double[totalVars];
        objCoeffs[dIndex] = 1.0;
        IVector chebyshevObj = Linalg.vector(objCoeffs);

        // 构建约束
        List<double[]> aubRows = new ArrayList<>();
        List<Double> bubValues = new ArrayList<>();

        // 原不等式约束
        if (A_ub != null) {
            for (int i = 0; i < A_ub.rows(); i++) {
                double[] row = new double[totalVars];
                for (int j = 0; j < numVariables; j++) {
                    row[j] = (Double) A_ub.get(i, j);
                }
                aubRows.add(row);
                bubValues.add((Double) b_ub.get(i));
            }
        }

        // Chebyshev约束：w_i * d_i^+ - D ≤ 0
        for (int i = 0; i < numObjectives; i++) {
            double[] row = new double[totalVars];
            row[numVariables + 2 * i] = weights[i]; // +w_i * d_i^+
            row[dIndex] = -1.0;                        // -D
            aubRows.add(row);
            bubValues.add(0.0);
        }

        // Chebyshev约束：w_i * d_i^- - D ≤ 0
        for (int i = 0; i < numObjectives; i++) {
            double[] row = new double[totalVars];
            row[numVariables + 2 * i + 1] = weights[i]; // +w_i * d_i^-
            row[dIndex] = -1.0;                            // -D
            aubRows.add(row);
            bubValues.add(0.0);
        }

        IMatrix combinedAub = Linalg.matrix(aubRows.toArray(new double[0][]));
        IVector combinedBub = Linalg.vector(bubValues.stream().mapToDouble(Double::doubleValue).toArray());

        // 等式约束：原等式（扩展到totalVars列） + 目标约束
        List<double[]> aeqRows = new ArrayList<>();
        if (A_eq != null) {
            for (int i = 0; i < A_eq.rows(); i++) {
                double[] row = new double[totalVars];
                for (int j = 0; j < numVariables; j++) {
                    row[j] = (Double) A_eq.get(i, j);
                }
                aeqRows.add(row);
            }
        }
        double[][] goalRows = new double[numObjectives][totalVars];
        for (int i = 0; i < numObjectives; i++) {
            for (int j = 0; j < numVariables; j++) {
                goalRows[i][j] = (Double) c[i].get(j);
            }
            goalRows[i][numVariables + 2 * i] = 1.0;     // d_i^+
            goalRows[i][numVariables + 2 * i + 1] = -1.0; // d_i^-
            aeqRows.add(goalRows[i]);
        }
        IMatrix combinedAeq = aeqRows.isEmpty() ? null
            : Linalg.matrix(aeqRows.toArray(new double[0][]));
        IVector combinedBeq = combineEqualityBounds(b_eq, goals);

        // 初始点
        double[] initData = new double[totalVars];
        if (initX != null) {
            for (int i = 0; i < numVariables && i < initX.length(); i++) {
                initData[i] = (Double) initX.get(i);
            }
        } else {
            for (int i = 0; i < numVariables; i++) initData[i] = 1.0;
        }
        IVector chebyshevInitX = Linalg.vector(initData);

        OptResult result = baseSolver.solve(chebyshevObj, combinedAub, combinedBub, combinedAeq, combinedBeq, chebyshevInitX);

        IVector optimalX = extractOriginalVariables(result.getOptimalPoint(), numVariables);
        double[] actualObjVals = new double[numObjectives];
        for (int i = 0; i < numObjectives; i++) {
            actualObjVals[i] = c[i].innerProductValue(optimalX);
        }

        List<IVector> solutions = new ArrayList<>();
        List<double[]> objectiveValues = new ArrayList<>();
        List<OptResult> individualResults = new ArrayList<>();
        solutions.add(optimalX);
        objectiveValues.add(actualObjVals);
        individualResults.add(result);

        double[] idealPoint = computeIdealPoint(c, A_ub, b_ub, A_eq, b_eq);
        double[] nadirPoint = computeNadirPoint(c, A_ub, b_ub, A_eq, b_eq);
        double achievementDegree = computeAchievementDegree(actualObjVals, goals, weights);

        return new MclpResult.Builder()
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
     * 计算默认目标值（各目标的单目标最优值）
     */
    private double[] computeDefaultGoals(IVector[] c, IMatrix A_ub, IVector b_ub,
                                         IMatrix A_eq, IVector b_eq) {
        double[] defaultGoals = new double[c.length];
        for (int i = 0; i < c.length; i++) {
            OptResult result = baseSolver.solve(c[i], A_ub, b_ub, A_eq, b_eq);
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
     * 构造目标规划目标函数：minimize Σ(w_i * (d_i^+ + d_i^-))
     */
    private IVector constructGoalProgrammingObjective(int numVariables, int numObjectives) {
        double[] cData = new double[numVariables + 2 * numObjectives];

        // 原变量系数为0
        for (int i = 0; i < numVariables; i++) {
            cData[i] = 0.0;
        }

        // 偏差变量系数为权重（正负偏差同样重要）
        for (int i = 0; i < numObjectives; i++) {
            cData[numVariables + 2 * i] = weights[i];     // d_i^+
            cData[numVariables + 2 * i + 1] = weights[i]; // d_i^-
        }

        return Linalg.vector(cData);
    }

    /**
     * 构造扩展不等式约束
     */
    private IMatrix constructExtendedInequalityConstraints(IMatrix A_ub,
                                                          int numVariables, int numObjectives) {
        if (A_ub == null) {
            return null;
        }

        int numRows = A_ub.rows();
        int totalCols = numVariables + 2 * numObjectives;
        double[][] extendedData = new double[numRows][totalCols];

        for (int i = 0; i < numRows; i++) {
            // 复制原约束系数
            for (int j = 0; j < numVariables; j++) {
                extendedData[i][j] = (Double) A_ub.get(i, j);
            }
            // 偏差变量系数为0
            for (int j = numVariables; j < totalCols; j++) {
                extendedData[i][j] = 0.0;
            }
        }

        return Linalg.matrix(extendedData);
    }

    /**
     * 构造扩展等式约束
     */
    private IMatrix constructExtendedEqualityConstraints(IMatrix A_eq,
                                                        int numVariables, int numObjectives) {
        if (A_eq == null) {
            return null;
        }

        int numRows = A_eq.rows();
        int totalCols = numVariables + 2 * numObjectives;
        double[][] extendedData = new double[numRows][totalCols];

        for (int i = 0; i < numRows; i++) {
            // 复制原约束系数
            for (int j = 0; j < numVariables; j++) {
                extendedData[i][j] = (Double) A_eq.get(i, j);
            }
            // 偏差变量系数为0
            for (int j = numVariables; j < totalCols; j++) {
                extendedData[i][j] = 0.0;
            }
        }

        return Linalg.matrix(extendedData);
    }

    /**
     * 构造目标约束：c_i^T * x + d_i^+ - d_i^- = g_i
     */
    private IMatrix constructGoalConstraints(IVector[] c, int numVariables, int numObjectives) {
        double[][] goalData = new double[numObjectives][numVariables + 2 * numObjectives];

        for (int i = 0; i < numObjectives; i++) {
            // 目标函数系数
            for (int j = 0; j < numVariables; j++) {
                goalData[i][j] = (Double) c[i].get(j);
            }
            // 正偏差系数为+1
            goalData[i][numVariables + 2 * i] = 1.0;
            // 负偏差系数为-1
            goalData[i][numVariables + 2 * i + 1] = -1.0;
        }

        return Linalg.matrix(goalData);
    }

    /**
     * 合并等式约束
     */
    private IMatrix combineEqualityConstraints(IMatrix existingAeq, IMatrix goalConstraints) {
        if (existingAeq == null) {
            return goalConstraints;
        }

        int existingRows = existingAeq.rows();
        int goalRows = goalConstraints.rows();
        int cols = existingAeq.cols();

        double[][] combined = new double[existingRows + goalRows][cols];

        for (int i = 0; i < existingRows; i++) {
            for (int j = 0; j < cols; j++) {
                combined[i][j] = (Double) existingAeq.get(i, j);
            }
        }

        for (int i = 0; i < goalRows; i++) {
            for (int j = 0; j < cols; j++) {
                combined[existingRows + i][j] = (Double) goalConstraints.get(i, j);
            }
        }

        return Linalg.matrix(combined);
    }

    /**
     * 合并等式约束右端向量
     */
    private IVector combineEqualityBounds(IVector existingBeq, double[] goals) {
        int existingLen = existingBeq != null ? existingBeq.length() : 0;
        int totalLen = existingLen + goals.length;
        double[] combined = new double[totalLen];

        if (existingBeq != null) {
            for (int i = 0; i < existingLen; i++) {
                combined[i] = (Double) existingBeq.get(i);
            }
        }

        for (int i = 0; i < goals.length; i++) {
            combined[existingLen + i] = goals[i];
        }

        return Linalg.vector(combined);
    }

    /**
     * 创建扩展初始点
     */
    private IVector createExtendedInitX(IVector initX, int numVariables, int numObjectives) {
        double[] initData = new double[numVariables + 2 * numObjectives];

        if (initX != null) {
            for (int i = 0; i < numVariables; i++) {
                initData[i] = (Double) initX.get(i);
            }
        } else {
            for (int i = 0; i < numVariables; i++) {
                initData[i] = 1.0;
            }
        }

        // 偏差变量初始值为0
        for (int i = 0; i < numObjectives; i++) {
            initData[numVariables + 2 * i] = 0.0;     // d_i^+
            initData[numVariables + 2 * i + 1] = 0.0; // d_i^-
        }

        return Linalg.vector(initData);
    }

    /**
     * 从扩展解中提取原变量
     */
    private IVector extractOriginalVariables(IVector extendedX, int numVariables) {
        double[] original = new double[numVariables];
        for (int i = 0; i < numVariables; i++) {
            original[i] = (Double) extendedX.get(i);
        }
        return Linalg.vector(original);
    }

    /**
     * 计算达成度
     */
    private double computeAchievementDegree(double[] actualVals, double[] goals, double[] weights) {
        double weightedDeviation = 0.0;
        double totalWeight = 0.0;

        for (int i = 0; i < actualVals.length; i++) {
            double deviation = Math.abs(actualVals[i] - goals[i]);
            // 考虑目标值大小的归一化
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
     * 设置目标值
     *
     * @param goals 目标值数组
     * @return this
     */
    public RereGoalProgramming setGoals(double[] goals) {
        Objects.requireNonNull(goals, "参数不能为 null");
        this.goals = goals.clone();
        return this;
    }

    /**
     * 设置权重
     *
     * @param weights 权重数组
     * @return this
     */
    public RereGoalProgramming setWeights(double[] weights) {
        Objects.requireNonNull(weights, "参数不能为 null");
        this.weights = weights.clone();
        return this;
    }

    /**
     * 设置目标规划方法类型
     *
     * @param methodType 方法类型
     * @return this
     */
    public RereGoalProgramming setMethodType(GoalProgrammingType methodType) {
        this.methodType = methodType;
        return this;
    }

    /**
     * 设置底层求解器
     *
     * @param baseSolver 底层求解器
     * @return this
     */
    public RereGoalProgramming setBaseSolver(ILinProgSolver baseSolver) {
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
        return "目标规划法通过设定目标值并最小化实际值与目标值的偏差来求解多目标问题。";
    }

    @Override
    public MclpSolverType getSolverType() {
        return solverType;
    }

    @Override
    public String toString() {
        return "RereGoalProgramming{" +
                "goals=" + (goals != null ? java.util.Arrays.toString(goals) : "null") +
                ", weights=" + (weights != null ? java.util.Arrays.toString(weights) : "null") +
                ", methodType=" + methodType +
                '}';
    }
}
