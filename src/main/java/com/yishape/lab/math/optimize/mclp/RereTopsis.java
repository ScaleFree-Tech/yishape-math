package com.yishape.lab.math.optimize.mclp;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * TOPSIS方法求解多目标线性规划
 * TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) for MCLP
 *
 * <p>TOPSIS是一种多准则决策分析方法，其基本思想是：
 * 所选择的方案应距离理想解最近，距离负理想解最远。
 *
 * <p>算法步骤：
 * 1. 计算各目标的单目标最优解，构成候选解集
 * 2. 构建决策矩阵（各解对应的目标函数值）
 * 3. 归一化决策矩阵
 * 4. 加权归一化决策矩阵
 * 5. 计算理想解（各目标的最好值）和负理想解（各目标的最差值）
 * 6. 计算各解到理想解和负理想解的距离
 * 7. 计算相对贴近度，选择最高者
 *
 * <p>特点：
 * - 不需要权重计算，直接使用给定的权重
 * - 考虑了解与理想解和负理想解的距离
 * - 适用于有明确偏好的决策场景
 *
 * @author lteb2
 * @see IMclpSolver
 */
public class RereTopsis implements IMclpSolver {

    /** 权重向量 / Weight vector */
    private double[] weights;

    /** 底层单目标线性规划求解器 / Underlying single-objective LP solver */
    private ILinProgSolver baseSolver;

    /** 是否最大化（默认false，即最小化）/ Whether to maximize (default false, i.e., minimize) */
    private boolean[] isMaximization;

    /** 求解器类型 / Solver type */
    private final MclpSolverType solverType = MclpSolverType.Topsis;

    /** 求解器名称 / Solver name */
    private final String solverName = "TOPSIS法";

    /**
     * 默认构造函数
     */
    public RereTopsis() {
        this.baseSolver = ILinProgSolver.of();
    }

    /**
     * 使用指定权重的构造函数
     *
     * @param weights 权重向量
     */
    public RereTopsis(double[] weights) {
        this.weights = weights.clone();
        this.baseSolver = ILinProgSolver.of();
    }

    /**
     * 使用权重和底层求解器的构造函数
     *
     * @param weights 权重向量
     * @param baseSolver 底层求解器
     */
    public RereTopsis(double[] weights, ILinProgSolver baseSolver) {
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
            weights = initializeDefaultWeights(numObjectives);
        }

        // 初始化最大化标志（默认都是最小化）
        if (isMaximization == null || isMaximization.length != numObjectives) {
            isMaximization = new boolean[numObjectives];
            for (int i = 0; i < numObjectives; i++) {
                isMaximization[i] = false;
            }
        }

        // 步骤1：计算各目标的极端解
        List<IVector> candidateSolutions = new ArrayList<>();
        List<double[]> candidateObjectiveValues = new ArrayList<>();
        List<OptResult> candidateResults = new ArrayList<>();

        for (int i = 0; i < numObjectives; i++) {
            IVector objC = isMaximization[i] ? c[i].multiplyByScalar(-1.0) : c[i];
            OptResult result = baseSolver.solve(objC, A_ub, b_ub, A_eq, b_eq, initX);

            if (result.isConverged() && result.getOptimalPoint() != null) {
                IVector optimalX = result.getOptimalPoint();
                candidateSolutions.add(optimalX);
                candidateResults.add(result);

                double[] objVals = new double[numObjectives];
                for (int j = 0; j < numObjectives; j++) {
                    objVals[j] = c[j].innerProductValue(optimalX);
                }
                candidateObjectiveValues.add(objVals);
            }
        }

        // 如果极端解不够，添加一些中间权重解
        addIntermediateSolutions(c, A_ub, b_ub, A_eq, b_eq, initX,
                                candidateSolutions, candidateObjectiveValues, candidateResults);

        // 步骤2：构建决策矩阵
        double[][] decisionMatrix = candidateObjectiveValues.toArray(new double[0][]);
        int numCandidates = decisionMatrix.length;

        // 步骤3：归一化决策矩阵
        double[][] normalizedMatrix = normalizeDecisionMatrix(decisionMatrix);

        // 步骤4：加权归一化
        double[][] weightedMatrix = applyWeights(normalizedMatrix);

        // 步骤5：计算理想解和负理想解
        double[] idealSolution = computeIdealSolution(weightedMatrix);
        double[] negativeIdealSolution = computeNegativeIdealSolution(weightedMatrix);

        // 步骤6：计算距离
        double[] distancesToIdeal = computeDistances(weightedMatrix, idealSolution);
        double[] distancesToNegativeIdeal = computeDistances(weightedMatrix, negativeIdealSolution);

        // 步骤7：计算相对贴近度
        double[] closenessRatios = computeClosenessRatios(distancesToIdeal, distancesToNegativeIdeal);

        // 选择最佳解
        int bestIndex = findBestSolution(closenessRatios);

        // 构建最终结果
        IVector optimalX = candidateSolutions.get(bestIndex);
        double[] optimalObjVals = candidateObjectiveValues.get(bestIndex);

        // 计算理想点和反理想点
        double[] idealPoint = computeIdealPoint(c, A_ub, b_ub, A_eq, b_eq);
        double[] nadirPoint = computeNadirPoint(c, A_ub, b_ub, A_eq, b_eq);

        List<IVector> solutions = new ArrayList<>();
        List<double[]> objectiveValues = new ArrayList<>();
        List<OptResult> individualResults = new ArrayList<>();

        solutions.add(optimalX);
        objectiveValues.add(optimalObjVals);
        individualResults.add(candidateResults.get(bestIndex));

        int totalIterations = candidateResults.stream()
            .mapToInt(OptResult::getIterations)
            .sum();

        return new MclpResult.Builder()
            .solutions(solutions)
            .objectiveValues(objectiveValues)
            .individualResults(individualResults)
            .numObjectives(numObjectives)
            .numConstraints((A_ub != null ? A_ub.rows() : 0) + (A_eq != null ? A_eq.rows() : 0))
            .numVariables(numVariables)
            .weights(weights)
            .solverType(solverType)
            .solverName(solverName)
            .converged(true)
            .convergenceReason("TOPSIS选择最佳解，贴近度=" + String.format("%.4f", closenessRatios[bestIndex]))
            .totalIterations(totalIterations)
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .idealPoint(idealPoint)
            .nadirPoint(nadirPoint)
            .diversityMetric(closenessRatios[bestIndex])
            .selectedSolutionIndex(0)
            .build();
    }

    /**
     * 添加中间权重解以丰富候选集
     */
    private void addIntermediateSolutions(IVector[] c, IMatrix A_ub, IVector b_ub,
                                          IMatrix A_eq, IVector b_eq, IVector initX,
                                          List<IVector> solutions, List<double[]> objectiveValues,
                                          List<OptResult> results) {
        int numObjectives = c.length;
        int numToAdd = Math.max(3, numObjectives); // 至少添加numObjectives个中间解

        // 生成中间权重
        for (int i = 0; i < numToAdd; i++) {
            double[] intermediateWeights = new double[numObjectives];
            for (int j = 0; j < numObjectives; j++) {
                // 生成0.2到0.8之间的权重
                intermediateWeights[j] = 0.2 + 0.6 * (i + 1) / (numToAdd + 1.0);
            }
            // 归一化
            double sum = 0.0;
            for (double w : intermediateWeights) sum += w;
            for (int j = 0; j < numObjectives; j++) intermediateWeights[j] /= sum;

            // 求解
            IVector weightedC = computeWeightedObjective(c, intermediateWeights);
            OptResult result = baseSolver.solve(weightedC, A_ub, b_ub, A_eq, b_eq, initX);

            if (result.isConverged() && result.getOptimalPoint() != null) {
                IVector optimalX = result.getOptimalPoint();

                // 检查是否重复
                boolean isDuplicate = false;
                for (IVector existing : solutions) {
                    if (isSameSolution(existing, optimalX)) {
                        isDuplicate = true;
                        break;
                    }
                }

                if (!isDuplicate) {
                    solutions.add(optimalX);
                    results.add(result);

                    double[] objVals = new double[numObjectives];
                    for (int j = 0; j < numObjectives; j++) {
                        objVals[j] = c[j].innerProductValue(optimalX);
                    }
                    objectiveValues.add(objVals);
                }
            }
        }
    }

    /**
     * 归一化决策矩阵
     */
    private double[][] normalizeDecisionMatrix(double[][] decisionMatrix) {
        int numCandidates = decisionMatrix.length;
        int numObjectives = decisionMatrix[0].length;

        // 计算每个目标的平方和根
        double[] sqrtSumSquares = new double[numObjectives];
        for (int j = 0; j < numObjectives; j++) {
            double sum = 0.0;
            for (int i = 0; i < numCandidates; i++) {
                sum += decisionMatrix[i][j] * decisionMatrix[i][j];
            }
            sqrtSumSquares[j] = Math.sqrt(sum);
        }

        // 归一化
        double[][] normalized = new double[numCandidates][numObjectives];
        for (int i = 0; i < numCandidates; i++) {
            for (int j = 0; j < numObjectives; j++) {
                if (sqrtSumSquares[j] > 1e-12) {
                    normalized[i][j] = decisionMatrix[i][j] / sqrtSumSquares[j];
                } else {
                    normalized[i][j] = 0.0;
                }
            }
        }

        return normalized;
    }

    /**
     * 应用权重
     */
    private double[][] applyWeights(double[][] normalizedMatrix) {
        int numCandidates = normalizedMatrix.length;
        int numObjectives = normalizedMatrix[0].length;

        double[][] weighted = new double[numCandidates][numObjectives];
        for (int i = 0; i < numCandidates; i++) {
            for (int j = 0; j < numObjectives; j++) {
                weighted[i][j] = normalizedMatrix[i][j] * weights[j];
            }
        }

        return weighted;
    }

    /**
     * 计算理想解
     */
    private double[] computeIdealSolution(double[][] weightedMatrix) {
        int numCandidates = weightedMatrix.length;
        int numObjectives = weightedMatrix[0].length;

        double[] ideal = new double[numObjectives];
        for (int j = 0; j < numObjectives; j++) {
            double bestValue = isMaximization != null && j < isMaximization.length && isMaximization[j]
                    ? -Double.MAX_VALUE : Double.MAX_VALUE;
            for (int i = 0; i < numCandidates; i++) {
                if (isMaximization != null && j < isMaximization.length && isMaximization[j]) {
                    bestValue = Math.max(bestValue, weightedMatrix[i][j]);
                } else {
                    bestValue = Math.min(bestValue, weightedMatrix[i][j]);
                }
            }
            ideal[j] = bestValue;
        }

        return ideal;
    }

    /**
     * 计算负理想解
     */
    private double[] computeNegativeIdealSolution(double[][] weightedMatrix) {
        int numCandidates = weightedMatrix.length;
        int numObjectives = weightedMatrix[0].length;

        double[] negativeIdeal = new double[numObjectives];
        for (int j = 0; j < numObjectives; j++) {
            double worstValue = isMaximization != null && j < isMaximization.length && isMaximization[j]
                    ? Double.MAX_VALUE : -Double.MAX_VALUE;
            for (int i = 0; i < numCandidates; i++) {
                if (isMaximization != null && j < isMaximization.length && isMaximization[j]) {
                    worstValue = Math.min(worstValue, weightedMatrix[i][j]);
                } else {
                    worstValue = Math.max(worstValue, weightedMatrix[i][j]);
                }
            }
            negativeIdeal[j] = worstValue;
        }

        return negativeIdeal;
    }

    /**
     * 计算距离
     */
    private double[] computeDistances(double[][] weightedMatrix, double[] reference) {
        int numCandidates = weightedMatrix.length;
        int numObjectives = weightedMatrix[0].length;

        double[] distances = new double[numCandidates];
        for (int i = 0; i < numCandidates; i++) {
            double sum = 0.0;
            for (int j = 0; j < numObjectives; j++) {
                double diff = weightedMatrix[i][j] - reference[j];
                sum += diff * diff;
            }
            distances[i] = Math.sqrt(sum);
        }

        return distances;
    }

    /**
     * 计算相对贴近度
     */
    private double[] computeClosenessRatios(double[] distancesToIdeal, double[] distancesToNegativeIdeal) {
        int numCandidates = distancesToIdeal.length;
        double[] ratios = new double[numCandidates];

        for (int i = 0; i < numCandidates; i++) {
            double sum = distancesToIdeal[i] + distancesToNegativeIdeal[i];
            if (sum > 1e-12) {
                ratios[i] = distancesToNegativeIdeal[i] / sum;
            } else {
                ratios[i] = 0.0;
            }
        }

        return ratios;
    }

    /**
     * 找到最佳解
     */
    private int findBestSolution(double[] closenessRatios) {
        int bestIndex = 0;
        double maxRatio = closenessRatios[0];

        for (int i = 1; i < closenessRatios.length; i++) {
            if (closenessRatios[i] > maxRatio) {
                maxRatio = closenessRatios[i];
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    /**
     * 判断两个解是否相同
     */
    private boolean isSameSolution(IVector sol1, IVector sol2) {
        if (sol1.length() != sol2.length()) {
            return false;
        }
        for (int i = 0; i < sol1.length(); i++) {
            if (Math.abs((Double) sol1.get(i) - (Double) sol2.get(i)) > 1e-8) {
                return false;
            }
        }
        return true;
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
     * 初始化默认权重
     */
    private double[] initializeDefaultWeights(int numObjectives) {
        double[] defaultWeights = new double[numObjectives];
        for (int i = 0; i < numObjectives; i++) {
            defaultWeights[i] = 1.0 / numObjectives;
        }
        return defaultWeights;
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
     * 设置权重向量
     *
     * @param weights 权重向量
     * @return this
     */
    public RereTopsis setWeights(double[] weights) {
        Objects.requireNonNull(weights, "参数不能为 null");
        this.weights = weights.clone();
        return this;
    }

    /**
     * 设置最大化目标标志
     *
     * @param isMaximization 是否最大化数组
     * @return this
     */
    public RereTopsis setMaximizationFlags(boolean[] isMaximization) {
        this.isMaximization = isMaximization.clone();
        return this;
    }

    /**
     * 设置底层求解器
     *
     * @param baseSolver 底层求解器
     * @return this
     */
    public RereTopsis setBaseSolver(ILinProgSolver baseSolver) {
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
        return "TOPSIS法基于理想解和负理想解的距离选择最佳方案，距离理想解越近且负理想解越远越好。";
    }

    @Override
    public MclpSolverType getSolverType() {
        return solverType;
    }

    @Override
    public String toString() {
        return "RereTopsis{" +
                "weights=" + (weights != null ? java.util.Arrays.toString(weights) : "null") +
                '}';
    }
}
