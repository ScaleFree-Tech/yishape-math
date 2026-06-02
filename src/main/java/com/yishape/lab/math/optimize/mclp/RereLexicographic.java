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
 * 字典序法求解多目标线性规划
 * Lexicographic Method for Multi-Objective Linear Programming
 *
 * <p>该方法按照目标的优先级顺序依次求解：
 * 1. 首先优化最重要的目标，达到最优后记录最优值
 * 2. 在该最优值的约束下，优化第二重要目标
 * 3. 重复上述过程直到所有目标都被优化
 *
 * <p>数学形式：
 * minimize c_1^T * x
 * subject to constraints
 *       c_1^T * x = z_1* (第一目标最优值)
 *
 * minimize c_2^T * x
 * subject to constraints
 *       c_1^T * x = z_1*
 *       c_2^T * x = z_2* (第二目标最优值)
 * ...</p>
 *
 * <p>特点：
 * - 适用于目标有明确优先级顺序的场景
 * - 不需要权重设置
 * - 每个目标的容差范围可调
 * - 如果低优先级目标在容差范围内无法优化，会跳过</p>
 *
 * @author lteb2
 * @see IMclpSolver
 */
public class RereLexicographic implements IMclpSolver {

    /** 默认目标容差（相对于最优值的百分比）/ Default objective tolerance */
    private static final double DEFAULT_TOLERANCE = 1e-6;

    /** 底层单目标线性规划求解器 / Underlying single-objective LP solver */
    private ILinProgSolver baseSolver;

    /** 目标优先级顺序（默认为0,1,2,...）/ Objective priority order */
    private int[] priorityOrder;

    /** 目标的容差范围（epsilon约束）/ Objective tolerance range */
    private double tolerance;

    /** 求解器类型 / Solver type */
    private final MclpSolverType solverType = MclpSolverType.Lexicographic;

    /** 求解器名称 / Solver name */
    private final String solverName = "字典序法";

    /**
     * 默认构造函数
     */
    public RereLexicographic() {
        this.baseSolver = ILinProgSolver.of();
        this.tolerance = DEFAULT_TOLERANCE;
    }

    /**
     * 使用指定优先级顺序的构造函数
     *
     * @param priorityOrder 优先级顺序数组，例如[0,2,1]表示先优化目标0，再优化目标2，最后目标1
     */
    public RereLexicographic(int[] priorityOrder) {
        this.baseSolver = ILinProgSolver.of();
        this.priorityOrder = priorityOrder.clone();
        this.tolerance = DEFAULT_TOLERANCE;
    }

    /**
     * 使用指定优先级顺序和底层求解器的构造函数
     *
     * @param priorityOrder 优先级顺序数组
     * @param baseSolver 底层求解器
     */
    public RereLexicographic(int[] priorityOrder, ILinProgSolver baseSolver) {
        this.baseSolver = baseSolver;
        this.priorityOrder = priorityOrder.clone();
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

        // 初始化优先级顺序（如果未设置）
        if (priorityOrder == null || priorityOrder.length != numObjectives) {
            initializePriorityOrder(numObjectives);
        }

        // 记录各目标的最优值
        double[] optimalValues = new double[numObjectives];
        List<IVector> solutions = new ArrayList<>();
        List<double[]> objectiveValues = new ArrayList<>();
        List<OptResult> individualResults = new ArrayList<>();
        List<IVector> parameterHistory = new ArrayList<>();

        // 保存原始约束（用于第二遍重放）
        IMatrix originalAeq = A_eq;
        IVector originalBeq = b_eq;

        // 当前解向量
        IVector currentX = initX != null ? initX.copy() : Linalg.ones(numVariables);
        int totalIterations = 0;

        // 依次优化各目标，逐步添加松弛约束
        for (int p = 0; p < numObjectives; p++) {
            int objIdx = priorityOrder[p];

            OptResult result = baseSolver.solve(c[objIdx], A_ub, b_ub, A_eq, b_eq, currentX);

            if (!result.isConverged()) {
                continue;
            }

            IVector optimalX = result.getOptimalPoint();
            double optimalValue = result.getOptimalValue();
            optimalValues[objIdx] = optimalValue;
            totalIterations += result.getIterations();

            double relaxedValue = optimalValue + tolerance * Math.max(1.0, Math.abs(optimalValue));
            IMatrix newAeq = addEpsilonConstraint(A_eq, c[objIdx], relaxedValue);
            IVector newBeq = addEpsilonBound(b_eq, relaxedValue);

            A_eq = newAeq;
            b_eq = newBeq;

            currentX = optimalX;
            parameterHistory.add(optimalX.copy());
        }

        double[] finalObjVals = new double[numObjectives];
        for (int i = 0; i < numObjectives; i++) {
            finalObjVals[i] = c[i].innerProductValue(currentX);
        }

        IVector finalSolution = currentX;
        solutions.add(finalSolution);
        objectiveValues.add(finalObjVals);

        // 第二遍：使用原始约束重新构建，逐步添加松弛约束以记录中间结果
        IVector tempX = initX != null ? initX.copy() : Linalg.ones(numVariables);
        IMatrix tempAeq = originalAeq;
        IVector tempBeq = originalBeq;

        for (int p = 0; p < numObjectives; p++) {
            int objIdx = priorityOrder[p];

            OptResult result = baseSolver.solve(c[objIdx], A_ub, b_ub, tempAeq, tempBeq, tempX);

            if (result.isConverged()) {
                tempX = result.getOptimalPoint();

                double[] intermediateObjVals = new double[numObjectives];
                for (int i = 0; i < numObjectives; i++) {
                    intermediateObjVals[i] = c[i].innerProductValue(tempX);
                }

                if (p == 0 || !solutions.contains(tempX)) {
                    if (p > 0) {
                        solutions.add(tempX.copy());
                        objectiveValues.add(intermediateObjVals);
                    }
                }

                double relaxedValue = optimalValues[objIdx] + tolerance * Math.max(1.0, Math.abs(optimalValues[objIdx]));
                tempAeq = addEpsilonConstraint(tempAeq, c[objIdx], relaxedValue);
                tempBeq = addEpsilonBound(tempBeq, relaxedValue);
            }
        }

        // 计算理想点和反理想点
        double[] idealPoint = computeIdealPoint(c, A_ub, b_ub, A_eq, b_eq);
        double[] nadirPoint = computeNadirPoint(c, A_ub, b_ub, A_eq, b_eq);

        return new MclpResult.Builder()
            .solutions(solutions)
            .objectiveValues(objectiveValues)
            .numObjectives(numObjectives)
            .numConstraints((A_ub != null ? A_ub.rows() : 0) + (A_eq != null ? A_eq.rows() : 0))
            .numVariables(numVariables)
            .priorityOrder(priorityOrder)
            .solverType(solverType)
            .solverName(solverName)
            .converged(true)
            .convergenceReason("字典序法完成，所有优先级目标已依次优化")
            .totalIterations(totalIterations)
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .idealPoint(idealPoint)
            .nadirPoint(nadirPoint)
            .diversityMetric(0.0)
            .build();
    }

    /**
     * 添加epsilon约束到现有等式约束
     */
    private IMatrix addEpsilonConstraint(IMatrix existingAeq, IVector c, double optimalValue) {
        if (existingAeq == null) {
            return Linalg.matrix(new double[][]{c.toDoubleArray()});
        }
        // 添加一行 c^T * x <= optimalValue + epsilon
        int existingRows = existingAeq.rows();
        double[][] newData = new double[existingRows + 1][];
        for (int i = 0; i < existingRows; i++) {
            newData[i] = existingAeq.getRow(i).toDoubleArray();
        }
        newData[existingRows] = c.toDoubleArray();
        return Linalg.matrix(newData);
    }

    /**
     * 添加epsilon约束右端值
     */
    private IVector addEpsilonBound(IVector existingBeq, double optimalValue) {
        if (existingBeq == null) {
            return Linalg.vector(optimalValue);
        }
        int existingLen = existingBeq.length();
        double[] newData = new double[existingLen + 1];
        for (int i = 0; i < existingLen; i++) {
            newData[i] = (Double) existingBeq.get(i);
        }
        newData[existingLen] = optimalValue;
        return Linalg.vector(newData);
    }

    /**
     * 初始化默认优先级顺序
     */
    private void initializePriorityOrder(int numObjectives) {
        this.priorityOrder = new int[numObjectives];
        for (int i = 0; i < numObjectives; i++) {
            priorityOrder[i] = i;
        }
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
     * 设置优先级顺序
     *
     * @param priorityOrder 优先级顺序数组
     * @return this
     */
    public RereLexicographic setPriorityOrder(int[] priorityOrder) {
        Objects.requireNonNull(priorityOrder, "参数不能为 null");
        this.priorityOrder = priorityOrder.clone();
        return this;
    }

    /**
     * 设置容差
     *
     * @param tolerance 容差值（相对值）
     * @return this
     */
    public RereLexicographic setTolerance(double tolerance) {
        this.tolerance = tolerance;
        return this;
    }

    /**
     * 设置底层求解器
     *
     * @param baseSolver 底层求解器
     * @return this
     */
    public RereLexicographic setBaseSolver(ILinProgSolver baseSolver) {
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
        return "字典序法按优先级顺序依次优化各目标，适合目标有明确优先级的场景。";
    }

    @Override
    public MclpSolverType getSolverType() {
        return solverType;
    }

    @Override
    public String toString() {
        return "RereLexicographic{" +
                "priorityOrder=" + java.util.Arrays.toString(priorityOrder) +
                ", tolerance=" + tolerance +
                '}';
    }
}
