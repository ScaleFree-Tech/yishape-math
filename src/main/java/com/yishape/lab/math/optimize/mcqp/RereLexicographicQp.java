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
 * 字典序法求解多目标二次规划
 * Lexicographic Method for Multi-Criteria Quadratic Programming
 *
 * <p>该方法按照目标的优先级顺序依次求解。
 * 适用于目标有明确优先级顺序的场景。
 *
 * @author lteb2
 * @see IMcqpSolver
 */
public class RereLexicographicQp implements IMcqpSolver {

    /** 默认目标容差（相对于最优值的百分比）/ Default objective tolerance */
    private static final double DEFAULT_TOLERANCE = 1e-6;

    /** 底层单目标二次规划求解器 / Underlying single-objective QP solver */
    private IQpSolver baseSolver;

    /** 目标优先级顺序（默认为0,1,2,...）/ Objective priority order */
    private int[] priorityOrder;

    /** 目标的容差范围 / Objective tolerance range */
    private double tolerance;

    /** 求解器类型 / Solver type */
    private final McqpSolverType solverType = McqpSolverType.Lexicographic;

    /** 求解器名称 / Solver name */
    private final String solverName = "字典序法(QP)";

    /**
     * 默认构造函数
     */
    public RereLexicographicQp() {
        this.baseSolver = IQpSolver.of();
        this.tolerance = DEFAULT_TOLERANCE;
    }

    /**
     * 使用指定优先级顺序的构造函数
     *
     * @param priorityOrder 优先级顺序数组，例如[0,2,1]表示先优化目标0，再优化目标2，最后目标1
     */
    public RereLexicographicQp(int[] priorityOrder) {
        this.baseSolver = IQpSolver.of();
        this.priorityOrder = priorityOrder.clone();
        this.tolerance = DEFAULT_TOLERANCE;
    }

    /**
     * 使用指定优先级顺序和底层求解器的构造函数
     *
     * @param priorityOrder 优先级顺序数组
     * @param baseSolver 底层求解器
     */
    public RereLexicographicQp(int[] priorityOrder, IQpSolver baseSolver) {
        this.baseSolver = baseSolver;
        this.priorityOrder = priorityOrder.clone();
        this.tolerance = DEFAULT_TOLERANCE;
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

        // 初始化优先级顺序（如果未设置）
        if (priorityOrder == null || priorityOrder.length != numObjectives) {
            priorityOrder = new int[numObjectives];
            for (int i = 0; i < numObjectives; i++) {
                priorityOrder[i] = i;
            }
        }

        int numOriginalConstraints = (A_ub != null ? A_ub.rows() : 0) + (A_eq != null ? A_eq.rows() : 0);

        // 逐层求解
        IVector currentX = initX != null ? initX.copy() : Linalg.ones(numVariables);
        double[] optimalValues = new double[numObjectives];
        int totalIterations = 0;
        List<OptResult> individualResults = new ArrayList<>();

        for (int p = 0; p < numObjectives; p++) {
            int idx = priorityOrder[p];
            if (idx < 0 || idx >= numObjectives) {
                throw new IllegalArgumentException(
                    "priorityOrder[" + p + "] = " + idx + " is out of range [0, " + numObjectives + ")");
            }

            // 求解当前优先级目标
            OptResult result = baseSolver.solve(Q[idx], c[idx], A_ub, b_ub, A_eq, b_eq, currentX);

            if (result.isConverged() && result.getOptimalPoint() != null) {
                currentX = result.getOptimalPoint();
                optimalValues[idx] = result.getOptimalValue();
                totalIterations += result.getIterations();
                individualResults.add(result);
            } else {
                individualResults.add(null);
            }
        }

        // 构建结果
        List<IVector> solutions = new ArrayList<>();
        List<double[]> objectiveValues = new ArrayList<>();
        solutions.add(currentX);

        double[] finalObjVals = new double[numObjectives];
        for (int i = 0; i < numObjectives; i++) {
            finalObjVals[i] = computeQuadraticObjective(Q[i], c[i], currentX);
        }
        objectiveValues.add(finalObjVals);

        // 计算理想点和反理想点
        double[] idealPoint = computeIdealPoint(Q, c, A_ub, b_ub, A_eq, b_eq);
        double[] nadirPoint = computeNadirPoint(Q, c, A_ub, b_ub, A_eq, b_eq);

        return new McqpResult.Builder()
            .solutions(solutions)
            .objectiveValues(objectiveValues)
            .individualResults(individualResults)
            .numObjectives(numObjectives)
            .numConstraints(numOriginalConstraints)
            .numVariables(numVariables)
            .priorityOrder(priorityOrder)
            .solverType(solverType)
            .solverName(solverName)
            .converged(true)
            .convergenceReason("字典序求解完成")
            .totalIterations(totalIterations)
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .idealPoint(idealPoint)
            .nadirPoint(nadirPoint)
            .build();
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
     * 计算理想点
     */
    private double[] computeIdealPoint(IMatrix[] Q, IVector[] c, IMatrix A_ub, IVector b_ub,
                                     IMatrix A_eq, IVector b_eq) {
        double[] ideal = new double[Q.length];
        for (int i = 0; i < Q.length; i++) {
            OptResult result = baseSolver.solve(Q[i], c[i], A_ub, b_ub, A_eq, b_eq, null);
            ideal[i] = result.isConverged() ? result.getOptimalValue() : Double.NaN;
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
            nadir[i] = result.isConverged() ? -result.getOptimalValue() : Double.NaN;
        }
        return nadir;
    }

    // ==================== 配置方法 ====================

    public RereLexicographicQp setPriorityOrder(int[] priorityOrder) {
        Objects.requireNonNull(priorityOrder, "参数不能为 null");
        this.priorityOrder = priorityOrder.clone();
        return this;
    }

    public RereLexicographicQp setTolerance(double tolerance) {
        this.tolerance = tolerance;
        return this;
    }

    public RereLexicographicQp setBaseSolver(IQpSolver baseSolver) {
        Objects.requireNonNull(baseSolver, "baseSolver 不能为 null");
        this.baseSolver = baseSolver;
        return this;
    }

    // ==================== IMcqpSolver 接口实现 ====================

    @Override
    public String getName() {
        return solverName;
    }

    @Override
    public String getDescription() {
        return "字典序法按优先级顺序依次优化各目标。";
    }

    @Override
    public McqpSolverType getSolverType() {
        return solverType;
    }

    @Override
    public String toString() {
        return "RereLexicographicQp{" +
                "priorityOrder=" + (priorityOrder != null ? java.util.Arrays.toString(priorityOrder) : "null") +
                '}';
    }
}
