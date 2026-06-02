package com.yishape.lab.math.optimize.mcqp;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IQpSolver;
import com.yishape.lab.math.optimize.OptResult;

import java.util.*;

/**
 * 交互式STEM方法求解多目标二次规划
 * Interactive STEM Method for Multi-Criteria Quadratic Programming
 *
 * <p>STEM是一种交互式多目标优化方法，通过决策者逐步提供偏好信息来引导搜索满意解。</p>
 *
 * @author lteb2
 * @see IMcqpSolver
 */
public class RereInteractiveQp implements IMcqpSolver {

    /** 默认最大迭代次数 / Default maximum iterations */
    private static final int DEFAULT_MAX_ITERATIONS = 10;

    /** 底层单目标二次规划求解器 / Underlying single-objective QP solver */
    private IQpSolver baseSolver;

    /** 最大迭代次数 / Maximum iterations */
    private int maxIterations;

    /** 收敛容差 / Convergence tolerance */
    private double tolerance;

    /** 初始参考点 / Initial reference point */
    private double[] initialReferencePoint;

    /** 交互回调接口 / Interactive callback interface */
    private transient DecisionMakerCallback callback;

    /** 求解器类型 / Solver type */
    private final McqpSolverType solverType = McqpSolverType.Interactive;

    /** 求解器名称 / Solver name */
    private final String solverName = "交互式STEM法(QP)";

    /**
     * 决策者回调接口
     */
    public interface DecisionMakerCallback {
        DecisionMakerResponse provideFeedback(IVector currentSolution, double[] objectiveValues, int iteration);
    }

    /**
     * 决策者响应
     */
    public static class DecisionMakerResponse {
        private final double[] aspirationLevels;
        private final boolean satisfied;
        private final String comment;

        public DecisionMakerResponse(double[] aspirationLevels, boolean satisfied, String comment) {
            this.aspirationLevels = aspirationLevels;
            this.satisfied = satisfied;
            this.comment = comment;
        }

        public double[] getAspirationLevels() { return aspirationLevels; }
        public boolean isSatisfied() { return satisfied; }
        public String getComment() { return comment; }
    }

    public RereInteractiveQp() {
        this.baseSolver = IQpSolver.of();
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
        this.tolerance = 1e-6;
    }

    public RereInteractiveQp(int maxIterations) {
        this.baseSolver = IQpSolver.of();
        this.maxIterations = maxIterations;
        this.tolerance = 1e-6;
    }

    public RereInteractiveQp(int maxIterations, IQpSolver baseSolver) {
        this.baseSolver = baseSolver;
        this.maxIterations = maxIterations;
        this.tolerance = 1e-6;
    }

    @Override
    public McqpResult solve(IMatrix[] Q, IVector[] c, IMatrix A_ub, IVector b_ub,
                            IMatrix A_eq, IVector b_eq, IVector initX) {
        long startTime = System.currentTimeMillis();

        if (Q == null || Q.length == 0 || c == null || c.length == 0) {
            throw new IllegalArgumentException("Q和c不能为空");
        }
        if (baseSolver == null) throw new IllegalStateException("baseSolver 未设置");
        if (callback == null) {
            throw new IllegalStateException("需要设置DecisionMakerCallback才能使用交互式方法");
        }

        int numObjectives = Q.length;
        int numVariables = c[0].length();

        // 计算理想点
        double[] idealPoint = computeIdealPoint(Q, c, A_ub, b_ub, A_eq, b_eq);
        double[] nadirPoint = computeNadirPoint(Q, c, A_ub, b_ub, A_eq, b_eq);

        // 初始化参考点为理想点
        double[] referencePoint = initialReferencePoint != null ? initialReferencePoint.clone() : idealPoint.clone();

        List<IVector> allSolutions = new ArrayList<>();
        List<double[]> allObjectiveValues = new ArrayList<>();
        int totalIterations = 0;

        for (int iter = 0; iter < maxIterations; iter++) {
            // 使用参考点方法求解
            IVector currentX = solveWithReferencePoint(Q, c, A_ub, b_ub, A_eq, b_eq, referencePoint, initX);

            if (currentX == null) break;

            allSolutions.add(currentX);
            double[] objVals = new double[numObjectives];
            for (int i = 0; i < numObjectives; i++) {
                objVals[i] = computeQuadraticObjective(Q[i], c[i], currentX);
            }
            allObjectiveValues.add(objVals);

            // 获取决策者反馈
            DecisionMakerResponse response = callback.provideFeedback(currentX, objVals, iter);

            if (response.isSatisfied()) {
                return buildResult(allSolutions, allObjectiveValues, Q, c, numVariables,
                        A_ub, b_ub, A_eq, b_eq, idealPoint, nadirPoint, totalIterations, startTime, true, "决策者满意");
            }

            // 更新参考点
            if (response.getAspirationLevels() != null) {
                referencePoint = response.getAspirationLevels();
            }
        }

        return buildResult(allSolutions, allObjectiveValues, Q, c, numVariables,
                A_ub, b_ub, A_eq, b_eq, idealPoint, nadirPoint, totalIterations, startTime, false, "达到最大迭代次数");
    }

    private IVector solveWithReferencePoint(IMatrix[] Q, IVector[] c, IMatrix A_ub, IVector b_ub,
                                          IMatrix A_eq, IVector b_eq, double[] referencePoint, IVector initX) {
        int numObjectives = Q.length;
        int numVariables = c[0].length();

        // 参考点方法：minimize max_i (w_i * (f_i(x) - r_i))
        // 转化为：minimize t
        // subject to: w_i * (f_i(x) - r_i) <= t for all i

        // 扩展变量：[x; t]
        int totalVars = numVariables + 1;
        double[] weights = new double[numObjectives];
        Arrays.fill(weights, 1.0 / numObjectives);

        // 构造扩展问题
        // 目标函数：t (最小化)
        double[] cExtData = new double[totalVars];
        cExtData[numVariables] = 1.0; // t的系数为1
        IVector extendedC = Linalg.vector(cExtData);

        // Q矩阵（扩展）：t没有二次项
        double[][] qExtData = new double[totalVars][totalVars];
        for (int i = 0; i < numVariables; i++) {
            for (int j = 0; j < numVariables; j++) {
                qExtData[i][j] = 0.0; // 线性化处理
            }
        }
        IMatrix extendedQ = Linalg.matrix(qExtData);

        // 不等式约束
        List<double[]> aubRows = new ArrayList<>();
        List<Double> bubValues = new ArrayList<>();

        // 原不等式约束
        if (A_ub != null && b_ub != null) {
            for (int i = 0; i < A_ub.rows(); i++) {
                double[] row = new double[totalVars];
                for (int j = 0; j < numVariables; j++) {
                    row[j] = ((Number) A_ub.get(i, j)).doubleValue();
                }
                row[numVariables] = 0.0;
                aubRows.add(row);
                bubValues.add(((Number) b_ub.get(i)).doubleValue());
            }
        }

        // 参考点约束：w_i * (c_i^T * x - r_i) <= t
        for (int i = 0; i < numObjectives; i++) {
            double[] row = new double[totalVars];
            for (int j = 0; j < numVariables; j++) {
                row[j] = weights[i] * ((Number) c[i].get(j)).doubleValue();
            }
            row[numVariables] = -1.0; // -t
            aubRows.add(row);
            bubValues.add(weights[i] * referencePoint[i]);
        }

        IMatrix extendedAub = aubRows.isEmpty() ? null : Linalg.matrix(aubRows.toArray(new double[0][]));
        IVector extendedBub = bubValues.isEmpty() ? null : Linalg.vector(bubValues.stream().mapToDouble(Double::doubleValue).toArray());

        // 等式约束
        IMatrix extendedAeq = buildExtendedAeq(A_eq, numVariables);
        IVector extendedBeq = b_eq != null ? b_eq.copy() : null;

        // 初始点
        double[] initData = new double[totalVars];
        if (initX != null) {
            for (int i = 0; i < numVariables && i < initX.length(); i++) {
                initData[i] = ((Number) initX.get(i)).doubleValue();
            }
        }
        initData[numVariables] = 0.0; // t初始为0
        IVector extendedInitX = Linalg.vector(initData);

        OptResult result = baseSolver.solve(extendedQ, extendedC, extendedAub, extendedBub, extendedAeq, extendedBeq, extendedInitX);

        if (!result.isConverged() || result.getOptimalPoint() == null) return null;

        // 提取原变量
        double[] xData = new double[numVariables];
        for (int i = 0; i < numVariables; i++) {
            xData[i] = ((Number) result.getOptimalPoint().get(i)).doubleValue();
        }
        return Linalg.vector(xData);
    }

    private IMatrix buildExtendedAeq(IMatrix A_eq, int numVariables) {
        if (A_eq == null) return null;
        int m = A_eq.rows();
        double[][] data = new double[m][numVariables + 1];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < numVariables; j++) {
                data[i][j] = ((Number) A_eq.get(i, j)).doubleValue();
            }
            data[i][numVariables] = 0.0;
        }
        return Linalg.matrix(data);
    }

    private McqpResult buildResult(List<IVector> solutions, List<double[]> objectiveValues,
                                  IMatrix[] Q, IVector[] c, int numVariables,
                                  IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq,
                                  double[] idealPoint, double[] nadirPoint,
                                  int totalIterations, long startTime,
                                  boolean converged, String reason) {

        return new McqpResult.Builder()
            .solutions(solutions)
            .objectiveValues(objectiveValues)
            .numObjectives(Q.length)
            .numConstraints((A_ub != null ? A_ub.rows() : 0) + (A_eq != null ? A_eq.rows() : 0))
            .numVariables(numVariables)
            .solverType(solverType)
            .solverName(solverName)
            .converged(converged)
            .convergenceReason(reason)
            .totalIterations(totalIterations)
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .idealPoint(idealPoint)
            .nadirPoint(nadirPoint)
            .build();
    }

    private double computeQuadraticObjective(IMatrix Q, IVector c, IVector x) {
        IVector qx = Q.mmul(x);
        return 0.5 * x.dotValue(qx) + c.dotValue(x);
    }

    private double[] computeIdealPoint(IMatrix[] Q, IVector[] c, IMatrix A_ub, IVector b_ub,
                                     IMatrix A_eq, IVector b_eq) {
        double[] ideal = new double[Q.length];
        for (int i = 0; i < Q.length; i++) {
            OptResult result = baseSolver.solve(Q[i], c[i], A_ub, b_ub, A_eq, b_eq, null);
            ideal[i] = result.getOptimalValue();
        }
        return ideal;
    }

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

    public RereInteractiveQp setCallback(DecisionMakerCallback callback) {
        this.callback = callback;
        return this;
    }

    public RereInteractiveQp setMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
        return this;
    }

    public RereInteractiveQp setTolerance(double tolerance) {
        this.tolerance = tolerance;
        return this;
    }

    public RereInteractiveQp setInitialReferencePoint(double[] referencePoint) {
        this.initialReferencePoint = referencePoint.clone();
        return this;
    }

    public RereInteractiveQp setBaseSolver(IQpSolver baseSolver) {
        this.baseSolver = baseSolver;
        return this;
    }

    @Override public String getName() { return solverName; }
    @Override public String getDescription() { return "交互式STEM法通过决策者反馈逐步引导搜索满意解。"; }
    @Override public McqpSolverType getSolverType() { return solverType; }
}
