package com.yishape.lab.math.optimize.mcqp;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IQpSolver;
import com.yishape.lab.math.optimize.OptResult;

import java.util.Objects;

/**
 * 层次分析法（AHP）求解多目标二次规划
 * Analytic Hierarchy Process (AHP) for Multi-Criteria Quadratic Programming
 *
 * <p>AHP通过构建成对比较矩阵来计算各目标的权重，
 * 然后使用计算得到的权重进行加权求解。</p>
 *
 * @author lteb2
 * @see IMcqpSolver
 */
public class RereAhpQp implements IMcqpSolver {

    /** 成对比较矩阵 / Pairwise comparison matrix */
    private IMatrix comparisonMatrix;

    /** 计算得到的权重向量 / Computed weight vector */
    private double[] weights;

    /** 一致性比率 / Consistency ratio */
    private double consistencyRatio;

    /** 底层单目标二次规划求解器 / Underlying single-objective QP solver */
    private IQpSolver baseSolver;

    /** 求解器类型 / Solver type */
    private final McqpSolverType solverType = McqpSolverType.Ahp;

    /** 求解器名称 / Solver name */
    private final String solverName = "层次分析法(QP)";

    /** Saaty随机一致性指标表 / Random consistency index table */
    private static final double[] RANDOM_INDEX = {
        0.0, 0.0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49
    };

    public RereAhpQp() {
        this.baseSolver = IQpSolver.of();
    }

    public RereAhpQp(IMatrix comparisonMatrix) {
        this.comparisonMatrix = comparisonMatrix;
        this.baseSolver = IQpSolver.of();
        computeWeights();
    }

    public RereAhpQp(IMatrix comparisonMatrix, IQpSolver baseSolver) {
        this.comparisonMatrix = comparisonMatrix;
        this.baseSolver = baseSolver;
        computeWeights();
    }

    @Override
    public McqpResult solve(IMatrix[] Q, IVector[] c, IMatrix A_ub, IVector b_ub,
                            IMatrix A_eq, IVector b_eq, IVector initX) {
        long startTime = System.currentTimeMillis();

        if (Q == null || Q.length == 0 || c == null || c.length == 0) {
            throw new IllegalArgumentException("Q和c不能为空");
        }
        if (baseSolver == null) throw new IllegalStateException("baseSolver 未设置");

        int numObjectives = Q.length;
        int numVariables = c[0].length();

        // 如果没有提供比较矩阵，使用等权重
        if (weights == null || weights.length != numObjectives) {
            weights = new double[numObjectives];
            for (int i = 0; i < numObjectives; i++) {
                weights[i] = 1.0 / numObjectives;
            }
        }

        // 构造加权目标函数
        IMatrix weightedQ = computeWeightedQ(Q, weights);
        IVector weightedC = computeWeightedC(c, weights);

        // 求解
        OptResult result = baseSolver.solve(weightedQ, weightedC, A_ub, b_ub, A_eq, b_eq, initX);

        // 构建结果
        IVector optimalX = result.getOptimalPoint();
        double[] objVals = new double[numObjectives];
        for (int i = 0; i < numObjectives; i++) {
            objVals[i] = computeQuadraticObjective(Q[i], c[i], optimalX);
        }

        double[] idealPoint = computeIdealPoint(Q, c, A_ub, b_ub, A_eq, b_eq);
        double[] nadirPoint = computeNadirPoint(Q, c, A_ub, b_ub, A_eq, b_eq);

        return new McqpResult.Builder()
            .solutions(java.util.Collections.singletonList(optimalX))
            .objectiveValues(java.util.Collections.singletonList(objVals))
            .individualResults(java.util.Collections.singletonList(result))
            .numObjectives(numObjectives)
            .numConstraints((A_ub != null ? A_ub.rows() : 0) + (A_eq != null ? A_eq.rows() : 0))
            .numVariables(numVariables)
            .weights(weights)
            .solverType(solverType)
            .solverName(solverName)
            .converged(result.isConverged())
            .convergenceReason(result.getConvergenceReason())
            .totalIterations(result.getIterations())
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .idealPoint(idealPoint)
            .nadirPoint(nadirPoint)
            .build();
    }

    /**
     * 从比较矩阵计算权重
     */
    private void computeWeights() {
        if (comparisonMatrix == null) return;

        int n = comparisonMatrix.rows();
        double[] eigenvector = computePrincipalEigenvector(comparisonMatrix);
        weights = normalizeWeights(eigenvector);
        consistencyRatio = computeConsistencyRatio(comparisonMatrix, weights);
    }

    /**
     * 幂迭代法计算主特征向量
     */
    private double[] computePrincipalEigenvector(IMatrix matrix) {
        int n = matrix.rows();
        double[] eigenvector = new double[n];
        for (int i = 0; i < n; i++) eigenvector[i] = 1.0 / n;

        double tolerance = 1e-10;
        int maxIterations = 100;

        IVector eigenVec = Linalg.vector(eigenvector);
        for (int iter = 0; iter < maxIterations; iter++) {
            IVector newEigenVec = matrix.mmul(eigenVec);
            double norm = Math.sqrt(newEigenVec.dotValue(newEigenVec));
            if (norm < 1e-12) break;
            newEigenVec = newEigenVec.multiplyByScalar(1.0 / norm);

            double maxDiff = 0.0;
            for (int i = 0; i < n; i++) {
                maxDiff = Math.max(maxDiff,
                    Math.abs(((Number) newEigenVec.get(i)).doubleValue() - ((Number) eigenVec.get(i)).doubleValue()));
            }
            eigenVec = newEigenVec;
            if (maxDiff < tolerance) break;
        }

        double[] result = new double[n];
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            result[i] = ((Number) eigenVec.get(i)).doubleValue();
            sum += result[i];
        }
        if (sum > 1e-12) {
            for (int i = 0; i < n; i++) result[i] /= sum;
        }
        return result;
    }

    private double[] normalizeWeights(double[] eigenvector) {
        double sum = 0.0;
        for (double w : eigenvector) sum += w;
        if (Math.abs(sum) < 1e-12) {
            double[] equalWeights = new double[eigenvector.length];
            for (int i = 0; i < equalWeights.length; i++) equalWeights[i] = 1.0 / eigenvector.length;
            return equalWeights;
        }
        double[] weights = new double[eigenvector.length];
        for (int i = 0; i < eigenvector.length; i++) weights[i] = eigenvector[i] / sum;
        return weights;
    }

    private double computeConsistencyRatio(IMatrix matrix, double[] weights) {
        int n = matrix.rows();
        IVector weightVec = Linalg.vector(weights);
        IVector rowDotWeights = matrix.mmul(weightVec);
        double maxEigenvalue = 0.0;
        for (int i = 0; i < n; i++) {
            maxEigenvalue += ((Number) rowDotWeights.get(i)).doubleValue() / weights[i];
        }
        maxEigenvalue /= n;

        double consistencyIndex = (maxEigenvalue - n) / (n - 1);
        int riIndex = Math.min(n, RANDOM_INDEX.length - 1);
        return RANDOM_INDEX[riIndex] > 0 ? consistencyIndex / RANDOM_INDEX[riIndex] : 0.0;
    }

    private IMatrix computeWeightedQ(IMatrix[] Q, double[] weights) {
        int n = Q[0].rows();
        double[][] weightedQData = new double[n][n];
        for (int k = 0; k < Q.length; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    weightedQData[i][j] += weights[k] * ((Number) Q[k].get(i, j)).doubleValue();
                }
            }
        }
        return Linalg.matrix(weightedQData);
    }

    private IVector computeWeightedC(IVector[] c, double[] weights) {
        IVector weightedC = Linalg.zeros(c[0].length());
        for (int i = 0; i < c.length; i++) {
            weightedC = weightedC.add(c[i].multiplyByScalar(weights[i]));
        }
        return weightedC;
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

    public RereAhpQp setComparisonMatrix(IMatrix comparisonMatrix) {
        this.comparisonMatrix = comparisonMatrix;
        computeWeights();
        return this;
    }

    public RereAhpQp setWeights(double[] weights) {
        this.weights = weights.clone();
        return this;
    }

    public RereAhpQp setBaseSolver(IQpSolver baseSolver) {
        this.baseSolver = Objects.requireNonNull(baseSolver);
        return this;
    }

    public double getConsistencyRatio() { return consistencyRatio; }

    @Override public String getName() { return solverName; }
    @Override public String getDescription() { return "层次分析法通过成对比较计算权重来求解多目标问题。"; }
    @Override public McqpSolverType getSolverType() { return solverType; }
}
