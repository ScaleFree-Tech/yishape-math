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
 * 层次分析法（AHP）求解多目标线性规划
 * Analytic Hierarchy Process (AHP) for Multi-Objective Linear Programming
 *
 * <p>AHP是一种多准则决策分析方法，通过构建成对比较矩阵来计算各目标的权重。
 * 该权重可以用于加权求和法等其他MCLP求解方法。
 *
 * <p>算法步骤：
 * 1. 构建成对比较矩阵A（n×n矩阵，a_ij表示目标i相对于目标j的重要性）
 * 2. 计算比较矩阵的特征向量作为权重
 * 3. 进行一致性检验（{@code CR < 0.1} 为可接受）
 * 4. 使用计算得到的权重进行加权求和求解
 *
 * <p>比较矩阵的标度（Saaty 1-9标度）：
 * 1：同样重要
 * 3：稍微重要
 * 5：明显重要
 * 7：强烈重要
 * 9：极端重要
 * 2,4,6,8：上述相邻判断的中间值
 *
 * @author lteb2
 * @see IMclpSolver
 */
public class RereAHP implements IMclpSolver {

    /** 成对比较矩阵 / Pairwise comparison matrix */
    private IMatrix comparisonMatrix;

    /** 计算得到的权重向量 / Computed weight vector */
    private double[] weights;

    /** 一致性比率 / Consistency ratio */
    private double consistencyRatio;

    /** 一致性指标 / Consistency index */
    private double consistencyIndex;

    /** 底层单目标线性规划求解器 / Underlying single-objective LP solver */
    private ILinProgSolver baseSolver;

    /** 求解器类型 / Solver type */
    private final MclpSolverType solverType = MclpSolverType.Ahp;

    /** 求解器名称 / Solver name */
    private final String solverName = "层次分析法(AHP)";

    /** Saaty随机一致性指标表 / Saaty's random consistency index table */
    private static final double[] RANDOM_INDEX = {
        0.0,    // n=1 (不存在)
        0.0,    // n=2
        0.58,   // n=3
        0.90,   // n=4
        1.12,   // n=5
        1.24,   // n=6
        1.32,   // n=7
        1.41,   // n=8
        1.45,   // n=9
        1.49    // n=10
    };

    /**
     * 默认构造函数
     */
    public RereAHP() {
        this.baseSolver = ILinProgSolver.of();
    }

    /**
     * 使用比较矩阵的构造函数
     *
     * @param comparisonMatrix 成对比较矩阵（n×n）
     */
    public RereAHP(IMatrix comparisonMatrix) {
        this.comparisonMatrix = comparisonMatrix;
        this.baseSolver = ILinProgSolver.of();
        computeWeights();
    }

    /**
     * 使用比较矩阵和底层求解器的构造函数
     *
     * @param comparisonMatrix 成对比较矩阵
     * @param baseSolver 底层求解器
     */
    public RereAHP(IMatrix comparisonMatrix, ILinProgSolver baseSolver) {
        this.comparisonMatrix = comparisonMatrix;
        this.baseSolver = baseSolver;
        computeWeights();
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

        // 确保权重存在且数量与目标数量一致（预设置权重优先于默认等权重）
        if (weights == null || weights.length != numObjectives) {
            this.weights = new double[numObjectives];
            for (int i = 0; i < numObjectives; i++) {
                weights[i] = 1.0 / numObjectives;
            }
        }

        // 使用AHP权重进行加权求和求解
        IVector weightedC = computeWeightedObjective(c, weights);
        OptResult result = baseSolver.solve(weightedC, A_ub, b_ub, A_eq, b_eq, initX);

        // 构建解
        IVector optimalX = result.getOptimalPoint();
        double[] objVals = new double[numObjectives];
        for (int i = 0; i < numObjectives; i++) {
            objVals[i] = c[i].innerProductValue(optimalX);
        }

        // 计算理想点和反理想点
        double[] idealPoint = computeIdealPoint(c, A_ub, b_ub, A_eq, b_eq);
        double[] nadirPoint = computeNadirPoint(c, A_ub, b_ub, A_eq, b_eq);

        // 构建结果
        List<IVector> solutions = new ArrayList<>();
        List<double[]> objectiveValues = new ArrayList<>();
        List<OptResult> individualResults = new ArrayList<>();

        solutions.add(optimalX);
        objectiveValues.add(objVals);
        individualResults.add(result);

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
            .converged(result.isConverged())
            .convergenceReason(result.getConvergenceReason() + " | AHP权重已计算，一致性比率CR=" + String.format("%.4f", consistencyRatio))
            .totalIterations(result.getIterations())
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .idealPoint(idealPoint)
            .nadirPoint(nadirPoint)
            .diversityMetric(1.0 - consistencyRatio) // 一致性越好，多样性指标越高
            .build();
    }

    /**
     * 计算AHP权重向量
     * <p>使用特征值方法：
     * 1. 计算比较矩阵的最大特征值和对应特征向量
     * 2. 归一化特征向量得到权重
     */
    public void computeWeights() {
        if (comparisonMatrix == null) {
            return;
        }

        int n = comparisonMatrix.rows();

        // 计算特征值和特征向量
        // 使用幂迭代法计算主特征向量
        double[] principalEigenvector = computePrincipalEigenvector(comparisonMatrix);

        // 归一化得到权重
        this.weights = normalizeWeights(principalEigenvector);

        // 计算一致性指标
        double maxEigenvalue = computeMaxEigenvalue(comparisonMatrix, weights);
        this.consistencyIndex = (maxEigenvalue - n) / (n - 1);

        // 计算一致性比率 (2×2 矩阵完全一致，RANDOM_INDEX[2]=0)
        if (n <= 2) {
            this.consistencyRatio = 0.0;
        } else {
            int riIndex = Math.min(n, RANDOM_INDEX.length - 1);
            this.consistencyRatio = this.consistencyIndex / RANDOM_INDEX[riIndex];
        }
    }

    /**
     * 使用幂迭代法计算主特征向量
     */
    private double[] computePrincipalEigenvector(IMatrix matrix) {
        int n = matrix.rows();
        double[] eigenvector = new double[n];

        // 初始化为等权重向量
        for (int i = 0; i < n; i++) {
            eigenvector[i] = 1.0 / n;
        }

        // 幂迭代
        double tolerance = 1e-10;
        int maxIterations = 100;

        IVector eigenVec = Linalg.vector(eigenvector);
        for (int iter = 0; iter < maxIterations; iter++) {
            // 使用库的 mmul API 计算 matrix * eigenvector
            IVector newEigenVec = matrix.mmul(eigenVec);

            // 归一化
            double norm = Math.sqrt(newEigenVec.dotValue(newEigenVec));
            if (norm < 1e-12) {
                break;
            }
            newEigenVec = newEigenVec.multiplyByScalar(1.0 / norm);

            // 检查收敛
            double maxDiff = 0.0;
            for (int i = 0; i < n; i++) {
                maxDiff = Math.max(maxDiff,
                    Math.abs(((Number) newEigenVec.get(i)).doubleValue() - ((Number) eigenVec.get(i)).doubleValue()));
            }

            eigenVec = newEigenVec;

            if (maxDiff < tolerance) {
                break;
            }
        }

        // 转换回 double[] 并归一化
        double[] result = new double[n];
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            result[i] = ((Number) eigenVec.get(i)).doubleValue();
            sum += result[i];
        }
        if (sum > 1e-12) {
            for (int i = 0; i < n; i++) {
                result[i] /= sum;
            }
        }
        return result;
    }

    /**
     * 归一化权重向量
     */
    private double[] normalizeWeights(double[] eigenvector) {
        double sum = 0.0;
        for (double w : eigenvector) {
            sum += w;
        }

        if (Math.abs(sum) < 1e-12) {
            double[] equalWeights = new double[eigenvector.length];
            for (int i = 0; i < equalWeights.length; i++) {
                equalWeights[i] = 1.0 / eigenvector.length;
            }
            return equalWeights;
        }

        double[] weights = new double[eigenvector.length];
        for (int i = 0; i < eigenvector.length; i++) {
            weights[i] = eigenvector[i] / sum;
        }

        return weights;
    }

    /**
     * 计算最大特征值
     */
    private double computeMaxEigenvalue(IMatrix matrix, double[] weights) {
        int n = matrix.rows();
        IVector weightVec = Linalg.vector(weights);
        IVector rowDotWeights = matrix.mmul(weightVec);
        double maxEigenvalue = 0.0;

        for (int i = 0; i < n; i++) {
            maxEigenvalue += (Double) rowDotWeights.get(i) / weights[i];
        }

        return maxEigenvalue / n;
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
     * 设置成对比较矩阵
     *
     * @param comparisonMatrix 成对比较矩阵
     * @return this
     */
    public RereAHP setComparisonMatrix(IMatrix comparisonMatrix) {
        Objects.requireNonNull(comparisonMatrix, "参数不能为 null");
        this.comparisonMatrix = comparisonMatrix;
        computeWeights();
        return this;
    }

    /**
     * 设置预先计算好的权重（跳过AHP计算）
     *
     * @param weights 权重向量
     * @return this
     */
    public RereAHP setWeights(double[] weights) {
        Objects.requireNonNull(weights, "参数不能为 null");
        this.weights = weights.clone();
        this.consistencyRatio = 0.0; // 假定已验证
        this.consistencyIndex = 0.0;
        return this;
    }

    /**
     * 设置底层求解器
     *
     * @param baseSolver 底层求解器
     * @return this
     */
    public RereAHP setBaseSolver(ILinProgSolver baseSolver) {
        Objects.requireNonNull(baseSolver, "baseSolver 不能为 null");
        this.baseSolver = baseSolver;
        return this;
    }

    /**
     * 验证一致性
     *
     * @return 是否通过一致性检验（{@code CR < 0.1}）
     */
    public boolean isConsistent() {
        return consistencyRatio < 0.1;
    }

    /**
     * 获取一致性比率
     *
     * @return 一致性比率
     */
    public double getConsistencyRatio() {
        return consistencyRatio;
    }

    /**
     * 获取一致性指标
     *
     * @return 一致性指标
     */
    public double getConsistencyIndex() {
        return consistencyIndex;
    }

    /**
     * 获取权重向量
     *
     * @return 权重向量
     */
    public double[] getWeights() {
        return weights != null ? weights.clone() : null;
    }

    // ==================== IMclpSolver 接口实现 ====================

    @Override
    public String getName() {
        return solverName;
    }

    @Override
    public String getDescription() {
        return "层次分析法(AHP)基于成对比较矩阵计算目标权重，一致性比率CR<0.1为可接受。";
    }

    @Override
    public MclpSolverType getSolverType() {
        return solverType;
    }

    @Override
    public String toString() {
        return "RereAhp{" +
                "weights=" + (weights != null ? java.util.Arrays.toString(weights) : "null") +
                ", consistencyRatio=" + consistencyRatio +
                ", isConsistent=" + isConsistent() +
                '}';
    }

    // ==================== 静态工具方法 ====================

    /**
     * 创建一致矩阵（从权重反向构造）
     *
     * @param weights 权重向量
     * @return 一致矩阵
     */
    public static IMatrix createConsistentMatrix(double[] weights) {
        int n = weights.length;
        double[][] matrix = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Math.abs(weights[j]) < 1e-12) {
                    matrix[i][j] = 1.0;
                } else {
                    matrix[i][j] = weights[i] / weights[j];
                }
            }
        }

        return Linalg.matrix(matrix);
    }

    /**
     * 从比较矩阵检查一致性
     *
     * @param comparisonMatrix 比较矩阵
     * @return 是否一致
     */
    public static boolean checkConsistency(IMatrix comparisonMatrix) {
        RereAHP ahp = new RereAHP(comparisonMatrix);
        return ahp.isConsistent();
    }
}
