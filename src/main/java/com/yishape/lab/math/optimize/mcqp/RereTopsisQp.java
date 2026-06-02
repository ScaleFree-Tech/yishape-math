package com.yishape.lab.math.optimize.mcqp;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IQpSolver;
import com.yishape.lab.math.optimize.OptResult;

import java.util.*;

/**
 * TOPSIS方法求解多目标二次规划
 * TOPSIS for Multi-Criteria Quadratic Programming
 *
 * <p>TOPSIS是一种多准则决策分析方法，其基本思想是：
 * 所选择的方案应距离理想解最近，距离负理想解最远。</p>
 *
 * @author lteb2
 * @see IMcqpSolver
 */
public class RereTopsisQp implements IMcqpSolver {

    /** 权重向量 / Weight vector */
    private double[] weights;

    /** 底层单目标二次规划求解器 / Underlying single-objective QP solver */
    private IQpSolver baseSolver;

    /** 求解器类型 / Solver type */
    private final McqpSolverType solverType = McqpSolverType.Topsis;

    /** 求解器名称 / Solver name */
    private final String solverName = "TOPSIS法(QP)";

    public RereTopsisQp() {
        this.baseSolver = IQpSolver.of();
    }

    public RereTopsisQp(double[] weights) {
        this.weights = weights.clone();
        this.baseSolver = IQpSolver.of();
    }

    public RereTopsisQp(double[] weights, IQpSolver baseSolver) {
        this.weights = weights.clone();
        this.baseSolver = baseSolver;
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

        // 初始化权重
        if (weights == null || weights.length != numObjectives) {
            weights = new double[numObjectives];
            Arrays.fill(weights, 1.0 / numObjectives);
        }

        // 计算各目标的单目标最优解
        List<IVector> candidateSolutions = new ArrayList<>();
        List<double[]> candidateObjValues = new ArrayList<>();

        for (int i = 0; i < numObjectives; i++) {
            OptResult result = baseSolver.solve(Q[i], c[i], A_ub, b_ub, A_eq, b_eq, initX);
            if (result.isConverged() && result.getOptimalPoint() != null) {
                IVector sol = result.getOptimalPoint();
                candidateSolutions.add(sol);
                double[] vals = new double[numObjectives];
                for (int j = 0; j < numObjectives; j++) {
                    vals[j] = computeQuadraticObjective(Q[j], c[j], sol);
                }
                candidateObjValues.add(vals);
            }
        }

        // 计算理想解和负理想解
        double[] idealPoint = new double[numObjectives];
        double[] nadirPoint = new double[numObjectives];
        for (int j = 0; j < numObjectives; j++) {
            double minVal = Double.MAX_VALUE;
            double maxVal = -Double.MAX_VALUE;
            for (double[] vals : candidateObjValues) {
                minVal = Math.min(minVal, vals[j]);
                maxVal = Math.max(maxVal, vals[j]);
            }
            idealPoint[j] = minVal;
            nadirPoint[j] = maxVal;
        }

        // 归一化并计算贴近度
        double[] bestCloseness = new double[]{-1.0};
        int bestIndex = 0;

        for (int i = 0; i < candidateSolutions.size(); i++) {
            double[] vals = candidateObjValues.get(i);
            double[] normalizedIdeal = new double[numObjectives];
            double[] normalizedNadir = new double[numObjectives];
            double[] normalizedVals = new double[numObjectives];

            for (int j = 0; j < numObjectives; j++) {
                double range = nadirPoint[j] - idealPoint[j];
                if (range > 1e-12) {
                    normalizedVals[j] = (vals[j] - idealPoint[j]) / range;
                    normalizedIdeal[j] = 0.0;
                    normalizedNadir[j] = 1.0;
                } else {
                    normalizedVals[j] = 0.0;
                    normalizedIdeal[j] = 0.0;
                    normalizedNadir[j] = 0.0;
                }
            }

            // 加权
            double distIdeal = 0.0;
            double distNadir = 0.0;
            for (int j = 0; j < numObjectives; j++) {
                double wDiffIdeal = weights[j] * (normalizedVals[j] - normalizedIdeal[j]);
                double wDiffNadir = weights[j] * (normalizedVals[j] - normalizedNadir[j]);
                distIdeal += wDiffIdeal * wDiffIdeal;
                distNadir += wDiffNadir * wDiffNadir;
            }
            distIdeal = Math.sqrt(distIdeal);
            distNadir = Math.sqrt(distNadir);

            double closeness = distNadir / (distIdeal + distNadir + 1e-12);
            if (closeness > bestCloseness[0]) {
                bestCloseness[0] = closeness;
                bestIndex = i;
            }
        }

        // 处理无解情况
        if (candidateSolutions.isEmpty()) {
            return new McqpResult.Builder()
                .solutions(Collections.emptyList())
                .objectiveValues(Collections.emptyList())
                .numObjectives(numObjectives)
                .numConstraints((A_ub != null ? A_ub.rows() : 0) + (A_eq != null ? A_eq.rows() : 0))
                .numVariables(numVariables)
                .weights(weights)
                .solverType(solverType)
                .solverName(solverName)
                .converged(false)
                .convergenceReason("无法找到可行解")
                .executionTimeMs(System.currentTimeMillis() - startTime)
                .build();
        }

        IVector optimalX = candidateSolutions.get(bestIndex);
        double[] optimalObjVals = candidateObjValues.get(bestIndex);

        List<IVector> solutions = Collections.singletonList(optimalX);
        List<double[]> objectiveValues = Collections.singletonList(optimalObjVals);

        return new McqpResult.Builder()
            .solutions(solutions)
            .objectiveValues(objectiveValues)
            .numObjectives(numObjectives)
            .numConstraints((A_ub != null ? A_ub.rows() : 0) + (A_eq != null ? A_eq.rows() : 0))
            .numVariables(numVariables)
            .weights(weights)
            .solverType(solverType)
            .solverName(solverName)
            .converged(true)
            .convergenceReason("TOPSIS选择最佳解")
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .idealPoint(idealPoint)
            .nadirPoint(nadirPoint)
            .build();
    }

    private double computeQuadraticObjective(IMatrix Q, IVector c, IVector x) {
        IVector qx = Q.mmul(x);
        return 0.5 * x.dotValue(qx) + c.dotValue(x);
    }

    public RereTopsisQp setWeights(double[] weights) {
        this.weights = weights.clone();
        return this;
    }

    public RereTopsisQp setBaseSolver(IQpSolver baseSolver) {
        this.baseSolver = baseSolver;
        return this;
    }

    @Override public String getName() { return solverName; }
    @Override public String getDescription() { return "TOPSIS法选择距离理想解最近、距离负理想解最远的解。"; }
    @Override public McqpSolverType getSolverType() { return solverType; }
}
