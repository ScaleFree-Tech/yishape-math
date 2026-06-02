package com.yishape.lab.math.optimize.mcqp;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IQpSolver;
import com.yishape.lab.math.optimize.OptResult;

import java.util.*;

/**
 * Pareto最优解法求解多目标二次规划
 * Pareto Optimal Method for Multi-Criteria Quadratic Programming
 *
 * <p>该方法通过系统地探索权重空间，生成完整的Pareto前沿。
 *
 * @author lteb2
 * @see IMcqpSolver
 */
public class RereParetoQp implements IMcqpSolver {

    /** 默认采样点数 / Default number of sample points */
    private static final int DEFAULT_NUM_SAMPLES = 30;

    /** 底层单目标二次规划求解器 / Underlying single-objective QP solver */
    private IQpSolver baseSolver;

    /** 采样点数量 / Number of sample points */
    private int numSamples;

    /** 采样方法 / Sampling method */
    private SamplingMethod samplingMethod = SamplingMethod.UniformGrid;

    /** 收敛容差 / Convergence tolerance */
    private double tolerance = 1e-8;

    /** 求解器类型 / Solver type */
    private final McqpSolverType solverType = McqpSolverType.Pareto;

    /** 求解器名称 / Solver name */
    private final String solverName = "Pareto最优解法(QP)";

    public enum SamplingMethod {
        UniformGrid("均匀网格"),
        Random("随机采样"),
        Adaptive("自适应采样");

        private final String chineseName;
        SamplingMethod(String chineseName) { this.chineseName = chineseName; }
    }

    public RereParetoQp() {
        this.baseSolver = IQpSolver.of();
        this.numSamples = DEFAULT_NUM_SAMPLES;
    }

    public RereParetoQp(int numSamples) {
        this.baseSolver = IQpSolver.of();
        this.numSamples = numSamples;
    }

    public RereParetoQp(int numSamples, IQpSolver baseSolver) {
        this.baseSolver = baseSolver;
        this.numSamples = numSamples;
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
        int numOriginalConstraints = (A_ub != null ? A_ub.rows() : 0) + (A_eq != null ? A_eq.rows() : 0);

        // 计算理想点和反理想点
        double[] idealPoint = computeIdealPoint(Q, c, A_ub, b_ub, A_eq, b_eq);
        double[] nadirPoint = computeNadirPoint(Q, c, A_ub, b_ub, A_eq, b_eq);

        // 生成权重向量
        List<double[]> weightVectors = generateWeightVectors(numObjectives, numSamples);

        // 对每个权重求解
        List<IVector> allSolutions = new ArrayList<>();
        List<double[]> allObjectiveValues = new ArrayList<>();
        List<OptResult> allResults = new ArrayList<>();

        for (double[] weights : weightVectors) {
            IMatrix weightedQ = computeWeightedQ(Q, weights);
            IVector weightedC = computeWeightedC(c, weights);

            OptResult result = baseSolver.solve(weightedQ, weightedC, A_ub, b_ub, A_eq, b_eq, initX);
            if (result.isConverged() && result.getOptimalPoint() != null) {
                IVector sol = result.getOptimalPoint();
                allSolutions.add(sol);
                allResults.add(result);

                double[] objVals = new double[numObjectives];
                for (int i = 0; i < numObjectives; i++) {
                    objVals[i] = computeQuadraticObjective(Q[i], c[i], sol);
                }
                allObjectiveValues.add(objVals);
            }
        }

        // 过滤Pareto最优解
        List<Integer> paretoIndices = filterParetoOptimal(allObjectiveValues);
        List<IVector> paretoSolutions = new ArrayList<>();
        List<double[]> paretoValues = new ArrayList<>();
        List<OptResult> paretoResults = new ArrayList<>();

        for (int idx : paretoIndices) {
            paretoSolutions.add(allSolutions.get(idx));
            paretoValues.add(allObjectiveValues.get(idx));
            paretoResults.add(allResults.get(idx));
        }

        int totalIterations = paretoResults.stream().mapToInt(r -> r != null ? r.getIterations() : 0).sum();

        return new McqpResult.Builder()
            .solutions(paretoSolutions)
            .objectiveValues(paretoValues)
            .individualResults(paretoResults)
            .numObjectives(numObjectives)
            .numConstraints(numOriginalConstraints)
            .numVariables(numVariables)
            .solverType(solverType)
            .solverName(solverName)
            .converged(true)
            .convergenceReason("Pareto前沿生成完成")
            .totalIterations(totalIterations)
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .idealPoint(idealPoint)
            .nadirPoint(nadirPoint)
            .diversityMetric(computeDiversityMetric(paretoValues, idealPoint, nadirPoint))
            .build();
    }

    private List<double[]> generateWeightVectors(int numObjectives, int numSamples) {
        List<double[]> weights = new ArrayList<>();
        if (numObjectives == 2) {
            double step = 1.0 / (numSamples - 1);
            for (int i = 0; i < numSamples; i++) {
                weights.add(new double[]{i * step, 1.0 - i * step});
            }
        } else {
            weights = McqpUtil.generateUniformWeights(numObjectives, numSamples);
        }
        return weights;
    }

    private List<Integer> filterParetoOptimal(List<double[]> objectiveValues) {
        List<Integer> paretoIndices = new ArrayList<>();
        for (int i = 0; i < objectiveValues.size(); i++) {
            boolean isDominated = false;
            for (int j = 0; j < objectiveValues.size(); j++) {
                if (i != j && McqpResult.paretoDominates(objectiveValues.get(j), objectiveValues.get(i)) > 0) {
                    isDominated = true;
                    break;
                }
            }
            if (!isDominated) paretoIndices.add(i);
        }
        return paretoIndices;
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
            ideal[i] = result.isConverged() ? result.getOptimalValue() : Double.NaN;
        }
        return ideal;
    }

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

    private double computeDiversityMetric(List<double[]> paretoFront, double[] idealPoint, double[] nadirPoint) {
        if (paretoFront.size() < 2) return 0.0;
        double totalDist = 0.0;
        List<double[]> sorted = new ArrayList<>(paretoFront);
        sorted.sort(Comparator.comparingDouble(a -> a[0]));
        for (int i = 0; i < sorted.size() - 1; i++) {
            totalDist += euclideanDistance(sorted.get(i), sorted.get(i + 1));
        }
        double range = euclideanDistance(idealPoint, nadirPoint);
        return range > 0 ? totalDist / range : 0.0;
    }

    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    public RereParetoQp setNumSamples(int numSamples) { this.numSamples = numSamples; return this; }
    public RereParetoQp setBaseSolver(IQpSolver baseSolver) { this.baseSolver = baseSolver; return this; }

    @Override public String getName() { return solverName; }
    @Override public String getDescription() { return "Pareto最优解法生成完整Pareto前沿。"; }
    @Override public McqpSolverType getSolverType() { return solverType; }
}
