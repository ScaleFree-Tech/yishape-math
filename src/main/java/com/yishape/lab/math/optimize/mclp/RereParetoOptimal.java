package com.yishape.lab.math.optimize.mclp;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;

import java.util.*;

/**
 * Pareto最优解法求解多目标线性规划
 * Pareto Optimal Method for Multi-Objective Linear Programming
 *
 * <p>该方法通过系统地探索权重空间，生成完整的Pareto前沿，
 * 帮助决策者了解各目标之间的权衡关系。
 *
 * <p>算法步骤：
 * 1. 计算理想点和反理想点
 * 2. 在目标空间中均匀采样权重向量
 * 3. 对每个权重向量求解加权问题
 * 4. 过滤非支配解，构建Pareto前沿
 *
 * <p>特点：
 * - 生成完整的Pareto前沿
 * - 需要多次调用单目标求解器
 * - 结果可供决策者进一步分析
 * - 支持非凸Pareto前沿的发现（通过极端点求解辅助）
 *
 * @author lteb2
 * @see IMclpSolver
 */
public class RereParetoOptimal implements IMclpSolver {

    /** 默认采样点数 / Default number of sample points */
    private static final int DEFAULT_NUM_SAMPLES = 50;

    /** 底层单目标线性规划求解器 / Underlying single-objective LP solver */
    private ILinProgSolver baseSolver;

    /** 采样点数量 / Number of sample points */
    private int numSamples;

    /** 采样方法 / Sampling method */
    private SamplingMethod samplingMethod = SamplingMethod.UniformGrid;

    /** 收敛容差 / Convergence tolerance for Pareto filtering */
    private double tolerance = 1e-8;

    /** 求解器类型 / Solver type */
    private final MclpSolverType solverType = MclpSolverType.Pareto;

    /** 求解器名称 / Solver name */
    private final String solverName = "Pareto最优解法";

    /**
     * 采样方法枚举
     */
    public enum SamplingMethod {
        /** 均匀网格采样 / Uniform grid sampling */
        UniformGrid("均匀网格"),
        /** 随机采样 / Random sampling */
        Random("随机采样"),
        /** 自适应采样 / Adaptive sampling */
        Adaptive("自适应采样");

        private final String chineseName;

        SamplingMethod(String chineseName) {
            this.chineseName = chineseName;
        }

        public String getChineseName() {
            return chineseName;
        }
    }

    /**
     * 默认构造函数
     */
    public RereParetoOptimal() {
        this.baseSolver = ILinProgSolver.of();
        this.numSamples = DEFAULT_NUM_SAMPLES;
    }

    /**
     * 使用指定采样点数量的构造函数
     *
     * @param numSamples 采样点数量
     */
    public RereParetoOptimal(int numSamples) {
        this.baseSolver = ILinProgSolver.of();
        this.numSamples = numSamples;
    }

    /**
     * 使用指定采样点和底层求解器的构造函数
     *
     * @param numSamples 采样点数量
     * @param baseSolver 底层求解器
     */
    public RereParetoOptimal(int numSamples, ILinProgSolver baseSolver) {
        this.baseSolver = baseSolver;
        this.numSamples = numSamples;
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

        // 步骤1：计算理想点和反理想点
        double[] idealPoint = computeIdealPoint(c, A_ub, b_ub, A_eq, b_eq);
        double[] nadirPoint = computeNadirPoint(c, A_ub, b_ub, A_eq, b_eq);

        // 步骤2：添加极端点（确保Pareto前沿边界）
        List<double[]> allWeights = new ArrayList<>();
        allWeights.addAll(generateExtremePoints(numObjectives));
        allWeights.addAll(generateSampledWeights(numObjectives, numSamples));

        // 步骤3：对每个权重向量求解
        List<IVector> solutions = new ArrayList<>();
        List<double[]> objectiveValues = new ArrayList<>();
        List<OptResult> individualResults = new ArrayList<>();

        for (double[] weights : allWeights) {
            IVector weightedC = computeWeightedObjective(c, weights);
            OptResult result = baseSolver.solve(weightedC, A_ub, b_ub, A_eq, b_eq, initX);

            if (result.isConverged() && result.getOptimalPoint() != null) {
                IVector optimalX = result.getOptimalPoint();

                // 检查是否已有相同解
                boolean isDuplicate = false;
                for (IVector existing : solutions) {
                    if (isSameSolution(existing, optimalX, tolerance)) {
                        isDuplicate = true;
                        break;
                    }
                }

                if (!isDuplicate) {
                    solutions.add(optimalX);

                    double[] objVals = new double[numObjectives];
                    for (int i = 0; i < numObjectives; i++) {
                        objVals[i] = c[i].innerProductValue(optimalX);
                    }
                    objectiveValues.add(objVals);
                    individualResults.add(result);
                }
            }
        }

        // 步骤4：过滤非支配解
        List<Integer> paretoIndices = filterParetoOptimal(objectiveValues);
        List<IVector> paretoSolutions = new ArrayList<>();
        List<double[]> paretoObjectiveValues = new ArrayList<>();
        List<OptResult> paretoIndividualResults = new ArrayList<>();

        for (int idx : paretoIndices) {
            paretoSolutions.add(solutions.get(idx));
            paretoObjectiveValues.add(objectiveValues.get(idx));
            if (idx < individualResults.size()) {
                paretoIndividualResults.add(individualResults.get(idx));
            }
        }

        // 步骤5：计算Pareto前沿质量指标
        double diversityMetric = computeDiversityMetric(paretoObjectiveValues, idealPoint, nadirPoint);
        double paretoCoverage = computeParetoCoverage(paretoObjectiveValues, idealPoint, nadirPoint);
        double hypervolume = computeHypervolume(paretoObjectiveValues, nadirPoint);

        int totalIterations = individualResults.stream()
            .mapToInt(OptResult::getIterations)
            .sum();

        return new MclpResult.Builder()
            .solutions(paretoSolutions)
            .objectiveValues(paretoObjectiveValues)
            .individualResults(paretoIndividualResults)
            .numObjectives(numObjectives)
            .numConstraints((A_ub != null ? A_ub.rows() : 0) + (A_eq != null ? A_eq.rows() : 0))
            .numVariables(numVariables)
            .solverType(solverType)
            .solverName(solverName)
            .converged(true)
            .convergenceReason("Pareto前沿生成完成")
            .totalIterations(totalIterations)
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .idealPoint(idealPoint)
            .nadirPoint(nadirPoint)
            .diversityMetric(diversityMetric)
            .paretoCoverage(paretoCoverage)
            .hypervolume(hypervolume)
            .selectedSolutionIndex(0)
            .build();
    }

    /**
     * 生成极端点权重（对应各目标的单目标最优解）
     */
    private List<double[]> generateExtremePoints(int numObjectives) {
        List<double[]> extremePoints = new ArrayList<>();

        for (int i = 0; i < numObjectives; i++) {
            double[] weights = new double[numObjectives];
            weights[i] = 1.0;
            extremePoints.add(weights);
        }

        return extremePoints;
    }

    /**
     * 生成采样权重向量
     */
    private List<double[]> generateSampledWeights(int numObjectives, int totalSamples) {
        List<double[]> weights = new ArrayList<>();

        switch (samplingMethod) {
            case UniformGrid -> {
                weights.addAll(generateUniformGridWeights(numObjectives, totalSamples));
            }
            case Random -> {
                weights.addAll(generateRandomWeights(numObjectives, totalSamples));
            }
            case Adaptive -> {
                weights.addAll(generateAdaptiveWeights(numObjectives, totalSamples));
            }
        }

        return weights;
    }

    /**
     * 均匀网格采样
     */
    private List<double[]> generateUniformGridWeights(int numObjectives, int totalSamples) {
        List<double[]> weights = new ArrayList<>();

        // 计算每个维度的采样点数
        int pointsPerDim = Math.max(2, (int) Math.pow(totalSamples, 1.0 / numObjectives));

        // 生成网格点
        double[][] gridPoints = new double[pointsPerDim][];
        for (int i = 0; i < pointsPerDim; i++) {
            gridPoints[i] = new double[]{i / (double)(pointsPerDim - 1)};
        }

        // 使用递归生成所有组合
        generateGridCombinations(weights, new double[numObjectives], 0, gridPoints);

        return weights;
    }

    private void generateGridCombinations(List<double[]> weights, double[] current,
                                         int dim, double[][] gridPoints) {
        if (dim == current.length) {
            weights.add(current.clone());
            return;
        }

        for (double value : gridPoints[dim]) {
            current[dim] = value;
            generateGridCombinations(weights, current, dim + 1, gridPoints);
        }
    }

    /**
     * 随机采样
     */
    private List<double[]> generateRandomWeights(int numObjectives, int totalSamples) {
        List<double[]> weights = new ArrayList<>();
        Random random = new Random(42); // 固定种子以保证可重复性

        for (int i = 0; i < totalSamples; i++) {
            double[] w = new double[numObjectives];
            double sum = 0.0;

            for (int j = 0; j < numObjectives; j++) {
                w[j] = random.nextDouble();
                sum += w[j];
            }

            // 归一化
            for (int j = 0; j < numObjectives; j++) {
                w[j] /= sum;
            }

            weights.add(w);
        }

        return weights;
    }

    /**
     * 自适应采样：先在极端点附近进行密集采样，再在中间区域均匀填充。
     * 极端点附近通常对应Pareto前沿曲率最大的区域。
     */
    private List<double[]> generateAdaptiveWeights(int numObjectives, int totalSamples) {
        List<double[]> weights = new ArrayList<>();

        // 分配：40% 样本给极端点附近的高密度区域，60% 给均匀填充
        int extremeBudget = Math.max(numObjectives, totalSamples * 2 / 5);
        int uniformBudget = totalSamples - extremeBudget;

        // 极端点附近采样（Beta分布偏斜到各个角）
        Random random = new Random(42);
        for (int i = 0; i < numObjectives; i++) {
            int perExtreme = extremeBudget / numObjectives;
            for (int k = 0; k < perExtreme; k++) {
                double[] w = new double[numObjectives];
                double sum = 0.0;
                for (int j = 0; j < numObjectives; j++) {
                    if (j == i) {
                        // 对焦点目标使用较大权重（0.5-1.0范围）
                        w[j] = 0.5 + 0.5 * random.nextDouble();
                    } else {
                        w[j] = 0.01 + 0.49 * random.nextDouble();
                    }
                    sum += w[j];
                }
                for (int j = 0; j < numObjectives; j++) {
                    w[j] /= sum;
                }
                weights.add(w);
            }
        }

        // 剩余样本均匀覆盖空间
        weights.addAll(generateUniformGridWeights(numObjectives, uniformBudget));

        return weights;
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
     * 判断两个解是否相同
     */
    private boolean isSameSolution(IVector sol1, IVector sol2, double tol) {
        if (sol1.length() != sol2.length()) {
            return false;
        }
        for (int i = 0; i < sol1.length(); i++) {
            if (Math.abs((Double) sol1.get(i) - (Double) sol2.get(i)) > tol) {
                return false;
            }
        }
        return true;
    }

    /**
     * 过滤Pareto最优解
     */
    private List<Integer> filterParetoOptimal(List<double[]> objectiveValues) {
        List<Integer> paretoIndices = new ArrayList<>();

        for (int i = 0; i < objectiveValues.size(); i++) {
            boolean isDominated = false;

            for (int j = 0; j < objectiveValues.size(); j++) {
                if (i != j) {
                    if (MclpResult.paretoDominates(objectiveValues.get(j), objectiveValues.get(i)) > 0) {
                        isDominated = true;
                        break;
                    }
                }
            }

            if (!isDominated) {
                paretoIndices.add(i);
            }
        }

        return paretoIndices;
    }

    /**
     * 计算多样性指标（基于目标空间分布）
     */
    private double computeDiversityMetric(List<double[]> paretoFront,
                                        double[] ideal, double[] nadir) {
        if (paretoFront.size() < 2) {
            return 0.0;
        }

        // 计算相邻解之间的欧氏距离
        double totalSpread = 0.0;
        List<double[]> sortedFront = new ArrayList<>(paretoFront);
        final int objIdx = 0; // 按第一个目标排序

        sortedFront.sort(Comparator.comparingDouble(a -> a[objIdx]));

        for (int i = 0; i < sortedFront.size() - 1; i++) {
            double dist = euclideanDistance(sortedFront.get(i), sortedFront.get(i + 1));
            totalSpread += dist;
        }

        return totalSpread;
    }

    /**
     * 计算欧氏距离
     */
    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * 计算Pareto前沿覆盖率
     */
    private double computeParetoCoverage(List<double[]> paretoFront,
                                        double[] ideal, double[] nadir) {
        if (paretoFront.isEmpty() || ideal == null || nadir == null) {
            return 0.0;
        }

        // 计算归一化后的目标空间范围
        double range = 0.0;
        for (int i = 0; i < ideal.length; i++) {
            range += (nadir[i] - ideal[i]) * (nadir[i] - ideal[i]);
        }
        range = Math.sqrt(range);

        // 简化的覆盖率计算
        return range > 0 ? Math.min(1.0, paretoFront.size() / (10.0 * ideal.length)) : 0.0;
    }

    /**
     * 计算超体积（仅适用于双目标问题）
     * <p>假设前沿已按第一目标升序排列且非支配，第二目标严格递减。
     * 超体积 = Σ_i (f1_{i+1} - f1_i) × (ref_f2 - f2_i)，其中 f1_{n} = ref_f1。
     */
    private double computeHypervolume(List<double[]> paretoFront, double[] referencePoint) {
        if (paretoFront.size() == 0 || referencePoint == null) {
            return 0.0;
        }

        if (referencePoint.length != 2) {
            return 0.0;
        }

        List<double[]> sorted = new ArrayList<>(paretoFront);
        sorted.sort(Comparator.comparingDouble(a -> a[0]));

        double hypervolume = 0.0;
        for (int i = 0; i < sorted.size(); i++) {
            double f1_i = sorted.get(i)[0];
            double f2_i = sorted.get(i)[1];
            double f1_next = (i + 1 < sorted.size()) ? sorted.get(i + 1)[0] : referencePoint[0];
            double width = f1_next - f1_i;
            double height = referencePoint[1] - f2_i;
            if (width > 0 && height > 0) {
                hypervolume += width * height;
            }
        }

        return hypervolume;
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
     * 设置采样点数量
     *
     * @param numSamples 采样点数量
     * @return this
     */
    public RereParetoOptimal setNumSamples(int numSamples) {
        this.numSamples = numSamples;
        return this;
    }

    /**
     * 设置采样方法
     *
     * @param samplingMethod 采样方法
     * @return this
     */
    public RereParetoOptimal setSamplingMethod(SamplingMethod samplingMethod) {
        this.samplingMethod = samplingMethod;
        return this;
    }

    /**
     * 设置容差
     *
     * @param tolerance 容差值
     * @return this
     */
    public RereParetoOptimal setTolerance(double tolerance) {
        this.tolerance = tolerance;
        return this;
    }

    /**
     * 设置底层求解器
     *
     * @param baseSolver 底层求解器
     * @return this
     */
    public RereParetoOptimal setBaseSolver(ILinProgSolver baseSolver) {
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
        return "Pareto最优解法通过系统探索权重空间生成完整Pareto前沿，帮助决策者了解目标间的权衡关系。";
    }

    @Override
    public MclpSolverType getSolverType() {
        return solverType;
    }

    @Override
    public String toString() {
        return "RereParetoOptimal{" +
                "numSamples=" + numSamples +
                ", samplingMethod=" + samplingMethod +
                ", tolerance=" + tolerance +
                '}';
    }
}
