package com.yishape.lab.math.optimize.mclp;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.util.Tuple2;

import java.io.Serializable;
import java.util.*;

/**
 * 多目标线性规划结果封装类
 * Multi-Criteria Linear Programming Result Container
 *
 * <p>封装多目标线性规划问题的求解结果，包括Pareto前沿、最优解集、各目标函数值等。
 * This class encapsulates the results of multi-criteria linear programming problems,
 * including Pareto frontier, optimal solution set, objective function values, etc.</p>
 *
 * @author lteb2
 */
public class MclpResult implements Serializable {

    // ==================== 基本信息 ====================

    /** 解的数量 / Number of solutions */
    private final int numSolutions;

    /** 各目标的数量 / Number of objectives */
    private final int numObjectives;

    /** 约束数量 / Number of constraints */
    private final int numConstraints;

    /** 变量数量 / Number of variables */
    private final int numVariables;

    // ==================== 解集信息 ====================

    /** 最优解集 / Optimal solution set */
    private final List<IVector> solutions;

    /** 各解对应的目标函数值矩阵（每行是一个解的各目标值）/ Objective values for each solution */
    private final List<double[]> objectiveValues;

    /** 各解对应的OptResult列表（单目标优化的详细结果）/ Individual OptResult for each solution */
    private final List<OptResult> individualResults;

    /** 偏好权重（如果使用加权方法）/ Preference weights (if weighted method used) */
    private final double[] weights;

    /** 各目标的重要性排序（如果使用字典序方法）/ Objective importance order (if lexicographic method used) */
    private final int[] priorityOrder;

    /** 目标值（如果使用目标规划方法）/ Goal values (if goal programming method used) */
    private final double[] goals;

    // ==================== 求解信息 ====================

    /** 求解器类型 / Solver type */
    private final MclpSolverType solverType;

    /** 求解器名称 / Solver name */
    private final String solverName;

    /** 收敛状态 / Convergence status */
    private final boolean converged;

    /** 收敛原因 / Convergence reason */
    private final String convergenceReason;

    /** 总迭代次数 / Total iterations */
    private final int totalIterations;

    /** 总执行时间（毫秒）/ Total execution time in milliseconds */
    private final long executionTimeMs;

    /** 各目标的最优值（理想点）/ Ideal point (best value for each objective) */
    private final double[] idealPoint;

    /** 各目标的最差值（反理想点）/ Nadir point (worst value for each objective) */
    private final double[] nadirPoint;

    // ==================== 评估指标 ====================

    /** Pareto前沿覆盖率 / Pareto frontier coverage */
    private final double paretoCoverage;

    /** 解的多样性指标 / Diversity metric of solutions */
    private final double diversityMetric;

    /** 超体积（仅用于双目标问题）/ Hypervolume (only for bi-objective problems) */
    private final double hypervolume;

    /** 决策者选择的最终解索引 / Final selected solution index by decision maker */
    private int selectedSolutionIndex;

    // ==================== 构造函数 ====================

    /**
     * 私有构造函数（使用Builder）
     */
    private MclpResult(Builder builder) {
        this.numSolutions = builder.solutions.size();
        this.numObjectives = builder.numObjectives;
        this.numConstraints = builder.numConstraints;
        this.numVariables = builder.numVariables;
        this.solutions = Collections.unmodifiableList(new ArrayList<>(builder.solutions));
        this.objectiveValues = Collections.unmodifiableList(new ArrayList<>(builder.objectiveValues));
        this.individualResults = builder.individualResults != null ?
            Collections.unmodifiableList(new ArrayList<>(builder.individualResults)) : new ArrayList<>();
        this.weights = builder.weights != null ? builder.weights.clone() : null;
        this.priorityOrder = builder.priorityOrder != null ? builder.priorityOrder.clone() : null;
        this.goals = builder.goals != null ? builder.goals.clone() : null;
        this.solverType = builder.solverType;
        this.solverName = builder.solverName;
        this.converged = builder.converged;
        this.convergenceReason = builder.convergenceReason;
        this.totalIterations = builder.totalIterations;
        this.executionTimeMs = builder.executionTimeMs;
        this.idealPoint = builder.idealPoint != null ? builder.idealPoint.clone() : null;
        this.nadirPoint = builder.nadirPoint != null ? builder.nadirPoint.clone() : null;
        this.paretoCoverage = builder.paretoCoverage;
        this.diversityMetric = builder.diversityMetric;
        this.hypervolume = builder.hypervolume;
        this.selectedSolutionIndex = builder.selectedSolutionIndex >= 0 ?
            builder.selectedSolutionIndex : 0;
    }

    // ==================== Getter 方法 ====================

    public int getNumSolutions() { return numSolutions; }
    public int getNumObjectives() { return numObjectives; }
    public int getNumConstraints() { return numConstraints; }
    public int getNumVariables() { return numVariables; }
    public List<IVector> getSolutions() { return solutions; }
    public List<double[]> getObjectiveValues() { return objectiveValues; }
    public List<OptResult> getIndividualResults() { return individualResults; }
    public double[] getWeights() { return weights != null ? weights.clone() : null; }
    public int[] getPriorityOrder() { return priorityOrder != null ? priorityOrder.clone() : null; }
    public double[] getGoals() { return goals != null ? goals.clone() : null; }
    public MclpSolverType getSolverType() { return solverType; }
    public String getSolverName() { return solverName; }
    public boolean isConverged() { return converged; }
    public String getConvergenceReason() { return convergenceReason; }
    public int getTotalIterations() { return totalIterations; }
    public long getExecutionTimeMs() { return executionTimeMs; }
    public double[] getIdealPoint() { return idealPoint != null ? idealPoint.clone() : null; }
    public double[] getNadirPoint() { return nadirPoint != null ? nadirPoint.clone() : null; }
    public double getParetoCoverage() { return paretoCoverage; }
    public double getDiversityMetric() { return diversityMetric; }
    public double getHypervolume() { return hypervolume; }
    public int getSelectedSolutionIndex() { return selectedSolutionIndex; }

    /**
     * 获取最终选择的解
     * @return 最终解向量
     */
    public IVector getSelectedSolution() {
        if (solutions.isEmpty() || selectedSolutionIndex < 0
                || selectedSolutionIndex >= solutions.size()) {
            return null;
        }
        return solutions.get(selectedSolutionIndex);
    }

    /**
     * 获取最终解对应的各目标函数值
     * @return 各目标函数值数组
     */
    public double[] getSelectedObjectiveValues() {
        if (objectiveValues.isEmpty() || selectedSolutionIndex < 0
                || selectedSolutionIndex >= objectiveValues.size()) {
            return null;
        }
        return objectiveValues.get(selectedSolutionIndex).clone();
    }

    /**
     * 获取最终解对应的OptResult
     * @return OptResult
     */
    public OptResult getSelectedIndividualResult() {
        if (individualResults.isEmpty() || selectedSolutionIndex < 0
                || selectedSolutionIndex >= individualResults.size()) {
            return null;
        }
        return individualResults.get(selectedSolutionIndex);
    }

    // ==================== 便利方法 ====================

    /**
     * 获取第i个解
     * @param index 解的索引
     * @return 解向量
     */
    public IVector getSolution(int index) {
        if (index < 0 || index >= solutions.size()) {
            return null;
        }
        return solutions.get(index);
    }

    /**
     * 获取第i个解的各目标函数值
     * @param index 解的索引
     * @return 各目标函数值数组
     */
    public double[] getObjectiveValues(int index) {
        if (index < 0 || index >= objectiveValues.size()) {
            return null;
        }
        return objectiveValues.get(index).clone();
    }

    /**
     * 计算两个解之间的Pareto支配关系
     * @param obj1 第一个解的目标值
     * @param obj2 第二个解的目标值
     * @return 如果obj1支配obj2返回1，如果obj2支配obj1返回-1，否则返回0
     */
    public static int paretoDominates(double[] obj1, double[] obj2) {
        boolean atLeastOneBetter = false;
        boolean atLeastOneWorse = false;

        for (int i = 0; i < obj1.length; i++) {
            if (obj1[i] < obj2[i]) { // 最小化问题
                atLeastOneBetter = true;
            } else if (obj1[i] > obj2[i]) {
                atLeastOneWorse = true;
            }
        }

        if (atLeastOneBetter && !atLeastOneWorse) return 1;  // obj1支配obj2
        if (atLeastOneWorse && !atLeastOneBetter) return -1; // obj2支配obj1
        return 0; // 互不支配
    }

    /**
     * 获取非支配解集（Pareto前沿）
     * @return 非支配解索引列表
     */
    public List<Integer> getNonDominatedSolutions() {
        List<Integer> paretoFront = new ArrayList<>();
        for (int i = 0; i < numSolutions; i++) {
            boolean isDominated = false;
            for (int j = 0; j < numSolutions; j++) {
                if (i != j && paretoDominates(objectiveValues.get(j), objectiveValues.get(i)) > 0) {
                    isDominated = true;
                    break;
                 }
            }
            if (!isDominated) {
                paretoFront.add(i);
            }
        }
        return paretoFront;
    }

    /**
     * 计算理想点（各目标的单目标最优值）
     * @return 理想点向量
     */
    public double[] computeIdealPoint() {
        if (objectiveValues.isEmpty()) {
            return null;
        }
        double[] ideal = new double[numObjectives];
        Arrays.fill(ideal, Double.MAX_VALUE);

        for (double[] objVals : objectiveValues) {
            for (int i = 0; i < numObjectives; i++) {
                if (objVals[i] < ideal[i]) {
                    ideal[i] = objVals[i];
                }
            }
        }
        return ideal;
    }

    /**
     * 计算反理想点（各目标的单目标最差值）
     * @return 反理想点向量
     */
    public double[] computeNadirPoint() {
        if (objectiveValues.isEmpty()) {
            return null;
        }
        double[] nadir = new double[numObjectives];
        Arrays.fill(nadir, -Double.MAX_VALUE);

        for (double[] objVals : objectiveValues) {
            for (int i = 0; i < numObjectives; i++) {
                if (objVals[i] > nadir[i]) {
                    nadir[i] = objVals[i];
                }
            }
        }
        return nadir;
    }

    /**
     * 验证Pareto最优性
     * @return 是否为Pareto最优
     */
    public boolean isParetoOptimal(int index) {
        return getNonDominatedSolutions().contains(index);
    }

    /**
     * 获取解的摘要信息
     * @return 摘要字符串
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 多目标线性规划结果摘要 / MCLP Result Summary ===\n");
        sb.append(String.format("求解器 / Solver: %s (%s)\n", solverName, solverType));
        sb.append(String.format("解数量 / Solutions: %d\n", numSolutions));
        sb.append(String.format("目标数量 / Objectives: %d\n", numObjectives));
        sb.append(String.format("变量数量 / Variables: %d\n", numVariables));
        sb.append(String.format("约束数量 / Constraints: %d\n", numConstraints));
        sb.append(String.format("收敛状态 / Converged: %s\n", converged ? "是/Yes" : "否/No"));
        sb.append(String.format("收敛原因 / Reason: %s\n", convergenceReason));
        sb.append(String.format("总迭代次数 / Total Iterations: %d\n", totalIterations));
        sb.append(String.format("执行时间 / Execution Time: %d ms\n", executionTimeMs));

        if (idealPoint != null) {
            sb.append(String.format("理想点 / Ideal Point: %s\n", Arrays.toString(idealPoint)));
        }
        if (nadirPoint != null) {
            sb.append(String.format("反理想点 / Nadir Point: %s\n", Arrays.toString(nadirPoint)));
        }

        sb.append(String.format("非支配解数量 / Non-dominated Solutions: %d\n", getNonDominatedSolutions().size()));
        sb.append(String.format("多样性指标 / Diversity Metric: %.6f\n", diversityMetric));

        if (weights != null) {
            sb.append(String.format("权重 / Weights: %s\n", Arrays.toString(weights)));
        }
        if (goals != null) {
            sb.append(String.format("目标值 / Goals: %s\n", Arrays.toString(goals)));
        }

        sb.append("\n--- 最终选择解 / Selected Solution ---\n");
        if (!solutions.isEmpty()) {
            IVector selected = getSelectedSolution();
            double[] selectedObjs = getSelectedObjectiveValues();
            sb.append(String.format("解向量 / Solution: %s\n", selected));
            sb.append(String.format("目标值 / Objective Values: %s\n", Arrays.toString(selectedObjs)));
        }

        return sb.toString();
    }

    /**
     * 获取详细报告
     * @return 详细报告字符串
     */
    public String getDetailedReport() {
        StringBuilder sb = new StringBuilder();
        sb.append(getSummary());

        sb.append("\n=== 所有解 / All Solutions ===\n");
        for (int i = 0; i < numSolutions; i++) {
            sb.append(String.format("\n--- 解 %d ---\n", i));
            sb.append(String.format("目标值 / Objectives: %s\n", Arrays.toString(objectiveValues.get(i))));
            if (individualResults.size() > i) {
                OptResult result = individualResults.get(i);
                sb.append(String.format("收敛 / Converged: %s\n", result.isConverged()));
                sb.append(String.format("迭代 / Iterations: %d\n", result.getIterations()));
            }
        }

        sb.append("\n=== Pareto前沿 / Pareto Frontier ===\n");
        List<Integer> paretoIndices = getNonDominatedSolutions();
        sb.append(String.format("非支配解索引 / Non-dominated Indices: %s\n", paretoIndices));

        return sb.toString();
    }

    @Override
    public String toString() {
        return String.format("MclpResult{solver=%s, solutions=%d, objectives=%d, converged=%s}",
                solverType, numSolutions, numObjectives, converged);
    }

    // ==================== 建造者模式 ====================

    public static class Builder {
        private List<IVector> solutions = new ArrayList<>();
        private List<double[]> objectiveValues = new ArrayList<>();
        private List<OptResult> individualResults = new ArrayList<>();
        private int numObjectives = 0;
        private int numConstraints = 0;
        private int numVariables = 0;
        private double[] weights = null;
        private int[] priorityOrder = null;
        private double[] goals = null;
        private MclpSolverType solverType = MclpSolverType.WeightedSum;
        private String solverName = "MCLP Solver";
        private boolean converged = true;
        private String convergenceReason = "Completed";
        private int totalIterations = 0;
        private long executionTimeMs = 0L;
        private double[] idealPoint = null;
        private double[] nadirPoint = null;
        private double paretoCoverage = 1.0;
        private double diversityMetric = 0.0;
        private double hypervolume = 0.0;
        private int selectedSolutionIndex = 0;

        public Builder() {}

        public Builder solutions(List<IVector> solutions) {
            this.solutions = solutions;
            if (!solutions.isEmpty() && numVariables == 0) {
                this.numVariables = solutions.get(0).length();
            }
            return this;
        }

        public Builder addSolution(IVector solution, double[] objValues) {
            this.solutions.add(solution);
            this.objectiveValues.add(objValues);
            return this;
        }

        public Builder addSolution(IVector solution, double[] objValues, OptResult individualResult) {
            this.solutions.add(solution);
            this.objectiveValues.add(objValues);
            if (individualResult != null) {
                this.individualResults.add(individualResult);
            }
            return this;
        }

        public Builder objectiveValues(List<double[]> objectiveValues) {
            this.objectiveValues = objectiveValues;
            return this;
        }

        public Builder individualResults(List<OptResult> individualResults) {
            this.individualResults = individualResults;
            return this;
        }

        public Builder numObjectives(int numObjectives) {
            this.numObjectives = numObjectives;
            return this;
        }

        public Builder numConstraints(int numConstraints) {
            this.numConstraints = numConstraints;
            return this;
        }

        public Builder numVariables(int numVariables) {
            this.numVariables = numVariables;
            return this;
        }

        public Builder weights(double[] weights) {
            this.weights = weights;
            return this;
        }

        public Builder priorityOrder(int[] priorityOrder) {
            this.priorityOrder = priorityOrder;
            return this;
        }

        public Builder goals(double[] goals) {
            this.goals = goals;
            return this;
        }

        public Builder solverType(MclpSolverType solverType) {
            this.solverType = solverType;
            return this;
        }

        public Builder solverName(String solverName) {
            this.solverName = solverName;
            return this;
        }

        public Builder converged(boolean converged) {
            this.converged = converged;
            return this;
        }

        public Builder convergenceReason(String convergenceReason) {
            this.convergenceReason = convergenceReason;
            return this;
        }

        public Builder totalIterations(int totalIterations) {
            this.totalIterations = totalIterations;
            return this;
        }

        public Builder executionTimeMs(long executionTimeMs) {
            this.executionTimeMs = executionTimeMs;
            return this;
        }

        public Builder idealPoint(double[] idealPoint) {
            this.idealPoint = idealPoint;
            return this;
        }

        public Builder nadirPoint(double[] nadirPoint) {
            this.nadirPoint = nadirPoint;
            return this;
        }

        public Builder paretoCoverage(double paretoCoverage) {
            this.paretoCoverage = paretoCoverage;
            return this;
        }

        public Builder diversityMetric(double diversityMetric) {
            this.diversityMetric = diversityMetric;
            return this;
        }

        public Builder hypervolume(double hypervolume) {
            this.hypervolume = hypervolume;
            return this;
        }

        public Builder selectedSolutionIndex(int selectedSolutionIndex) {
            this.selectedSolutionIndex = selectedSolutionIndex;
            return this;
        }

        public MclpResult build() {
            if (numObjectives == 0 && !objectiveValues.isEmpty()) {
                numObjectives = objectiveValues.get(0).length;
            }
            if (numVariables == 0 && !solutions.isEmpty()) {
                numVariables = solutions.get(0).length();
            }
            return new MclpResult(this);
        }
    }
}
