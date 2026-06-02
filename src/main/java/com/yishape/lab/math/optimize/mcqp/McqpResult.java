package com.yishape.lab.math.optimize.mcqp;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;

import java.io.Serializable;
import java.util.*;

/**
 * 多目标二次规划结果封装类
 * Multi-Criteria Quadratic Programming Result Container
 *
 * <p>封装多目标二次规划问题的求解结果，包括Pareto前沿、最优解集、各目标函数值等。
 * This class encapsulates the results of multi-criteria quadratic programming problems,
 * including Pareto frontier, optimal solution set, objective function values, etc.</p>
 *
 * @author lteb2
 */
public class McqpResult implements Serializable {

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
    private final McqpSolverType solverType;

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
    private McqpResult(Builder builder) {
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
        this.selectedSolutionIndex = builder.selectedSolutionIndex;
    }

    // ==================== 静态方法 ====================

    private static final double DEFAULT_DOMINANCE_EPSILON = 1e-12;

    /**
     * 判断向量A是否Pareto支配向量B（最小化问题）
     * Returns positive if A dominates B, negative if B dominates A, 0 if non-dominated
     */
    public static int paretoDominates(double[] A, double[] B) {
        return paretoDominates(A, B, DEFAULT_DOMINANCE_EPSILON);
    }

    /**
     * 判断向量A是否Pareto支配向量B（最小化问题），使用指定的容差
     * @param A vector A
     * @param B vector B
     * @param epsilon tolerance for floating-point comparison
     * @return positive if A dominates B, negative if B dominates A, 0 if non-dominated
     */
    public static int paretoDominates(double[] A, double[] B, double epsilon) {
        if (A == null || B == null || A.length != B.length) {
            return 0;
        }
        boolean atLeastOneBetter = false;
        for (int i = 0; i < A.length; i++) {
            if (A[i] < B[i] - epsilon) {
                atLeastOneBetter = true;
            } else if (A[i] > B[i] + epsilon) {
                return -1; // B is better
            }
        }
        return atLeastOneBetter ? 1 : 0;
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
    public McqpSolverType getSolverType() { return solverType; }
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
    public IVector getSelectedSolution() {
        if (selectedSolutionIndex >= 0 && selectedSolutionIndex < solutions.size()) {
            return solutions.get(selectedSolutionIndex);
        }
        return solutions.isEmpty() ? null : solutions.get(0);
    }

    public void setSelectedSolutionIndex(int index) {
        if (index >= 0 && index < numSolutions) {
            this.selectedSolutionIndex = index;
        }
    }

    /**
     * 获取指定索引的解
     */
    public IVector getSolution(int index) {
        if (index >= 0 && index < solutions.size()) {
            return solutions.get(index);
        }
        return null;
    }

    /**
     * 获取指定索引的目标函数值
     */
    public double[] getObjectiveValue(int index) {
        if (index >= 0 && index < objectiveValues.size()) {
            return objectiveValues.get(index).clone();
        }
        return null;
    }

    /**
     * 获取Pareto前沿（所有非支配解）
     */
    public List<IVector> getParetoFront() {
        if (objectiveValues.isEmpty()) {
            return new ArrayList<>();
        }
        List<Integer> paretoIndices = new ArrayList<>();
        for (int i = 0; i < objectiveValues.size(); i++) {
            boolean isDominated = false;
            for (int j = 0; j < objectiveValues.size(); j++) {
                if (i != j && paretoDominates(objectiveValues.get(j), objectiveValues.get(i)) > 0) {
                    isDominated = true;
                    break;
                }
            }
            if (!isDominated) {
                paretoIndices.add(i);
            }
        }
        List<IVector> paretoFront = new ArrayList<>();
        for (int idx : paretoIndices) {
            paretoFront.add(solutions.get(idx));
        }
        return paretoFront;
    }

    /**
     * 生成摘要信息
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("═══════════════════════════════════════════\n");
        sb.append("       MCQP Solver Result Summary\n");
        sb.append("═══════════════════════════════════════════\n");
        sb.append(String.format("Solver: %s (%s)\n", solverName, solverType));
        sb.append(String.format("Converged: %s\n", converged ? "Yes" : "No"));
        if (convergenceReason != null) {
            sb.append(String.format("Reason: %s\n", convergenceReason));
        }
        sb.append(String.format("Solutions Found: %d\n", numSolutions));
        sb.append(String.format("Objectives: %d\n", numObjectives));
        sb.append(String.format("Variables: %d\n", numVariables));
        sb.append(String.format("Constraints: %d\n", numConstraints));
        sb.append(String.format("Total Iterations: %d\n", totalIterations));
        sb.append(String.format("Execution Time: %d ms\n", executionTimeMs));

        if (idealPoint != null && nadirPoint != null) {
            sb.append("\n── Ideal & Nadir Points ──\n");
            sb.append("Ideal Point: ");
            for (int i = 0; i < idealPoint.length; i++) {
                sb.append(String.format("%.6f", idealPoint[i]));
                if (i < idealPoint.length - 1) sb.append(", ");
            }
            sb.append("\nNadir Point: ");
            for (int i = 0; i < nadirPoint.length; i++) {
                sb.append(String.format("%.6f", nadirPoint[i]));
                if (i < nadirPoint.length - 1) sb.append(", ");
            }
            sb.append("\n");
        }

        if (!objectiveValues.isEmpty()) {
            sb.append("\n── Objective Values ──\n");
            for (int i = 0; i < Math.min(objectiveValues.size(), 10); i++) {
                sb.append(String.format("Solution %d: ", i));
                double[] vals = objectiveValues.get(i);
                for (int j = 0; j < vals.length; j++) {
                    sb.append(String.format("f%d=%.6f", j, vals[j]));
                    if (j < vals.length - 1) sb.append(", ");
                }
                sb.append("\n");
            }
            if (objectiveValues.size() > 10) {
                sb.append(String.format("... and %d more solutions\n", objectiveValues.size() - 10));
            }
        }

        sb.append("═══════════════════════════════════════════\n");
        return sb.toString();
    }

    @Override
    public String toString() {
        return String.format("McqpResult{solver='%s', solutions=%d, objectives=%d, converged=%s}",
                solverName, numSolutions, numObjectives, converged);
    }

    // ==================== Builder ====================

    public static class Builder {
        private List<IVector> solutions = new ArrayList<>();
        private int numObjectives;
        private int numConstraints;
        private int numVariables;
        private List<double[]> objectiveValues = new ArrayList<>();
        private List<OptResult> individualResults;
        private double[] weights;
        private int[] priorityOrder;
        private double[] goals;
        private McqpSolverType solverType;
        private String solverName = "";
        private boolean converged = false;
        private String convergenceReason = "";
        private int totalIterations = 0;
        private long executionTimeMs = 0;
        private double[] idealPoint;
        private double[] nadirPoint;
        private double paretoCoverage = 0.0;
        private double diversityMetric = 0.0;
        private double hypervolume = 0.0;
        private int selectedSolutionIndex = 0;

        public Builder solutions(List<IVector> solutions) {
            this.solutions = solutions != null ? solutions : new ArrayList<>();
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

        public Builder objectiveValues(List<double[]> objectiveValues) {
            this.objectiveValues = objectiveValues != null ? objectiveValues : new ArrayList<>();
            return this;
        }

        public Builder individualResults(List<OptResult> individualResults) {
            this.individualResults = individualResults;
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

        public Builder solverType(McqpSolverType solverType) {
            this.solverType = solverType;
            return this;
        }

        public Builder solverName(String solverName) {
            this.solverName = solverName != null ? solverName : "";
            return this;
        }

        public Builder converged(boolean converged) {
            this.converged = converged;
            return this;
        }

        public Builder convergenceReason(String convergenceReason) {
            this.convergenceReason = convergenceReason != null ? convergenceReason : "";
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

        public McqpResult build() {
            if (!objectiveValues.isEmpty() && objectiveValues.size() != solutions.size()) {
                throw new IllegalStateException(
                    "objectiveValues size (" + objectiveValues.size() +
                    ") must match solutions size (" + solutions.size() + ")");
            }
            return new McqpResult(this);
        }
    }
}
