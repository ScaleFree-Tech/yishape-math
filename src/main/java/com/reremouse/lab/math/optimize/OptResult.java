package com.reremouse.lab.math.optimize;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.util.Tuple2;
import java.io.Serializable;
import java.util.List;
import java.util.ArrayList;

/**
 * 优化结果类
 * <p>
 * 封装优化算法的执行结果，包含最优解、收敛信息、执行统计等。
 * 支持无约束优化和约束优化的结果表示。
 * </p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class OptResult implements Serializable{
    
    // ==================== 基本优化结果 ====================
    
    /** 最优函数值 / Optimal function value */
    private final double optimalValue;
    
    /** 最优点 / Optimal point */
    private final IVector optimalPoint;
    
    /** 初始点 / Initial point */
    private final IVector initialPoint;
    
    /** 初始函数值 / Initial function value */
    private final double initialValue;
    
    // ==================== 收敛信息 ====================
    
    /** 是否收敛 / Whether converged */
    private boolean converged = true;
    
    /** 收敛原因 / Convergence reason */
    private final String convergenceReason;
    
    /** 实际迭代次数 / Actual number of iterations */
    private final int iterations;
    
    /** 最大迭代次数 / Maximum iterations allowed */
    private final int maxIterations;
    
    /** 最终梯度范数 / Final gradient norm */
    private final double finalGradientNorm;
    
    /** 收敛容差 / Convergence tolerance */
    private double tolerance;
    
    // ==================== 执行统计 ====================
    
    /** 执行时间（毫秒）/ Execution time in milliseconds */
    private final long executionTimeMs;
    
    /** 函数评估次数 / Number of function evaluations */
    private final int functionEvaluations;
    
    /** 梯度评估次数 / Number of gradient evaluations */
    private final int gradientEvaluations;
    
    // ==================== 优化过程历史 ====================
    
    /** 函数值历史 / Function value history */
    private final List<Double> functionValueHistory;
    
    /** 梯度范数历史 / Gradient norm history */
    private final List<Double> gradientNormHistory;
    
    /** 参数历史（可选，用于轨迹分析）/ Parameter history (optional, for trajectory analysis) */
    private final List<IVector> parameterHistory;
    
    // ==================== 约束优化相关 ====================
    
    /** 约束违反度 / Constraint violation */
    private final double constraintViolation;
    
    /** 拉格朗日乘数 / Lagrange multipliers */
    private final IVector lagrangeMultipliers;
    
    /** 是否满足KKT条件 / Whether KKT conditions are satisfied */
    private final boolean kktSatisfied;
    
    // ==================== 构造函数 ====================
    
    /**
     * 基本构造函数
     * @param optimalValue 最优函数值
     * @param optimalPoint 最优点
     */
    public OptResult(double optimalValue, IVector optimalPoint) {
        this(optimalValue, optimalPoint, null, Double.NaN, true, "Manual construction", 
             0, 0, 0.0, 1e-6, 0L, 0, 0, null, null, null, 0.0, null, false);
    }
    
    /**
     * 从Tuple2构造
     * @param result 优化器返回的结果元组
     */
    public OptResult(Tuple2<Double, IVector> result) {
        this(result._1, result._2);
    }
    
    /**
     * 完整构造函数
     */
    public OptResult(double optimalValue, IVector optimalPoint, IVector initialPoint, 
                    double initialValue, boolean converged, String convergenceReason,
                    int iterations, int maxIterations, double finalGradientNorm, 
                    double tolerance, long executionTimeMs, int functionEvaluations,
                    int gradientEvaluations, List<Double> functionValueHistory,
                    List<Double> gradientNormHistory, List<IVector> parameterHistory,
                    double constraintViolation, IVector lagrangeMultipliers, 
                    boolean kktSatisfied) {
        this.optimalValue = optimalValue;
        this.optimalPoint = optimalPoint;
        this.initialPoint = initialPoint;
        this.initialValue = initialValue;
        this.converged = converged;
        this.convergenceReason = convergenceReason;
        this.iterations = iterations;
        this.maxIterations = maxIterations;
        this.finalGradientNorm = finalGradientNorm;
        this.tolerance = tolerance;
        this.executionTimeMs = executionTimeMs;
        this.functionEvaluations = functionEvaluations;
        this.gradientEvaluations = gradientEvaluations;
        this.functionValueHistory = functionValueHistory != null ? 
            new ArrayList<>(functionValueHistory) : new ArrayList<>();
        this.gradientNormHistory = gradientNormHistory != null ? 
            new ArrayList<>(gradientNormHistory) : new ArrayList<>();
        this.parameterHistory = parameterHistory != null ? 
            new ArrayList<>(parameterHistory) : new ArrayList<>();
        this.constraintViolation = constraintViolation;
        this.lagrangeMultipliers = lagrangeMultipliers;
        this.kktSatisfied = kktSatisfied;
    }
    
    // ==================== Getter 方法 ====================
    
    /**
     * 获取最优函数值
     * @return 最优函数值
     */
    public double getOptimalValue() {
        return optimalValue;
    }
    
    /**
     * 获取最优点
     * @return 最优点向量
     */
    public IVector getOptimalPoint() {
        return optimalPoint;
    }
    
    /**
     * 获取初始点
     * @return 初始点向量
     */
    public IVector getInitialPoint() {
        return initialPoint;
    }
    
    /**
     * 获取初始函数值
     * @return 初始函数值
     */
    public double getInitialValue() {
        return initialValue;
    }
    
    /**
     * 是否收敛
     * @return 收敛状态
     */
    public boolean isConverged() {
        return converged;
    }
    
    /**
     * 获取收敛原因
     * @return 收敛原因描述
     */
    public String getConvergenceReason() {
        return convergenceReason;
    }
    
    /**
     * 获取迭代次数
     * @return 实际迭代次数
     */
    public int getIterations() {
        return iterations;
    }
    
    /**
     * 获取最大迭代次数
     * @return 最大迭代次数
     */
    public int getMaxIterations() {
        return maxIterations;
    }
    
    /**
     * 获取最终梯度范数
     * @return 最终梯度范数
     */
    public double getFinalGradientNorm() {
        return finalGradientNorm;
    }
    
    /**
     * 获取收敛容差
     * @return 收敛容差
     */
    public double getTolerance() {
        return tolerance;
    }

    public void setConverged(boolean converged) {
        this.converged = converged;
    }

    public void setTolerance(double tolerance) {
        this.tolerance = tolerance;
    }
    
    /**
     * 获取执行时间
     * @return 执行时间（毫秒）
     */
    public long getExecutionTimeMs() {
        return executionTimeMs;
    }
    
    /**
     * 获取函数评估次数
     * @return 函数评估次数
     */
    public int getFunctionEvaluations() {
        return functionEvaluations;
    }
    
    /**
     * 获取梯度评估次数
     * @return 梯度评估次数
     */
    public int getGradientEvaluations() {
        return gradientEvaluations;
    }
    
    /**
     * 获取函数值历史
     * @return 函数值历史列表的副本
     */
    public List<Double> getFunctionValueHistory() {
        return new ArrayList<>(functionValueHistory);
    }
    
    /**
     * 获取梯度范数历史
     * @return 梯度范数历史列表的副本
     */
    public List<Double> getGradientNormHistory() {
        return new ArrayList<>(gradientNormHistory);
    }
    
    /**
     * 获取参数历史
     * @return 参数历史列表的副本
     */
    public List<IVector> getParameterHistory() {
        return new ArrayList<>(parameterHistory);
    }
    
    /**
     * 获取约束违反度
     * @return 约束违反度
     */
    public double getConstraintViolation() {
        return constraintViolation;
    }
    
    /**
     * 获取拉格朗日乘数
     * @return 拉格朗日乘数向量
     */
    public IVector getLagrangeMultipliers() {
        return lagrangeMultipliers;
    }
    
    /**
     * 是否满足KKT条件
     * @return KKT条件满足状态
     */
    public boolean isKktSatisfied() {
        return kktSatisfied;
    }
    
    // ==================== 便利方法 ====================
    
    /**
     * 转换为Tuple2格式（兼容现有优化器接口）
     * @return 包含最优值和最优点的元组
     */
    public Tuple2<Double, IVector> toTuple2() {
        return new Tuple2<>(optimalValue, optimalPoint);
    }
    
    /**
     * 计算优化改进程度
     * @return 函数值改进程度，如果没有初始值则返回NaN
     */
    public double getImprovement() {
        if (Double.isNaN(initialValue)) {
            return Double.NaN;
        }
        return initialValue - optimalValue;
    }
    
    /**
     * 计算相对改进程度
     * @return 相对改进程度（百分比），如果没有初始值则返回NaN
     */
    public double getRelativeImprovement() {
        if (Double.isNaN(initialValue) || Math.abs(initialValue) < 1e-12) {
            return Double.NaN;
        }
        return (initialValue - optimalValue) / Math.abs(initialValue) * 100.0;
    }
    
    /**
     * 获取收敛效率（每次迭代的平均改进）
     * @return 收敛效率，如果迭代次数为0或没有初始值则返回NaN
     */
    public double getConvergenceEfficiency() {
        if (iterations == 0 || Double.isNaN(initialValue)) {
            return Double.NaN;
        }
        return getImprovement() / iterations;
    }
    
    /**
     * 验证结果的有效性
     * @return 结果验证信息
     */
    public ResultValidation validate() {
        List<String> warnings = new ArrayList<>();
        List<String> errors = new ArrayList<>();
        
        // 检查基本有效性
        if (optimalPoint == null) {
            errors.add("最优点为空");
        }
        if (Double.isInfinite(optimalValue)) {
            errors.add("最优值为无穷大");
        }
        if (Double.isNaN(optimalValue)) {
            errors.add("最优值为NaN");
        }
        
        // 检查收敛性
        if (!converged) {
            warnings.add("优化未收敛");
        }
        if (iterations >= maxIterations && maxIterations > 0) {
            warnings.add("达到最大迭代次数");
        }
        
        // 检查梯度
        if (finalGradientNorm > tolerance * 10) {
            warnings.add("最终梯度范数较大: " + finalGradientNorm);
        }
        
        // 检查约束
        if (constraintViolation > 1e-6) {
            warnings.add("存在约束违反: " + constraintViolation);
        }
        
        return new ResultValidation(errors.isEmpty(), errors, warnings);
    }
    
    /**
     * 比较两个优化结果
     * @param other 另一个优化结果
     * @return 比较结果
     */
    public ResultComparison compareTo(OptResult other) {
        if (other == null) {
            throw new IllegalArgumentException("比较对象不能为空");
        }
        
        double valueDiff = this.optimalValue - other.optimalValue;
        boolean thisBetter = valueDiff < 0; // 假设是最小化问题
        
        int iterationDiff = this.iterations - other.iterations;
        long timeDiff = this.executionTimeMs - other.executionTimeMs;
        
        return new ResultComparison(thisBetter, valueDiff, iterationDiff, timeDiff,
                                  this.converged, other.converged);
    }
    
    /**
     * 格式化输出优化结果摘要
     * @return 格式化的结果字符串
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 优化结果摘要 / Optimization Result Summary ===\n");
        sb.append(String.format("最优值 / Optimal Value: %.6e\n", optimalValue));
        sb.append(String.format("最优点 / Optimal Point: %s\n", optimalPoint));
        sb.append(String.format("收敛状态 / Converged: %s\n", converged ? "是/Yes" : "否/No"));
        sb.append(String.format("收敛原因 / Reason: %s\n", convergenceReason));
        sb.append(String.format("迭代次数 / Iterations: %d/%d\n", iterations, maxIterations));
        sb.append(String.format("最终梯度范数 / Final Gradient Norm: %.6e\n", finalGradientNorm));
        sb.append(String.format("执行时间 / Execution Time: %d ms\n", executionTimeMs));
        sb.append(String.format("函数评估 / Function Evaluations: %d\n", functionEvaluations));
        sb.append(String.format("梯度评估 / Gradient Evaluations: %d\n", gradientEvaluations));
        
        if (!Double.isNaN(initialValue)) {
            sb.append(String.format("函数值改进 / Improvement: %.6e\n", getImprovement()));
            sb.append(String.format("相对改进 / Relative Improvement: %.2f%%\n", getRelativeImprovement()));
        }
        
        if (constraintViolation > 0) {
            sb.append(String.format("约束违反度 / Constraint Violation: %.6e\n", constraintViolation));
        }
        
        return sb.toString();
    }
    
    /**
     * 格式化输出详细结果
     * @return 详细的结果字符串
     */
    public String getDetailedReport() {
        StringBuilder sb = new StringBuilder();
        sb.append(getSummary());
        
        if (!functionValueHistory.isEmpty()) {
            sb.append("\n=== 函数值历史 / Function Value History ===\n");
            for (int i = 0; i < Math.min(10, functionValueHistory.size()); i++) {
                sb.append(String.format("Iter %d: %.6e\n", i, functionValueHistory.get(i)));
            }
            if (functionValueHistory.size() > 10) {
                sb.append("... (显示前10个值)\n");
            }
        }
        
        if (!gradientNormHistory.isEmpty()) {
            sb.append("\n=== 梯度范数历史 / Gradient Norm History ===\n");
            for (int i = 0; i < Math.min(10, gradientNormHistory.size()); i++) {
                sb.append(String.format("Iter %d: %.6e\n", i, gradientNormHistory.get(i)));
            }
            if (gradientNormHistory.size() > 10) {
                sb.append("... (显示前10个值)\n");
            }
        }
        
        // 添加验证信息
        ResultValidation validation = validate();
        sb.append("\n=== 结果验证 / Result Validation ===\n");
        sb.append(String.format("有效性 / Valid: %s\n", validation.isValid() ? "是/Yes" : "否/No"));
        if (!validation.getErrors().isEmpty()) {
            sb.append("错误 / Errors:\n");
            for (String error : validation.getErrors()) {
                sb.append("  - ").append(error).append("\n");
            }
        }
        if (!validation.getWarnings().isEmpty()) {
            sb.append("警告 / Warnings:\n");
            for (String warning : validation.getWarnings()) {
                sb.append("  - ").append(warning).append("\n");
            }
        }
        
        return sb.toString();
    }
    
    @Override
    public String toString() {
        return String.format("OptResult{value=%.6e, converged=%s, iterations=%d}", 
                           optimalValue, converged, iterations);
    }
    
    // ==================== 内部类 ====================
    
    /**
     * 结果验证类
     */
    public static class ResultValidation {
        private final boolean valid;
        private final List<String> errors;
        private final List<String> warnings;
        
        public ResultValidation(boolean valid, List<String> errors, List<String> warnings) {
            this.valid = valid;
            this.errors = new ArrayList<>(errors);
            this.warnings = new ArrayList<>(warnings);
        }
        
        public boolean isValid() { return valid; }
        public List<String> getErrors() { return new ArrayList<>(errors); }
        public List<String> getWarnings() { return new ArrayList<>(warnings); }
    }
    
    /**
     * 结果比较类
     */
    public static class ResultComparison {
        private final boolean thisBetter;
        private final double valueDifference;
        private final int iterationDifference;
        private final long timeDifference;
        private final boolean thisConverged;
        private final boolean otherConverged;
        
        public ResultComparison(boolean thisBetter, double valueDifference, 
                              int iterationDifference, long timeDifference,
                              boolean thisConverged, boolean otherConverged) {
            this.thisBetter = thisBetter;
            this.valueDifference = valueDifference;
            this.iterationDifference = iterationDifference;
            this.timeDifference = timeDifference;
            this.thisConverged = thisConverged;
            this.otherConverged = otherConverged;
        }
        
        public boolean isThisBetter() { return thisBetter; }
        public double getValueDifference() { return valueDifference; }
        public int getIterationDifference() { return iterationDifference; }
        public long getTimeDifference() { return timeDifference; }
        public boolean isThisConverged() { return thisConverged; }
        public boolean isOtherConverged() { return otherConverged; }
        
        @Override
        public String toString() {
            return String.format("ResultComparison{thisBetter=%s, valueDiff=%.6e, iterDiff=%d, timeDiff=%d ms}",
                               thisBetter, valueDifference, iterationDifference, timeDifference);
        }
    }
    
    // ==================== 建造者模式 ====================
    
    /**
     * OptResult建造者类，用于方便地构建复杂的OptResult对象
     */
    public static class Builder {
        private double optimalValue;
        private IVector optimalPoint;
        private IVector initialPoint;
        private double initialValue = Double.NaN;
        private boolean converged = true;
        private String convergenceReason = "Unknown";
        private int iterations = 0;
        private int maxIterations = 0;
        private double finalGradientNorm = 0.0;
        private double tolerance = 1e-6;
        private long executionTimeMs = 0L;
        private int functionEvaluations = 0;
        private int gradientEvaluations = 0;
        private List<Double> functionValueHistory = new ArrayList<>();
        private List<Double> gradientNormHistory = new ArrayList<>();
        private List<IVector> parameterHistory = new ArrayList<>();
        private double constraintViolation = 0.0;
        private IVector lagrangeMultipliers = null;
        private boolean kktSatisfied = false;
        
        public Builder(double optimalValue, IVector optimalPoint) {
            this.optimalValue = optimalValue;
            this.optimalPoint = optimalPoint;
        }
        
        public Builder initialPoint(IVector initialPoint) {
            this.initialPoint = initialPoint;
            return this;
        }
        
        public Builder initialValue(double initialValue) {
            this.initialValue = initialValue;
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
        
        public Builder iterations(int iterations) {
            this.iterations = iterations;
            return this;
        }
        
        public Builder maxIterations(int maxIterations) {
            this.maxIterations = maxIterations;
            return this;
        }
        
        public Builder finalGradientNorm(double finalGradientNorm) {
            this.finalGradientNorm = finalGradientNorm;
            return this;
        }
        
        public Builder tolerance(double tolerance) {
            this.tolerance = tolerance;
            return this;
        }
        
        public Builder executionTimeMs(long executionTimeMs) {
            this.executionTimeMs = executionTimeMs;
            return this;
        }
        
        public Builder functionEvaluations(int functionEvaluations) {
            this.functionEvaluations = functionEvaluations;
            return this;
        }
        
        public Builder gradientEvaluations(int gradientEvaluations) {
            this.gradientEvaluations = gradientEvaluations;
            return this;
        }
        
        public Builder functionValueHistory(List<Double> functionValueHistory) {
            this.functionValueHistory = functionValueHistory;
            return this;
        }
        
        public Builder gradientNormHistory(List<Double> gradientNormHistory) {
            this.gradientNormHistory = gradientNormHistory;
            return this;
        }
        
        public Builder parameterHistory(List<IVector> parameterHistory) {
            this.parameterHistory = parameterHistory;
            return this;
        }
        
        public Builder constraintViolation(double constraintViolation) {
            this.constraintViolation = constraintViolation;
            return this;
        }
        
        public Builder lagrangeMultipliers(IVector lagrangeMultipliers) {
            this.lagrangeMultipliers = lagrangeMultipliers;
            return this;
        }
        
        public Builder kktSatisfied(boolean kktSatisfied) {
            this.kktSatisfied = kktSatisfied;
            return this;
        }
        
        public OptResult build() {
            return new OptResult(optimalValue, optimalPoint, initialPoint, initialValue,
                               converged, convergenceReason, iterations, maxIterations,
                               finalGradientNorm, tolerance, executionTimeMs,
                               functionEvaluations, gradientEvaluations,
                               functionValueHistory, gradientNormHistory, parameterHistory,
                               constraintViolation, lagrangeMultipliers, kktSatisfied);
        }
    }
}
