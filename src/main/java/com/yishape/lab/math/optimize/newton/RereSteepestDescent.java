package com.yishape.lab.math.optimize.newton;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.RereLineSearch;
import com.yishape.lab.math.util.RerePrecision;

import java.util.ArrayList;
import java.util.List;

/**
 * 最速下降法优化算法实现
 * Steepest Descent Method Optimization Algorithm Implementation
 *
 * <p>最速下降法是一种用于求解无约束优化问题的经典一阶优化方法。
 * 该算法在每一步都沿着目标函数梯度的负方向进行搜索，这是函数值下降最快的方向。
 * The steepest descent method is a classical first-order optimization algorithm for solving
 * unconstrained optimization problems. At each step, it searches in the negative gradient
 * direction, which is the direction of steepest descent of the function.</p>
 *
 * <h3>算法特点 / Algorithm Features:</h3>
 * <ul>
 *   <li>简单易实现：只需要目标函数和梯度信息 / Simple to implement: only requires objective function and gradient</li>
 *   <li>线性收敛：对于一般函数具有线性收敛速度 / Linear convergence: linear convergence rate for general functions</li>
 *   <li>无需Hessian矩阵：只使用一阶导数信息 / No Hessian computation: only uses first derivative information</li>
 *   <li>可能收敛较慢：对于病态问题收敛速度可能很慢 / May converge slowly: slow convergence for ill-conditioned problems</li>
 * </ul>
 *
 * <h3>算法描述 / Algorithm Description:</h3>
 * <pre>
 * 1. 初始化 x_0, 设置 k = 0
 * 2. 计算梯度 g_k = ∇f(x_k)
 * 3. 设置搜索方向 d_k = -g_k
 * 4. 使用线搜索确定步长 α_k
 * 5. 更新 x_{k+1} = x_k + α_k * d_k
 * 6. 如果收敛则停止，否则 k = k + 1，转到步骤2
 * </pre>
 *
 * @author lteb2
 * @see IOptimizer
 */
public class RereSteepestDescent implements IOptimizer {

    // 最速下降法参数 / Steepest descent algorithm parameters
    private double tolerance = 1e-6;       // 收敛容差 / Convergence tolerance
    private int maxIterations = 1000;      // 最大迭代次数 / Maximum iterations
    private double initialStepSize = 1.0;  // 初始步长 / Initial step size

    /**
     * 默认构造函数，使用标准参数
     * Default constructor with standard parameters
     *
     * <p>使用默认参数: tolerance=1e-6, maxIterations=1000, initialStepSize=1.0
     * Uses default parameters: tolerance=1e-6, maxIterations=1000, initialStepSize=1.0</p>
     */
    public RereSteepestDescent() {
    }

    /**
     * 自定义参数构造函数
     * Constructor with custom parameters
     *
     * @param tolerance 收敛容差，必须大于0 / Convergence tolerance, must be greater than 0
     * @param maxIterations 最大迭代次数，必须大于0 / Maximum iterations, must be greater than 0
     * @param initialStepSize 初始步长，必须大于0 / Initial step size, must be greater than 0
     * @throws IllegalArgumentException 如果任何参数无效 / If any parameter is invalid
     */
    public RereSteepestDescent(double tolerance, int maxIterations, double initialStepSize) {
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
        this.initialStepSize = initialStepSize;
    }

    /**
     * 使用最速下降法优化无约束非线性优化问题
     * Optimize Unconstrained Nonlinear Problem Using Steepest Descent Method
     *
     * @param initX 初始点向量 / Initial point vector
     * @param objFun 目标函数 / Objective function
     * @param grdFun 梯度函数 / Gradient function
     * @return 优化结果，包含最优解、迭代信息和收敛状态 / Optimization result containing optimal solution, iteration info, and convergence status
     * @throws IllegalArgumentException 如果任何参数为null / If any parameter is null
     */
    @Override
    public OptResult optimize(IVector initX, IObjectiveFunction objFun, IGradientFunction grdFun) {
        // 参数验证 / Parameter validation
        if (initX == null) {
            throw new IllegalArgumentException("初始点不能为空 / Initial point cannot be null");
        }
        if (objFun == null) {
            throw new IllegalArgumentException("目标函数不能为空 / Objective function cannot be null");
        }
        if (grdFun == null) {
            throw new IllegalArgumentException("梯度函数不能为空 / Gradient function cannot be null");
        }

        // 记录开始时间 / Record start time
        long startTime = System.currentTimeMillis();

        // 初始化变量 / Initialize variables
        IVector x = initX.copy();  // 当前点 / Current point
        IVector initialPoint = initX.copy(); // 保存初始点 / Save initial point
        double stepSize = initialStepSize;  // 当前步长 / Current step size

        // 计算初始函数值 / Compute initial function value
        double initialValue = objFun.computeObjective(x);

        // 计算初始梯度 / Compute initial gradient
        IVector grad = grdFun.computeGradient(x);
        double initialGradNorm = grad.norm2Value();
        double finalGradientNorm = initialGradNorm;

        // 收敛历史记录 / Convergence history tracking
        List<Double> functionValueHistory = new ArrayList<>();
        List<Double> gradientNormHistory = new ArrayList<>();
        List<IVector> parameterHistory = new ArrayList<>();

        // 评估计数 / Evaluation counters
        int functionEvaluations = 1; // 初始函数值计算 / Initial function evaluation
        int gradientEvaluations = 0; // 梯度计算将在循环中开始计数 / Gradient evaluations will start counting in loop

        // 添加初始历史记录 / Add initial history records
        functionValueHistory.add(initialValue);
        gradientNormHistory.add(initialGradNorm);
        parameterHistory.add(x.copy());

        boolean converged = false;
        String convergenceReason = "Maximum iterations reached";
        int actualIterations = 0;

        // 线搜索实例（循环外创建，避免反复分配）/ Line search instance (created once outside loop)
        RereLineSearch lineSearch = new RereLineSearch();

        // 主迭代循环 / Main iteration loop
        for (int iter = 0; iter < maxIterations; iter++) {
            actualIterations = iter + 1;

            // 检查收敛条件：梯度范数足够小 / Check convergence: gradient norm is small enough
            double gradNorm = grad.norm2Value();
            finalGradientNorm = gradNorm;
            double convergenceThreshold = tolerance * Math.max(1.0, initialGradNorm);
            if (RerePrecision.compareTo(gradNorm, convergenceThreshold, tolerance) <= 0) {
                converged = true;
                convergenceReason = "Gradient norm below tolerance";
                double optimalValue = objFun.computeObjective(x);
                functionEvaluations++;

                // 构建丰富的OptResult / Build rich OptResult
                OptResult.Builder builder = new OptResult.Builder(optimalValue, x)
                    .initialPoint(initialPoint)
                    .initialValue(initialValue)
                    .converged(converged)
                    .convergenceReason(convergenceReason)
                    .iterations(actualIterations)
                    .maxIterations(maxIterations)
                    .finalGradientNorm(finalGradientNorm)
                    .tolerance(tolerance)
                    .executionTimeMs(System.currentTimeMillis() - startTime)
                    .functionEvaluations(functionEvaluations)
                    .gradientEvaluations(gradientEvaluations)
                    .functionValueHistory(functionValueHistory)
                    .gradientNormHistory(gradientNormHistory)
                    .parameterHistory(parameterHistory);

                return builder.build();
            }

            // 线搜索确定步长 / Line search to determine step size
            // 搜索方向为负梯度方向 / Search direction is negative gradient direction
            IVector searchDirection = grad.multiplyByScalar(-1.0);
            stepSize = lineSearch.search(x, searchDirection, objFun, grdFun, grad);

            // 更新位置 / Update position
            IVector newX = x.add(searchDirection.multiplyByScalar(stepSize));

            // 计算新梯度 / Compute new gradient
            IVector newGrad = grdFun.computeGradient(newX);
            gradientEvaluations++;

            // 计算新函数值并记录 / Compute new function value and record
            double newValue = objFun.computeObjective(newX);
            functionEvaluations++;
            functionValueHistory.add(newValue);
            gradientNormHistory.add(newGrad.norm2Value());
            parameterHistory.add(newX.copy());

            // 更新变量 / Update variables
            x = newX;
            grad = newGrad;
        }

        // 达到最大迭代次数，返回当前最优解 / Maximum iterations reached, return current best solution
        double finalValue = objFun.computeObjective(x);
        functionEvaluations++;

        // 构建丰富的OptResult / Build rich OptResult
        OptResult.Builder builder = new OptResult.Builder(finalValue, x)
            .initialPoint(initialPoint)
            .initialValue(initialValue)
            .converged(converged)
            .convergenceReason(convergenceReason)
            .iterations(actualIterations)
            .maxIterations(maxIterations)
            .finalGradientNorm(finalGradientNorm)
            .tolerance(tolerance)
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .functionEvaluations(functionEvaluations)
            .gradientEvaluations(gradientEvaluations)
            .functionValueHistory(functionValueHistory)
            .gradientNormHistory(gradientNormHistory)
            .parameterHistory(parameterHistory);

        return builder.build();
    }

    // Getter和Setter方法 / Getter and Setter methods

    /**
     * 获取收敛容差
     * Get convergence tolerance
     *
     * @return 收敛容差 / Convergence tolerance
     */
    public double getTolerance() {
        return tolerance;
    }

    /**
     * 设置收敛容差
     * Set convergence tolerance
     *
     * @param tolerance 收敛容差，必须大于0 / Convergence tolerance, must be greater than 0
     * @throws IllegalArgumentException 如果容差不大于0 / If tolerance is not greater than 0
     */
    public void setTolerance(double tolerance) {
        this.tolerance = Math.max(1e-12, tolerance);
    }

    /**
     * 获取最大迭代次数
     * Get maximum iterations
     *
     * @return 最大迭代次数 / Maximum iterations
     */
    public int getMaxIterations() {
        return maxIterations;
    }

    /**
     * 设置最大迭代次数
     * Set maximum iterations
     *
     * @param maxIterations 最大迭代次数，必须大于0 / Maximum iterations, must be greater than 0
     * @throws IllegalArgumentException 如果最大迭代次数不大于0 / If max iterations is not greater than 0
     */
    public void setMaxIterations(int maxIterations) {
        this.maxIterations = Math.max(1, maxIterations);
    }

    /**
     * 获取初始步长
     * Get initial step size
     *
     * @return 初始步长 / Initial step size
     */
    public double getInitialStepSize() {
        return initialStepSize;
    }

    /**
     * 设置初始步长
     * Set initial step size
     *
     * @param initialStepSize 初始步长，必须大于0 / Initial step size, must be greater than 0
     * @throws IllegalArgumentException 如果初始步长不大于0 / If initial step size is not greater than 0
     */
    public void setInitialStepSize(double initialStepSize) {
        this.initialStepSize = Math.max(1e-12, initialStepSize);
    }
}