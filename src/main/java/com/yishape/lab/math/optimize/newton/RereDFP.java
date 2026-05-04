package com.yishape.lab.math.optimize.newton;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.RereLineSearch;
import com.yishape.lab.math.util.RerePrecision;

import java.util.ArrayList;
import java.util.List;

/**
 * DFP拟牛顿优化算法实现
 * DFP (Davidon-Fletcher-Powell) Quasi-Newton Optimization Algorithm Implementation
 *
 * <p>DFP算法是一种用于求解无约束非线性优化问题的拟牛顿方法。
 * 它通过迭代近似Hessian矩阵的逆来加速收敛，无需直接计算二阶导数。
 * DFP (Davidon-Fletcher-Powell) is a quasi-Newton method for solving unconstrained
 * nonlinear optimization problems. It accelerates convergence by iteratively approximating
 * the inverse of the Hessian matrix without directly computing second derivatives.</p>
 *
 * <h3>算法特点 / Algorithm Features:</h3>
 * <ul>
 *   <li>超线性收敛: 在最优解附近具有快速收敛速度 / Superlinear convergence: Fast convergence near optimal solution</li>
 *   <li>无需Hessian计算: 只需目标函数值和梯度信息 / No Hessian computation: Only requires objective function and gradient</li>
 *   <li>正定更新: 保持Hessian近似的正定性 / Positive definite updates: Maintains positive definiteness of Hessian approximation</li>
 *   <li>适用于中大规模问题 / Suitable for medium-scale problems</li>
 * </ul>
 *
 * <h3>更新公式 / Update Formula:</h3>
 * <pre>
 * H_{k+1} = H_k + (s * s^T) / (s^T * y) - (H_k * y * y^T * H_k) / (y^T * H_k * y)
 * 其中: s = x_{k+1} - x_k, y = g_{k+1} - g_k
 * where: s = x_{k+1} - x_k, y = g_{k+1} - g_k
 * </pre>
 *
 * @author lteb2
 * @see IOptimizer
 */
public class RereDFP implements IOptimizer {

    // DFP algorithm parameters
    private double tolerance = 1e-6;       // 收敛容差 / Convergence tolerance
    private int maxIterations = 1000;      // 最大迭代次数 / Maximum iterations

    /**
     * 默认构造函数，使用标准参数
     * Default constructor with standard parameters
     *
     * <p>使用默认参数: tolerance=1e-6, maxIterations=1000
     * Uses default parameters: tolerance=1e-6, maxIterations=1000</p>
     */
    public RereDFP() {
    }

    /**
     * 自定义参数构造函数
     * Constructor with custom parameters
     *
     * @param tolerance 收敛容差，必须大于0 / Convergence tolerance, must be greater than 0
     * @param maxIterations 最大迭代次数，必须大于0 / Maximum iterations, must be greater than 0
     * @throws IllegalArgumentException 如果任何参数无效 / If any parameter is invalid
     */
    public RereDFP(double tolerance, int maxIterations) {
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
    }

    /**
     * 使用DFP算法优化无约束非线性优化问题
     * Optimize Unconstrained Nonlinear Problem Using DFP Algorithm
     *
     * @param initX 初始点向量 / Initial point vector
     * @param objFun 目标函数 / Objective function
     * @param grdFun 梯度函数 / Gradient function
     * @return 优化结果，包含最优解、迭代信息和收敛状态 / Optimization result containing optimal solution, iteration info, and convergence status
     * @throws IllegalArgumentException 如果任何参数为null / If any parameter is null
     */
    @Override
    public OptResult optimize(IVector initX, IObjectiveFunction objFun, IGradientFunction grdFun) {
        // Parameter validation
        if (initX == null) {
            throw new IllegalArgumentException("初始点不能为空 / Initial point cannot be null");
        }
        if (objFun == null) {
            throw new IllegalArgumentException("目标函数不能为空 / Objective function cannot be null");
        }
        if (grdFun == null) {
            throw new IllegalArgumentException("梯度函数不能为空 / Gradient function cannot be null");
        }

        // Record start time
        long startTime = System.currentTimeMillis();

        // Initialize variables
        IVector x = initX.copy();  // 当前点 / Current point
        IVector initialPoint = initX.copy(); // 保存初始点 / Save initial point
        int n = x.length();       // 问题维度 / Problem dimension

        // Compute initial function value
        double initialValue = objFun.computeObjective(x);

        // Initialize inverse Hessian approximation as identity matrix
        IMatrix<Double> H = Linalg.eye(n);

        // Convergence history tracking
        List<Double> functionValueHistory = new ArrayList<>();
        List<Double> gradientNormHistory = new ArrayList<>();
        List<IVector> parameterHistory = new ArrayList<>();

        // Evaluation counters
        int functionEvaluations = 1; // Initial function value computation
        int gradientEvaluations = 0; // Gradient evaluations will start counting in loop

        // Compute initial gradient
        IVector grad = grdFun.computeGradient(x);
        gradientEvaluations++;
        double initialGradNorm = (Double) grad.norm2();
        double finalGradientNorm = initialGradNorm;

        // Add initial history records
        functionValueHistory.add(initialValue);
        gradientNormHistory.add(initialGradNorm);
        parameterHistory.add(x.copy());

        boolean converged = false;
        String convergenceReason = "Maximum iterations reached";
        int actualIterations = 0;

        // Main iteration loop
        for (int iter = 0; iter < maxIterations; iter++) {
            actualIterations = iter + 1;

            // Check convergence: gradient norm is small enough
            double gradNorm = (Double) grad.norm2();
            finalGradientNorm = gradNorm;
            double convergenceThreshold = tolerance * Math.max(1.0, initialGradNorm);
            if (RerePrecision.compareTo(gradNorm, convergenceThreshold, tolerance) < 0) {
                converged = true;
                convergenceReason = "Gradient norm below tolerance";
                double optimalValue = objFun.computeObjective(x);
                functionEvaluations++;

                // Build rich OptResult
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

            // Compute search direction: d = -H * g
            IVector searchDirection = H.mmul(grad).multiplyScalar(-1.0);

            // Line search to determine step size
            double stepSize = new RereLineSearch().search(x, searchDirection, objFun, grdFun, grad);

            // Update position
            IVector newX = x.add(searchDirection.multiplyScalar(stepSize));
            IVector newGrad = grdFun.computeGradient(newX);
            gradientEvaluations++;

            // Compute new function value and record
            double newValue = objFun.computeObjective(newX);
            functionEvaluations++;
            functionValueHistory.add(newValue);
            gradientNormHistory.add((Double) newGrad.norm2());
            parameterHistory.add(newX.copy());

            // Compute differences
            IVector s = newX.sub(x);           // s = x_{k+1} - x_k
            IVector y = newGrad.sub(grad);     // y = g_{k+1} - g_k

            // Update inverse Hessian approximation using DFP formula
            H = updateInverseHessian(H, s, y);

            // Update current point and gradient
            x = newX;
            grad = newGrad;
        }

        // Maximum iterations reached, return current best solution
        double finalValue = objFun.computeObjective(x);
        functionEvaluations++;

        // Build rich OptResult
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

    /**
     * 使用DFP公式更新Hessian矩阵的逆近似
     * Update Inverse Hessian Approximation Using DFP Formula
     *
     * <p>DFP更新公式:
     * H_{k+1} = H_k + (s * s^T) / (s^T * y) - (H_k * y * y^T * H_k) / (y^T * H_k * y)
     *
     * 其中:
     * - s = x_{k+1} - x_k (位置差向量)
     * - y = g_{k+1} - g_k (梯度差向量)
     * - H_k 是第k次迭代的Hessian逆近似
     *
     * DFP update formula:
     * H_{k+1} = H_k + (s * s^T) / (s^T * y) - (H_k * y * y^T * H_k) / (y^T * H_k * y)
     *
     * where:
     * - s = x_{k+1} - x_k (position difference vector)
     * - y = g_{k+1} - g_k (gradient difference vector)
     * - H_k is the inverse Hessian approximation at iteration k</p>
     *
     * @param H 当前Hessian矩阵逆近似 / Current inverse Hessian approximation
     * @param s 位置差向量 x_{k+1} - x_k / Position difference vector
     * @param y 梯度差向量 g_{k+1} - g_k / Gradient difference vector
     * @return 更新后的Hessian矩阵逆近似 / Updated inverse Hessian approximation
     */
    private IMatrix<Double> updateInverseHessian(IMatrix<Double> H, IVector s, IVector y) {
        // Compute s^T * y (needed for DFP update)
        double sTy = (Double) s.innerProduct(y);

        // Check curvature condition: s^T * y > 0, required for positive definiteness
        if (RerePrecision.compareTo(sTy, 1e-10, tolerance) <= 0) {
            // If curvature condition is not satisfied, return current Hessian approximation
            return H;
        }

        // Convert vectors to column matrices for matrix operations
        IMatrix<Double> sMatrix = s.asColumnVector();
        IMatrix<Double> yMatrix = y.asColumnVector();

        // Compute s * s^T
        IMatrix<Double> sOuter = sMatrix.mmul(sMatrix.transposeNew());

        // Compute (s * s^T) / (s^T * y)
        IMatrix<Double> term1 = sOuter.multiplyScalar(1.0 / sTy);

        // Compute H * y
        IVector Hy = H.mmul(y);
        IMatrix<Double> HyMatrix = Hy.asColumnVector();

        // Compute (H * y) * (H * y)^T
        IMatrix<Double> HyOuter = HyMatrix.mmul(HyMatrix.transposeNew());

        // Compute y^T * H * y
        double yTHy = (Double) y.innerProduct(Hy);

        // Compute (H * y * y^T * H) / (y^T * H * y)
        IMatrix<Double> term2 = HyOuter.multiplyScalar(1.0 / yTHy);

        // Apply DFP update formula: H_{k+1} = H_k + term1 - term2
        IMatrix<Double> newH = H.add(term1).sub(term2);

        return newH;
    }

    // Getter and Setter methods / 获取和设置方法

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
}