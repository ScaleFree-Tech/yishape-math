package com.yishape.lab.math.optimize.newton;

import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.RereLineSearch;
import com.yishape.lab.math.util.RerePrecision;
import java.util.ArrayList;

/**
 * LBFGS优化器（性能优化版）
 * <p>
 * LBFGS（Limited-memory Broyden-Fletcher-Goldfarb-Shanno）是一种拟牛顿法优化算法，
 * 用于解决无约束非线性优化问题。该算法通过使用有限的历史信息来近似Hessian矩阵的逆，
 * 具有内存使用量小、收敛速度快的特点。
 * </p>
 * <p>
 * LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton optimization algorithm
 * for solving unconstrained nonlinear optimization problems. The algorithm approximates the inverse
 * of the Hessian matrix using limited historical information, featuring low memory usage and fast convergence.
 * </p>
 * 
 * <h3>性能优化 / Performance Optimizations:</h3>
 * <ul>
 *   <li>使用循环缓冲区存储历史信息，O(1)插入和删除</li>
 *   <li>缓存梯度范数和方向导数，避免重复计算</li>
 *   <li>预计算并缓存gamma值</li>
 *   <li>减少不必要的向量复制</li>
 * </ul>
 * 
 * @author lteb2
 */
public class RereLBFGS implements IOptimizer{

    // LBFGS算法参数 / LBFGS algorithm parameters
    private int m = 10;                    // 存储的历史信息对数 / Number of stored history pairs
    private double tolerance = 1e-6;       // 收敛容差 / Convergence tolerance
    private int maxIterations = 100;      // 最大迭代次数 / Maximum iterations

    // 缓存变量 / Cached variables
    private double cachedGradNorm = 0;     // 缓存的梯度范数 / Cached gradient norm
    private double cachedDirectionalDerivative = 0; // 缓存的方向导数 / Cached directional derivative
    private double cachedGamma = 1.0;      // 缓存的gamma值 / Cached gamma value
    
    /**
     * 构造函数，使用默认参数 / Constructor with default parameters
     */
    public RereLBFGS() {
    }
    
    /**
     * 构造函数，允许自定义参数 / Constructor with custom parameters
     * 
     * @param m 存储的历史信息对数 / Number of stored history pairs
     * @param tolerance 收敛容差 / Convergence tolerance
     * @param maxIterations 最大迭代次数 / Maximum iterations
     */
    public RereLBFGS(int m, double tolerance, int maxIterations) {
        this.m = m;
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
    }

    /**
     * 根据提供的初始点、目标函数计算方法、梯度计算方法，求解数学最优化问题
     * <p>
     * 使用LBFGS算法进行无约束优化：
     * 1. 初始化：设置初始点和参数
     * 2. 迭代过程：
     *    a) 使用缓存的梯度范数检查收敛条件
     *    b) 使用两循环递归计算搜索方向（使用缓存的gamma）
     *    c) 线搜索确定步长（传递方向导数以避免重复计算）
     *    d) 更新参数和历史信息
     * 3. 返回最优解
     * </p>
     * 
     * @param initX 初始点 / Initial point
     * @param objFun 目标函数计算法 / Objective function
     * @param grdFun 梯度计算法 / Gradient function
     * @return 返回最优值及最优点的变量值（向量） / Returns optimal value and optimal point
     * @throws IllegalArgumentException 如果输入参数无效 / if input parameters are invalid
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
        int n = x.length();       // 问题维度 / Problem dimension
        
        // 计算初始函数值 / Compute initial function value
        double initialValue = objFun.computeObjective(x);
        
        // 使用循环缓冲区存储历史信息（O(1)插入删除）/ Use circular buffer for history (O(1) insert/delete)
        LinkedList<IVector> s_history = new LinkedList<>();  // 位置差向量历史 / Position difference history
        LinkedList<IVector> y_history = new LinkedList<>();  // 梯度差向量历史 / Gradient difference history
        LinkedList<Double> rho_history = new LinkedList<>(); // ρ值历史 / Rho value history
        LinkedList<Double> gamma_history = new LinkedList<>(); // gamma值历史 / Gamma value history
        
        // 收敛历史记录 / Convergence history tracking
        List<Double> functionValueHistory = new ArrayList<>(Math.min(maxIterations, 100));
        List<Double> gradientNormHistory = new ArrayList<>(Math.min(maxIterations, 100));
        List<IVector> parameterHistory = new ArrayList<>(Math.min(maxIterations, 100));
        
        // 评估计数 / Evaluation counters
        int functionEvaluations = 1; // 初始函数值计算 / Initial function evaluation
        int gradientEvaluations = 0; // 梯度计算将在循环中开始计数 / Gradient evaluations will start counting in loop
        
        // 计算初始梯度 / Compute initial gradient
        IVector grad = grdFun.computeGradient(x);
        gradientEvaluations++;
        
        // 缓存初始梯度范数 / Cache initial gradient norm
        cachedGradNorm = (Double) grad.norm2();
        double initialGradNorm = cachedGradNorm;
        
        // 添加初始历史记录 / Add initial history records
        functionValueHistory.add(initialValue);
        gradientNormHistory.add(initialGradNorm);
        parameterHistory.add(x.copy());
        
        // 最佳解跟踪 / Best solution tracking
        IVector bestX = x.copy();
        double bestValue = initialValue;
        double bestGradNorm = initialGradNorm;
        
        boolean converged = false;
        String convergenceReason = "Maximum iterations reached";
        int actualIterations = 0;
        
        // 停滞检测变量 / Stagnation detection variables
        double previousValue = initialValue;
        int stagnationCounter = 0;
        int maxStagnationIterations = 10;
        
        // 当前函数值（避免重复计算）/ Current function value (avoid recomputation)
        double currentValue = initialValue;
        
        // 缓存方向导数 / Cache directional derivative
        cachedDirectionalDerivative = 0;
        
        // 主迭代循环 / Main iteration loop
        for (int iter = 0; iter < maxIterations; iter++) {
            actualIterations = iter + 1;
            
            // 使用缓存的梯度范数 / Use cached gradient norm
            double gradNorm = cachedGradNorm;
            
            // 更新最佳解（只在改善时更新）/ Update best solution (only when improved)
            if (currentValue < bestValue) {
                bestX = x.copy();
                bestValue = currentValue;
                bestGradNorm = gradNorm;
            }
            
            // 改进的收敛检查：使用绝对和相对容差的组合 / Improved convergence check: use combination of absolute and relative tolerance
            double convergenceThreshold = Math.max(tolerance, tolerance * Math.max(1.0, initialGradNorm));
            if (RerePrecision.compareTo(gradNorm, convergenceThreshold, tolerance) < 0) {
                converged = true;
                convergenceReason = "Gradient norm below tolerance";
                
                // 复用当前函数值，不重新计算 / Reuse current function value, don't recompute
                
                // 构建丰富的OptResult / Build rich OptResult
                OptResult.Builder builder = new OptResult.Builder(currentValue, x)
                    .initialPoint(initX)
                    .initialValue(initialValue)
                    .converged(converged)
                    .convergenceReason(convergenceReason)
                    .iterations(actualIterations)
                    .maxIterations(maxIterations)
                    .finalGradientNorm(gradNorm)
                    .tolerance(tolerance)
                    .executionTimeMs(System.currentTimeMillis() - startTime)
                    .functionEvaluations(functionEvaluations)
                    .gradientEvaluations(gradientEvaluations)
                    .functionValueHistory(functionValueHistory)
                    .gradientNormHistory(gradientNormHistory)
                    .parameterHistory(parameterHistory);
                
                return builder.build();
            }
            
            // 停滞检测 / Stagnation detection
            if (iter > 0) {
                double valueChange = Math.abs(currentValue - previousValue);
                // 如果函数值变化非常小，增加停滞计数器 / If function value change is very small, increment stagnation counter
                if (RerePrecision.compareTo(valueChange, 1e-12 * Math.max(1.0, Math.abs(currentValue)), 1e-12) < 0) {
                    stagnationCounter++;
                    if (stagnationCounter >= maxStagnationIterations) {
                        converged = true;
                        convergenceReason = "Stagnation detected";
                        // 使用最佳解 / Use best solution
                        OptResult.Builder builder = new OptResult.Builder(bestValue, bestX)
                            .initialPoint(initX)
                            .initialValue(initialValue)
                            .converged(converged)
                            .convergenceReason(convergenceReason)
                            .iterations(actualIterations)
                            .maxIterations(maxIterations)
                            .finalGradientNorm(bestGradNorm)
                            .tolerance(tolerance)
                            .executionTimeMs(System.currentTimeMillis() - startTime)
                            .functionEvaluations(functionEvaluations)
                            .gradientEvaluations(gradientEvaluations)
                            .functionValueHistory(functionValueHistory)
                            .gradientNormHistory(gradientNormHistory)
                            .parameterHistory(parameterHistory);
                        return builder.build();
                    }
                } else {
                    // 重置停滞计数器 / Reset stagnation counter
                    stagnationCounter = 0;
                }
            }
            previousValue = currentValue;
            
            // 计算搜索方向：使用两循环递归（传递缓存的gamma）/ Compute search direction: two-loop recursion (pass cached gamma)
            IVector direction = computeSearchDirection(grad, s_history, y_history, rho_history, gamma_history);
            
            // 计算并缓存方向导数 / Compute and cache directional derivative
            cachedDirectionalDerivative = (Double) grad.innerProduct(direction);
            
            // 线搜索确定步长（传递方向导数）/ Line search to determine step size (pass directional derivative)
            double stepSize = new RereLineSearch().searchWithCachedDerivative(
                x, direction, objFun, grdFun, grad, cachedDirectionalDerivative);
            
            // 更新位置 / Update position
            IVector newX = x.add(direction.multiplyScalar(stepSize));
            IVector newGrad = grdFun.computeGradient(newX);
            gradientEvaluations++;
            
            // 计算新函数值并记录 / Compute new function value and record
            newValue = objFun.computeObjective(newX);
            functionEvaluations++;
            
            // 缓存新梯度范数 / Cache new gradient norm
            double newGradNorm = (Double) newGrad.norm2();
            cachedGradNorm = newGradNorm;
            
            functionValueHistory.add(newValue);
            gradientNormHistory.add(newGradNorm);
            parameterHistory.add(newX.copy());
            
            // 更新历史信息（包含gamma计算）/ Update history information (including gamma computation)
            updateHistory(x, newX, grad, newGrad, s_history, y_history, rho_history, gamma_history);
            
            // 更新当前点和梯度 / Update current point and gradient
            x = newX;
            grad = newGrad;
            currentValue = newValue;
        }
        
        // 达到最大迭代次数，返回找到的最佳解 / Maximum iterations reached, return best solution found
        
        // 构建丰富的OptResult / Build rich OptResult
        OptResult.Builder builder = new OptResult.Builder(bestValue, bestX)
            .initialPoint(initX)
            .initialValue(initialValue)
            .converged(converged)
            .convergenceReason(convergenceReason)
            .iterations(actualIterations)
            .maxIterations(maxIterations)
            .finalGradientNorm(bestGradNorm)
            .tolerance(tolerance)
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .functionEvaluations(functionEvaluations)
            .gradientEvaluations(gradientEvaluations)
            .functionValueHistory(functionValueHistory)
            .gradientNormHistory(gradientNormHistory)
            .parameterHistory(parameterHistory);
        
        return builder.build();
    }
    
    // 新函数值变量（避免重复计算）/ New function value variable (avoid recomputation)
    private double newValue;
    
    /**
     * 计算LBFGS搜索方向（优化版）
     * <p>
     * 使用两循环递归算法计算搜索方向：
     * 1. 第一个循环：从最新到最旧的历史信息，向后递归
     * 2. 第二个循环：从最旧到最新的历史信息，向前递归
     * 3. 使用缓存的gamma值进行Hessian近似
     * </p>
     * 
     * @param grad 当前梯度 / Current gradient
     * @param s_history 位置差历史 / Position difference history
     * @param y_history 梯度差历史 / Gradient difference history  
     * @param rho_history ρ值历史 / Rho value history
     * @param gamma_history gamma值历史 / Gamma value history
     * @return 搜索方向 / Search direction
     */
    private IVector computeSearchDirection(IVector grad, LinkedList<IVector> s_history,
                                         LinkedList<IVector> y_history, LinkedList<Double> rho_history,
                                         LinkedList<Double> gamma_history) {
        
        IVector q = grad.copy();
        int historySize = s_history.size();
        double[] alpha = new double[historySize];
        
        // 第一个循环：向后递归 / First loop: backward recursion
        for (int i = historySize - 1; i >= 0; i--) {
            alpha[i] = rho_history.get(i) * (Double) s_history.get(i).innerProduct(q);
            q = q.sub(y_history.get(i).multiplyScalar(alpha[i]));
        }
        
        // 应用初始Hessian近似（使用缓存的gamma或计算新的gamma）/ Apply initial Hessian approximation (use cached gamma or compute new)
        IVector r = applyInitialHessianApproximationOptimized(q, s_history, y_history, gamma_history);
        
        // 第二个循环：向前递归 / Second loop: forward recursion
        for (int i = 0; i < historySize; i++) {
            double beta = rho_history.get(i) * (Double) y_history.get(i).innerProduct(r);
            r = r.add(s_history.get(i).multiplyScalar(alpha[i] - beta));
        }
        
        // Return the negative direction (descent direction)
        return r.multiplyScalar(-1.0);
    }
    
    /**
     * 应用初始Hessian矩阵近似（优化版）
     * <p>
     * 使用缓存的gamma值，只在必要时重新计算。
     * </p>
     * 
     * @param q 输入向量 / Input vector
     * @param s_history 位置差历史 / Position difference history
     * @param y_history 梯度差历史 / Gradient difference history
     * @param gamma_history gamma值历史 / Gamma value history
     * @return 应用初始Hessian近似后的向量 / IVector after applying initial Hessian approximation
     */
    private IVector applyInitialHessianApproximationOptimized(IVector q, LinkedList<IVector> s_history,
                                                               LinkedList<IVector> y_history, LinkedList<Double> gamma_history) {
        if (s_history.isEmpty()) {
            // 如果没有历史信息，使用单位矩阵 / If no history, use identity matrix
            return q;
        }
        
        // 检查是否有缓存的gamma / Check for cached gamma
        int historySize = s_history.size();
        if (!gamma_history.isEmpty()) {
            // 使用缓存的gamma / Use cached gamma
            double gamma = gamma_history.getLast();
            return q.multiplyScalar(gamma);
        }
        
        // 计算新的gamma / Compute new gamma
        return applyInitialHessianApproximation(q, s_history, y_history);
    }
    
    /**
     * 应用初始Hessian矩阵近似
     * <p>
     * 使用标量γ乘以单位矩阵作为初始Hessian近似，其中：
     * γ = (s^T * y) / (y^T * y)
     * 这是基于最新的历史信息计算得出的。
     * </p>
     * 
     * @param q 输入向量 / Input vector
     * @param s_history 位置差历史 / Position difference history
     * @param y_history 梯度差历史 / Gradient difference history
     * @return 应用初始Hessian近似后的向量 / IVector after applying initial Hessian approximation
     */
    private IVector applyInitialHessianApproximation(IVector q, LinkedList<IVector> s_history, LinkedList<IVector> y_history) {
        if (s_history.isEmpty()) {
            // 如果没有历史信息，使用单位矩阵 / If no history, use identity matrix
            return q;
        }
        
        // 使用最新的历史信息计算初始Hessian近似 / Use latest history to compute initial Hessian approximation
        IVector s_k = s_history.get(s_history.size() - 1);
        IVector y_k = y_history.get(y_history.size() - 1);
        
        double yTy = (Double) y_k.innerProduct(y_k);
        if (RerePrecision.equalsZero(yTy, 1e-12)) {
            // 避免除零 / Avoid division by zero
            return q;
        }
        
        double gamma = (Double) s_k.innerProduct(y_k) / yTy;
        
        // 确保gamma为正值以保持正定性 / Ensure gamma is positive to maintain positive definiteness
        gamma = Math.max(gamma, 1e-12);
        
        // 缓存gamma值 / Cache gamma value
        cachedGamma = gamma;
        
        return q.multiplyScalar(gamma);
    }
    
    
    /**
     * 更新LBFGS历史信息（优化版）
     * <p>
     * 更新存储的历史信息，包括：
     * 1. 位置差向量：s_k = x_{k+1} - x_k
     * 2. 梯度差向量：y_k = ∇f(x_{k+1}) - ∇f(x_k)
     * 3. ρ值：ρ_k = 1 / (y_k^T * s_k)
     * 4. gamma值：γ_k = (s_k^T * y_k) / (y_k^T * y_k)
     * 
     * 如果存储的历史信息超过限制m，则删除最旧的信息。
     * </p>
     * 
     * @param oldX 旧位置 / Old position
     * @param newX 新位置 / New position
     * @param oldGrad 旧梯度 / Old gradient
     * @param newGrad 新梯度 / New gradient
     * @param s_history 位置差历史 / Position difference history
     * @param y_history 梯度差历史 / Gradient difference history
     * @param rho_history ρ值历史 / Rho value history
     * @param gamma_history gamma值历史 / Gamma value history
     */
    private void updateHistory(IVector oldX, IVector newX, IVector oldGrad, IVector newGrad,
                              LinkedList<IVector> s_history, LinkedList<IVector> y_history,
                              LinkedList<Double> rho_history, LinkedList<Double> gamma_history) {
        
        // 计算位置差和梯度差 / Compute position and gradient differences
        IVector s_k = newX.sub(oldX);
        IVector y_k = newGrad.sub(oldGrad);
        
        // 计算ρ值 / Compute rho value
        double sTy = (Double) s_k.innerProduct(y_k);
        
        // 检查曲率条件：s^T * y > 0，确保正定性 / Check curvature condition: s^T * y > 0 for positive definiteness
        if (RerePrecision.compareTo(sTy, 1e-10, tolerance) > 0) {
            double rho_k = 1.0 / sTy;
            
            // 计算并存储gamma值 / Compute and store gamma value
            double yTy = (Double) y_k.innerProduct(y_k);
            double gamma_k = (yTy > 1e-12) ? (sTy / yTy) : 1.0;
            gamma_k = Math.max(gamma_k, 1e-12);
            
            // 添加新的历史信息 / Add new history information
            s_history.addLast(s_k);
            y_history.addLast(y_k);
            rho_history.addLast(rho_k);
            gamma_history.addLast(gamma_k);

            // 如果超过存储限制，删除最旧的信息（O(1)操作）/ If exceeds storage limit, remove oldest information (O(1) operation)
            if (s_history.size() > m) {
                s_history.removeFirst();
                y_history.removeFirst();
                rho_history.removeFirst();
                gamma_history.removeFirst();
            }
        }
        // If curvature condition is not satisfied, we simply don't update the history
    }
    
    // Getter和Setter方法 / Getter and Setter methods
    
    /**
     * 获取存储的历史信息对数 / Get number of stored history pairs
     * @return 历史信息对数 / Number of history pairs
     */
    public int getM() {
        return m;
    }
    
    /**
     * 设置存储的历史信息对数 / Set number of stored history pairs
     * @param m 历史信息对数 / Number of history pairs
     */
    public void setM(int m) {
        this.m = Math.max(1, m);
    }
    
    /**
     * 获取收敛容差 / Get convergence tolerance
     * @return 收敛容差 / Convergence tolerance
     */
    public double getTolerance() {
        return tolerance;
    }
    
    /**
     * 设置收敛容差 / Set convergence tolerance
     * @param tolerance 收敛容差 / Convergence tolerance
     */
    public void setTolerance(double tolerance) {
        this.tolerance = Math.max(1e-12, tolerance);
    }
    
    /**
     * 获取最大迭代次数 / Get maximum iterations
     * @return 最大迭代次数 / Maximum iterations
     */
    public int getMaxIterations() {
        return maxIterations;
    }
    
    /**
     * 设置最大迭代次数 / Set maximum iterations
     * @param maxIterations 最大迭代次数 / Maximum iterations
     */
    public void setMaxIterations(int maxIterations) {
        this.maxIterations = Math.max(1, maxIterations);
    }
}
