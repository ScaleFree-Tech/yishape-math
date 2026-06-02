package com.yishape.lab.math.optimize.newton;

import java.util.ArrayList;
import java.util.List;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.RereLineSearch;
import com.yishape.lab.math.util.RerePrecision;

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
    private int maxIterations = 1000;      // 最大迭代次数 / Maximum iterations
    private boolean trackHistory = true;   // 是否记录完整历史 / Whether to track full history

    // 缓存变量 / Cached variables
    private double cachedGradNorm = 0;     // 缓存的梯度范数 / Cached gradient norm
    private double cachedDirectionalDerivative = 0; // 缓存的方向导数 / Cached directional derivative
    /**
     * 构造函数，使用默认参数 / Constructor with default parameters
     */
    public RereLBFGS() {
    }
    
    /**
     * 构造函数，允许自定义参数 / Constructor with custom parameters
     * 
     * @param tolerance 收敛容差 / Convergence tolerance
     * @param maxIterations 最大迭代次数 / Maximum iterations
     */
    public RereLBFGS(double tolerance, int maxIterations) {
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
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
        
        // 使用ArrayList存储历史信息，O(1)随机访问优于LinkedList的O(i) / Use ArrayList for O(1) random access
        ArrayList<IVector> s_history = new ArrayList<>(m + 1);  // 位置差向量历史 / Position difference history
        ArrayList<IVector> y_history = new ArrayList<>(m + 1);  // 梯度差向量历史 / Gradient difference history
        ArrayList<Double> rho_history = new ArrayList<>(m + 1); // ρ值历史 / Rho value history
        ArrayList<Double> gamma_history = new ArrayList<>(m + 1); // gamma值历史 / Gamma value history

        // 收敛历史记录 / Convergence history tracking
        List<Double> functionValueHistory = trackHistory ? new ArrayList<>(Math.min(maxIterations, 100)) : null;
        List<Double> gradientNormHistory = trackHistory ? new ArrayList<>(Math.min(maxIterations, 100)) : null;
        List<IVector> parameterHistory = trackHistory ? new ArrayList<>(Math.min(maxIterations, 100)) : null;
        
        // 评估计数 / Evaluation counters
        int functionEvaluations = 1; // 初始函数值计算 / Initial function evaluation
        int gradientEvaluations = 0; // 梯度计算将在循环中开始计数 / Gradient evaluations will start counting in loop
        
        // 计算初始梯度 / Compute initial gradient
        IVector grad = grdFun.computeGradient(x);
        gradientEvaluations++;
        
        // 缓存初始梯度范数 / Cache initial gradient norm
        cachedGradNorm = grad.norm2Value();
        double initialGradNorm = cachedGradNorm;
        
        // 添加初始历史记录 / Add initial history records
        if (trackHistory) {
            functionValueHistory.add(initialValue);
            gradientNormHistory.add(initialGradNorm);
            parameterHistory.add(x.copy());
        }
        
        // 最佳解跟踪 / Best solution tracking
        IVector bestX = x.copy();
        double bestValue = initialValue;
        double bestGradNorm = initialGradNorm;
        
        boolean converged = false;
        String convergenceReason = "Maximum iterations reached";
        int actualIterations = 0;
        
        // 停滞检测：仅当梯度也已足够小时才提前结束，避免在狭长山谷误判收敛
        double previousValue = initialValue;
        int stagnationCounter = 0;
        int maxStagnationIterations = 10;
        
        // 当前函数值（避免重复计算）/ Current function value (avoid recomputation)
        double currentValue = initialValue;
        
        // 缓存方向导数 / Cache directional derivative
        cachedDirectionalDerivative = 0;
        
        RereLineSearch lineSearch = new RereLineSearch();
        Double lastObjectiveForStepHint = null;
        
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
            if (RerePrecision.compareTo(gradNorm, convergenceThreshold, tolerance) <= 0) {
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
                    .gradientEvaluations(gradientEvaluations);
                if (trackHistory) {
                    builder.functionValueHistory(functionValueHistory)
                           .gradientNormHistory(gradientNormHistory)
                           .parameterHistory(parameterHistory);
                }
                return builder.build();
            }
            
            // 停滞检测：目标值几乎不变且梯度仍大时，视为假停滞，继续迭代
            if (iter > 0) {
                double valueChange = Math.abs(currentValue - previousValue);
                if (RerePrecision.compareTo(valueChange, 1e-12 * Math.max(1.0, Math.abs(currentValue)), 1e-12) < 0) {
                    stagnationCounter++;
                    if (stagnationCounter >= maxStagnationIterations) {
                        double stagnGradTol = Math.max(convergenceThreshold * 10.0, 1e-8);
                        if (RerePrecision.compareTo(gradNorm, stagnGradTol, tolerance) < 0) {
                            converged = true;
                            convergenceReason = "Stagnation detected";
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
                                .gradientEvaluations(gradientEvaluations);
                            if (trackHistory) {
                                builder.functionValueHistory(functionValueHistory)
                                       .gradientNormHistory(gradientNormHistory)
                                       .parameterHistory(parameterHistory);
                            }
                            return builder.build();
                        }
                        stagnationCounter = 0;
                    }
                } else {
                    stagnationCounter = 0;
                }
            }
            previousValue = currentValue;
            
            // 计算搜索方向：使用两循环递归（传递缓存的gamma）/ Compute search direction: two-loop recursion (pass cached gamma)
            IVector direction = computeSearchDirection(grad, s_history, y_history, rho_history, gamma_history);
            
            // 计算并缓存方向导数 / Compute and cache directional derivative
            cachedDirectionalDerivative = grad.innerProductValue(direction);
            
            double fAtStepStart = currentValue;
            // 强 Wolfe 线搜索：传入 f(x) 与上一迭代目标值以改善初值步长
            double stepSize = lineSearch.searchWithCachedDerivative(
                x, direction, objFun, grdFun, grad, fAtStepStart, lastObjectiveForStepHint, cachedDirectionalDerivative);
            lastObjectiveForStepHint = fAtStepStart;
            
            // 更新位置 / Update position
            IVector newX = x.add(direction.multiplyByScalar(stepSize));
            IVector newGrad = grdFun.computeGradient(newX);
            gradientEvaluations++;

            // 计算新函数值并记录 / Compute new function value and record
            double newValue = objFun.computeObjective(newX);
            functionEvaluations++;

            // 缓存新梯度范数 / Cache new gradient norm
            double newGradNorm = newGrad.norm2Value();
            cachedGradNorm = newGradNorm;

            if (trackHistory) {
                functionValueHistory.add(newValue);
                gradientNormHistory.add(newGradNorm);
                parameterHistory.add(newX.copy());
            }

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
    private IVector computeSearchDirection(IVector grad, ArrayList<IVector> s_history,
                                         ArrayList<IVector> y_history, ArrayList<Double> rho_history,
                                         ArrayList<Double> gamma_history) {

        int historySize = s_history.size();
        double[] alpha = new double[historySize];

        // 获取原始数组，预提取 s/y 历史数据避免循环内反复拆箱
        double[][] sData = new double[historySize][];
        double[][] yData = new double[historySize][];
        for (int i = 0; i < historySize; i++) {
            sData[i] = ((com.yishape.lab.math.linalg.IDoubleVector) s_history.get(i)).getData();
            yData[i] = ((com.yishape.lab.math.linalg.IDoubleVector) y_history.get(i)).getData();
        }

        // q 初始化为 grad 的拷贝，之后原地修改
        double[] qArr = ((com.yishape.lab.math.linalg.IDoubleVector) grad).getData().clone();
        int n = qArr.length;

        // 第一个循环：向后递归，原地更新 qArr
        for (int i = historySize - 1; i >= 0; i--) {
            double[] si = sData[i];
            double[] yi = yData[i];
            double dot = 0;
            for (int j = 0; j < n; j++) dot += si[j] * qArr[j];
            alpha[i] = rho_history.get(i) * dot;
            double ai = alpha[i];
            for (int j = 0; j < n; j++) qArr[j] -= ai * yi[j];
        }

        // 应用初始 Hessian 近似：r = gamma * q，qArr 就地变为 rArr
        double gamma;
        if (!gamma_history.isEmpty()) {
            gamma = gamma_history.get(gamma_history.size() - 1);
        } else if (!s_history.isEmpty()) {
            IVector sLast = s_history.get(historySize - 1);
            IVector yLast = y_history.get(historySize - 1);
            double sy = sLast.innerProductValue(yLast);
            double yy = yLast.innerProductValue(yLast);
            gamma = (Math.abs(sy) > 1e-16 && Math.abs(yy) > 1e-16) ? sy / yy : 1.0;
        } else {
            gamma = 1.0;
        }
        for (int j = 0; j < n; j++) qArr[j] *= gamma;

        // 第二个循环：向前递归，原地更新 qArr（现在即 r）
        for (int i = 0; i < historySize; i++) {
            double[] si = sData[i];
            double[] yi = yData[i];
            double dot = 0;
            for (int j = 0; j < n; j++) dot += yi[j] * qArr[j];
            double beta = rho_history.get(i) * dot;
            double diff = alpha[i] - beta;
            for (int j = 0; j < n; j++) qArr[j] += diff * si[j];
        }

        // 取负方向（下降方向）
        for (int j = 0; j < n; j++) qArr[j] = -qArr[j];

        return com.yishape.lab.math.linalg.IDoubleVector.of(qArr);
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
                              ArrayList<IVector> s_history, ArrayList<IVector> y_history,
                              ArrayList<Double> rho_history, ArrayList<Double> gamma_history) {
        
        // 计算位置差和梯度差 / Compute position and gradient differences
        IVector s_k = newX.sub(oldX);
        IVector y_k = newGrad.sub(oldGrad);
        
        // 计算ρ值 / Compute rho value
        double sTy = s_k.innerProductValue(y_k);
        
        // 检查曲率条件：s^T * y > 0，确保正定性 / Check curvature condition: s^T * y > 0 for positive definiteness
        if (sTy > 1e-10) {
            double rho_k = 1.0 / sTy;
            
            // 计算并存储gamma值 / Compute and store gamma value
            double yTy = y_k.innerProductValue(y_k);
            double gamma_k = (yTy > 1e-12) ? (sTy / yTy) : 1.0;
            gamma_k = Math.max(gamma_k, 1e-12);
            
            // 添加新的历史信息 / Add new history information
            s_history.add(s_k);
            y_history.add(y_k);
            rho_history.add(rho_k);
            gamma_history.add(gamma_k);

            // 如果超过存储限制，删除最旧的信息 / If exceeds storage limit, remove oldest information
            if (s_history.size() > m) {
                s_history.remove(0);
                y_history.remove(0);
                rho_history.remove(0);
                gamma_history.remove(0);
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

    /**
     * 获取是否记录完整历史 / Get whether to track full history
     * @return 是否记录完整历史 / Whether to track full history
     */
    public boolean isTrackHistory() {
        return trackHistory;
    }

    /**
     * 设置是否记录完整历史 / Set whether to track full history
     * @param trackHistory 是否记录完整历史 / Whether to track full history
     */
    public void setTrackHistory(boolean trackHistory) {
        this.trackHistory = trackHistory;
    }
}
