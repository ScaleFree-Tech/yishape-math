package com.reremouse.lab.math.optimize;

import com.reremouse.lab.util.Tuple2;
import java.util.ArrayList;
import java.util.List;
import com.reremouse.lab.math.IVector;

/**
 * LBFGS优化器
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
 * <h3>算法特点 / Algorithm Features:</h3>
 * <ul>
 *   <li>内存友好：只存储有限的历史信息 / Memory-friendly: stores only limited historical information</li>
 *   <li>超线性收敛：在接近最优解时收敛速度很快 / Superlinear convergence: fast convergence near optimal solution</li>
 *   <li>无需计算Hessian矩阵：只需要目标函数和梯度 / No Hessian computation: only requires objective function and gradient</li>
 *   <li>适用于大规模问题：内存使用量与问题维度线性相关 / Suitable for large-scale problems: memory usage linear in problem dimension</li>
 * </ul>
 * 
 * @author lteb2
 */
public class RereLBFGS implements IOptimizer{

    // LBFGS算法参数 / LBFGS algorithm parameters
    private int m = 10;                    // 存储的历史信息对数 / Number of stored history pairs
    private float tolerance = 1e-6f;       // 收敛容差 / Convergence tolerance
    private int maxIterations = 1000;      // 最大迭代次数 / Maximum iterations

    
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
    public RereLBFGS(int m, float tolerance, int maxIterations) {
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
     *    a) 计算梯度
     *    b) 检查收敛条件
     *    c) 使用两循环递归计算搜索方向
     *    d) 线搜索确定步长
     *    e) 更新参数和历史信息
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
    public Tuple2<Float, IVector> optimize(IVector initX, IObjectiveFunction objFun, IGradientFunction grdFun) {
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
        
        // 初始化变量 / Initialize variables
        IVector x = initX.copy();  // 当前点 / Current point
        int n = x.length();       // 问题维度 / Problem dimension
        
        // 历史信息存储 / History information storage
        List<IVector> s_history = new ArrayList<>();  // 位置差向量历史 / Position difference history
        List<IVector> y_history = new ArrayList<>();  // 梯度差向量历史 / Gradient difference history
        List<Float> rho_history = new ArrayList<>(); // ρ值历史 / Rho value history
        
        // 计算初始梯度 / Compute initial gradient
        IVector grad = grdFun.computeGradient(x);
        
        // 主迭代循环 / Main iteration loop
        for (int iter = 0; iter < maxIterations; iter++) {
            // 检查收敛条件：梯度范数足够小 / Check convergence: gradient norm is small enough
            float gradNorm = grad.norm2();
            if (gradNorm < tolerance) {
                float optimalValue = objFun.computeObjective(x);
                return new Tuple2<>(optimalValue, x);
            }
            
            // 计算搜索方向：使用两循环递归 / Compute search direction: two-loop recursion
            IVector direction = computeSearchDirection(grad, s_history, y_history, rho_history);
            direction = direction.multiplyScalar(-1.0f); // 负梯度方向 / Negative gradient direction
            
            // 线搜索确定步长 / Line search to determine step size
            float stepSize = new RereLineSearch().search(x, direction, objFun, grdFun, grad);
            
            // 更新位置 / Update position
            IVector newX = x.add(direction.multiplyScalar(stepSize));
            IVector newGrad = grdFun.computeGradient(newX);
            
            // 更新历史信息 / Update history information
            updateHistory(x, newX, grad, newGrad, s_history, y_history, rho_history);
            
            // 更新当前点和梯度 / Update current point and gradient
            x = newX;
            grad = newGrad;
        }
        
        // 达到最大迭代次数，返回当前最优解 / Maximum iterations reached, return current best solution
        float finalValue = objFun.computeObjective(x);
        return new Tuple2<>(finalValue, x);
    }
    
    /**
     * 计算LBFGS搜索方向
     * <p>
     * 使用两循环递归算法计算搜索方向：
     * 1. 第一个循环：从最新到最旧的历史信息，向后递归
     * 2. 第二个循环：从最旧到最新的历史信息，向前递归
     * 3. 使用初始Hessian近似（单位矩阵的标量倍数）
     * </p>
     * 
     * @param grad 当前梯度 / Current gradient
     * @param s_history 位置差历史 / Position difference history
     * @param y_history 梯度差历史 / Gradient difference history  
     * @param rho_history ρ值历史 / Rho value history
     * @return 搜索方向 / Search direction
     */
    private IVector computeSearchDirection(IVector grad, List<IVector> s_history, 
                                        List<IVector> y_history, List<Float> rho_history) {
        
        IVector q = grad.copy();
        int historySize = s_history.size();
        float[] alpha = new float[historySize];
        
        // 第一个循环：向后递归 / First loop: backward recursion
        for (int i = historySize - 1; i >= 0; i--) {
            alpha[i] = rho_history.get(i) * s_history.get(i).innerProduct(q);
            q = q.sub(y_history.get(i).multiplyScalar(alpha[i]));
        }
        
        // 应用初始Hessian近似 / Apply initial Hessian approximation
        IVector r = applyInitialHessianApproximation(q, s_history, y_history);
        
        // 第二个循环：向前递归 / Second loop: forward recursion
        for (int i = 0; i < historySize; i++) {
            float beta = rho_history.get(i) * y_history.get(i).innerProduct(r);
            r = r.add(s_history.get(i).multiplyScalar(alpha[i] - beta));
        }
        
        return r;
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
    private IVector applyInitialHessianApproximation(IVector q, List<IVector> s_history, List<IVector> y_history) {
        if (s_history.isEmpty()) {
            // 如果没有历史信息，使用单位矩阵 / If no history, use identity matrix
            return q;
        }
        
        // 使用最新的历史信息计算初始Hessian近似 / Use latest history to compute initial Hessian approximation
        IVector s_k = s_history.get(s_history.size() - 1);
        IVector y_k = y_history.get(y_history.size() - 1);
        
        float gamma = s_k.innerProduct(y_k) / y_k.innerProduct(y_k);
        
        return q.multiplyScalar(gamma);
    }
    
    
    /**
     * 更新LBFGS历史信息
     * <p>
     * 更新存储的历史信息，包括：
     * 1. 位置差向量：s_k = x_{k+1} - x_k
     * 2. 梯度差向量：y_k = ∇f(x_{k+1}) - ∇f(x_k)
     * 3. ρ值：ρ_k = 1 / (y_k^T * s_k)
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
     */
    private void updateHistory(IVector oldX, IVector newX, IVector oldGrad, IVector newGrad,
                             List<IVector> s_history, List<IVector> y_history, List<Float> rho_history) {
        
        // 计算位置差和梯度差 / Compute position and gradient differences
        IVector s_k = newX.sub(oldX);
        IVector y_k = newGrad.sub(oldGrad);
        
        // 计算ρ值 / Compute rho value
        float sTy = s_k.innerProduct(y_k);
        
        // 检查曲率条件：s^T * y > 0，确保正定性 / Check curvature condition: s^T * y > 0 for positive definiteness
        if (sTy > 1e-10f) {
            float rho_k = 1.0f / sTy;
            
            // 添加新的历史信息 / Add new history information
            s_history.add(s_k);
            y_history.add(y_k);
            rho_history.add(rho_k);
            
            // 如果超过存储限制，删除最旧的信息 / If exceeds storage limit, remove oldest information
            if (s_history.size() > m) {
                s_history.remove(0);
                y_history.remove(0);
                rho_history.remove(0);
            }
        }
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
    public float getTolerance() {
        return tolerance;
    }
    
    /**
     * 设置收敛容差 / Set convergence tolerance
     * @param tolerance 收敛容差 / Convergence tolerance
     */
    public void setTolerance(float tolerance) {
        this.tolerance = Math.max(1e-12f, tolerance);
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
