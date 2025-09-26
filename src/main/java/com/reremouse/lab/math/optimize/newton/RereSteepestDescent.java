package com.reremouse.lab.math.optimize.newton;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IOptimizer;
import com.reremouse.lab.math.optimize.RereLineSearch;
import com.reremouse.lab.util.Tuple2;

/**
 * 最速下降法 / steepest descent method
 * 一阶优化方法
 * 
 * 最速下降法是一种用于求解无约束优化问题的迭代算法。
 * 该算法在每一步都沿着目标函数梯度的负方向进行搜索，这是函数值下降最快的方向。
 * 
 * <h3>算法特点 / Algorithm Features:</h3>
 * <ul>
 *   <li>简单易实现：只需要目标函数和梯度信息 / Simple to implement: only requires objective function and gradient</li>
 *   <li>线性收敛：对于一般函数具有线性收敛速度 / Linear convergence: linear convergence rate for general functions</li>
 *   <li>无需计算Hessian矩阵：只需要目标函数和梯度 / No Hessian computation: only requires objective function and gradient</li>
 *   <li>可能收敛较慢：对于病态问题收敛速度可能很慢 / May converge slowly: slow convergence for ill-conditioned problems</li>
 * </ul>
 * 
 * @author lteb2
 */
public class RereSteepestDescent implements IOptimizer {

    // 最速下降法参数 / Steepest descent algorithm parameters
    private double tolerance = 1e-6;       // 收敛容差 / Convergence tolerance
    private int maxIterations = 1000;      // 最大迭代次数 / Maximum iterations
    private double initialStepSize = 1.0;  // 初始步长 / Initial step size

    /**
     * 构造函数，使用默认参数 / Constructor with default parameters
     */
    public RereSteepestDescent() {
    }
    
    /**
     * 构造函数，允许自定义参数 / Constructor with custom parameters
     * 
     * @param tolerance 收敛容差 / Convergence tolerance
     * @param maxIterations 最大迭代次数 / Maximum iterations
     * @param initialStepSize 初始步长 / Initial step size
     */
    public RereSteepestDescent(double tolerance, int maxIterations, double initialStepSize) {
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
        this.initialStepSize = initialStepSize;
    }

    @Override
    public Tuple2<Double, IVector> optimize(IVector initX, IObjectiveFunction objFun, IGradientFunction grdFun) {
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
        double stepSize = initialStepSize;  // 当前步长 / Current step size
        
        // 计算初始梯度 / Compute initial gradient
        IVector grad = grdFun.computeGradient(x);
        double initialGradNorm = (Double) grad.norm2();
        
        // 主迭代循环 / Main iteration loop
        for (int iter = 0; iter < maxIterations; iter++) {
            // 检查收敛条件：梯度范数足够小 / Check convergence: gradient norm is small enough
            double gradNorm = (Double) grad.norm2();
            if (gradNorm < tolerance * Math.max(1.0, initialGradNorm)) {
                double optimalValue = objFun.computeObjective(x);
                return new Tuple2<>(optimalValue, x);
            }
            
            // 线搜索确定步长 / Line search to determine step size
            // 搜索方向为负梯度方向 / Search direction is negative gradient direction
            IVector searchDirection = grad.multiplyScalar(-1.0);
            stepSize = new RereLineSearch().search(x, searchDirection, objFun, grdFun, grad);
            
            // 更新位置 / Update position
            IVector newX = x.add(searchDirection.multiplyScalar(stepSize));
            
            // 计算新梯度 / Compute new gradient
            IVector newGrad = grdFun.computeGradient(newX);
            
            // 更新变量 / Update variables
            x = newX;
            grad = newGrad;
        }
        
        // 达到最大迭代次数，返回当前最优解 / Maximum iterations reached, return current best solution
        double finalValue = objFun.computeObjective(x);
        return new Tuple2<>(finalValue, x);
    }
    
    // Getter和Setter方法 / Getter and Setter methods
    
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
     * 获取初始步长 / Get initial step size
     * @return 初始步长 / Initial step size
     */
    public double getInitialStepSize() {
        return initialStepSize;
    }
    
    /**
     * 设置初始步长 / Set initial step size
     * @param initialStepSize 初始步长 / Initial step size
     */
    public void setInitialStepSize(double initialStepSize) {
        this.initialStepSize = Math.max(1e-12, initialStepSize);
    }
}