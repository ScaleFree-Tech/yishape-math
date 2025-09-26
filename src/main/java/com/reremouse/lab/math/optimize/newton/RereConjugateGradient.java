package com.reremouse.lab.math.optimize.newton;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IOptimizer;
import com.reremouse.lab.math.optimize.RereLineSearch;
import com.reremouse.lab.util.Tuple2;

/**
 * 共轭梯度法 / conjugate gradient method
 * <p>
 * 共轭梯度法是一种用于求解无约束优化问题的迭代算法，特别适用于大规模问题。
 * 该算法结合了最速下降法的稳定性和牛顿法的快速收敛性。
 * </p>
 * <p>
 * The conjugate gradient method is an iterative algorithm for solving unconstrained optimization problems,
 * particularly suitable for large-scale problems. This algorithm combines the stability of steepest descent
 * with the fast convergence of Newton's method.
 * </p>
 * 
 * <h3>算法特点 / Algorithm Features:</h3>
 * <ul>
 *   <li>内存友好：只需要存储少量向量 / Memory-friendly: only needs to store a few vectors</li>
 *   <li>线性收敛：对于二次函数具有有限步收敛性 / Linear convergence: finite convergence for quadratic functions</li>
 *   <li>无需计算Hessian矩阵：只需要目标函数和梯度 / No Hessian computation: only requires objective function and gradient</li>
 *   <li>适用于大规模问题：内存使用量与问题维度线性相关 / Suitable for large-scale problems: memory usage linear in problem dimension</li>
 * </ul>
 * 
 * @author lteb2
 */
public class RereConjugateGradient implements IOptimizer{

    // 共轭梯度法参数 / Conjugate gradient algorithm parameters
    private double tolerance = 1e-6;       // 收敛容差 / Convergence tolerance
    private int maxIterations = 1000;      // 最大迭代次数 / Maximum iterations
    private double restartThreshold = 0.1; // 重启阈值 / Restart threshold

    /**
     * 构造函数，使用默认参数 / Constructor with default parameters
     */
    public RereConjugateGradient() {
    }
    
    /**
     * 构造函数，允许自定义参数 / Constructor with custom parameters
     * 
     * @param tolerance 收敛容差 / Convergence tolerance
     * @param maxIterations 最大迭代次数 / Maximum iterations
     * @param restartThreshold 重启阈值 / Restart threshold
     */
    public RereConjugateGradient(double tolerance, int maxIterations, double restartThreshold) {
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
        this.restartThreshold = restartThreshold;
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
        int n = x.length();       // 问题维度 / Problem dimension
        
        // 计算初始梯度 / Compute initial gradient
        IVector grad = grdFun.computeGradient(x);
        double initialGradNorm = (Double) grad.norm2();
        
        // 初始化搜索方向为负梯度方向 / Initialize search direction as negative gradient
        IVector direction = grad.multiplyScalar(-1.0);
        
        // 用于存储前一次迭代的梯度 / Store previous gradient
        IVector prevGrad = null;
        
        // 主迭代循环 / Main iteration loop
        for (int iter = 0; iter < maxIterations; iter++) {
            // 检查收敛条件：梯度范数足够小 / Check convergence: gradient norm is small enough
            double gradNorm = (Double) grad.norm2();
            if (gradNorm < tolerance * Math.max(1.0, initialGradNorm)) {
                double optimalValue = objFun.computeObjective(x);
                return new Tuple2<>(optimalValue, x);
            }
            
            // 线搜索确定步长 / Line search to determine step size
            double stepSize = new RereLineSearch().search(x, direction, objFun, grdFun, grad);
            
            // 更新位置 / Update position
            IVector newX = x.add(direction.multiplyScalar(stepSize));
            
            // 计算新梯度 / Compute new gradient
            IVector newGrad = grdFun.computeGradient(newX);
            
            // 计算β参数用于更新搜索方向 / Compute beta parameter for updating search direction
            // 使用Polak-Ribière公式 / Using Polak-Ribière formula
            double beta = 0.0;
            if (prevGrad != null) {
                IVector gradDiff = newGrad.sub(prevGrad);
                double numerator = (Double) newGrad.innerProduct(gradDiff);
                double denominator = (Double) prevGrad.innerProduct(prevGrad);
                
                if (Math.abs(denominator) > 1e-12) {
                    beta = numerator / denominator;
                    // 确保β非负 / Ensure beta is non-negative
                    beta = Math.max(beta, 0.0);
                }
            }
            
            // 更新搜索方向 / Update search direction
            // 使用公式: d_{k+1} = -g_{k+1} + β_k * d_k
            IVector newDirection = newGrad.multiplyScalar(-1.0).add(direction.multiplyScalar(beta));
            
            // 检查是否需要重启 / Check if restart is needed
            // 当梯度与搜索方向的夹角过大时重启 / Restart when angle between gradient and search direction is too large
            double gradDotDirection = Math.abs((Double) newGrad.innerProduct(newDirection));
            double gradNormNew = (Double) newGrad.norm2();
            double directionNorm = (Double) newDirection.norm2();
            
            if (gradNormNew > 1e-12 && directionNorm > 1e-12) {
                double cosAngle = gradDotDirection / (gradNormNew * directionNorm);
                // 如果夹角余弦值小于阈值，重启搜索方向 / If cosine of angle is less than threshold, restart search direction
                if (cosAngle < restartThreshold) {
                    newDirection = newGrad.multiplyScalar(-1.0);
                }
            }
            
            // 更新变量 / Update variables
            x = newX;
            prevGrad = grad;
            grad = newGrad;
            direction = newDirection;
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
     * 获取重启阈值 / Get restart threshold
     * @return 重启阈值 / Restart threshold
     */
    public double getRestartThreshold() {
        return restartThreshold;
    }
    
    /**
     * 设置重启阈值 / Set restart threshold
     * @param restartThreshold 重启阈值 / Restart threshold
     */
    public void setRestartThreshold(double restartThreshold) {
        this.restartThreshold = Math.max(0.0, Math.min(1.0, restartThreshold));
    }
}