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
    private double restartThreshold = 0.5; // 重启阈值 / Restart threshold
    private boolean useAdaptiveRestart = true; // 是否使用自适应重启 / Whether to use adaptive restart

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
        int n = x.length();       // 问题维度 / Problem dimension
        
        // 计算初始函数值 / Compute initial function value
        double initialValue = objFun.computeObjective(x);
        
        // 计算初始梯度 / Compute initial gradient
        IVector grad = grdFun.computeGradient(x);
        double initialGradNorm = (Double) grad.norm2();
        double finalGradientNorm = initialGradNorm;
        
        // 初始化搜索方向为负梯度方向 / Initialize search direction as negative gradient
        IVector direction = grad.multiplyScalar(-1.0);
        
        // 用于存储前一次迭代的梯度 / Store previous gradient
        IVector prevGrad = null;
        IVector prevDirection = null;
        double prevValue = initialValue;
        
        // 存储最佳解 / Store best solution
        IVector bestX = x.copy();
        double bestValue = initialValue;
        
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
        double finalValue = initialValue; // 初始化最终值 / Initialize final value
        
        // 重启计数器 / Restart counter
        int restartCounter = 0;
        int maxRestarts = 10;
        
        // 主迭代循环 / Main iteration loop
        for (int iter = 0; iter < maxIterations; iter++) {
            actualIterations = iter + 1;
            
            // 检查收敛条件：梯度范数足够小 / Check convergence: gradient norm is small enough
            double gradNorm = (Double) grad.norm2();
            finalGradientNorm = gradNorm;
            
            // 更新最佳解 / Update best solution
            if (objFun.computeObjective(x) < bestValue) {
                bestX = x.copy();
                bestValue = objFun.computeObjective(x);
            }
            
            // 改进的收敛检查 / Improved convergence check
            // 使用绝对和相对容差的组合 / Use combination of absolute and relative tolerance
            double convergenceThreshold = Math.max(tolerance, tolerance * Math.max(1.0, initialGradNorm));
            if (RerePrecision.compareTo(gradNorm, convergenceThreshold, tolerance) < 0) {
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
            
            // 检查是否陷入局部停滞 / Check for local stagnation
            if (iter > 0 && RerePrecision.compareTo(Math.abs(prevValue - objFun.computeObjective(x)), 1e-12, tolerance) < 0) {
                // 如果函数值没有显著变化，尝试重启 / If function value hasn't changed significantly, try restart
                direction = grad.multiplyScalar(-1.0);
                restartCounter++;
                if (restartCounter > maxRestarts) {
                    convergenceReason = "Stagnation detected";
                    break;
                }
            }
            
            // 线搜索确定步长 / Line search to determine step size
            double stepSize = new RereLineSearch().search(x, direction, objFun, grdFun, grad);
            
            // 更新位置 / Update position
            IVector newX = x.add(direction.multiplyScalar(stepSize));
            
            // 计算新梯度 / Compute new gradient
            IVector newGrad = grdFun.computeGradient(newX);
            gradientEvaluations++;
            
            // 计算新函数值并记录 / Compute new function value and record
            double newValue = objFun.computeObjective(newX);
            functionEvaluations++;
            functionValueHistory.add(newValue);
            gradientNormHistory.add((Double) newGrad.norm2());
            parameterHistory.add(newX.copy());
            
            // 如果新点更差，尝试重启 / If new point is worse, try restart
            if (RerePrecision.compareTo(newValue, prevValue, tolerance) > 0 && iter > 0) {
                direction = newGrad.multiplyScalar(-1.0);
                newX = x; // 回退到前一个点 / Revert to previous point
                newGrad = grad;
                newValue = prevValue;
                restartCounter++;
                if (restartCounter > maxRestarts) {
                    convergenceReason = "Repeated poor steps";
                    break;
                }
                // 继续下一次迭代 / Continue to next iteration
                prevValue = newValue;
                prevGrad = grad;
                prevDirection = direction;
                x = newX;
                grad = newGrad;
                continue;
            }
            
            // 计算β参数用于更新搜索方向 / Compute beta parameter for updating search direction
            // 使用Polak-Ribière公式 / Using Polak-Ribière formula
            double beta = 0.0;
            if (prevGrad != null) {
                IVector gradDiff = newGrad.sub(prevGrad);
                double numerator = (Double) newGrad.innerProduct(gradDiff);
                double denominator = (Double) prevGrad.innerProduct(prevGrad);
                
                if (!RerePrecision.equalsZero(denominator, 1e-12)) {
                    beta = numerator / denominator;
                    // 确保β非负 / Ensure beta is non-negative
                    beta = Math.max(beta, 0.0);
                }
                
                // 如果β为0或负数，强制重启 / If beta is zero or negative, force restart
                if (RerePrecision.compareTo(beta, 0.0, tolerance) <= 0) {
                    beta = 0.0;
                    direction = newGrad.multiplyScalar(-1.0);
                }
            }
            
            // 更新搜索方向 / Update search direction
            // 使用公式: d_{k+1} = -g_{k+1} + β_k * d_k
            IVector newDirection = newGrad.multiplyScalar(-1.0).add(direction.multiplyScalar(beta));
            
            // 改进的重启检查 / Improved restart check
            if (useAdaptiveRestart) {
                // 检查是否需要重启 / Check if restart is needed
                // 当梯度与搜索方向的夹角过大时重启 / Restart when angle between gradient and search direction is too large
                double gradDotDirection = (Double) newGrad.innerProduct(newDirection); // 不再取绝对值 / No longer take absolute value
                double gradNormNew = (Double) newGrad.norm2();
                double directionNorm = (Double) newDirection.norm2();
                
                if (RerePrecision.compareTo(gradNormNew, 1e-12, tolerance) > 0 && RerePrecision.compareTo(directionNorm, 1e-12, tolerance) > 0) {
                    // 使用更合理的重启条件 / Use more reasonable restart condition
                    double cosAngle = gradDotDirection / (gradNormNew * directionNorm);
                    // 如果搜索方向不是下降方向，或者夹角余弦值小于阈值，重启搜索方向 / 
                    // If search direction is not descent direction, or cosine of angle is less than threshold, restart search direction
                    if (cosAngle >= 0 || RerePrecision.compareTo(cosAngle, -restartThreshold, tolerance) < 0) {
                        newDirection = newGrad.multiplyScalar(-1.0);
                    }
                }
            } else {
                // 原始重启逻辑 / Original restart logic
                double gradDotDirection = Math.abs((Double) newGrad.innerProduct(newDirection));
                double gradNormNew = (Double) newGrad.norm2();
                double directionNorm = (Double) newDirection.norm2();
                
                if (RerePrecision.compareTo(gradNormNew, 1e-12, tolerance) > 0 && RerePrecision.compareTo(directionNorm, 1e-12, tolerance) > 0) {
                    double cosAngle = gradDotDirection / (gradNormNew * directionNorm);
                    // 如果夹角余弦值小于阈值，重启搜索方向 / If cosine of angle is less than threshold, restart search direction
                    if (RerePrecision.compareTo(cosAngle, restartThreshold, tolerance) < 0) {
                        newDirection = newGrad.multiplyScalar(-1.0);
                    }
                }
            }
            
            // 更新变量 / Update variables
            prevValue = newValue;
            prevGrad = grad;
            prevDirection = direction;
            x = newX;
            grad = newGrad;
            direction = newDirection;
        }
        
        // 如果没有收敛，返回找到的最佳解 / If not converged, return best solution found
        if (!converged) {
            x = bestX;
            finalValue = bestValue;
        } else {
            finalValue = objFun.computeObjective(x);
        }
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
    
    /**
     * 获取是否使用自适应重启 / Get whether to use adaptive restart
     * @return 是否使用自适应重启 / Whether to use adaptive restart
     */
    public boolean isUseAdaptiveRestart() {
        return useAdaptiveRestart;
    }
    
    /**
     * 设置是否使用自适应重启 / Set whether to use adaptive restart
     * @param useAdaptiveRestart 是否使用自适应重启 / Whether to use adaptive restart
     */
    public void setUseAdaptiveRestart(boolean useAdaptiveRestart) {
        this.useAdaptiveRestart = useAdaptiveRestart;
    }
}