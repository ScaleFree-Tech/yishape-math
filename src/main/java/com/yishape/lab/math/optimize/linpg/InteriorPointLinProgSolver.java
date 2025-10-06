package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.newton.RereLBFGS;
import com.yishape.lab.math.util.Precision;

import java.util.ArrayList;
import java.util.List;

/**
 * 基于内点法的线性规划求解器
 * 使用对数障碍函数方法处理不等式约束
 * @author lteb2
 */
public class InteriorPointLinProgSolver implements ILinProgSolver{

    // 障碍参数的衰减因子
    private static final double MU_DECAY = 0.9;
    // 障碍参数的初始值
    private static final double MU_INITIAL = 1.0;
    // 障碍参数的最小值
    private static final double MU_MIN = 1e-10;
    // 收敛容差
    private static final double TOLERANCE = 1e-8;
    // 最大迭代次数
    private static final int MAX_ITERATIONS = 100;
    
    // 数值稳定性常量
    private static final double MIN_VARIABLE_VALUE = 1e-12;  // 变量的最小值
    private static final double PENALTY_FACTOR = 1e6;       // 惩罚因子
    private static final double INITIAL_VARIABLE_VALUE = 1e-3; // 初始变量值



    @Override
    public OptResult solve(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq, IVector initX) {
        //todo: 改造后暂时未处理A_ub, b_ub
        
        // 记录开始时间
        long startTime = System.currentTimeMillis();
        
        // 初始化障碍参数
        double mu = MU_INITIAL;
        
        // 创建初始解向量，确保满足非负约束
        IVector x;
        if (initX != null) {
            x = initX.copy();
            // 确保所有变量都是正数
            for (int i = 0; i < x.length(); i++) {
                double value = (Double) x.get(i);
                if (value <= 0) {
                    x.set(i, INITIAL_VARIABLE_VALUE);
                }
            }
        } else {
            // 如果没有提供初始解，创建一个小的正数向量
            x = IVector.ones(c.length()).multiplyScalar(INITIAL_VARIABLE_VALUE);
        }
        
        // 投影初始点到满足等式约束的空间
        if (A_eq != null && b_eq != null) {
            x = projectToFeasibleSet(x, A_eq, b_eq);
            // 投影后再次确保非负性
            for (int i = 0; i < x.length(); i++) {
                double value = (Double) x.get(i);
                if (value <= 0) {
                    x.set(i, INITIAL_VARIABLE_VALUE);
                }
            }
        }
        
        // 初始化迭代计数和评估计数
        int iterations = 0;
        int functionEvaluations = 0;
        int gradientEvaluations = 0;
        
        // 收敛历史记录
        List<Double> functionValueHistory = new ArrayList<>();
        List<Double> gradientNormHistory = new ArrayList<>();
        List<IVector> parameterHistory = new ArrayList<>();
        
        // 记录初始点和目标函数值
        double initialObjectiveValue = (Double) c.innerProduct(x);
        functionValueHistory.add(initialObjectiveValue);
        parameterHistory.add(x.copy());
        functionEvaluations++;
        
        // 主循环：逐步减小障碍参数直到达到最小值
        IVector previousX = null;
        boolean converged = false;
        
        while (mu > MU_MIN && iterations < MAX_ITERATIONS && !converged) {
            iterations++;
            
            // 创建带障碍项的目标函数
            IObjectiveFunction objectiveFunction = createBarrierObjectiveFunction(c, A_eq, b_eq, mu);
            
            // 创建带障碍项的梯度函数
            IGradientFunction gradientFunction = createBarrierGradientFunction(c, A_eq, b_eq, mu);
            
            // 使用LBFGS求解当前障碍问题
            RereLBFGS optimizer = new RereLBFGS();
            OptResult result = optimizer.optimize(x, objectiveFunction, gradientFunction);
            
            // 检查LBFGS是否成功
            if (result == null || result.getOptimalPoint() == null) {
                break; // 优化失败，退出循环
            }
            
            // 更新评估计数
            functionEvaluations += result.getFunctionEvaluations();
            gradientEvaluations += result.getGradientEvaluations();
            
            // 保存前一个解用于收敛检查
            previousX = x.copy();
            
            // 更新解
            x = result.getOptimalPoint();
            
            // 记录历史信息
            functionValueHistory.add(result.getOptimalValue());
            if (result.getGradientNormHistory() != null && !result.getGradientNormHistory().isEmpty()) {
                gradientNormHistory.addAll(result.getGradientNormHistory());
            }
            if (result.getParameterHistory() != null && !result.getParameterHistory().isEmpty()) {
                parameterHistory.addAll(result.getParameterHistory());
            }
            
            // 确保解满足等式约束和非负约束
            if (A_eq != null && b_eq != null) {
                x = projectToFeasibleSet(x, A_eq, b_eq);
            }
            
            // 确保非负性
            for (int i = 0; i < x.length(); i++) {
                double value = (Double) x.get(i);
                if (Precision.compareTo(value, 0.0, TOLERANCE) <= 0) {
                    x.set(i, MIN_VARIABLE_VALUE);
                }
            }
            
            // 检查收敛性
            if (previousX != null) {
                double changeNorm = ((Number) x.sub(previousX).norm2()).doubleValue();
                if (changeNorm < TOLERANCE) {
                    converged = true;
                }
            }
            
            // 减小障碍参数
            mu *= MU_DECAY;
        }
        
        // 计算最终的目标函数值
        double objectiveValue = c.dot(x).doubleValue();
        functionEvaluations++;
        functionValueHistory.add(objectiveValue);
        parameterHistory.add(x.copy());
        
        // 检查最终解的有效性
        if (x == null) {
            throw new RuntimeException("内点法求解失败：无法找到有效解 / Interior point method failed: unable to find valid solution");
        }
        
        // 验证解的可行性
        boolean feasible = true;
        StringBuilder errorMsg = new StringBuilder();
        
        // 检查非负约束
        for (int i = 0; i < x.length(); i++) {
            double value = ((Number) x.get(i)).doubleValue();
            if (Precision.compareTo(value, -TOLERANCE, TOLERANCE) < 0) {
                feasible = false;
                errorMsg.append("变量 ").append(i).append(" 违反非负约束: ").append(value).append("\n");
            }
        }
        
        // 检查等式约束（如果存在）
        if (A_eq != null && b_eq != null) {
            IVector constraintViolation = A_eq.mmul(x).sub(b_eq);
            for (int i = 0; i < constraintViolation.length(); i++) {
                double violation = Math.abs(((Number) constraintViolation.get(i)).doubleValue());
                if (Precision.compareTo(violation, TOLERANCE, TOLERANCE) > 0) {
                    feasible = false;
                    errorMsg.append("等式约束 ").append(i).append(" 违反: 误差 = ").append(violation).append("\n");
                }
            }
        }
        
        if (!feasible) {
            System.err.println("警告：最终解不满足约束条件 / Warning: Final solution does not satisfy constraints");
            System.err.println(errorMsg.toString());
        }
        
        // 构建丰富的OptResult
        OptResult.Builder builder = new OptResult.Builder(objectiveValue, x)
            .converged(converged)
            .convergenceReason(converged ? "Interior point method converged" : "Maximum iterations reached")
            .iterations(iterations)
            .maxIterations(MAX_ITERATIONS)
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .functionEvaluations(functionEvaluations)
            .gradientEvaluations(gradientEvaluations)
            .functionValueHistory(functionValueHistory)
            .gradientNormHistory(gradientNormHistory)
            .parameterHistory(parameterHistory);
        
        return builder.build();
    }
    
    /**
     * 将点投影到等式约束的可行集上
     */
    private IVector projectToFeasibleSet(IVector x, IMatrix A_eq, IVector b_eq) {
        try {
            // 使用最小二乘法求解投影: minimize ||x - x0||^2 subject to A_eq * x = b_eq
            // 解为: x_proj = x - A_eq^T * (A_eq * A_eq^T)^(-1) * (A_eq * x - b_eq)
            
            IVector residual = A_eq.mmul(x).sub(b_eq);  // A_eq * x - b_eq
            IMatrix AtA = A_eq.mmul(A_eq.transpose());  // A_eq * A_eq^T
            IVector lagrangeMult = AtA.solve(residual); // (A_eq * A_eq^T)^(-1) * (A_eq * x - b_eq)
            IVector correction = A_eq.transpose().mmul(lagrangeMult); // A_eq^T * lagrangeMult
            
            return x.sub(correction);
        } catch (Exception e) {
            // 如果投影失败，返回原始点
            return x;
        }
    }
    
    /**
     * 创建带对数障碍项的目标函数
     */
    private IObjectiveFunction createBarrierObjectiveFunction(IVector c, IMatrix A_eq, IVector b_eq, double mu) {
        return new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                // 原始线性目标函数: f(x) = c^T * x
                double objectiveValue = (Double) c.innerProduct(x);
                
                // 添加对数障碍项: -mu * Σ ln(xi)
                double barrierTerm = 0.0;
                for (int i = 0; i < x.length(); i++) {
                    double xi = (Double) x.get(i);
                    // 如果变量接近0或为负，添加大的惩罚
                    if (Precision.compareTo(xi, 1e-12, TOLERANCE) <= 0) {
                        barrierTerm += 1e6; // 大的惩罚值
                    } else {
                        // 对数障碍函数: -mu * ln(xi)
                        barrierTerm -= mu * Math.log(xi);
                    }
                }
                
                // 如果有等式约束，添加惩罚项
                if (A_eq != null && b_eq != null) {
                    IVector constraintViolation = A_eq.mmul(x).sub(b_eq);
                    double penalty = (Double) constraintViolation.innerProduct(constraintViolation);
                    objectiveValue += 1e6 * penalty; // 大的惩罚因子
                }
                
                return objectiveValue + barrierTerm;
            }
        };
    }
    
    /**
     * 创建带对数障碍项的梯度函数
     */
    private IGradientFunction createBarrierGradientFunction(IVector c, IMatrix A_eq, IVector b_eq, double mu) {
        return new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                // 原始线性函数的梯度是常数向量c
                IVector gradient = c.copy();
                
                // 添加对数障碍函数的梯度项: -mu / xi
                for (int i = 0; i < x.length(); i++) {
                    double xi = (Double) x.get(i);
                    // 如果变量接近0或为负，添加大的梯度
                    if (Precision.compareTo(xi, 1e-12, TOLERANCE) <= 0) {
                        // 添加大的负梯度使变量变为正
                        gradient.set(i, (Double) gradient.get(i) - 1e6);
                    } else {
                        // 对数障碍函数的梯度: -mu / xi
                        double barrierGradient = -mu / xi;
                        gradient.set(i, (Double) gradient.get(i) + barrierGradient);
                    }
                }
                
                // 如果有等式约束，添加惩罚项的梯度
                if (A_eq != null && b_eq != null) {
                    IVector constraintViolation = A_eq.mmul(x).sub(b_eq);
                    IVector penaltyGradient = A_eq.transpose().mmul(constraintViolation).multiplyScalar(2e6);
                    gradient = gradient.add(penaltyGradient);
                }
                
                return gradient;
            }
        };
    }
}