package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.constraint.LagrangeMultiplierSolver;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.util.Precision;

import java.util.ArrayList;
import java.util.List;

/**
 * 基于拉格朗日乘子法求解线性规划求解器
 * @author lteb2
 */
public class LangMultiplierLinProgSolver implements ILinProgSolver{

    // 障碍参数的衰减因子
    private static final double MU_DECAY = 0.9;
    // 障碍参数的初始值
    private static final double MU_INITIAL = 1.0;
    // 障碍参数的最小值
    private static final double MU_MIN = 1e-10;
    
    LagrangeMultiplierSolver baseSolver;//拉格朗日乘子法求解器
    
    
    /**
     * 求解带非负约束的线性规划问题
     * minimize c^T * x
     * subject to A_eq * x = b_eq
     *            x >= 0
     * 
     * @param c 目标函数系数向量
     * @param A_eq 等式约束矩阵
     * @param b_eq 等式约束右侧向量
     * @return 最优值和最优解
     */
    @Override
    public OptResult solve(IVector c,  IMatrix A_ub, IVector b_ub,IMatrix A_eq, IVector b_eq, IVector initX) {
        
        //todo: 改造后暂时未处理A_ub, b_ub
        
        // 记录开始时间
        long startTime = System.currentTimeMillis();
        
        // 初始化障碍参数
        double mu = MU_INITIAL;
        
        // 创建初始解向量（小的正数，确保满足非负约束）
        IVector x = initX.copy();
        
        // 投影初始点到满足等式约束的空间
        if (A_eq != null && b_eq != null) {
            x = projectToFeasibleSet(x, A_eq, b_eq);
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
        while (mu > MU_MIN) {
            iterations++;
            
            // 创建带障碍项的目标函数
            IObjectiveFunction objectiveFunction = createBarrierObjectiveFunction(c, A_eq, b_eq, mu);
            
            // 创建带障碍项的梯度函数
            IGradientFunction gradientFunction = createBarrierGradientFunction(c, A_eq, b_eq, mu);
            
            // 创建基础求解器
            this.baseSolver = new LagrangeMultiplierSolver(A_eq, b_eq);
            
            // 使用拉格朗日乘子法求解
            var result = baseSolver.optimize(x, objectiveFunction, gradientFunction);
            
            // 更新评估计数
            functionEvaluations += result.getFunctionEvaluations();
            gradientEvaluations += result.getGradientEvaluations();
            
            // 更新解
            x = result.getOptimalPoint();
            
            // 记录历史信息
            functionValueHistory.add(result.getOptimalValue());
            if (result.getParameterHistory() != null && !result.getParameterHistory().isEmpty()) {
                parameterHistory.addAll(result.getParameterHistory());
            }
            
            // 确保解满足等式约束
            if (A_eq != null && b_eq != null) {
                x = projectToFeasibleSet(x, A_eq, b_eq);
            }
            
            // 减小障碍参数
            mu *= MU_DECAY;
        }
        
        // 计算最终的目标函数值
        double objectiveValue = (Double) c.innerProduct(x);
        functionEvaluations++;
        functionValueHistory.add(objectiveValue);
        parameterHistory.add(x.copy());
        
        // 构建丰富的OptResult
        OptResult.Builder builder = new OptResult.Builder(objectiveValue, x)
            .converged(true)
            .convergenceReason("Lagrange multiplier method completed")
            .iterations(iterations)
            .maxIterations(100) // Using default max iterations
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .functionEvaluations(functionEvaluations)
            .gradientEvaluations(gradientEvaluations)
            .functionValueHistory(functionValueHistory)
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
                    if (Precision.compareTo(xi, 1e-12, 1e-15) <= 0) {
                        barrierTerm += 1e6; // 大的惩罚值
                    } else {
                        // 对数障碍函数: -mu * ln(xi)
                        barrierTerm -= mu * Math.log(xi);
                    }
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
                    if (Precision.compareTo(xi, 1e-12, 1e-15) <= 0) {
                        // 添加大的负梯度使变量变为正
                        gradient.set(i, (Double) gradient.get(i) - 1e6);
                    } else {
                        // 对数障碍函数的梯度: -mu / xi
                        double barrierGradient = -mu / xi;
                        gradient.set(i, (Double) gradient.get(i) + barrierGradient);
                    }
                }
                
                return gradient;
            }
        };
    }
}