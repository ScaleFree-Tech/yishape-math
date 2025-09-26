package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.newton.RereLBFGS;
import com.reremouse.lab.util.Tuple2;

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

    @Override
    public Tuple2<Double, IVector> solve(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq) {
        return ILinProgSolver.super.solve(c, A_ub, b_ub, A_eq, b_eq);
    }

    @Override
    public Tuple2<Double, IVector> solveWithNonNegativeEqualConstraints(IVector c, IMatrix A_eq, IVector b_eq) {
        // 初始化障碍参数
        double mu = MU_INITIAL;
        
        // 创建初始解向量（小的正数，确保满足非负约束）
        IVector x = IVector.ones(c.length()).multiplyScalar(1.0);
        
        // 投影初始点到满足等式约束的空间
        if (A_eq != null && b_eq != null) {
            x = projectToFeasibleSet(x, A_eq, b_eq);
        }
        
        // 主循环：逐步减小障碍参数直到达到最小值
        while (mu > MU_MIN) {
            // 创建带障碍项的目标函数
            IObjectiveFunction objectiveFunction = createBarrierObjectiveFunction(c, A_eq, b_eq, mu);
            
            // 创建带障碍项的梯度函数
            IGradientFunction gradientFunction = createBarrierGradientFunction(c, A_eq, b_eq, mu);
            
            // 使用LBFGS求解当前障碍问题
            RereLBFGS optimizer = new RereLBFGS();
            Tuple2<Double, IVector> result = optimizer.optimize(x, objectiveFunction, gradientFunction);
            
            // 更新解
            x = result._2;
            
            // 确保解满足等式约束
            if (A_eq != null && b_eq != null) {
                x = projectToFeasibleSet(x, A_eq, b_eq);
            }
            
            // 减小障碍参数
            mu *= MU_DECAY;
        }
        
        // 计算最终的目标函数值
        double objectiveValue = (Double) c.innerProduct(x);
        
        return new Tuple2<>(objectiveValue, x);
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
                    if (xi <= 1e-12) {
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
                    if (xi <= 1e-12) {
                        // 添加大的负梯度使变量变为正
                        gradient = gradient.set(i, (Double) gradient.get(i) - 1e6);
                    } else {
                        // 对数障碍函数的梯度: -mu / xi
                        double barrierGradient = -mu / xi;
                        gradient = gradient.set(i, (Double) gradient.get(i) + barrierGradient);
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