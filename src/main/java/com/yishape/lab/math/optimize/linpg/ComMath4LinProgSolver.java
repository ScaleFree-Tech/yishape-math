package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;
import org.apache.commons.math4.legacy.optim.linear.LinearObjectiveFunction;
import org.apache.commons.math4.legacy.optim.linear.LinearConstraint;
import org.apache.commons.math4.legacy.optim.linear.Relationship;
import org.apache.commons.math4.legacy.optim.linear.LinearConstraintSet;
import org.apache.commons.math4.legacy.optim.linear.NonNegativeConstraint;
import org.apache.commons.math4.legacy.optim.linear.SimplexSolver;
import org.apache.commons.math4.legacy.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math4.legacy.optim.PointValuePair;
import org.apache.commons.math4.legacy.optim.MaxIter;
import org.apache.commons.math4.legacy.optim.InitialGuess;

import java.util.ArrayList;
import java.util.List;

/**
 * 基于Commons Math 4的线性规划求解器
 * @author lteb2
 */
public class ComMath4LinProgSolver implements ILinProgSolver{

    // 收敛容差
    private static final double TOLERANCE = 1e-9;
    // 最大迭代次数
    private static final int MAX_ITERATIONS = 1000;

    /**
     * 求解线性规划问题（处理不等式和等式约束）
     *
     * @param c 目标函数系数（最小化问题）
     * @param A_ub 小于等于约束矩阵系数
     * @param b_ub 小于等于约束值（不等式右方）
     * @param A_eq 等式约束矩阵系数
     * @param b_eq 等式约束值（等式右方）
     * @param initX
     * @return 优化结果
     */
    @Override
    public OptResult solve(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq, IVector initX) {
        //todo: 改造后暂时未处理A_ub, b_ub
        
        // 记录开始时间
        long startTime = System.currentTimeMillis();
        
        try {
            // 获取问题维度
            int n = c.length(); // 变量数量
            int m = A_ub != null ? A_ub.rows() : 0; // 不等式约束数量
            
            int m2 = A_eq != null ? A_eq.rows() : 0; // 等式约束数量
            
            // 创建目标函数系数数组
            double[] objectiveCoefficients = new double[n];
            for (int i = 0; i < n; i++) {
                objectiveCoefficients[i] = (Double) c.get(i);
            }
            
            // 创建目标函数
            LinearObjectiveFunction objectiveFunction = new LinearObjectiveFunction(objectiveCoefficients, 0);
            
            // 创建约束条件列表
            List<LinearConstraint> constraints = new ArrayList<>();
            
            // 添加等式约束（如果有）
            if (A_ub != null && b_ub != null && m > 0) {
                for (int i = 0; i < m; i++) {
                    double[] constraintCoefficients = new double[n];
                    for (int j = 0; j < n; j++) {
                        constraintCoefficients[j] = (Double) A_ub.get(i, j);
                    }
                    // 小于等于约束
                    constraints.add(new LinearConstraint(constraintCoefficients, Relationship.LEQ, (Double) b_ub.get(i)));
                }
            }
            
            // 添加等式约束（如果有）
            if (A_eq != null && b_eq != null && m2 > 0) {
                for (int i = 0; i < m2; i++) {
                    double[] constraintCoefficients = new double[n];
                    for (int j = 0; j < n; j++) {
                        constraintCoefficients[j] = (Double) A_eq.get(i, j);
                    }
                    // 等式约束
                    constraints.add(new LinearConstraint(constraintCoefficients, Relationship.EQ, (Double) b_eq.get(i)));
                }
            }
            
            // 创建约束集
            LinearConstraintSet constraintSet = new LinearConstraintSet(constraints);
            
            // 创建初始猜测值（默认全1向量）
            double[] initialGuess = new double[n];
            for (int i = 0; i < n; i++) {
                initialGuess[i] = 1.0;
            }
            
            // 创建求解器
            SimplexSolver solver = new SimplexSolver(TOLERANCE);
            
            // 求解
            PointValuePair solution = solver.optimize(
                new MaxIter(MAX_ITERATIONS),
                objectiveFunction,
                constraintSet,
                GoalType.MINIMIZE,  // 我们的接口要求最小化问题
                new NonNegativeConstraint(true),  // 非负约束
                new InitialGuess(initialGuess)
            );
            
            // 计算执行时间
            long executionTimeMs = System.currentTimeMillis() - startTime;
            
            // 检查求解结果
            if (solution != null) {
                // 提取解向量
                double[] solutionArray = solution.getPoint();
                IVector optimalPoint = IVector.of(solutionArray);
                
                // 计算目标函数值
                double objectiveValue = solution.getValue();
                
                // 构建结果
                OptResult.Builder builder = new OptResult.Builder(objectiveValue, optimalPoint)
                    .converged(true)
                    .convergenceReason("Simplex algorithm converged")
                    .iterations(0) // SimplexSolver doesn't provide iteration count directly
                    .maxIterations(MAX_ITERATIONS)
                    .finalGradientNorm(0.0) // 线性规划不需要梯度
                    .tolerance(TOLERANCE)
                    .executionTimeMs(executionTimeMs)
                    .functionEvaluations(1) // 一次求解调用
                    .gradientEvaluations(0); // 线性规划不需要梯度
                
                return builder.build();
            } else {
                // 无解或求解失败
                OptResult.Builder builder = new OptResult.Builder(Double.NaN, null)
                    .converged(false)
                    .convergenceReason("No feasible solution found")
                    .iterations(0)
                    .maxIterations(MAX_ITERATIONS)
                    .finalGradientNorm(0.0)
                    .tolerance(TOLERANCE)
                    .executionTimeMs(executionTimeMs)
                    .functionEvaluations(0)
                    .gradientEvaluations(0);
                
                return builder.build();
            }
        } catch (Exception e) {
            // 求解过程中出现异常
            long executionTimeMs = System.currentTimeMillis() - startTime;
            OptResult.Builder builder = new OptResult.Builder(Double.NaN, null)
                .converged(false)
                .convergenceReason("Exception occurred: " + e.getMessage())
                .iterations(0)
                .maxIterations(MAX_ITERATIONS)
                .finalGradientNorm(0.0)
                .tolerance(TOLERANCE)
                .executionTimeMs(executionTimeMs)
                .functionEvaluations(0)
                .gradientEvaluations(0);
            
            return builder.build();
        }
    }
    
    
    
    
    
    

    @Override
    public OptResult solveWithNonNegativeEqualConstraints(IVector c, IMatrix A_eq, IVector b_eq, IVector initX) {
        return this.solve(c, null,null, A_eq, b_eq, initX);
    }
}