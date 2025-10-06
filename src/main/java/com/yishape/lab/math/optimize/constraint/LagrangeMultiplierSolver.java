package com.yishape.lab.math.optimize.constraint;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.newton.RereLBFGS;

import java.util.ArrayList;
import java.util.List;

/**
 * 拉格朗日乘子法求解器
 * <p>
 * 该求解器用于解决带有线性等式约束的优化问题，形式为：
 * minimize f(x)
 * subject to A_eq * x = b_eq
 * </p>
 * <p>
 * 使用拉格朗日乘子法将约束优化问题转换为无约束优化问题：
 * L(x, λ) = f(x) + λ^T * (A_eq * x - b_eq)
 * </p>
 * 
 * @author lteb2
 */
public class LagrangeMultiplierSolver implements IOptimizer {

    private IOptimizer baseOptimizer;
    private IMatrix A_eq;//线性等式约束系数
    private IVector b_eq;//线性等式约束值
    private double penaltyFactor = 1.0; // 惩罚因子
    private double penaltyIncreaseRate = 10.0; // 惩罚因子增长速率
    private int maxPenaltyIterations = 100; // 最大惩罚迭代次数

    public LagrangeMultiplierSolver(IMatrix A_eq, IVector b_eq) {
        this.A_eq = A_eq;
        this.b_eq = b_eq;
        baseOptimizer = new RereLBFGS();
    }
    
    public LagrangeMultiplierSolver(IMatrix A_eq, IVector b_eq, IOptimizer baseOptimizer) {
        this.A_eq = A_eq;
        this.b_eq = b_eq;
        this.baseOptimizer = baseOptimizer;
    }

    @Override
    public OptResult optimize(IVector initX, IObjectiveFunction objFun, IGradientFunction grdFun) {
        // 记录开始时间 / Record start time
        long startTime = System.currentTimeMillis();
        
        // 初始化惩罚因子
        double currentPenalty = penaltyFactor;
        
        // 当前解
        IVector currentX = initX.copy();
        IVector initialPoint = initX.copy(); // 保存初始点 / Save initial point
        
        // 计算初始函数值
        double initialValue = objFun.computeObjective(currentX);
        
        double constraintError = 0;
        
        // 收敛历史记录 / Convergence history tracking
        List<Double> functionValueHistory = new ArrayList<>();
        List<Double> gradientNormHistory = new ArrayList<>();
        List<IVector> parameterHistory = new ArrayList<>();
        
        // 评估计数 / Evaluation counters
        int functionEvaluations = 1; // 初始函数值计算 / Initial function evaluation
        int gradientEvaluations = 0; // 梯度计算将在循环中开始计数 / Gradient evaluations will start counting in loop
        int actualIterations = 0;
        
        // 添加初始历史记录 / Add initial history records
        functionValueHistory.add(initialValue);
        
        boolean converged = false;
        String convergenceReason = "Maximum penalty iterations reached";
        
        // 迭代增大惩罚系数求解
        for (int iter = 0; iter < maxPenaltyIterations; iter++) {
            actualIterations = iter + 1;
            
            // 创建带惩罚项的目标函数和梯度函数
            IObjectiveFunction penalizedObjFun = createPenalizedObjectiveFunction(objFun, currentPenalty);
            IGradientFunction penalizedGrdFun = createPenalizedGradientFunction(grdFun, currentPenalty);
            
            // 使用基础优化器求解
            OptResult result = baseOptimizer.optimize(currentX, penalizedObjFun, penalizedGrdFun);
            
            // 更新评估计数 / Update evaluation counters
            functionEvaluations += result.getFunctionEvaluations();
            gradientEvaluations += result.getGradientEvaluations();
            
            // 更新当前解
            currentX = result.getOptimalPoint();
            
            // 记录历史信息 / Record history information
            functionValueHistory.add(result.getOptimalValue());
            parameterHistory.add(currentX.copy());
            
            // 检查收敛性：计算约束违反程度
            IVector constraintViolation = A_eq.mmul(currentX).sub(b_eq);
            constraintError = (Double) constraintViolation.norm2();
            
            // 如果约束满足足够好，则提前退出
            if (constraintError < 1e-6) {
                converged = true;
                convergenceReason = "Constraint violation below tolerance";
                break;
            }
            
            // 增大惩罚因子
            currentPenalty *= penaltyIncreaseRate;
        }
        
        // 返回最终结果
        double finalValue = objFun.computeObjective(currentX);
        functionEvaluations++;
        
        // 构建丰富的OptResult / Build rich OptResult
        OptResult.Builder builder = new OptResult.Builder(finalValue, currentX)
            .initialPoint(initialPoint)
            .initialValue(initialValue)
            .converged(converged)
            .convergenceReason(convergenceReason)
            .iterations(actualIterations)
            .maxIterations(maxPenaltyIterations)
            .constraintViolation(constraintError)
            .executionTimeMs(System.currentTimeMillis() - startTime)
            .functionEvaluations(functionEvaluations)
            .gradientEvaluations(gradientEvaluations)
            .functionValueHistory(functionValueHistory)
            .parameterHistory(parameterHistory);
        
        return builder.build();
    }
    
    /**
     * 创建带惩罚项的目标函数
     * <p>
     * 惩罚项形式：penalty/2 * ||A_eq * x - b_eq||^2
     * </p>
     * 
     * @param originalObjFun 原始目标函数
     * @param penalty 惩罚因子
     * @return 带惩罚项的目标函数
     */
    private IObjectiveFunction createPenalizedObjectiveFunction(IObjectiveFunction originalObjFun, double penalty) {
        return new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                // 原始目标函数值
                double originalValue = originalObjFun.computeObjective(x);
                
                // 计算约束违反程度: A_eq * x - b_eq
                IVector constraintViolation = A_eq.mmul(x).sub(b_eq);
                
                // 惩罚项: penalty/2 * ||constraintViolation||^2
                double penaltyValue = penalty / 2.0 * (Double) constraintViolation.innerProduct(constraintViolation);
                
                return originalValue + penaltyValue;
            }
        };
    }
    
    /**
     * 创建带惩罚项的梯度函数
     * <p>
     * 惩罚项梯度形式：penalty * A_eq^T * (A_eq * x - b_eq)
     * </p>
     * 
     * @param originalGrdFun 原始梯度函数
     * @param penalty 惩罚因子
     * @return 带惩罚项的梯度函数
     */
    private IGradientFunction createPenalizedGradientFunction(IGradientFunction originalGrdFun, double penalty) {
        return new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                // 原始梯度
                IVector originalGradient = originalGrdFun.computeGradient(x);
                
                // 计算约束违反程度: A_eq * x - b_eq
                IVector constraintViolation = A_eq.mmul(x).sub(b_eq);
                
                // 计算惩罚项梯度: penalty * A_eq^T * (A_eq * x - b_eq)
                IVector penaltyGradient = A_eq.transposeNew().mmul(constraintViolation).multiplyScalar(penalty);
                
                // 返回总梯度
                return originalGradient.add(penaltyGradient);
            }
        };
    }
    
    // Getter和Setter方法
    
    public double getPenaltyFactor() {
        return penaltyFactor;
    }
    
    public void setPenaltyFactor(double penaltyFactor) {
        this.penaltyFactor = penaltyFactor;
    }
    
    public double getPenaltyIncreaseRate() {
        return penaltyIncreaseRate;
    }
    
    public void setPenaltyIncreaseRate(double penaltyIncreaseRate) {
        this.penaltyIncreaseRate = penaltyIncreaseRate;
    }
    
    public int getMaxPenaltyIterations() {
        return maxPenaltyIterations;
    }
    
    public void setMaxPenaltyIterations(int maxPenaltyIterations) {
        this.maxPenaltyIterations = maxPenaltyIterations;
    }
}
