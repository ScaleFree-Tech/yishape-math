package com.reremouse.lab.math.optimize.constraint;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IOptimizer;
import com.reremouse.lab.math.optimize.newton.RereLBFGS;
import com.reremouse.lab.util.Tuple2;

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
    public Tuple2<Double, IVector> optimize(IVector initX, IObjectiveFunction objFun, IGradientFunction grdFun) {
        // 初始化惩罚因子
        double currentPenalty = penaltyFactor;
        
        // 当前解
        IVector currentX = initX.copy();
        
        // 迭代增大惩罚系数求解
        for (int iter = 0; iter < maxPenaltyIterations; iter++) {
            // 创建带惩罚项的目标函数和梯度函数
            IObjectiveFunction penalizedObjFun = createPenalizedObjectiveFunction(objFun, currentPenalty);
            IGradientFunction penalizedGrdFun = createPenalizedGradientFunction(grdFun, currentPenalty);
            
            // 使用基础优化器求解
            Tuple2<Double, IVector> result = baseOptimizer.optimize(currentX, penalizedObjFun, penalizedGrdFun);
            
            // 更新当前解
            currentX = result.getSecond();
            
            // 检查收敛性：计算约束违反程度
            IVector constraintViolation = A_eq.mmul(currentX).sub(b_eq);
            double constraintError = (Double) constraintViolation.norm2();
            
            // 如果约束满足足够好，则提前退出
            if (constraintError < 1e-6) {
                break;
            }
            
            // 增大惩罚因子
            currentPenalty *= penaltyIncreaseRate;
        }
        
        // 返回最终结果
        double finalValue = objFun.computeObjective(currentX);
        return new Tuple2<>(finalValue, currentX);
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
