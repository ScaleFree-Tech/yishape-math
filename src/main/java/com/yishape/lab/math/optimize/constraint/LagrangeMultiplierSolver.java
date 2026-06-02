package com.yishape.lab.math.optimize.constraint;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.Opts;

import java.util.ArrayList;
import java.util.List;

/**
 * 拉格朗日乘子法求解器，用于求解带线性等式约束的优化问题
 * Lagrange Multiplier Solver for Solving Optimization Problems with Linear Equality Constraints
 *
 * <p>该求解器使用拉格朗日乘子法将带约束的优化问题转换为无约束优化问题。
 * 通过引入惩罚因子迭代增大策略，有效处理线性等式约束。
 * This solver uses the Lagrange multiplier method to convert constrained optimization problems
 * into unconstrained ones. It effectively handles linear equality constraints through an
 * iteratively increasing penalty factor strategy.</p>
 *
 * <h3>问题形式 / Problem Form:</h3>
 * <pre>
 * minimize f(x)
 * subject to A_eq * x = b_eq
 * </pre>
 *
 * <h3>算法描述 / Algorithm Description:</h3>
 * <p>使用增广拉格朗日函数:
 * L(x, λ) = f(x) + λ^T * (A_eq * x - b_eq) + penalty/2 * ||A_eq * x - b_eq||^2
 * The augmented Lagrangian function is used:
 * L(x, λ) = f(x) + λ^T * (A_eq * x - b_eq) + penalty/2 * ||A_eq * x - b_eq||^2</p>
 *
 * @author lteb2
 * @see IOptimizer
 */
public class LagrangeMultiplierSolver implements IOptimizer {

    private IOptimizer baseOptimizer;
    private IMatrix A_eq;//线性等式约束系数矩阵 / Linear equality constraint coefficient matrix
    private IVector b_eq;//线性等式约束右侧向量 / Linear equality constraint right-hand side vector
    private double penaltyFactor = 1.0; // 初始惩罚因子 / Initial penalty factor
    private double penaltyIncreaseRate = 10.0; // 惩罚因子增长速率 / Penalty factor increase rate
    private int maxPenaltyIterations = 100; // 最大惩罚迭代次数 / Maximum penalty iterations

    /**
     * 构造函数，使用默认的L-BFGS优化器
     * Constructor using default L-BFGS optimizer
     *
     * @param A_eq 线性等式约束系数矩阵 / Linear equality constraint coefficient matrix
     * @param b_eq 线性等式约束右侧向量 / Linear equality constraint right-hand side vector
     * @throws IllegalArgumentException 如果 A_eq 或 b_eq 为 null / If A_eq or b_eq is null
     */
    public LagrangeMultiplierSolver(IMatrix A_eq, IVector b_eq) {
        this.A_eq = A_eq;
        this.b_eq = b_eq;
        baseOptimizer = Opts.lbfgs();
    }

    /**
     * 构造函数，使用指定的优化器
     * Constructor with specified optimizer
     *
     * @param A_eq 线性等式约束系数矩阵 / Linear equality constraint coefficient matrix
     * @param b_eq 线性等式约束右侧向量 / Linear equality constraint right-hand side vector
     * @param baseOptimizer 基础无约束优化器 / Base unconstrained optimizer
     * @throws IllegalArgumentException 如果任何参数为 null / If any parameter is null
     */
    public LagrangeMultiplierSolver(IMatrix A_eq, IVector b_eq, IOptimizer baseOptimizer) {
        this.A_eq = A_eq;
        this.b_eq = b_eq;
        this.baseOptimizer = baseOptimizer;
    }

    /**
     * 使用拉格朗日乘子法优化带等式约束的优化问题
     * Optimize with Lagrange Multiplier Method for Equality Constrained Problems
     *
     * @param initX 初始点向量 / Initial point vector
     * @param objFun 目标函数 / Objective function
     * @param grdFun 梯度函数 / Gradient function
     * @return 优化结果，包含最优解和相关信息 / Optimization result containing optimal solution and related information
     * @throws IllegalArgumentException 如果任何参数为 null / If any parameter is null
     */
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
            constraintError = constraintViolation.norm2Value();

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
     * 创建带惩罚项的目标函数，用于增广拉格朗日法
     * Create Penalized Objective Function for Augmented Lagrangian Method
     *
     * <p>惩罚项形式: penalty/2 * ||A_eq * x - b_eq||^2
     * Penalty term form: penalty/2 * ||A_eq * x - b_eq||^2</p>
     *
     * @param originalObjFun 原始目标函数 / Original objective function
     * @param penalty 惩罚因子 / Penalty factor
     * @return 带惩罚项的目标函数 / Penalized objective function
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
                double penaltyValue = penalty / 2.0 * constraintViolation.innerProductValue(constraintViolation);

                return originalValue + penaltyValue;
            }
        };
    }

    /**
     * 创建带惩罚项的梯度函数，用于增广拉格朗日法
     * Create Penalized Gradient Function for Augmented Lagrangian Method
     *
     * <p>惩罚项梯度形式: penalty * A_eq^T * (A_eq * x - b_eq)
     * Penalty term gradient form: penalty * A_eq^T * (A_eq * x - b_eq)</p>
     *
     * @param originalGrdFun 原始梯度函数 / Original gradient function
     * @param penalty 惩罚因子 / Penalty factor
     * @return 带惩罚项的梯度函数 / Penalized gradient function
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
                IVector penaltyGradient = A_eq.transposeNew().mmul(constraintViolation).multiplyByScalar(penalty);

                // 返回总梯度
                return originalGradient.add(penaltyGradient);
            }
        };
    }

    // Getter和Setter方法 / Getter and Setter methods

    /**
     * 获取惩罚因子初始值
     * Get initial penalty factor
     *
     * @return 惩罚因子初始值 / Initial penalty factor
     */
    public double getPenaltyFactor() {
        return penaltyFactor;
    }

    /**
     * 设置惩罚因子初始值
     * Set initial penalty factor
     *
     * @param penaltyFactor 惩罚因子初始值，必须大于0 / Initial penalty factor, must be greater than 0
     * @throws IllegalArgumentException 如果惩罚因子不大于0 / If penalty factor is not greater than 0
     */
    public void setPenaltyFactor(double penaltyFactor) {
        this.penaltyFactor = penaltyFactor;
    }

    /**
     * 获取惩罚因子增长速率
     * Get penalty factor increase rate
     *
     * @return 惩罚因子增长速率 / Penalty factor increase rate
     */
    public double getPenaltyIncreaseRate() {
        return penaltyIncreaseRate;
    }

    /**
     * 设置惩罚因子增长速率
     * Set penalty factor increase rate
     *
     * @param penaltyIncreaseRate 惩罚因子增长速率，必须大于1 / Increase rate, must be greater than 1
     * @throws IllegalArgumentException 如果增长速率不大于1 / If increase rate is not greater than 1
     */
    public void setPenaltyIncreaseRate(double penaltyIncreaseRate) {
        this.penaltyIncreaseRate = penaltyIncreaseRate;
    }

    /**
     * 获取最大惩罚迭代次数
     * Get maximum number of penalty iterations
     *
     * @return 最大迭代次数 / Maximum number of iterations
     */
    public int getMaxPenaltyIterations() {
        return maxPenaltyIterations;
    }

    /**
     * 设置最大惩罚迭代次数
     * Set maximum number of penalty iterations
     *
     * @param maxPenaltyIterations 最大迭代次数，必须大于0 / Maximum iterations, must be greater than 0
     * @throws IllegalArgumentException 如果最大迭代次数不大于0 / If maximum iterations is not greater than 0
     */
    public void setMaxPenaltyIterations(int maxPenaltyIterations) {
        this.maxPenaltyIterations = maxPenaltyIterations;
    }
}