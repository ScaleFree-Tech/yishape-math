package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.Opts;
import com.yishape.lab.math.util.RerePrecision;

import java.util.ArrayList;
import java.util.List;
import com.yishape.lab.util.YishapeLogger;

/**
 * 基于内点法的线性规划求解器
 * Interior Point Linear Programming Solver
 *
 * <p>内点法是一种用于求解线性规划和二次规划问题的有效算法。
 * 该实现使用对数障碍函数方法处理非负约束，通过逐步减小障碍参数来逼近最优解。
 * The interior point method is an efficient algorithm for solving linear and quadratic
 * programming problems. This implementation uses logarithmic barrier function to handle
 * non-negative constraints, approaching the optimal solution by gradually reducing the
 * barrier parameter.</p>
 *
 * <h3>算法描述 / Algorithm Description:</h3>
 * <p>求解标准形式的线性规划:
 * minimize c^T * x
 * subject to A_eq * x = b_eq
 *            x >= 0
 *
 * 使用对数障碍函数:
 * f(x, mu) = c^T * x - mu * Σ ln(xi)
 *
 * Solves standard form linear programming:
 * minimize c^T * x
 * subject to A_eq * x = b_eq
 *            x >= 0
 *
 * Using logarithmic barrier function:
 * f(x, mu) = c^T * x - mu * Σ ln(xi)</p>
 *
 * <h3>算法特点 / Algorithm Features:</h3>
 * <ul>
 *   <li>多项式时间复杂度: 相比单纯形法有更好的理论复杂度保证 / Polynomial time complexity: better theoretical complexity than simplex method</li>
 *   <li>适合大规模问题: 对于稀疏矩阵结构效率较高 / Suitable for large-scale problems: efficient for sparse matrix structures</li>
 *   <li>路径跟踪: 沿着中心路径逐步逼近最优解 / Path following: gradually approaches optimal solution along central path</li>
 * </ul>
 *
 * @author lteb2
 * @see ILinProgSolver
 */
public class InteriorPointLinProgSolver implements ILinProgSolver{

    private static final YishapeLogger log = YishapeLogger.getLogger(InteriorPointLinProgSolver.class);


    // 障碍参数的衰减因子 / Barrier parameter decay factor
    private static final double MU_DECAY = 0.9;
    // 障碍参数的初始值 / Initial barrier parameter value
    private static final double MU_INITIAL = 1.0;
    // 障碍参数的最小值 / Minimum barrier parameter value
    private static final double MU_MIN = 1e-10;
    // 收敛容差 / Convergence tolerance
    private static final double TOLERANCE = 1e-6;
    // 最大迭代次数 / Maximum iterations
    private static final int MAX_ITERATIONS = 100;

    // 数值稳定性常量 / Numerical stability constants
    private static final double MIN_VARIABLE_VALUE = 1e-12;  // 变量的最小值 / Minimum variable value
    private static final double PENALTY_FACTOR = 1e6;       // 惩罚因子 / Penalty factor
    private static final double INITIAL_VARIABLE_VALUE = 1e-3; // 初始变量值 / Initial variable value



    /**
     * 使用内点法求解线性规划问题
     * Solve Linear Programming Problem Using Interior Point Method
     *
     * @param c 目标函数系数向量 / Objective function coefficient vector
     * @param A_ub 不等式约束矩阵（当前版本部分支持）/ Inequality constraint matrix (partially supported in current version)
     * @param b_ub 不等式约束右侧向量（当前版本部分支持）/ Inequality constraint RHS (partially supported in current version)
     * @param A_eq 等式约束矩阵 / Equality constraint matrix
     * @param b_eq 等式约束右侧向量 / Equality constraint right-hand side vector
     * @param initX 初始点向量（可选）/ Initial point vector (optional)
     * @return 优化结果，包含最优值、最优解和收敛信息 / Optimization result containing optimal value, solution, and convergence info
     * @throws IllegalArgumentException 如果 c 为 null / If c is null
     * @throws RuntimeException 如果内点法无法找到有效解 / If interior point method fails to find valid solution
     */
    @Override
    public OptResult solve(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq, IVector initX) {
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
            x = IVector.ones(c.length()).multiplyByScalar(INITIAL_VARIABLE_VALUE);
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
        double initialObjectiveValue = c.innerProductValue(x);
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
            IOptimizer optimizer = Opts.lbfgs();
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

            // 检查收敛性（在投影之前，避免投影操作干扰收敛判断）
            if (previousX != null) {
                double changeNorm = x.sub(previousX).norm2Value();
                if (changeNorm < TOLERANCE) {
                    converged = true;
                }
            }

            // 确保解满足等式约束和非负约束
            if (A_eq != null && b_eq != null) {
                x = projectToFeasibleSet(x, A_eq, b_eq);
            }

            // 确保非负性
            for (int i = 0; i < x.length(); i++) {
                double value = (Double) x.get(i);
                if (RerePrecision.compareTo(value, 0.0, TOLERANCE) <= 0) {
                    x.set(i, MIN_VARIABLE_VALUE);
                }
            }

            // 减小障碍参数
            mu *= MU_DECAY;
        }

        // 计算最终的目标函数值
        double objectiveValue = c.dotValue(x);
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
            if (RerePrecision.compareTo(value, -TOLERANCE, TOLERANCE) < 0) {
                feasible = false;
                errorMsg.append("变量 ").append(i).append(" 违反非负约束: ").append(value).append("\n");
            }
        }

        // 检查等式约束（如果存在）
        if (A_eq != null && b_eq != null) {
            IVector constraintViolation = A_eq.mmul(x).sub(b_eq);
            for (int i = 0; i < constraintViolation.length(); i++) {
                double violation = Math.abs(((Number) constraintViolation.get(i)).doubleValue());
                if (RerePrecision.compareTo(violation, TOLERANCE, TOLERANCE) > 0) {
                    feasible = false;
                    errorMsg.append("等式约束 ").append(i).append(" 违反: 误差 = ").append(violation).append("\n");
                }
            }
        }

        if (!feasible) {
            log.warn("警告：最终解不满足约束条件 / Warning: Final solution does not satisfy constraints");
            log.warn(errorMsg.toString());
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
     * Project a Point onto the Feasible Set Defined by Equality Constraints
     *
     * <p>使用最小二乘法求解投影:
     * minimize ||x - x0||^2 subject to A_eq * x = b_eq
     * Uses least squares for projection:
     * minimize ||x - x0||^2 subject to A_eq * x = b_eq</p>
     *
     * @param x 要投影的点向量 / Point vector to project
     * @param A_eq 等式约束矩阵 / Equality constraint matrix
     * @param b_eq 等式约束右侧向量 / Equality constraint right-hand side vector
     * @return 投影后的点向量，如果投影失败则返回原始点 / Projected point vector, returns original point if projection fails
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
     * Create Objective Function with Logarithmic Barrier Term
     *
     * <p>原始线性目标函数: f(x) = c^T * x
     * 带障碍项: f_barrier(x) = c^T * x - mu * Σ ln(xi)
     * Original linear objective: f(x) = c^T * x
     * With barrier: f_barrier(x) = c^T * x - mu * Σ ln(xi)</p>
     *
     * @param c 目标函数系数向量 / Objective function coefficient vector
     * @param A_eq 等式约束矩阵（未使用）/ Equality constraint matrix (unused)
     * @param b_eq 等式约束右侧向量（未使用）/ Equality constraint RHS (unused)
     * @param mu 障碍参数 / Barrier parameter
     * @return 带障碍项的目标函数 / Objective function with barrier term
     */
    private IObjectiveFunction createBarrierObjectiveFunction(IVector c, IMatrix A_eq, IVector b_eq, double mu) {
        return new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                // 原始线性目标函数: f(x) = c^T * x
                double objectiveValue = c.innerProductValue(x);

                // 添加对数障碍项: -mu * Σ ln(xi)
                double barrierTerm = 0.0;
                for (int i = 0; i < x.length(); i++) {
                    double xi = (Double) x.get(i);
                    // 如果变量接近0或为负，添加大的惩罚
                    if (RerePrecision.compareTo(xi, MIN_VARIABLE_VALUE, 1e-15) <= 0) {
                        barrierTerm += PENALTY_FACTOR;
                    } else {
                        // 对数障碍函数: -mu * ln(xi)
                        barrierTerm -= mu * Math.log(xi);
                    }
                }

                // 添加等式约束的惩罚项，引导优化器在投影前逼近可行域
                if (A_eq != null && b_eq != null) {
                    IVector constraintViolation = A_eq.mmul(x).sub(b_eq);
                    double penalty = constraintViolation.innerProductValue(constraintViolation);
                    objectiveValue += PENALTY_FACTOR * penalty;
                }

                return objectiveValue + barrierTerm;
            }
        };
    }

    /**
     * 创建带对数障碍项的梯度函数
     * Create Gradient Function with Logarithmic Barrier Term
     *
     * <p>原始线性函数的梯度是常数向量c
     * 对数障碍函数的梯度: -mu / xi
     * The gradient of original linear function is constant vector c
     * Gradient of logarithmic barrier: -mu / xi</p>
     *
     * @param c 目标函数系数向量 / Objective function coefficient vector
     * @param A_eq 等式约束矩阵 / Equality constraint matrix
     * @param b_eq 等式约束右侧向量 / Equality constraint RHS
     * @param mu 障碍参数 / Barrier parameter
     * @return 带障碍项的梯度函数 / Gradient function with barrier term
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
                    if (RerePrecision.compareTo(xi, MIN_VARIABLE_VALUE, 1e-15) <= 0) {
                        gradient.set(i, (Double) gradient.get(i) - PENALTY_FACTOR);
                    } else {
                        // 对数障碍函数的梯度: -mu / xi
                        double barrierGradient = -mu / xi;
                        gradient.set(i, (Double) gradient.get(i) + barrierGradient);
                    }
                }

                // 添加等式约束惩罚项的梯度，引导优化器在投影前逼近可行域
                if (A_eq != null && b_eq != null) {
                    IVector constraintViolation = A_eq.mmul(x).sub(b_eq);
                    IVector penaltyGradient = A_eq.transpose().mmul(constraintViolation).multiplyByScalar(2 * PENALTY_FACTOR);
                    gradient = gradient.add(penaltyGradient);
                }

                return gradient;
            }
        };
    }
}