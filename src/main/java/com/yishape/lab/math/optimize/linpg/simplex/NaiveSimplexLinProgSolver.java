package com.yishape.lab.math.optimize.linpg.simplex;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.math.RereMathUtil;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 简化单纯形法线性规划求解器
 * Naive Simplex Linear Programming Solver
 *
 * <p>基于单纯形法的线性规划求解器简化实现，专注于稳定性和正确性。
 * A simplified linear programming solver based on the simplex method,
 * focusing on stability and correctness rather than complex optimizations.</p>
 *
 * <h3>算法描述 / Algorithm Description:</h3>
 * <p>单纯形法通过在可行域的顶点之间移动来寻找最优解。
 * 每次迭代选择一个入基变量和一个出基变量进行枢轴操作。
 * The simplex method finds the optimal solution by moving between vertices of the
 * feasible region. At each iteration, it selects an entering variable and a leaving
 * variable for pivot operations.</p>
 *
 * <h3>支持的约束类型 / Supported Constraint Types:</h3>
 * <ul>
 *   <li>等式约束: A_eq * x = b_eq</li>
 *   <li>非负约束: x >= 0 (隐含)</li>
 *   <li>不等式约束通过引入松弛变量转换为等式约束 / Inequality constraints converted to equality via slack variables</li>
 * </ul>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 * @see ISimplexLinProgSolver
 */
public class NaiveSimplexLinProgSolver implements ISimplexLinProgSolver {

    private static final Logger log = LoggerFactory.getLogger(NaiveSimplexLinProgSolver.class);


    private boolean verbose = false;
    private static final double DEFAULT_EPSILON = 1e-6;
    private static final int MAX_ITERATIONS = 1000;
    private final double epsilon;

    /**
     * 默认构造函数，使用标准参数
     * Default constructor with standard parameters
     *
     * <p>使用默认精度: epsilon=1e-6
     * Uses default precision: epsilon=1e-6</p>
     */
    public NaiveSimplexLinProgSolver() {
        this.epsilon = DEFAULT_EPSILON;
    }

    /**
     * 自定义精度构造函数
     * Constructor with custom precision
     *
     * @param epsilon 数值精度容差，必须大于0 / Numerical precision tolerance, must be greater than 0
     * @throws IllegalArgumentException 如果 epsilon 无效 / If epsilon is invalid
     */
    public NaiveSimplexLinProgSolver(double epsilon) {
        this.epsilon = epsilon;
    }

    /**
     * 设置详细输出模式
     * Set verbose mode
     *
     * @param verbose 是否输出详细信息 / Whether to output detailed information
     */
    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    /**
     * 使用单纯形法求解线性规划问题（以最大化形式处理）
     * Solve Linear Programming Problem Using Simplex Method (Treated as Maximization)
     *
     * <p>按单纯形法常用的最大化来处理问题，防止在程序中来回转换目标函数出现最大误差。
     * The problem is treated as maximization following common simplex practice to prevent
     * accumulation of numerical errors from converting objective functions.</p>
     *
     * @param c 目标函数系数向量（最大化问题）/ Objective function coefficient vector (maximization problem)
     * @param A_ub 不等式约束矩阵系数（当前版本部分支持）/ Inequality constraint matrix (partially supported)
     * @param b_ub 不等式约束右侧值（当前版本部分支持）/ Inequality constraint RHS (partially supported)
     * @param A_eq 等式约束矩阵系数 / Equality constraint matrix
     * @param b_eq 等式约束右侧值 / Equality constraint RHS
     * @param initX 初始点（热启动点，可选）/ Initial point (warm start point, optional)
     * @return 优化结果，包含最优解和收敛信息 / Optimization result containing optimal solution and convergence info
     * @throws IllegalArgumentException 如果必需的参数为 null 或无效 / If required parameters are null or invalid
     */
    @Override
    public OptResult maximize(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq, IVector initX) {
        if (verbose) {
            log.debug("=== BetterSimplexLinProgSolver.maximizeWithNonNegativeEqualConstraints 开始 ===");
            log.debug("c = " + c);
            log.debug("A_eq = " + A_eq);
            log.debug("b_eq = " + b_eq);
        }

        try {
            // 基本验证
            if (c == null || A_eq == null || b_eq == null) {
                throw new IllegalArgumentException("输入参数不能为null / Input parameters cannot be null");
            }
            if (c.length() != A_eq.cols()) {
                throw new IllegalArgumentException("目标函数系数与约束矩阵列数不匹配 / Objective function coefficient dimension does not match constraint matrix columns");
            }
            if (A_eq.rows() != b_eq.length()) {
                throw new IllegalArgumentException("约束矩阵行数与约束向量长度不匹配 / Constraint matrix rows do not match constraint vector length");
            }

            int m = A_eq.rows();  // 约束数量
            int n = A_eq.cols();  // 变量数量

            if (verbose) {
                log.debug("问题规模: " + m + " 个约束, " + n + " 个变量");
            }

            // 检查是否为方阵系统（等式约束数 = 变量数）
            if (m == n) {
                // 方阵系统，直接求解
                return solveDeterminateSystem(c, A_eq, b_eq);
            } else if (m < n) {
                // 欠定系统，使用单纯形法
                return solveUnderdeterminedSystem(c, A_eq, b_eq);
            } else {
                // 超定系统，可能无解或多解
                throw new IllegalArgumentException("约束数量超过变量数量，系统可能无解 / Constraint count exceeds variable count, system may be infeasible");
            }
        } catch (Exception e) {
            log.warn("求解失败: " + e.getMessage());
            if (verbose) {
                log.error("exception", e);
            }

            // 返回失败结果
            IVector fallbackSolution = IVector.zeros(c.length());
            return new OptResult.Builder(Double.NEGATIVE_INFINITY, fallbackSolution)
                .converged(false)
                .convergenceReason("Solver failed: " + e.getMessage())
                .iterations(0)
                .build();
        }
    }

    /**
     * 求解方阵系统 (m = n)
     * Solve Determinate System (m = n)
     *
     * <p>当等式约束数量等于变量数量时，直接求解线性方程组。
     * When the number of equality constraints equals the number of variables,
     * solve the linear system directly.</p>
     *
     * @param c 目标函数系数向量 / Objective function coefficient vector
     * @param A_eq 等式约束矩阵 / Equality constraint matrix
     * @param b_eq 等式约束右侧向量 / Equality constraint RHS vector
     * @return 优化结果 / Optimization result
     */
    private OptResult solveDeterminateSystem(IVector c, IMatrix A_eq, IVector b_eq) {
        if (verbose) {
            log.debug("求解方阵系统...");
        }

        try {
            // 检查系统是否一致
            double det = computeDeterminant(A_eq);
            if (Math.abs(det) < epsilon) {
                // 矩阵奇异，可能不可行或有无穷解
                if (verbose) {
                    log.debug("矩阵奇异，检查一致性...");
                }
                return checkConsistency(c, A_eq, b_eq);
            }

            // 直接求解 A_eq * x = b_eq
            IVector solution = A_eq.solve(b_eq);

            // 检查非负性约束
            boolean feasible = true;
            for (int i = 0; i < solution.length(); i++) {
                if (RereMathUtil.safeDoubleValue(solution.get(i)) < -epsilon) {
                    feasible = false;
                    break;
                }
            }

            if (!feasible) {
                if (verbose) {
                    log.debug("解不满足非负性约束");
                }
                IVector fallbackSolution = IVector.zeros(c.length());
                return new OptResult.Builder(Double.NEGATIVE_INFINITY, fallbackSolution)
                    .converged(false)
                    .convergenceReason("解不满足非负性约束 / Solution does not satisfy non-negative constraints")
                    .iterations(1)
                    .build();
            }

            // 计算目标函数值
            double objectiveValue = 0.0;
            for (int i = 0; i < Math.min(c.length(), solution.length()); i++) {
                objectiveValue += RereMathUtil.safeDoubleValue(c.get(i)) * RereMathUtil.safeDoubleValue(solution.get(i));
            }

            if (verbose) {
                log.debug("找到可行解: " + solution);
                log.debug("目标函数值: " + objectiveValue);
            }

            return new OptResult.Builder(objectiveValue, solution)
                .converged(true)
                .convergenceReason("直接求解成功 / Direct solving successful")
                .iterations(1)
                .build();

        } catch (Exception e) {
            if (verbose) {
                log.debug("方阵求解失败: " + e.getMessage());
            }
            IVector fallbackSolution = IVector.zeros(c.length());
            return new OptResult.Builder(Double.NEGATIVE_INFINITY, fallbackSolution)
                .converged(false)
                .convergenceReason("矩阵求解失败 / Matrix solving failed: " + e.getMessage())
                .iterations(1)
                .build();
        }
    }

    /**
     * 计算矩阵行列式（简化版本）
     * Compute Matrix Determinant (Simplified Version)
     *
     * @param matrix 输入矩阵 / Input matrix
     * @return 行列式值 / Determinant value
     */
    private double computeDeterminant(IMatrix matrix) {
        // 简化实现：对于2x2矩阵
        if (matrix.rows() == 2 && matrix.cols() == 2) {
            double a = RereMathUtil.safeDoubleValue(matrix.get(0, 0));
            double b = RereMathUtil.safeDoubleValue(matrix.get(0, 1));
            double c = RereMathUtil.safeDoubleValue(matrix.get(1, 0));
            double d = RereMathUtil.safeDoubleValue(matrix.get(1, 1));
            return a * d - b * c;
        }
        // 对于其他情况，简单检查是否可逆
        try {
            IVector testVec = IVector.zeros(matrix.rows());
            testVec.set(0, 1.0);
            matrix.solve(testVec);
            return 1.0; // 可逆
        } catch (Exception e) {
            return 0.0; // 奇异
        }
    }

    /**
     * 检查系统一致性
     * Check System Consistency
     *
     * <p>检查增广矩阵的秩是否等于系数矩阵的秩。
     * Checks if the rank of the augmented matrix equals the rank of the coefficient matrix.</p>
     *
     * @param c 目标函数系数向量 / Objective function coefficient vector
     * @param A_eq 等式约束矩阵 / Equality constraint matrix
     * @param b_eq 等式约束右侧向量 / Equality constraint RHS vector
     * @return 优化结果 / Optimization result
     */
    private OptResult checkConsistency(IVector c, IMatrix A_eq, IVector b_eq) {
        // 检查增广矩阵的秩是否等于系数矩阵的秩
        // 简化实现：检查是否存在矛盾的约束
        int m = A_eq.rows();
        int n = A_eq.cols();

        for (int i = 0; i < m - 1; i++) {
            for (int j = i + 1; j < m; j++) {
                // 检查第i行和第j行是否成比例但右侧不成比例
                boolean proportional = true;
                double ratio = 0.0;
                boolean ratioSet = false;

                for (int k = 0; k < n; k++) {
                    double ai = RereMathUtil.safeDoubleValue(A_eq.get(i, k));
                    double aj = RereMathUtil.safeDoubleValue(A_eq.get(j, k));

                    if (Math.abs(ai) > epsilon || Math.abs(aj) > epsilon) {
                        if (Math.abs(aj) < epsilon && Math.abs(ai) > epsilon) {
                            proportional = false;
                            break;
                        }
                        if (Math.abs(ai) < epsilon && Math.abs(aj) > epsilon) {
                            proportional = false;
                            break;
                        }

                        double currentRatio = ai / aj;
                        if (!ratioSet) {
                            ratio = currentRatio;
                            ratioSet = true;
                        } else if (Math.abs(currentRatio - ratio) > epsilon) {
                            proportional = false;
                            break;
                        }
                    }
                }

                if (proportional && ratioSet) {
                    // 检查右侧是否也成比例
                    double bi = RereMathUtil.safeDoubleValue(b_eq.get(i));
                    double bj = RereMathUtil.safeDoubleValue(b_eq.get(j));

                    if (Math.abs(bj) > epsilon) {
                        double bRatio = bi / bj;
                        if (Math.abs(bRatio - ratio) > epsilon) {
                            // 不一致系统
                            if (verbose) {
                                log.debug("检测到不一致的约束：行" + i + "和行" + j);
                            }
                            IVector fallbackSolution = IVector.zeros(c.length());
                            return new OptResult.Builder(Double.NEGATIVE_INFINITY, fallbackSolution)
                                .converged(false)
                                .convergenceReason("约束系统不一致 / Constraint system is inconsistent")
                                .iterations(1)
                                .build();
                        }
                    } else if (Math.abs(bi) > epsilon) {
                        // bj = 0 但 bi != 0，不一致
                        if (verbose) {
                            log.debug("检测到不一致的约束：行" + i + "和行" + j);
                        }
                        IVector fallbackSolution = IVector.zeros(c.length());
                        return new OptResult.Builder(Double.NEGATIVE_INFINITY, fallbackSolution)
                            .converged(false)
                            .convergenceReason("约束系统不一致 / Constraint system is inconsistent")
                            .iterations(1)
                            .build();
                    }
                }
            }
        }

        // 如果没有发现矛盾，尝试求解（可能有无穷解）
        try {
            IVector solution = A_eq.solve(b_eq);

            // 计算目标函数值
            double objectiveValue = 0.0;
            for (int i = 0; i < Math.min(c.length(), solution.length()); i++) {
                objectiveValue += RereMathUtil.safeDoubleValue(c.get(i)) * RereMathUtil.safeDoubleValue(solution.get(i));
            }

            return new OptResult.Builder(objectiveValue, solution)
                .converged(true)
                .convergenceReason("找到一个解（可能有无穷解）/ Found a solution (possibly infinite solutions)")
                .iterations(1)
                .build();
        } catch (Exception e) {
            IVector fallbackSolution = IVector.zeros(c.length());
            return new OptResult.Builder(Double.NEGATIVE_INFINITY, fallbackSolution)
                .converged(false)
                .convergenceReason("无法求解 / Cannot solve: " + e.getMessage())
                .iterations(1)
                .build();
        }
    }

    /**
     * 求解欠定系统 (m < n) - 使用单纯形法
     * Solve Underdetermined System (m < n) - Using Simplex Method
     *
     * @param c 目标函数系数向量 / Objective function coefficient vector
     * @param A_eq 等式约束矩阵 / Equality constraint matrix
     * @param b_eq 等式约束右侧向量 / Equality constraint RHS vector
     * @return 优化结果 / Optimization result
     */
    private OptResult solveUnderdeterminedSystem(IVector c, IMatrix A_eq, IVector b_eq) {
        if (verbose) {
            log.debug("求解欠定系统，使用单纯形法...");
        }

        int m = A_eq.rows();
        int n = A_eq.cols();

        // 构建单纯形表：[A_eq | I | b_eq; -c^T | 0 | 0]
        // 表结构：(m+1) 行 x (n+m+1) 列
        IMatrix tableau = IMatrix.zeros(m + 1, n + m + 1);

        // 填充约束矩阵 A_eq
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                tableau.set(i, j, A_eq.get(i, j));
            }
        }

        // 添加单位矩阵（人工变量）
        for (int i = 0; i < m; i++) {
            tableau.set(i, n + i, 1.0);
        }

        // 填充右侧向量 b_eq
        for (int i = 0; i < m; i++) {
            tableau.set(i, n + m, RereMathUtil.safeDoubleValue(b_eq.get(i)));
        }

        // 设置目标函数行（最大化问题，系数取负）
        for (int j = 0; j < n; j++) {
            tableau.set(m, j, -RereMathUtil.safeDoubleValue(c.get(j)));
        }

        if (verbose) {
            log.debug("初始单纯形表已构建，维度: " + tableau.rows() + "x" + tableau.cols());
            printTableau(tableau, m, n);
        }

        // 执行单纯形迭代
        int iteration = 0;
        while (iteration < MAX_ITERATIONS) {
            // 检查最优性
            int enteringVar = findEnteringVariable(tableau, m, n);
            if (enteringVar == -1) {
                // 达到最优解
                break;
            }

            // 选择出基变量
            int leavingVar = findLeavingVariable(tableau, enteringVar, m);
            if (leavingVar == -1) {
                // 无界解
                if (verbose) {
                    log.debug("检测到无界解");
                }
                IVector unboundedSolution = extractCurrentSolution(tableau, n, m);
                return new OptResult.Builder(Double.POSITIVE_INFINITY, unboundedSolution)
                    .converged(false)
                    .convergenceReason("问题具有无界解 / Problem has unbounded solution")
                    .iterations(iteration)
                    .build();
            }

            // 执行枢轴操作
            performPivotOperation(tableau, leavingVar, enteringVar);
            iteration++;

            if (verbose && iteration % 10 == 0) {
                log.debug("迭代 " + iteration + "，入基: " + enteringVar + ", 出基: " + leavingVar);
            }
        }

        if (iteration >= MAX_ITERATIONS) {
            log.warn("达到最大迭代次数 / Maximum iterations reached");
        }

        // 提取最终解
        IVector solution = extractCurrentSolution(tableau, n, m);
        double objectiveValue = RereMathUtil.safeDoubleValue(tableau.get(m, n + m));

        if (verbose) {
            log.debug("单纯形法完成，迭代次数: " + iteration);
            log.debug("最终解: " + solution);
            log.debug("目标函数值: " + objectiveValue);
            printTableau(tableau, m, n);
        }

        return new OptResult.Builder(objectiveValue, solution)
            .converged(iteration < MAX_ITERATIONS)
            .convergenceReason(iteration < MAX_ITERATIONS ? "单纯形法成功 / Simplex method successful" : "达到迭代限制 / Iteration limit reached")
            .iterations(iteration)
            .build();
    }

    /**
     * 查找入基变量（最负系数规则）
     * Find Entering Variable (Most Negative Coefficient Rule)
     *
     * @param tableau 单纯形表 / Simplex tableau
     * @param m 约束数量 / Number of constraints
     * @param n 变量数量 / Number of variables
     * @return 入基变量索引，如果无变量可入基返回-1 / Entering variable index, -1 if no variable can enter
     */
    private int findEnteringVariable(IMatrix tableau, int m, int n) {
        int objectiveRow = m;
        int enteringVar = -1;
        double mostNegative = 0.0;

        // 在原始变量中寻找最负的系数
        for (int j = 0; j < n; j++) {
            double coeff = RereMathUtil.safeDoubleValue(tableau.get(objectiveRow, j));
            if (coeff < mostNegative) {
                mostNegative = coeff;
                enteringVar = j;
            }
        }

        return enteringVar;
    }

    /**
     * 查找出基变量（最小比值规则）
     * Find Leaving Variable (Minimum Ratio Test)
     *
     * @param tableau 单纯形表 / Simplex tableau
     * @param enteringVar 入基变量索引 / Entering variable index
     * @param m 约束数量 / Number of constraints
     * @return 出基变量索引，如果无变量可出基返回-1 / Leaving variable index, -1 if no variable can leave
     */
    private int findLeavingVariable(IMatrix tableau, int enteringVar, int m) {
        int leavingVar = -1;
        double minRatio = Double.POSITIVE_INFINITY;

        for (int i = 0; i < m; i++) {
            double coeff = RereMathUtil.safeDoubleValue(tableau.get(i, enteringVar));
            double rhs = RereMathUtil.safeDoubleValue(tableau.get(i, tableau.cols() - 1));

            if (coeff > epsilon) {  // 只考虑正系数
                double ratio = rhs / coeff;
                if (ratio >= 0 && ratio < minRatio) {
                    minRatio = ratio;
                    leavingVar = i;
                }
            }
        }

        return leavingVar;
    }

    /**
     * 执行枢轴操作
     * Perform Pivot Operation
     *
     * @param tableau 单纯形表 / Simplex tableau
     * @param pivotRow 枢轴行 / Pivot row
     * @param pivotCol 枢轴列 / Pivot column
     * @throws RuntimeException 如果枢轴元素太小 / If pivot element is too small
     */
    private void performPivotOperation(IMatrix tableau, int pivotRow, int pivotCol) {
        double pivot = RereMathUtil.safeDoubleValue(tableau.get(pivotRow, pivotCol));

        if (Math.abs(pivot) < epsilon) {
            throw new RuntimeException("枢轴元素太小: pivot / Pivot element too small: " + pivot);
        }

        // 归一化枢轴行
        for (int j = 0; j < tableau.cols(); j++) {
            double value = RereMathUtil.safeDoubleValue(tableau.get(pivotRow, j));
            tableau.set(pivotRow, j, value / pivot);
        }

        // 消除其他行的枢轴列
        for (int i = 0; i < tableau.rows(); i++) {
            if (i != pivotRow) {
                double multiplier = RereMathUtil.safeDoubleValue(tableau.get(i, pivotCol));
                for (int j = 0; j < tableau.cols(); j++) {
                    double currentValue = RereMathUtil.safeDoubleValue(tableau.get(i, j));
                    double pivotValue = RereMathUtil.safeDoubleValue(tableau.get(pivotRow, j));
                    tableau.set(i, j, currentValue - multiplier * pivotValue);
                }
            }
        }
    }

    /**
     * 从当前单纯形表提取最优解
     * Extract Optimal Solution from Current Simplex Tableau
     *
     * @param tableau 单纯形表 / Simplex tableau
     * @param n 原始变量数量 / Number of original variables
     * @param m 约束数量 / Number of constraints
     * @return 最优解向量 / Optimal solution vector
     */
    private IVector extractCurrentSolution(IMatrix tableau, int n, int m) {
        IVector solution = IVector.zeros(n);

        // 找到基变量
        for (int i = 0; i < m; i++) {
            // 寻找第i行的基变量（只有一个1，其他都是0的列）
            for (int j = 0; j < n; j++) {
                double value = RereMathUtil.safeDoubleValue(tableau.get(i, j));
                if (Math.abs(value - 1.0) < epsilon) {
                    // 检查这一列在其他行是否都为0
                    boolean isBasic = true;
                    for (int k = 0; k < m; k++) {
                        if (k != i) {
                            double otherValue = RereMathUtil.safeDoubleValue(tableau.get(k, j));
                            if (Math.abs(otherValue) > epsilon) {
                                isBasic = false;
                                break;
                            }
                        }
                    }
                    if (isBasic) {
                        double rhsValue = RereMathUtil.safeDoubleValue(tableau.get(i, tableau.cols() - 1));
                        solution.set(j, rhsValue);
                        break;
                    }
                }
            }
        }

        return solution;
    }

    /**
     * 打印单纯形表（调试用）
     * Print Simplex Tableau (For Debugging)
     *
     * @param tableau 单纯形表 / Simplex tableau
     * @param m 约束数量 / Number of constraints
     * @param n 变量数量 / Number of variables
     */
    private void printTableau(IMatrix tableau, int m, int n) {
        if (!verbose) return;

        log.debug("单纯形表:");
        for (int i = 0; i < tableau.rows(); i++) {
            for (int j = 0; j < tableau.cols(); j++) {
                log.debug(String.format("%8.3f ", RereMathUtil.safeDoubleValue(tableau.get(i, j))));
            }
            log.debug("");
        }
        log.debug("");
    }
}