package com.yishape.lab.math.optimize.linpg.simplex;

import com.yishape.lab.math.RereMathUtil;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 简化的线性规划求解器 - 基于commons-math4的核心算法
 * 专注于稳定性和正确性，而不是复杂的优化
 */
public class NaiveSimplexLinProgSolver implements ISimplexLinProgSolver {

    private boolean verbose = false;
    private static final double DEFAULT_EPSILON = 1e-6;
    private static final int MAX_ITERATIONS = 1000;
    private final double epsilon;

    public NaiveSimplexLinProgSolver() {
        this.epsilon = DEFAULT_EPSILON;
    }

    public NaiveSimplexLinProgSolver(double epsilon) {
        this.epsilon = epsilon;
    }

    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    /**
     * 按单纯形法常用的最大化来处理问题，防止在程序中来回转换目标函数出现最大错误
     * @param c 目标函数系数（最大化问题）
     * @param A_ub 小于等于约束矩阵系数
     * @param b_ub 小于等于约束值（不等式右方）
     * @param A_eq 等式约束矩阵系数
     * @param b_eq 等式约束值（等式右方）
     * @param initX 初始点（热启动点）
     * @return 优化结果
     */
    @Override
    public OptResult maximize(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq, IVector initX) {
        //todo: 改造后暂时未处理A_ub, b_ub
        
        
        if (verbose) {
            System.out.println("=== BetterSimplexLinProgSolver.maximizeWithNonNegativeEqualConstraints 开始 ===");
            System.out.println("c = " + c);
            System.out.println("A_eq = " + A_eq);
            System.out.println("b_eq = " + b_eq);
        }

        try {
            // 基本验证
            if (c == null || A_eq == null || b_eq == null) {
                throw new IllegalArgumentException("输入参数不能为null");
            }
            if (c.length() != A_eq.cols()) {
                throw new IllegalArgumentException("目标函数系数与约束矩阵列数不匹配");
            }
            if (A_eq.rows() != b_eq.length()) {
                throw new IllegalArgumentException("约束矩阵行数与约束向量长度不匹配");
            }
            
            int m = A_eq.rows();  // 约束数量
            int n = A_eq.cols();  // 变量数量
            
            if (verbose) {
                System.out.println("问题规模: " + m + " 个约束, " + n + " 个变量");
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
                throw new IllegalArgumentException("约束数量超过变量数量，系统可能无解");
            }
        } catch (Exception e) {
            System.err.println("求解失败: " + e.getMessage());
            if (verbose) {
                e.printStackTrace();
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
     */
    private OptResult solveDeterminateSystem(IVector c, IMatrix A_eq, IVector b_eq) {
        if (verbose) {
            System.out.println("求解方阵系统...");
        }
        
        try {
            // 检查系统是否一致
            double det = computeDeterminant(A_eq);
            if (Math.abs(det) < epsilon) {
                // 矩阵奇异，可能不可行或有无穷解
                if (verbose) {
                    System.out.println("矩阵奇异，检查一致性...");
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
                    System.out.println("解不满足非负性约束");
                }
                IVector fallbackSolution = IVector.zeros(c.length());
                return new OptResult.Builder(Double.NEGATIVE_INFINITY, fallbackSolution)
                    .converged(false)
                    .convergenceReason("解不满足非负性约束")
                    .iterations(1)
                    .build();
            }
            
            // 计算目标函数值
            double objectiveValue = 0.0;
            for (int i = 0; i < Math.min(c.length(), solution.length()); i++) {
                objectiveValue += RereMathUtil.safeDoubleValue(c.get(i)) * RereMathUtil.safeDoubleValue(solution.get(i));
            }
            
            if (verbose) {
                System.out.println("找到可行解: " + solution);
                System.out.println("目标函数值: " + objectiveValue);
            }
            
            return new OptResult.Builder(objectiveValue, solution)
                .converged(true)
                .convergenceReason("直接求解成功")
                .iterations(1)
                .build();
                
        } catch (Exception e) {
            if (verbose) {
                System.out.println("方阵求解失败: " + e.getMessage());
            }
            IVector fallbackSolution = IVector.zeros(c.length());
            return new OptResult.Builder(Double.NEGATIVE_INFINITY, fallbackSolution)
                .converged(false)
                .convergenceReason("矩阵求解失败: " + e.getMessage())
                .iterations(1)
                .build();
        }
    }
    
    /**
     * 计算矩阵行列式（简化版本）
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
                                System.out.println("检测到不一致的约束：行" + i + "和行" + j);
                            }
                            IVector fallbackSolution = IVector.zeros(c.length());
                            return new OptResult.Builder(Double.NEGATIVE_INFINITY, fallbackSolution)
                                .converged(false)
                                .convergenceReason("约束系统不一致")
                                .iterations(1)
                                .build();
                        }
                    } else if (Math.abs(bi) > epsilon) {
                        // bj = 0 但 bi != 0，不一致
                        if (verbose) {
                            System.out.println("检测到不一致的约束：行" + i + "和行" + j);
                        }
                        IVector fallbackSolution = IVector.zeros(c.length());
                        return new OptResult.Builder(Double.NEGATIVE_INFINITY, fallbackSolution)
                            .converged(false)
                            .convergenceReason("约束系统不一致")
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
                .convergenceReason("找到一个解（可能有无穷解）")
                .iterations(1)
                .build();
        } catch (Exception e) {
            IVector fallbackSolution = IVector.zeros(c.length());
            return new OptResult.Builder(Double.NEGATIVE_INFINITY, fallbackSolution)
                .converged(false)
                .convergenceReason("无法求解：" + e.getMessage())
                .iterations(1)
                .build();
        }
    }
    
    /**
     * 求解欠定系统 (m < n) - 使用单纯形法
     */
    private OptResult solveUnderdeterminedSystem(IVector c, IMatrix A_eq, IVector b_eq) {
        if (verbose) {
            System.out.println("求解欠定系统，使用单纯形法...");
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
            System.out.println("初始单纯形表已构建，维度: " + tableau.rows() + "x" + tableau.cols());
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
                    System.out.println("检测到无界解");
                }
                IVector unboundedSolution = extractCurrentSolution(tableau, n, m);
                return new OptResult.Builder(Double.POSITIVE_INFINITY, unboundedSolution)
                    .converged(false)
                    .convergenceReason("问题具有无界解")
                    .iterations(iteration)
                    .build();
            }
            
            // 执行枢轴操作
            performPivotOperation(tableau, leavingVar, enteringVar);
            iteration++;
            
            if (verbose && iteration % 10 == 0) {
                System.out.println("迭代 " + iteration + "，入基: " + enteringVar + ", 出基: " + leavingVar);
            }
        }
        
        if (iteration >= MAX_ITERATIONS) {
            System.err.println("达到最大迭代次数");
        }
        
        // 提取最终解
        IVector solution = extractCurrentSolution(tableau, n, m);
        double objectiveValue = RereMathUtil.safeDoubleValue(tableau.get(m, n + m));
        
        if (verbose) {
            System.out.println("单纯形法完成，迭代次数: " + iteration);
            System.out.println("最终解: " + solution);
            System.out.println("目标函数值: " + objectiveValue);
            printTableau(tableau, m, n);
        }
        
        return new OptResult.Builder(objectiveValue, solution)
            .converged(iteration < MAX_ITERATIONS)
            .convergenceReason(iteration < MAX_ITERATIONS ? "单纯形法成功" : "达到迭代限制")
            .iterations(iteration)
            .build();
    }
    
    /**
     * 查找入基变量（最负系数规则）
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
     */
    private void performPivotOperation(IMatrix tableau, int pivotRow, int pivotCol) {
        double pivot = RereMathUtil.safeDoubleValue(tableau.get(pivotRow, pivotCol));
        
        if (Math.abs(pivot) < epsilon) {
            throw new RuntimeException("枢轴元素太小: " + pivot);
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
     * 提取当前解
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
     */
    private void printTableau(IMatrix tableau, int m, int n) {
        if (!verbose) return;
        
        System.out.println("单纯形表:");
        for (int i = 0; i < tableau.rows(); i++) {
            for (int j = 0; j < tableau.cols(); j++) {
                System.out.printf("%8.3f ", RereMathUtil.safeDoubleValue(tableau.get(i, j)));
            }
            System.out.println();
        }
        System.out.println();
    }
}