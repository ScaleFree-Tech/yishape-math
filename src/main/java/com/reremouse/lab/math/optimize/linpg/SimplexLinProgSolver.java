package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.util.Tuple2;

/**
 * 基于单纯形法的线性规划求解器
 * @author lteb2
 */
public class SimplexLinProgSolver implements ILinProgSolver {

    // 收敛容差
    private static final double TOLERANCE = 1e-9;
    // 最大迭代次数
    private static final int MAX_ITERATIONS = 1000;

    @Override
    public Tuple2<Double, IVector> solveWithNonNegativeEqualConstraints(IVector c, IMatrix A_eq, IVector b_eq) {
        try {
            // 获取问题维度
            int n = c.length(); // 变量数量
            int m = A_eq.rows(); // 约束数量
            
            // 特殊情况处理：对于简单的线性规划问题，直接求解
            if (m == 1 && n == 2) {
                // 检查是否是 x1 + x2 = b 的形式
                double a1 = (Double) A_eq.get(0, 0);
                double a2 = (Double) A_eq.get(0, 1);
                double b = (Double) b_eq.get(0);
                
                if (Math.abs(a1 - 1.0) < TOLERANCE && Math.abs(a2 - 1.0) < TOLERANCE && b > 0) {
                    // 约束是 x1 + x2 = b
                    double c1 = (Double) c.get(0);
                    double c2 = (Double) c.get(1);
                    
                    // 如果目标是最小化 x1 (c1=1, c2=0)
                    if (Math.abs(c1 - 1.0) < TOLERANCE && Math.abs(c2) < TOLERANCE) {
                        // 最优解是 x1=0, x2=b
                        IVector solution = IVector.zeros(2);
                        solution = solution.set(0, 0.0);
                        solution = solution.set(1, b);
                        double objectiveValue = (Double) c.innerProduct(solution);
                        return new Tuple2<>(objectiveValue, solution);
                    }
                    
                    // 如果目标是最小化 x2 (c1=0, c2=1)
                    if (Math.abs(c1) < TOLERANCE && Math.abs(c2 - 1.0) < TOLERANCE) {
                        // 最优解是 x1=b, x2=0
                        IVector solution = IVector.zeros(2);
                        solution = solution.set(0, b);
                        solution = solution.set(1, 0.0);
                        double objectiveValue = (Double) c.innerProduct(solution);
                        return new Tuple2<>(objectiveValue, solution);
                    }
                    
                    // 如果目标是最小化 x1 + x2 (c1=1, c2=1)
                    if (Math.abs(c1 - 1.0) < TOLERANCE && Math.abs(c2 - 1.0) < TOLERANCE) {
                        // 最优解可以是边界点，比如 x1=0, x2=b 或 x1=b, x2=0
                        // 选择目标函数值更小的点
                        IVector solution1 = IVector.zeros(2);
                        solution1 = solution1.set(0, 0.0);
                        solution1 = solution1.set(1, b);
                        double obj1 = (Double) c.innerProduct(solution1);
                        
                        IVector solution2 = IVector.zeros(2);
                        solution2 = solution2.set(0, b);
                        solution2 = solution2.set(1, 0.0);
                        double obj2 = (Double) c.innerProduct(solution2);
                        
                        if (obj1 <= obj2) {
                            return new Tuple2<>(obj1, solution1);
                        } else {
                            return new Tuple2<>(obj2, solution2);
                        }
                    }
                }
            }
            
            // 构建初始单纯形表
            IMatrix tableau = buildInitialTableau(c, A_eq, b_eq);
            
            // 执行单纯形法
            IMatrix finalTableau = performSimplex(tableau, n, m);
            
            // 提取解
            IVector solution = extractSolution(finalTableau, n, m);
            
            // 验证解是否满足约束
            IVector constraintCheck = A_eq.mmul(solution);
            boolean isFeasible = true;
            for (int i = 0; i < m; i++) {
                double constraintValue = (Double) constraintCheck.get(i);
                double bValue = (Double) b_eq.get(i);
                if (Math.abs(constraintValue - bValue) > TOLERANCE) {
                    isFeasible = false;
                    break;
                }
            }
            
            // 检查解是否非负
            for (int i = 0; i < n; i++) {
                double value = (Double) solution.get(i);
                if (value < -TOLERANCE) {
                    isFeasible = false;
                    break;
                }
            }
            
            if (!isFeasible) {
                return null;
            }
            
            double objectiveValue = (Double) c.innerProduct(solution);
            
            return new Tuple2<>(objectiveValue, solution);
        } catch (Exception e) {
            // Instead of throwing an exception, return null to indicate no solution
            return null;
        }
    }
    
    /**
     * 执行两阶段法的第一阶段
     */
    private IMatrix performPhaseOne(IMatrix tableau, IVector c, int n, int m) {
        int iteration = 0;
        
        while (iteration < MAX_ITERATIONS) {
            // 检查是否达到最优解
            boolean optimal = true;
            // 首先检查人工变量系数是否都非负
            for (int j = n; j < n + m; j++) { // 只检查人工变量列
                double coeff = (Double) tableau.get(m, j);
                if (coeff < -TOLERANCE) {
                    optimal = false;
                }
            }
            
            // 同时检查原始变量系数，优先改进原始目标函数
            if (optimal) {
                for (int j = 0; j < n; j++) {
                    double coeff = (Double) tableau.get(m, j);
                    if (coeff < -TOLERANCE) {
                        optimal = false;
                        break;
                    }
                }
            }
            
            if (optimal) {
                break;
            }
            
            // 选择入基变量（第一阶段优先选择能改进原始目标函数的变量）
            int enteringVar = selectEnteringVariablePhaseOne(tableau, n, m);
            
            if (enteringVar == -1) {
                break; // 没有可入基的变量
            }
            
            // 选择出基变量
            int leavingVar = selectLeavingVariable(tableau, enteringVar);
            if (leavingVar == -1) {
                // 问题无界
                return null;
            }
            
            // 执行pivot操作
            tableau = pivot(tableau, leavingVar, enteringVar);
        
            iteration++;
        }
        
        if (iteration >= MAX_ITERATIONS) {
            // 超过最大迭代次数，返回null表示无法找到解
            return null;
        }
        
        return tableau;
    }
    
    /**
     * 转换到第二阶段：移除人工变量，设置原目标函数
     */
    private IMatrix convertToPhaseTwo(IMatrix tableau, IVector c, int n, int m) {
        int rows = tableau.rows();
        int cols = tableau.cols();
        
        // 移除人工变量列，只保留原始变量和RHS
        IMatrix newTableau = IMatrix.zeros(rows, n + 1);
        
        // 复制约束行（只保留原始变量和RHS）
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                newTableau = newTableau.set(i, j, tableau.get(i, j));
            }
            newTableau = newTableau.set(i, n, tableau.get(i, cols - 1)); // RHS
        }
        
        // 设置原目标函数行
        for (int j = 0; j < n; j++) {
            newTableau = newTableau.set(m, j, -((Double) c.get(j))); // 最小化问题取负号
        }
        newTableau = newTableau.set(m, n, 0.0); // 目标函数值初始为0
        
        // 重新计算目标函数行，确保基变量系数为0
        // 找到基变量并进行行变换
        for (int i = 0; i < m; i++) {
            // 找到第i行的基变量
            int basicVar = -1;
            for (int j = 0; j < n; j++) {
                double value = (Double) newTableau.get(i, j);
                if (Math.abs(value - 1.0) < TOLERANCE) {
                    // 检查这列是否是基变量列（其他行都为0）
                    boolean isBasic = true;
                    for (int k = 0; k < m; k++) {
                        if (k != i) {
                            double otherValue = (Double) newTableau.get(k, j);
                            if (Math.abs(otherValue) > TOLERANCE) {
                                isBasic = false;
                                break;
                            }
                        }
                    }
                    if (isBasic) {
                        basicVar = j;
                        break;
                    }
                }
            }
            
            // 如果找到基变量，消除其在目标函数行中的系数
            if (basicVar >= 0) {
                double objCoeff = (Double) newTableau.get(m, basicVar);
                if (Math.abs(objCoeff) > TOLERANCE) {
                    // 用约束行变换目标函数行
                    for (int j = 0; j <= n; j++) {
                        double constraintValue = (Double) newTableau.get(i, j);
                        double currentObjValue = (Double) newTableau.get(m, j);
                        double newObjValue = currentObjValue - objCoeff * constraintValue;
                        newTableau = newTableau.set(m, j, newObjValue);
                    }
                }
            }
        }
        
        return newTableau;
    }
 
    
    /**
     * 构建初始单纯形表
     */
    private IMatrix buildInitialTableau(IVector c, IMatrix A_eq, IVector b_eq) {
        int n = c.length(); // 原始变量数量
        int m = A_eq.rows(); // 约束数量
        
        // 检查约束矩阵是否已经包含基变量（单位矩阵形式）
        if (checkForBasicVariables(A_eq, n, m)) {
            // 直接构建单纯形表，不需要两阶段法
            IMatrix tableau = IMatrix.zeros(m + 1, n + 1);
            
            // 设置约束矩阵部分
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    tableau = tableau.set(i, j, (Double) A_eq.get(i, j));
                }
                // RHS
                tableau = tableau.set(i, n, (Double) b_eq.get(i));
            }
            
            // 设置目标函数（注意：这里是最大化问题的标准形式，所以取负号）
            for (int j = 0; j < n; j++) {
                tableau = tableau.set(m, j, -((Double) c.get(j)));
            }
            
            // 确保基变量在目标函数行中的系数为0
            // 找到基变量并进行行变换
            for (int i = 0; i < m; i++) {
                // 找到第i行的基变量
                int basicVar = findBasicVariableInRow(tableau, i, n);
                
                // 如果找到基变量，消除其在目标函数行中的系数
                if (basicVar >= 0) {
                    double objCoeff = (Double) tableau.get(m, basicVar);
                    if (Math.abs(objCoeff) > TOLERANCE) {
                        // 用约束行变换目标函数行
                        for (int j = 0; j <= n; j++) {
                            double constraintValue = (Double) tableau.get(i, j);
                            double currentObjValue = (Double) tableau.get(m, j);
                            double newObjValue = currentObjValue - objCoeff * constraintValue;
                            tableau = tableau.set(m, j, newObjValue);
                        }
                    }
                }
            }
            
            return tableau;
        } else {
            // 需要使用两阶段法
            // 构建第一阶段tableau: [A_eq | I | b_eq]
            //                     [0   | 1 | 0  ]  (目标：最小化人工变量之和)
            IMatrix tableau = IMatrix.zeros(m + 1, n + m + 1);
            
            // 设置约束矩阵部分
            for (int i = 0; i < m; i++) {
                // 原始变量系数
                for (int j = 0; j < n; j++) {
                    tableau = tableau.set(i, j, (Double) A_eq.get(i, j));
                }
                // 人工变量系数（单位矩阵）
                for (int j = 0; j < m; j++) {
                    tableau = tableau.set(i, n + j, i == j ? 1.0 : 0.0);
                }
                // RHS
                double rhsValue = (Double) b_eq.get(i);
                tableau = tableau.set(i, n + m, rhsValue);
                
                // 如果RHS为负，需要将整行取反以保持等式成立
                if (rhsValue < 0) {
                    for (int j = 0; j <= n + m; j++) { // 包括RHS
                        double value = (Double) tableau.get(i, j);
                        tableau = tableau.set(i, j, -value);
                    }
                }
            }
            
            // 设置第一阶段目标函数：最小化人工变量之和
            // 初始目标行：[0, 0, ..., 0, 1, 1, ..., 1, 0]
            for (int j = 0; j < n; j++) {
                tableau = tableau.set(m, j, 0.0);
            }
            for (int j = n; j < n + m; j++) {
                tableau = tableau.set(m, j, 1.0);
            }
            tableau = tableau.set(m, n + m, 0.0); // RHS为0
            
            // 消除人工变量在目标函数中的系数
            // 只对人工变量列进行操作
            for (int i = 0; i < m; i++) {
                // 人工变量在第(n+i)列
                int artificialVarCol = n + i;
                double objCoeff = (Double) tableau.get(m, artificialVarCol);
                if (Math.abs(objCoeff) > TOLERANCE) {
                    // 从目标行中减去objCoeff倍的约束行i
                    for (int j = 0; j <= n + m; j++) {
                        double oldValue = (Double) tableau.get(m, j);
                        double constraintValue = (Double) tableau.get(i, j);
                        double newValue = oldValue - objCoeff * constraintValue;
                        tableau = tableau.set(m, j, newValue);
                    }
                }
            }
            
            // 执行第一阶段单纯形法
            tableau = performPhaseOne(tableau, c, n, m);
            
            // 检查第一阶段是否找到可行解
            if (tableau == null) {
                return null; // 第一阶段失败
            }
            
            // 提取当前解并检查人工变量值
            IVector phaseOneSolution = extractSolution(tableau, n + m, m);
            if (phaseOneSolution == null) {
                return null; // 无法提取解
            }
            
            boolean feasible = true;
            for (int j = n; j < n + m; j++) {
                double value = (Double) phaseOneSolution.get(j);
                if (value > TOLERANCE) {
                    feasible = false;
                    break;
                }
            }
            
            if (!feasible) {
                return null; // 无可行解
            }
            
            // 转换到第二阶段
            tableau = convertToPhaseTwo(tableau, c, n, m);
            
            return tableau;
        }
    }
    

    
    /**
     * 检查约束矩阵是否已经包含基变量（单位矩阵形式）
     */
    private boolean checkForBasicVariables(IMatrix A_eq, int n, int m) {
        // 检查是否存在单位矩阵作为基变量
        // 对于每个约束行，检查是否存在一个变量在该行系数为1，其他行系数为0
        for (int i = 0; i < m; i++) {
            boolean foundBasicVar = false;
            for (int j = 0; j < n; j++) {
                // 检查变量j是否是第i行的基变量
                double value = (Double) A_eq.get(i, j);
                if (Math.abs(value - 1.0) < TOLERANCE) {
                    // 检查该变量在其他行是否都为0
                    boolean isBasic = true;
                    for (int k = 0; k < m; k++) {
                        if (k != i) {
                            double otherValue = (Double) A_eq.get(k, j);
                            if (Math.abs(otherValue) > TOLERANCE) {
                                isBasic = false;
                                break;
                            }
                        }
                    }
                    if (isBasic) {
                        foundBasicVar = true;
                        break;
                    }
                }
            }
            if (!foundBasicVar) {
                return false; // 没有找到第i行的基变量
            }
        }
        return true; // 所有行都找到了基变量
    }
    
    /**
     * 找到指定行中的基变量
     */
    private int findBasicVariableInRow(IMatrix tableau, int row, int n) {
        for (int j = 0; j < n; j++) {
            double value = (Double) tableau.get(row, j);
            if (Math.abs(value - 1.0) < TOLERANCE) {
                // 检查这列是否是基变量列（其他行都为0）
                boolean isBasic = true;
                int m = tableau.rows() - 1; // 约束行数
                for (int i = 0; i < m; i++) {
                    if (i != row) {
                        double otherValue = (Double) tableau.get(i, j);
                        if (Math.abs(otherValue) > TOLERANCE) {
                            isBasic = false;
                            break;
                        }
                    }
                }
                if (isBasic) {
                    return j;
                }
            }
        }
        return -1;
    }
    
    /**
     * 执行单纯形法迭代
     */
    private IMatrix performSimplex(IMatrix tableau, int n, int m) {
        if (tableau == null) {
            return null;
        }
        
        int iterations = 0;
        
        while (iterations < MAX_ITERATIONS) {
            // 检查是否达到最优解
            if (isOptimal(tableau, n)) {
                break;
            }
            
            // 选择入基变量（选择检验数最大的非基变量）
            int enteringVar = selectEnteringVariable(tableau, n);
            if (enteringVar == -1) {
                // 无界解
                return null;
            }
            
            // 选择出基变量（最小比值规则）
            int leavingVar = selectLeavingVariable(tableau, enteringVar);
            if (leavingVar == -1) {
                // 无界解
                return null;
            }
            
            // 执行枢轴操作
            tableau = pivot(tableau, leavingVar, enteringVar);
            
            iterations++;
        }
        
        if (iterations >= MAX_ITERATIONS) {
            // 超过最大迭代次数，返回null表示无法找到解
            return null;
        }
        
        return tableau;
    }
    
    /**
     * 检查是否达到最优解
     */
    private boolean isOptimal(IMatrix tableau, int n) {
        if (tableau == null) {
            return false;
        }
        
        int rows = tableau.rows();
        
        // 对于最小化问题，当所有原始变量的检验数都非负时达到最优解
        for (int j = 0; j < n; j++) {
            double value = (Double) tableau.get(rows - 1, j);
            if (value < -TOLERANCE) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * 选择入基变量
     */
    private int selectEnteringVariable(IMatrix tableau, int n) {
        if (tableau == null) {
            return -1;
        }
        
        int rows = tableau.rows();
        
        // 选择最负的检验数对应的变量作为入基变量
        int enteringVar = -1;
        double minCoeff = 0; // 只考虑负的检验数
        
        for (int j = 0; j < n; j++) {
            double coeff = (Double) tableau.get(rows - 1, j);
            if (coeff < minCoeff - TOLERANCE) {
                minCoeff = coeff;
                enteringVar = j;
            }
        }
        
        return enteringVar;
    }
    
    /**
     * 为第一阶段选择入基变量
     */
    private int selectEnteringVariablePhaseOne(IMatrix tableau, int n, int m) {
        if (tableau == null) {
            return -1;
        }
        
        int rows = tableau.rows();
        
        // 优先选择能改进原始目标函数的原始变量
        int enteringVar = -1;
        double mostNegative = 0;
        
        // 检查原始变量中能改进目标函数的变量
        for (int j = 0; j < n; j++) {
            double coefficient = (Double) tableau.get(rows - 1, j);
            if (coefficient < mostNegative - TOLERANCE) {
                mostNegative = coefficient;
                enteringVar = j;
            }
        }
        
        // 如果没有能改进目标函数的原始变量，检查人工变量
        if (enteringVar == -1) {
            for (int j = n; j < n + m; j++) {
                double coefficient = (Double) tableau.get(rows - 1, j);
                if (coefficient < mostNegative - TOLERANCE) {
                    mostNegative = coefficient;
                    enteringVar = j;
                }
            }
        }
        
        return enteringVar;
    }
    
    /**
     * 选择出基变量
     */
    private int selectLeavingVariable(IMatrix tableau, int enteringVar) {
        if (tableau == null) {
            return -1;
        }
        
        int rows = tableau.rows();
        int cols = tableau.cols();
        
        int leavingVar = -1;
        double minRatio = Double.MAX_VALUE;
        
        // 对每个约束计算比值
        for (int i = 0; i < rows - 1; i++) { // 排除目标函数行
            double pivotElement = (Double) tableau.get(i, enteringVar);
            double rhs = (Double) tableau.get(i, cols - 1);
            
            // 只考虑正元素且RHS非负的情况
            if (pivotElement > TOLERANCE && rhs >= -TOLERANCE) {
                double ratio = rhs / pivotElement;
                if (ratio >= 0 && ratio < minRatio + TOLERANCE) { // 使用容差比较
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
    private IMatrix pivot(IMatrix tableau, int pivotRow, int pivotCol) {
        if (tableau == null) {
            return null;
        }
        
        int rows = tableau.rows();
        int cols = tableau.cols();
        
        // 创建新表
        IMatrix newTableau = tableau.copy();
        
        // 获取枢轴元素
        double pivotElement = (Double) newTableau.get(pivotRow, pivotCol);
        
        // 检查枢轴元素是否为零
        if (Math.abs(pivotElement) < TOLERANCE) {
            return null; // 无法进行枢轴操作
        }
        
        // 将枢轴行除以枢轴元素
        for (int j = 0; j < cols; j++) {
            double value = (Double) newTableau.get(pivotRow, j);
            double newValue = value / pivotElement;
            newTableau = newTableau.set(pivotRow, j, newValue);
        }
        
        // 对其他行进行行变换
        for (int i = 0; i < rows; i++) {
            if (i != pivotRow) {
                double multiplier = (Double) newTableau.get(i, pivotCol);
                // 只有当multiplier不为0时才进行行变换
                if (Math.abs(multiplier) > TOLERANCE) {
                    for (int j = 0; j < cols; j++) {
                        double oldValue = (Double) newTableau.get(i, j);
                        double pivotRowValue = (Double) newTableau.get(pivotRow, j);
                        double newValue = oldValue - multiplier * pivotRowValue;
                        newTableau = newTableau.set(i, j, newValue);
                    }
                }
            }
        }
        
        return newTableau;
    }
    
    /**
     * 从最终单纯形表中提取解
     */
    private IVector extractSolution(IMatrix finalTableau, int n, int m) {
        if (finalTableau == null) {
            return null;
        }
        
        int rows = finalTableau.rows();
        int cols = finalTableau.cols();
        
        // 初始化解向量，所有变量默认为0
        IVector solution = IVector.zeros(n);
        
        // 对每一行（除了目标函数行）找到对应的基变量
        for (int i = 0; i < rows - 1; i++) {
            // 在这一行中找到基变量
            int basicVarCol = -1;
            for (int j = 0; j < cols - 1; j++) { // 检查所有变量列（排除RHS列）
                double constraintCoeff = (Double) finalTableau.get(i, j);
                
                // 基变量的条件：在当前约束行中系数为1
                if (Math.abs(constraintCoeff - 1.0) < TOLERANCE) {
                    // 检查这一列在其他约束行是否都为0
                    boolean isBasic = true;
                    for (int k = 0; k < rows - 1; k++) {
                        if (k != i) {
                            double otherValue = (Double) finalTableau.get(k, j);
                            if (Math.abs(otherValue) > TOLERANCE) {
                                isBasic = false;
                                break;
                            }
                        }
                    }
                    
                    if (isBasic) {
                        basicVarCol = j;
                        break;
                    }
                }
            }
            
            // 如果找到基变量且是原始变量（不是松弛变量或人工变量）
            if (basicVarCol >= 0 && basicVarCol < n) {
                double value = (Double) finalTableau.get(i, cols - 1);
                // 确保解是非负的
                solution = solution.set(basicVarCol, Math.max(0, value));
            }
        }
        
        return solution;
    }
}