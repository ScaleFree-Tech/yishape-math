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
            
            // 构建初始单纯形表
            IMatrix tableau = buildInitialTableau(c, A_eq, b_eq);
            
            // 执行单纯形法
            IMatrix finalTableau = performSimplex(tableau, n, m);
            
            // 提取解
            IVector solution = extractSolution(finalTableau, n, m);
            
            // 验证解是否满足约束
            IVector constraintCheck = A_eq.mmul(solution);
            
            double objectiveValue = (Double) c.innerProduct(solution);
            
            return new Tuple2<>(objectiveValue, solution);
        } catch (Exception e) {
            throw new RuntimeException("单纯形法求解失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 构建初始单纯形表
     */
    private IMatrix buildInitialTableau(IVector c, IMatrix A_eq, IVector b_eq) {
        int n = c.length(); // 变量数量
        int m = A_eq.rows(); // 约束数量
        
        // 对于等式约束，我们添加人工变量
        // 表格结构: [A_eq | I | b_eq]
        //          [c^T  | M | 0  ] 其中M是大数
        IMatrix identity = IMatrix.eye(m);
        IMatrix augmentedA = A_eq.hstack(identity);
        IMatrix constraints = augmentedA.hstack(b_eq.asColumnVector());
        
        // 构建目标函数行 (注意符号，因为我们要最小化)
        // 对人工变量使用大M法，设置大的正系数
        double bigM = 1e6;
        IVector artificialCoeffs = IVector.ones(m).multiplyScalar(-bigM);
        IVector objectiveRow = c.multiplyScalar(-1.0).concat(artificialCoeffs).concat(IVector.of(new double[]{0.0}));
        IMatrix tableau = constraints.vstack(objectiveRow.asColumnVector().transpose());
        
        // 消除人工变量在目标函数中的系数
        // 对每个人工变量，从目标函数行中减去对应约束行乘以bigM
        for (int i = 0; i < m; i++) {
            int artificialVarCol = n + i;
            for (int j = 0; j < tableau.cols(); j++) {
                double constraintValue = (Double) tableau.get(i, j);
                double currentObjValue = (Double) tableau.get(m, j);
                tableau = tableau.set(m, j, currentObjValue - bigM * constraintValue);
            }
        }
        
        return tableau;
    }
    
    /**
     * 执行单纯形法迭代
     */
    private IMatrix performSimplex(IMatrix tableau, int n, int m) {
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
                throw new RuntimeException("线性规划问题无界");
            }
            
            // 选择出基变量（最小比值规则）
            int leavingVar = selectLeavingVariable(tableau, enteringVar);
            if (leavingVar == -1) {
                // 无界解
                throw new RuntimeException("线性规划问题无界");
            }
            
            // 执行枢轴操作
            tableau = pivot(tableau, leavingVar, enteringVar);
            
            iterations++;
        }
        
        if (iterations >= MAX_ITERATIONS) {
            throw new RuntimeException("单纯形法超过最大迭代次数");
        }
        
        return tableau;
    }
    
    /**
     * 检查是否达到最优解
     */
    private boolean isOptimal(IMatrix tableau, int n) {
        int rows = tableau.rows();
        int cols = tableau.cols();
        
        // 首先检查目标函数行的所有检验数是否非正
        // 因为我们将目标函数系数乘以了-1，所以对于最小化问题，
        // 当所有检验数都非正时达到最优解
        for (int j = 0; j < cols - 1; j++) { // 最后一列是右侧常数
            double value = (Double) tableau.get(rows - 1, j);
            if (value > TOLERANCE) {
                return false;
            }
        }
        
        // 对于大M法，还需要检查是否有人工变量在基中且值不为0
        // 如果有，说明还需要继续迭代将人工变量移出基
        for (int i = 0; i < rows - 1; i++) {
            // 在这一行中找到基变量
            int basicVarCol = -1;
            for (int j = 0; j < cols - 1; j++) {
                double value = (Double) tableau.get(i, j);
                if (Math.abs(value - 1.0) < TOLERANCE) {
                    // 检查这一列在其他行是否都为0
                    boolean isBasic = true;
                    for (int k = 0; k < rows - 1; k++) {
                        if (k != i) {
                            double otherValue = (Double) tableau.get(k, j);
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
            
            // 如果找到的基变量是人工变量且值不为0，继续迭代
             if (basicVarCol >= 0) {
                 double rhsValue = (Double) tableau.get(i, cols - 1);
                 // 如果基变量是人工变量（列索引 >= n）且值不为0，说明还需要继续迭代
                 if (basicVarCol >= n && Math.abs(rhsValue) > TOLERANCE) {
                     return false; // 还需要继续迭代将人工变量移出基
                 }
             }
        }
        
        return true;
    }
    
    /**
     * 选择入基变量
     */
    private int selectEnteringVariable(IMatrix tableau, int n) {
        int rows = tableau.rows();
        int cols = tableau.cols();
        
        // 在目标函数行中找到最小的负检验数（绝对值最大的负数）
        // 因为我们将目标函数系数乘以了-1，所以要寻找负值
        int enteringVar = -1;
        double minCoeff = -TOLERANCE;
        
        for (int j = 0; j < n; j++) {
            double coeff = (Double) tableau.get(rows - 1, j);
            if (coeff < minCoeff) {
                minCoeff = coeff;
                enteringVar = j;
            }
        }
        
        return enteringVar;
    }
    
    /**
     * 选择出基变量
     */
    private int selectLeavingVariable(IMatrix tableau, int enteringVar) {
        int rows = tableau.rows();
        int cols = tableau.cols();
        
        int leavingVar = -1;
        double minRatio = Double.MAX_VALUE;
        
        // 对每个约束计算比值
        for (int i = 0; i < rows - 1; i++) { // 排除目标函数行
            double pivotElement = (Double) tableau.get(i, enteringVar);
            if (pivotElement > TOLERANCE) { // 只考虑正元素
                double rhs = (Double) tableau.get(i, cols - 1);
                double ratio = rhs / pivotElement;
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
    private IMatrix pivot(IMatrix tableau, int pivotRow, int pivotCol) {
        int rows = tableau.rows();
        int cols = tableau.cols();
        
        // 创建新表
        IMatrix newTableau = tableau.copy();
        
        // 获取枢轴元素
        double pivotElement = (Double) newTableau.get(pivotRow, pivotCol);
        
        // 将枢轴行除以枢轴元素
        for (int j = 0; j < cols; j++) {
            double value = (Double) newTableau.get(pivotRow, j);
            newTableau = newTableau.set(pivotRow, j, value / pivotElement);
        }
        
        // 对其他行进行行变换
        for (int i = 0; i < rows; i++) {
            if (i != pivotRow) {
                double multiplier = (Double) newTableau.get(i, pivotCol);
                for (int j = 0; j < cols; j++) {
                    double oldValue = (Double) newTableau.get(i, j);
                    double pivotRowValue = (Double) newTableau.get(pivotRow, j);
                    newTableau = newTableau.set(i, j, oldValue - multiplier * pivotRowValue);
                }
            }
        }
        
        return newTableau;
    }
    
    /**
     * 从最终单纯形表中提取解
     */
    private IVector extractSolution(IMatrix finalTableau, int n, int m) {
        int rows = finalTableau.rows();
        int cols = finalTableau.cols();
        
        // 初始化解向量，所有变量默认为0
        IVector solution = IVector.zeros(n);
        
        // 对每一行（除了目标函数行）找到对应的基变量
        for (int i = 0; i < rows - 1; i++) {
            // 在这一行中找到系数为1的列（基变量）
            int basicVarCol = -1;
            for (int j = 0; j < n + m; j++) { // 检查所有变量列（原始变量+人工变量）
                double value = (Double) finalTableau.get(i, j);
                if (Math.abs(value - 1.0) < TOLERANCE) {
                    // 检查这一列在其他行是否都为0
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
            
            // 如果找到基变量
            if (basicVarCol >= 0) {
                double value = (Double) finalTableau.get(i, cols - 1);
                
                // 如果是原始变量，设置其值
                if (basicVarCol < n) {
                    solution = solution.set(basicVarCol, Math.max(0, value));
                }
                // 如果是人工变量且值不为0，说明原问题无解
                else if (basicVarCol >= n && Math.abs(value) > TOLERANCE) {
                    throw new RuntimeException("原问题无可行解：人工变量 " + (basicVarCol - n) + " 的值为 " + value);
                }
            }
        }
        
        return solution;
    }
}