package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;
import java.util.*;

/**
 * 整数规划求解器，基于分支定界法
 * Integer Programming Solver using Branch and Bound method
 *
 * @author lteb2
 */
public class RereIntegerProg implements IIntegerProg {

    // 默认参数
    private static final double DEFAULT_TOLERANCE = 1e-6;
    private static final int DEFAULT_MAX_ITERATIONS = 1000;
    private static final double INFINITY = Double.POSITIVE_INFINITY;
    
    // 线性规划基求解器
    private ILinProgSolver baseSolver;
    
    // 整数变量索引集合
    private Set<Integer> integerVariables;
    
    // 0-1变量索引集合
    private Set<Integer> binaryVariables;
    
    // 算法参数
    private double tolerance = DEFAULT_TOLERANCE;
    private int maxIterations = DEFAULT_MAX_ITERATIONS;
    private boolean verbose = false;
    
    // 性能优化参数
    private double gapTolerance = 1e-6;  // 最优性间隙容忍度
    private int maxDepth = 50;           // 最大搜索深度

    /**
     * 构造函数，使用默认的单纯形法求解器
     */
    public RereIntegerProg() {
        this(new SimplexLinProgSolver());
    }

    /**
     * 构造函数，指定线性规划求解器
     * @param baseSolver 线性规划求解器
     */
    public RereIntegerProg(ILinProgSolver baseSolver) {
        this.baseSolver = baseSolver;
        this.integerVariables = new HashSet<>();
        this.binaryVariables = new HashSet<>();
    }

    /**
     * 设置整数变量
     * @param variableIndex 变量索引
     */
    public void setIntegerVariable(int variableIndex) {
        integerVariables.add(variableIndex);
    }

    /**
     * 设置0-1变量（二进制变量）
     * @param variableIndex 变量索引
     */
    public void setBinaryVariable(int variableIndex) {
        binaryVariables.add(variableIndex);
        // 0-1变量也是整数变量
        integerVariables.add(variableIndex);
    }

    /**
     * 添加整数变量
     * @param variableIndices 变量索引数组
     */
    public void addIntegerVariables(int... variableIndices) {
        for (int index : variableIndices) {
            integerVariables.add(index);
        }
    }

    /**
     * 添加0-1变量（二进制变量）
     * @param variableIndices 变量索引数组
     */
    public void addBinaryVariables(int... variableIndices) {
        for (int index : variableIndices) {
            binaryVariables.add(index);
            // 0-1变量也是整数变量
            integerVariables.add(index);
        }
    }

    /**
     * 设置所有变量为整数变量
     * @param numVariables 变量总数
     */
    private void setAllVariablesInteger(int numVariables) {
        integerVariables.clear();
        binaryVariables.clear();
        for (int i = 0; i < numVariables; i++) {
            integerVariables.add(i);
        }
    }

    /**
     * 设置所有变量为0-1变量（二进制变量）
     * @param numVariables 变量总数
     */
    private void setAllVariablesBinary(int numVariables) {
        integerVariables.clear();
        binaryVariables.clear();
        for (int i = 0; i < numVariables; i++) {
            binaryVariables.add(i);
            integerVariables.add(i);
        }
    }

    @Override
    public void setAllVariablesInteger() {
        // Record intent to set all variables as integer without throwing exception
        // Will be applied during solve() when variable count is known
        this.allVariablesInteger = true;
        this.allVariablesBinary = false; // Reset binary flag if set
    }

    @Override
    public void setAllVariablesBinary() {
        // Record intent to set all variables as binary without throwing exception
        // Will be applied during solve() when variable count is known
        this.allVariablesBinary = true;
        this.allVariablesInteger = false; // Reset integer flag if set
    }
    
    // Flags to indicate if all variables should be integer or binary
    private boolean allVariablesInteger = false;
    private boolean allVariablesBinary = false;
    
    // Store the constraint matrix dimensions for variable count inference
    private int constraintVariableCount = -1;
    
    // Method to set constraint variable count
    private void setConstraintVariableCount(IVector c) {
        this.constraintVariableCount = c.length();
    }
    
    /**
     * 设置收敛容差
     * @param tolerance 容差值
     */
    public void setTolerance(double tolerance) {
        this.tolerance = tolerance;
    }

    /**
     * 设置最大迭代次数
     * @param maxIterations 最大迭代次数
     */
    public void setMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
    }

    /**
     * 设置是否输出详细信息
     * @param verbose 是否详细输出
     */
    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }
    
    /**
     * 设置最优性间隙容忍度
     */
    public void setGapTolerance(double gapTolerance) {
        this.gapTolerance = Math.max(gapTolerance, 1e-12);
    }
    
    /**
     * 设置最大搜索深度
     */
    public void setMaxDepth(int maxDepth) {
        this.maxDepth = Math.max(maxDepth, 1);
    }
    
    // 保存原始变量数量，用于在扩展目标函数向量后正确提取解
    private int originalVariableCount = -1;
    
    // 临时保存变量数量，用于在未解决问题前设置所有变量为二进制
    private int tempVariableCount = -1;
    
    @Override
    public Tuple2<Double, IVector> solveWithNonNegativeEqualConstraints(IVector c, IMatrix A_eq, IVector b_eq) {
        // Save original variable count
        originalVariableCount = c.length();
        
        // Apply all variables integer or binary constraints if flagged
        if (allVariablesInteger && integerVariables.isEmpty() && binaryVariables.isEmpty()) {
            setAllVariablesInteger(originalVariableCount);
        } else if (allVariablesBinary && binaryVariables.isEmpty()) {
            setAllVariablesBinary(originalVariableCount);
        }
        
        // If no integer variables, directly use linear programming solver
        if (integerVariables.isEmpty()) {
            Tuple2<Double, IVector> result = baseSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
            // If needed, extract the original variables solution
            if (result != null && originalVariableCount > 0 && result.getSecond().length() > originalVariableCount) {
                IVector originalSolution = result.getSecond().slice(0, originalVariableCount);
                return new Tuple2<>(result.getFirst(), originalSolution);
            }
            return result;
        }
        
        // Use branch and bound algorithm to solve
        Tuple2<Double, IVector> result = branchAndBound(c, A_eq, b_eq);
        // If needed, extract the original variables solution
        if (result != null && originalVariableCount > 0 && result.getSecond().length() > originalVariableCount) {
            IVector originalSolution = result.getSecond().slice(0, originalVariableCount);
            return new Tuple2<>(result.getFirst(), originalSolution);
        }
        // Return null instead of throwing exception to be consistent with the interface
        return result;
    }

    /**
     * 分支定界算法核心实现
     */
    private Tuple2<Double, IVector> branchAndBound(IVector c, IMatrix A_eq, IVector b_eq) {
        // 初始化最优解
        double bestObjectiveValue = INFINITY;
        IVector bestSolution = null;
        
        // 使用优先队列存储分支节点，按下界排序，然后按创建顺序排序
        PriorityQueue<BranchNode> nodeQueue = new PriorityQueue<>(
            Comparator.comparingDouble((BranchNode node) -> node.lowerBound)
                     .thenComparingInt(node -> node.id)
        );
        
        // 创建根节点
        BranchNode rootNode = new BranchNode();
        nodeQueue.offer(rootNode);
        
        int iterations = 0;
        int prunedNodes = 0;
        
        while (!nodeQueue.isEmpty() && iterations < maxIterations) {
            iterations++;
            
            BranchNode currentNode = nodeQueue.poll();
            
            if (verbose) {
                System.out.println("迭代 " + iterations + ": 处理节点，下界 = " + currentNode.lowerBound + ", 深度 = " + currentNode.depth + ", ID = " + currentNode.id);
                if (!currentNode.variableBounds.isEmpty()) {
                    System.out.println("  节点变量界限: " + currentNode.variableBounds);
                }
            }
            
            // 深度剪枝
            if (currentNode.depth > maxDepth) {
                prunedNodes++;
                if (verbose) {
                    System.out.println("  深度剪枝");
                }
                continue;
            }
            
            // 求解当前节点的线性规划松弛问题
            Tuple2<Double, IVector> lpResult = null;
            try {
                lpResult = solveLPRelaxation(c, A_eq, b_eq, currentNode);
            } catch (Exception e) {
                if (verbose) {
                    System.out.println("求解LP松弛问题时出错: " + e.getMessage());
                }
                prunedNodes++;
                continue;
            }
            
            if (lpResult == null) {
                // 无可行解，剪枝
                prunedNodes++;
                if (verbose) {
                    System.out.println("  无可行解，剪枝");
                }
                continue;
            }
            
            double objectiveValue = lpResult.getFirst();
            IVector solution = lpResult.getSecond();
            
            // 检查解的有效性
            if (Double.isNaN(objectiveValue) || Double.isInfinite(objectiveValue)) {
                prunedNodes++;
                if (verbose) {
                    System.out.println("  无效解，剪枝");
                }
                continue;
            }
            
            // 更新节点的下界
            currentNode.lowerBound = objectiveValue;
            currentNode.solution = solution;
            
            if (verbose) {
                System.out.println("  LP松弛解: " + solution + ", 目标值: " + objectiveValue);
            }
            
            // 检查是否为整数解
            if (isIntegerSolution(solution)) {
                // 找到更好的整数解
                if (objectiveValue < bestObjectiveValue - tolerance) {
                    bestObjectiveValue = objectiveValue;
                    bestSolution = solution.copy(); // 创建副本
                    
                    if (verbose) {
                        System.out.println("找到新的最优整数解，目标值 = " + String.format("%.2f", bestObjectiveValue));
                        System.out.println("解: " + bestSolution);
                    }
                } else if (verbose) {
                    System.out.println("找到整数解，但不是更优解: 目标值 = " + String.format("%.2f", objectiveValue));
                    System.out.println("解: " + solution);
                }
                continue; // 整数解不需要再分支
            }
            
            // 界限剪枝：如果下界大于等于已知最优解（考虑容差）
            // 只有在已经找到整数解的情况下才进行界限剪枝
            if (bestSolution != null && objectiveValue >= bestObjectiveValue + gapTolerance) {
                prunedNodes++;
                if (verbose) {
                    System.out.println("  界限剪枝: 下界(" + objectiveValue + ") >= 最优值(" + bestObjectiveValue + ") + 容差(" + gapTolerance + ")");
                }
                continue;
            }
            
            // 需要分支
            int branchingVariable = selectBranchingVariable(solution);
            if (branchingVariable >= 0) {
                // 创建子节点
                double variableValue = (Double) solution.get(branchingVariable);
                
                if (verbose) {
                    System.out.println("  分支变量: x" + branchingVariable + " = " + variableValue);
                }
                
                // 标准的分支策略
                // 左子节点：变量 <= floor(value)
                int floorValue = (int) Math.floor(variableValue);
                BranchNode leftChild = createChildNode(currentNode, branchingVariable, 
                                                     Double.NEGATIVE_INFINITY, floorValue);
                nodeQueue.offer(leftChild);
                
                // 右子节点：变量 >= ceil(value)
                int ceilValue = (int) Math.ceil(variableValue);
                BranchNode rightChild = createChildNode(currentNode, branchingVariable, 
                                                      ceilValue, Double.POSITIVE_INFINITY);
                nodeQueue.offer(rightChild);
                
                if (verbose) {
                    System.out.println("  创建子节点: 左节点(x" + branchingVariable + " <= " + floorValue + 
                                     "), 右节点(x" + branchingVariable + " >= " + ceilValue + ")");
                    System.out.println("    左节点 ID = " + leftChild.id + ", 右节点 ID = " + rightChild.id);
                }
            } else if (verbose) {
                System.out.println("  无法选择分支变量");
            }
        }
        
        if (verbose) {
            System.out.println("分支定界算法完成，总迭代次数: " + iterations + ", 剪枝节点数: " + prunedNodes);
            if (bestSolution != null) {
                System.out.println("最优解: " + bestSolution + ", 最优值: " + bestObjectiveValue);
            } else {
                System.out.println("未找到可行的整数解");
            }
        }
        
        // Instead of throwing an exception, return null to indicate no solution found
        if (bestSolution == null) {
            return null;
        }
        
        return new Tuple2<>(bestObjectiveValue, bestSolution);
    }

    /**
     * 求解线性规划松弛问题
     */
    private Tuple2<Double, IVector> solveLPRelaxation(IVector c, IMatrix A_eq, IVector b_eq, BranchNode node) {
        try {
            // 如果节点有变量界限，需要添加到约束中
            if (!node.variableBounds.isEmpty()) {
                Tuple3<IVector, IMatrix, IVector> modifiedConstraints = addVariableBounds(c, A_eq, b_eq, node.variableBounds);
                IVector newC = modifiedConstraints.getFirst();
                IMatrix newA = modifiedConstraints.getSecond();
                IVector newB = modifiedConstraints.getThird();
                
                Tuple2<Double, IVector> result = baseSolver.solveWithNonNegativeEqualConstraints(newC, newA, newB);
                if (result != null) {
                    // 验证解是否满足所有约束，包括变量界限
                    IVector solution = result.getSecond();
                    boolean feasible = true;
                    
                    // 检查变量界限
                    for (Map.Entry<Integer, Tuple2<Double, Double>> entry : node.variableBounds.entrySet()) {
                        int varIndex = entry.getKey();
                        double lowerBound = entry.getValue().getFirst();
                        double upperBound = entry.getValue().getSecond();
                        
                        if (varIndex < solution.length()) {
                            double value = (Double) solution.get(varIndex);
                            if (value < lowerBound - tolerance || value > upperBound + tolerance) {
                                feasible = false;
                                break;
                            }
                        }
                    }
                    
                    if (feasible) {
                        // 只返回原始变量的解
                        IVector originalSolution = solution.slice(0, originalVariableCount);
                        return new Tuple2<>(result.getFirst(), originalSolution);
                    }
                }
                return null;
            } else {
                // 添加0-1变量的显式边界约束
                Map<Integer, Tuple2<Double, Double>> bounds = new HashMap<>();
                for (int varIndex : binaryVariables) {
                    bounds.put(varIndex, new Tuple2<>(0.0, 1.0));
                }
                
                if (!bounds.isEmpty()) {
                    Tuple3<IVector, IMatrix, IVector> modifiedConstraints = addVariableBounds(c, A_eq, b_eq, bounds);
                    IVector newC = modifiedConstraints.getFirst();
                    IMatrix newA = modifiedConstraints.getSecond();
                    IVector newB = modifiedConstraints.getThird();
                    
                    Tuple2<Double, IVector> result = baseSolver.solveWithNonNegativeEqualConstraints(newC, newA, newB);
                    if (result != null) {
                        // 只返回原始变量的解
                        IVector originalSolution = result.getSecond().slice(0, originalVariableCount);
                        return new Tuple2<>(result.getFirst(), originalSolution);
                    }
                    return null;
                } else {
                    return baseSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
                }
            }
        } catch (Exception e) {
            // 无可行解
            if (verbose) {
                System.out.println("LP松弛问题无可行解: " + e.getMessage());
            }
            return null;
        }
    }

    /**
     * 添加变量界限约束到约束矩阵中
     * 将变量界限转换为不等式约束，然后使用LinProgUtil转换为等式约束
     */
    private Tuple3<IVector, IMatrix, IVector> addVariableBounds(IVector c, IMatrix A_eq, IVector b_eq, Map<Integer, Tuple2<Double, Double>> bounds) {
        if (bounds.isEmpty()) {
            return new Tuple3<>(c, A_eq, b_eq);
        }
        
        int numVars = A_eq.cols();
        List<double[]> ubConstraints = new ArrayList<>();
        List<Double> ubValues = new ArrayList<>();
        
        // 为每个有界限的变量添加约束
        for (Map.Entry<Integer, Tuple2<Double, Double>> entry : bounds.entrySet()) {
            int varIndex = entry.getKey();
            double lowerBound = entry.getValue().getFirst();
            double upperBound = entry.getValue().getSecond();
            
            if (varIndex >= numVars) continue; // 跳过无效的变量索引
            
            // 添加下界约束：-x_i <= -lowerBound (即 x_i >= lowerBound)
            if (!Double.isInfinite(lowerBound)) {
                double[] constraint = new double[numVars];
                constraint[varIndex] = -1.0;
                ubConstraints.add(constraint);
                ubValues.add(-lowerBound);
            }
            
            // 添加上界约束：x_i <= upperBound
            if (!Double.isInfinite(upperBound)) {
                double[] constraint = new double[numVars];
                constraint[varIndex] = 1.0;
                ubConstraints.add(constraint);
                ubValues.add(upperBound);
            }
        }
        
        if (ubConstraints.isEmpty()) {
            return new Tuple3<>(c, A_eq, b_eq);
        }
        
        // 构建不等式约束矩阵
        double[][] ubMatrix = ubConstraints.toArray(new double[0][]);
        IMatrix A_ub = Linalg.matrix(ubMatrix);
        IVector b_ub = Linalg.vector(ubValues.stream().mapToDouble(Double::doubleValue).toArray());
        
        // 使用LinProgUtil将不等式约束转换为等式约束
        // 注意：转换后的约束矩阵会增加松弛变量，所以目标函数向量也需要扩展
        Tuple3<IVector, IMatrix, IVector> result = LinProgUtil.convertUbEqToEqConstraits(c, A_ub, b_ub, A_eq, b_eq);
        
        return result;
    }

    /**
     * 检查解是否满足整数约束
     */
    private boolean isIntegerSolution(IVector solution) {
        // 检查普通整数变量
        for (int index : integerVariables) {
            if (index < solution.length()) {
                double value = (Double) solution.get(index);
                if (Math.abs(value - Math.round(value)) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * 检查解是否满足0-1约束
     */
    private boolean isBinarySolution(IVector solution) {
        // 检查0-1变量是否在[0,1]范围内
        for (int index : binaryVariables) {
            if (index < solution.length()) {
                double value = (Double) solution.get(index);
                if (value < -tolerance || value > 1 + tolerance) {
                    return false;
                }
                // 检查是否接近0或1
                if (Math.abs(value) > tolerance && Math.abs(value - 1) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * 选择分支变量（使用改进的分支策略）
     */
    private int selectBranchingVariable(IVector solution) {
        int bestVar = -1;
        double bestScore = -1.0; // 分数越高越优先选择
        
        // 遍历所有整数变量，选择最不适定的变量进行分支
        for (int index : integerVariables) {
            if (index < solution.length()) {
                double value = (Double) solution.get(index);
                // 对于0-1变量，优先选择接近0.5的值
                if (binaryVariables.contains(index)) {
                    // 计算与最近整数的距离，越接近0.5越不适定
                    double fractionalPart = Math.abs(value - Math.round(value));
                    double score = 0.5 - Math.abs(fractionalPart - 0.5);
                    
                    // 优先选择分数部分接近0.5的变量
                    if (score > bestScore) {
                        bestScore = score;
                        bestVar = index;
                    }
                } else {
                    // 对于一般整数变量，使用原来的策略
                    double fractionalPart = Math.abs(value - Math.round(value));
                    double score = 0.5 - Math.abs(fractionalPart - 0.5);
                    
                    // 优先选择分数部分接近0.5的变量
                    if (score > bestScore) {
                        bestScore = score;
                        bestVar = index;
                    }
                }
            }
        }
        
        return bestVar;
    }
    
    /**
     * 创建子节点
     */
    private BranchNode createChildNode(BranchNode parent, int branchingVariable, double lowerBound, double upperBound) {
        BranchNode child = new BranchNode();
        child.depth = parent.depth + 1;
        // Don't set the lower bound here, let the LP relaxation compute it
        child.lowerBound = Double.NEGATIVE_INFINITY;
        
        // 复制父节点的变量界限
        child.variableBounds = new HashMap<>(parent.variableBounds);
        
        // 添加新的变量界限
        Tuple2<Double, Double> currentBounds = child.variableBounds.getOrDefault(branchingVariable, 
            new Tuple2<>(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY));
        double newLower = Math.max(currentBounds.getFirst(), lowerBound);
        double newUpper = Math.min(currentBounds.getSecond(), upperBound);
        child.variableBounds.put(branchingVariable, new Tuple2<>(newLower, newUpper));
        
        return child;
    }
    
    /**
     * 为等式约束创建子节点
     */
    private BranchNode createChildNodeForEquality(BranchNode parent, int branchingVariable, int value) {
        BranchNode child = new BranchNode();
        child.depth = parent.depth + 1;
        // Don't set the lower bound here, let the LP relaxation compute it
        child.lowerBound = Double.NEGATIVE_INFINITY;
        
        // 复制父节点的变量界限
        child.variableBounds = new HashMap<>(parent.variableBounds);
        
        // 添加等式约束: 变量 = value
        child.variableBounds.put(branchingVariable, new Tuple2<>((double) value, (double) value));
        
        return child;
    }
    
    /**
     * 分支节点类
     */
    private static class BranchNode {
        // 节点深度
        int depth = 0;
        
        // 节点的下界（LP松弛问题的最优值）
        double lowerBound = Double.NEGATIVE_INFINITY;
        
        // 当前节点的解
        IVector solution = null;
        
        // 变量界限约束：变量索引 -> (下界, 上界)
        Map<Integer, Tuple2<Double, Double>> variableBounds = new HashMap<>();
        
        // 节点创建顺序，用于优先队列排序
        static int nextId = 0;
        int id = nextId++;
    }
}