package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.util.Tuple2;
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
    }

    /**
     * 设置整数变量
     * @param variableIndex 变量索引
     */
    public void setIntegerVariable(int variableIndex) {
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
     * 设置所有变量为整数变量
     * @param numVariables 变量总数
     */
    public void setAllVariablesInteger(int numVariables) {
        integerVariables.clear();
        for (int i = 0; i < numVariables; i++) {
            integerVariables.add(i);
        }
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

    @Override
    public Tuple2<Double, IVector> solveWithNonNegativeEqualConstraints(IVector c, IMatrix A_eq, IVector b_eq) {
        // 如果没有整数变量，直接使用线性规划求解器
        if (integerVariables.isEmpty()) {
            return baseSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        }
        
        // 使用分支定界法求解
        return branchAndBound(c, A_eq, b_eq);
    }

    /**
     * 分支定界算法核心实现
     */
    private Tuple2<Double, IVector> branchAndBound(IVector c, IMatrix A_eq, IVector b_eq) {
        // 初始化最优解
        double bestObjectiveValue = INFINITY;
        IVector bestSolution = null;
        
        // 使用优先队列存储分支节点，按下界排序
        PriorityQueue<BranchNode> nodeQueue = new PriorityQueue<>(
            Comparator.comparingDouble(node -> node.lowerBound)
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
                System.out.println("迭代 " + iterations + ": 处理节点，下界 = " + currentNode.lowerBound + ", 深度 = " + currentNode.depth);
            }
            
            // 深度剪枝
            if (currentNode.depth > maxDepth) {
                prunedNodes++;
                continue;
            }
            
            // 界限剪枝：如果当前节点的下界已经大于等于已知最优解
            if (currentNode.lowerBound >= bestObjectiveValue - tolerance) {
                prunedNodes++;
                continue;
            }
            
            // 最优性间隙剪枝
            if (bestSolution != null && (bestObjectiveValue - currentNode.lowerBound) <= gapTolerance) {
                prunedNodes++;
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
                continue;
            }
            
            if (lpResult == null) {
                // 无可行解，剪枝
                prunedNodes++;
                continue;
            }
            
            double objectiveValue = lpResult.getFirst();
            IVector solution = lpResult.getSecond();
            
            // 检查解的有效性
            if (Double.isNaN(objectiveValue) || Double.isInfinite(objectiveValue)) {
                prunedNodes++;
                continue;
            }
            
            // 更新节点的下界
            currentNode.lowerBound = objectiveValue;
            currentNode.solution = solution;
            
            // 界限剪枝：如果下界大于等于已知最优解
            if (objectiveValue >= bestObjectiveValue - tolerance) {
                prunedNodes++;
                continue;
            }
            
            // 检查是否为整数解
            if (isIntegerSolution(solution)) {
                // 找到更好的整数解
                if (objectiveValue < bestObjectiveValue) {
                    bestObjectiveValue = objectiveValue;
                    bestSolution = solution.copy(); // 创建副本
                    
                    if (verbose) {
                        System.out.println("找到新的最优整数解，目标值 = " + String.format("%.2f", bestObjectiveValue));
                    }
                }
            } else {
                // 需要分支
                int branchingVariable = selectBranchingVariable(solution);
                if (branchingVariable >= 0) {
                    // 创建子节点
                    double variableValue = (Double) solution.get(branchingVariable);
                    
                    // 左子节点：变量 <= floor(value)
                    BranchNode leftChild = createChildNode(currentNode, branchingVariable, 
                                                         Double.NEGATIVE_INFINITY, Math.floor(variableValue));
                    nodeQueue.offer(leftChild);
                    
                    // 右子节点：变量 >= ceil(value)
                    BranchNode rightChild = createChildNode(currentNode, branchingVariable, 
                                                          Math.ceil(variableValue), Double.POSITIVE_INFINITY);
                    nodeQueue.offer(rightChild);
                }
            }
        }
        
        if (verbose) {
            System.out.println("分支定界算法完成，总迭代次数: " + iterations + ", 剪枝节点数: " + prunedNodes);
            if (bestSolution != null) {
                System.out.println("最优解: " + bestSolution + ", 最优值: " + bestObjectiveValue);
            }
        }
        
        if (bestSolution == null) {
            throw new RuntimeException("未找到可行的整数解。可能原因：问题无可行解、迭代次数不足或搜索深度限制");
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
                Tuple2<IMatrix, IVector> modifiedConstraints = addVariableBounds(A_eq, b_eq, node.variableBounds);
                IMatrix newA = modifiedConstraints.getFirst();
                IVector newB = modifiedConstraints.getSecond();
                
                // 扩展目标函数向量以匹配新的变量数量（原变量 + 松弛变量）
                int originalVars = c.length();
                int newVars = newA.cols();
                
                if (newVars > originalVars) {
                    // 为松弛变量添加0系数
                    double[] extendedC = new double[newVars];
                    for (int i = 0; i < originalVars; i++) {
                        extendedC[i] = (Double) c.get(i);
                    }
                    // 松弛变量的系数为0
                    for (int i = originalVars; i < newVars; i++) {
                        extendedC[i] = 0.0;
                    }
                    IVector newC = Linalg.vector(extendedC);
                    
                    Tuple2<Double, IVector> result = baseSolver.solveWithNonNegativeEqualConstraints(newC, newA, newB);
                    if (result != null) {
                        // 只返回原始变量的解
                        IVector originalSolution = result.getSecond().slice(0, originalVars);
                        return new Tuple2<>(result.getFirst(), originalSolution);
                    }
                    return null;
                } else {
                    return baseSolver.solveWithNonNegativeEqualConstraints(c, newA, newB);
                }
            } else {
                return baseSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
            }
        } catch (Exception e) {
            // 无可行解
            return null;
        }
    }

    /**
     * 添加变量界限约束到约束矩阵中
     * 将变量界限转换为不等式约束，然后使用LinProgUtil转换为等式约束
     */
    private Tuple2<IMatrix, IVector> addVariableBounds(IMatrix A_eq, IVector b_eq, Map<Integer, Tuple2<Double, Double>> bounds) {
        if (bounds.isEmpty()) {
            return new Tuple2<>(A_eq, b_eq);
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
            return new Tuple2<>(A_eq, b_eq);
        }
        
        // 构建不等式约束矩阵
        double[][] ubMatrix = ubConstraints.toArray(new double[0][]);
        IMatrix A_ub = Linalg.matrix(ubMatrix);
        IVector b_ub = Linalg.vector(ubValues.stream().mapToDouble(Double::doubleValue).toArray());
        
        // 使用LinProgUtil将不等式约束转换为等式约束
        // 注意：转换后的约束矩阵会增加松弛变量，所以目标函数向量也需要扩展
        Tuple2<IMatrix, IVector> result = LinProgUtil.convertUbEqToEqConstraits(A_ub, b_ub, A_eq, b_eq);
        
        return result;
    }

    /**
     * 检查解是否满足整数约束
     */
    private boolean isIntegerSolution(IVector solution) {
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
     * 选择分支变量（使用改进的分支策略）
     */
    private int selectBranchingVariable(IVector solution) {
        int bestVar = -1;
        double bestScore = -1;
        
        for (int var : integerVariables) {
            if (var < solution.length()) {
                double value = (Double) solution.get(var);
                double fractionalPart = value - Math.floor(value);
                
                // 跳过已经是整数的变量
                if (Math.abs(fractionalPart) < tolerance || Math.abs(fractionalPart - 1.0) < tolerance) {
                    continue;
                }
                
                // 使用最大小数部分策略（Most Fractional）
                // 选择小数部分最接近0.5的变量，这样分支更平衡
                double score = 0.5 - Math.abs(fractionalPart - 0.5);
                
                // 可以考虑添加变量重要性权重
                // 这里简单地使用目标函数系数的绝对值作为权重
                // double weight = Math.abs((Double) c.get(var));
                // score *= weight;
                
                if (score > bestScore) {
                    bestScore = score;
                    bestVar = var;
                }
            }
        }
        
        return bestVar;
    }

    /**
     * 创建子节点
     */
    private BranchNode createChildNode(BranchNode parent, int variableIndex, double lowerBound, double upperBound) {
        BranchNode child = new BranchNode();
        child.variableBounds.putAll(parent.variableBounds);
        child.variableBounds.put(variableIndex, new Tuple2<>(lowerBound, upperBound));
        child.lowerBound = parent.lowerBound; // 初始下界
        child.depth = parent.depth + 1; // 增加深度
        return child;
    }

    /**
     * 分支节点类
     */
    private static class BranchNode {
        double lowerBound = Double.NEGATIVE_INFINITY;
        IVector solution = null;
        Map<Integer, Tuple2<Double, Double>> variableBounds = new HashMap<>();
        int depth = 0; // 节点深度
    }
}
