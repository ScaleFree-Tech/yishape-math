package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.util.Tuple2;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ILinProgSolverTest {

    /**
     * 测试带有不等式约束的线性规划问题，验证是否正确移除了松弛变量
     */
    @Test
    public void testSolveWithInequalityConstraintsStripsSlackVariables() {
        // 创建一个简单的线性规划问题:
        // minimize: -x1 - x2 (equivalent to maximizing x1 + x2)
        // subject to: 
        //   x1 + x2 <= 2
        //   x1, x2 >= 0
        
        IVector c = IVector.of(-1.0, -1.0);  // 目标函数系数
        IMatrix A_ub = IMatrix.of(new double[][]{{1.0, 1.0}});  // 不等式约束矩阵
        IVector b_ub = IVector.of(2.0);  // 不等式约束值
        
        // 使用单纯形法求解器
        ILinProgSolver solver = new SimplexLinProgSolver();
        
        // 求解问题
        Tuple2<Double, IVector> result = solver.solve(c, A_ub, b_ub, null, null);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertNotNull(result._2, "解向量不应为null");
        
        // 验证解向量的长度是否正确（应该是2，而不是3（2个原始变量+1个松弛变量））
        assertEquals(2, result._2.length(), "解向量应该只包含原始变量，不包含松弛变量");
        
        // 验证目标函数值
        // 最优解应该是 x1=2, x2=0 或 x1=0, x2=2，目标函数值为-2
        assertEquals(-2.0, result._1, 1e-9, "目标函数值应该为-2");
    }
    
    /**
     * 测试只带有等式约束的线性规划问题
     */
    @Test
    public void testSolveWithEqualityConstraints() {
        // 创建一个简单的线性规划问题:
        // minimize: x1 + x2
        // subject to: 
        //   x1 + x2 = 2
        //   x1, x2 >= 0
        
        IVector c = IVector.of(1.0, 1.0);  // 目标函数系数
        IMatrix A_eq = IMatrix.of(new double[][]{{1.0, 1.0}});  // 等式约束矩阵
        IVector b_eq = IVector.of(2.0);  // 等式约束值
        
        // 使用单纯形法求解器
        ILinProgSolver solver = new SimplexLinProgSolver();
        
        // 求解问题
        Tuple2<Double, IVector> result = solver.solve(c, A_eq, b_eq);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertNotNull(result._2, "解向量不应为null");
        
        // 验证解向量的长度是否正确（应该是2个原始变量）
        assertEquals(2, result._2.length(), "解向量应该只包含原始变量");
        
        // 验证目标函数值
        // 最优解应该是 x1=0, x2=2 或 x1=2, x2=0，目标函数值为2
        assertEquals(2.0, result._1, 1e-9, "目标函数值应该为2");
    }
    
    /**
     * 测试解向量维度是否正确
     */
    @Test
    public void testSolutionVectorDimension() {
        // 创建一个简单的线性规划问题:
        // minimize: x1
        // subject to: 
        //   x1 <= 1
        //   x1 >= 0
        
        IVector c = IVector.of(1.0);  // 目标函数系数 (minimize x1)
        IMatrix A_ub = IMatrix.of(new double[][]{{1.0}});  // 不等式约束矩阵
        IVector b_ub = IVector.of(1.0);  // 不等式约束值
        
        // 使用单纯形法求解器
        ILinProgSolver solver = new SimplexLinProgSolver();
        
        // 求解问题
        Tuple2<Double, IVector> result = solver.solve(c, A_ub, b_ub, null, null);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertNotNull(result._2, "解向量不应为null");
        
        // 验证解向量的长度是否正确（应该是1个原始变量）
        assertEquals(1, result._2.length(), "解向量应该只包含原始变量");
        
        // 验证最优解
        assertEquals(0.0, (Double)result._2.get(0), 1e-9, "x1应该为0（因为我们要最小化x1且x1>=0）");
    }
}