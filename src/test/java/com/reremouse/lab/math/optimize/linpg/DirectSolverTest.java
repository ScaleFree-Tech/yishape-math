package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class DirectSolverTest {

    /**
     * 测试直接调用solveWithNonNegativeEqualConstraints方法
     */
    @Test
    public void testDirectSolveWithNonNegativeEqualConstraints() {
        // 创建一个简单的线性规划问题:
        // minimize: x1
        // subject to: 
        //   x1 + s1 = 1 (where s1 is a slack variable)
        //   x1 >= 0, s1 >= 0
        
        IVector c = IVector.of(1.0, 0.0);  // 目标函数系数 (minimize x1, s1 not in objective)
        IMatrix A_eq = IMatrix.of(new double[][]{{1.0, 1.0}});  // 等式约束矩阵
        IVector b_eq = IVector.of(1.0);  // 等式约束值
        
        // 使用单纯形法求解器
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        
        // 直接求解
        Tuple2<Double, IVector> result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertNotNull(result._2, "解向量不应为null");
        
        // 验证解向量的长度（应该是2个变量：x1和s1）
        assertEquals(2, result._2.length(), "解向量应该包含2个变量");
        
        // 验证最优解
        assertEquals(0.0, (Double)result._2.get(0), 1e-9, "x1应该为0（因为我们要最小化x1且x1>=0）");
        assertEquals(1.0, (Double)result._2.get(1), 1e-9, "s1应该为1（因为x1+s1=1且x1=0）");
        
        // 验证目标函数值
        assertEquals(0.0, result._1, 1e-9, "目标函数值应该为0");
    }
    
    /**
     * 测试通过LinProgUtil转换后的约束
     */
    @Test
    public void testConvertedConstraints() {
        // 创建一个简单的线性规划问题:
        // minimize: x1
        // subject to: 
        //   x1 <= 1
        //   x1 >= 0
        
        IVector c = IVector.of(1.0);  // 目标函数系数
        IMatrix A_ub = IMatrix.of(new double[][]{{1.0}});  // 不等式约束矩阵
        IVector b_ub = IVector.of(1.0);  // 不等式约束值
        
        // 转换约束
        Tuple3<IVector, IMatrix, IVector> converted = LinProgUtil.convertUbEqToEqConstraits(c, A_ub, b_ub, null, null);
        
        // 验证转换结果
        assertEquals(2, converted._1.length(), "扩展的目标函数向量应该包含2个变量");
        assertEquals(1, converted._2.rows(), "约束矩阵应该有1行");
        assertEquals(2, converted._2.cols(), "约束矩阵应该有2列");
        assertEquals(1, converted._3.length(), "约束向量应该有1个元素");
        
        // 验证目标函数系数：原始变量系数为1，松弛变量系数为0
        assertEquals(1.0, (Double)converted._1.get(0), 1e-9, "原始变量系数应为1");
        assertEquals(0.0, (Double)converted._1.get(1), 1e-9, "松弛变量系数应为0");
        
        // 验证约束矩阵：应该是 [1, 1]（原始变量系数 + 松弛变量系数）
        assertEquals(1.0, (Double)converted._2.get(0, 0), 1e-9, "原始变量约束系数应为1");
        assertEquals(1.0, (Double)converted._2.get(0, 1), 1e-9, "松弛变量约束系数应为1");
        
        // 验证约束向量：应该是 [1]
        assertEquals(1.0, (Double)converted._3.get(0), 1e-9, "约束值应为1");
        
        // 使用单纯形法求解器
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        
        // 求解转换后的问题
        Tuple2<Double, IVector> result = solver.solveWithNonNegativeEqualConstraints(converted._1, converted._2, converted._3);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertNotNull(result._2, "解向量不应为null");
        
        // 验证解向量的长度（应该是2个变量：x1和s1）
        assertEquals(2, result._2.length(), "解向量应该包含2个变量");
        
        // 验证最优解
        assertEquals(0.0, (Double)result._2.get(0), 1e-9, "x1应该为0（因为我们要最小化x1且x1>=0）");
        assertEquals(1.0, (Double)result._2.get(1), 1e-9, "s1应该为1（因为x1+s1=1且x1=0）");
        
        // 验证目标函数值
        assertEquals(0.0, result._1, 1e-9, "目标函数值应该为0");
    }
}