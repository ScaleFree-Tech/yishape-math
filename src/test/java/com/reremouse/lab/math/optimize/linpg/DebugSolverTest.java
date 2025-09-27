package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class DebugSolverTest {

    /**
     * 测试直接调用solveWithNonNegativeEqualConstraints方法并打印调试信息
     */
    @Test
    public void testDirectSolveWithDebug() {
        // 创建一个简单的线性规划问题:
        // minimize: x1
        // subject to: 
        //   x1 + s1 = 1 (where s1 is a slack variable)
        //   x1 >= 0, s1 >= 0
        
        IVector c = IVector.of(1.0, 0.0);  // 目标函数系数 (minimize x1, s1 not in objective)
        IMatrix A_eq = IMatrix.of(new double[][]{{1.0, 1.0}});  // 等式约束矩阵
        IVector b_eq = IVector.of(1.0);  // 等式约束值
        
        System.out.println("目标函数系数: " + c);
        System.out.println("约束矩阵: " + A_eq);
        System.out.println("约束向量: " + b_eq);
        
        // 使用单纯形法求解器
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        
        // 直接求解
        Tuple2<Double, IVector> result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        System.out.println("求解结果: " + result);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertNotNull(result._2, "解向量不应为null");
        
        // 验证解向量的长度（应该是2个变量：x1和s1）
        assertEquals(2, result._2.length(), "解向量应该包含2个变量");
        
        // 验证最优解
        double x1 = (Double)result._2.get(0);
        double s1 = (Double)result._2.get(1);
        System.out.println("x1 = " + x1 + ", s1 = " + s1);
        
        // 验证约束是否满足
        double constraintValue = x1 + s1;
        System.out.println("约束值 (x1 + s1) = " + constraintValue);
        assertEquals(1.0, constraintValue, 1e-9, "约束 x1 + s1 = 1 应该满足");
        
        // 验证目标函数值
        double objectiveValue = x1;
        System.out.println("目标函数值 = " + objectiveValue);
        assertEquals(x1, result._1, 1e-9, "目标函数值应该等于x1");
    }
}