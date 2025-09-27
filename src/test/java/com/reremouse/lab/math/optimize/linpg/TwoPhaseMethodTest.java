package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.util.Tuple2;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TwoPhaseMethodTest {

    /**
     * 测试两阶段法的第一阶段
     */
    @Test
    public void testTwoPhaseMethod() {
        // 创建一个简单的线性规划问题:
        // minimize: x1
        // subject to: 
        //   x1 + s1 = 1
        //   x1, s1 >= 0
        
        // 这对应于:
        // c = [1, 0] (minimize x1, s1 not in objective)
        // A_eq = [[1, 1]] (constraint matrix)
        // b_eq = [1] (constraint values)
        
        IVector c = IVector.of(1.0, 0.0);  // 目标函数系数
        IMatrix A_eq = IMatrix.of(new double[][]{{1.0, 1.0}});  // 约束矩阵
        IVector b_eq = IVector.of(1.0);  // 约束值
        
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
        double x1 = (Double)result._2.get(0);
        double s1 = (Double)result._2.get(1);
        
        // 验证约束是否满足
        double constraintValue = x1 + s1;
        assertEquals(1.0, constraintValue, 1e-9, "约束 x1 + s1 = 1 应该满足");
        
        // 验证变量非负
        assertTrue(x1 >= -1e-9, "x1应非负");
        assertTrue(s1 >= -1e-9, "s1应非负");
        
        // 验证目标函数值
        double objectiveValue = x1;
        assertEquals(x1, result._1, 1e-9, "目标函数值应该等于x1");
        
        // 最重要的是验证最优性：x1应该为0，因为我们要最小化x1
        assertEquals(0.0, x1, 1e-9, "x1应该为0（因为我们要最小化x1且x1>=0）");
        assertEquals(1.0, s1, 1e-9, "s1应该为1（因为x1+s1=1且x1=0）");
        assertEquals(0.0, objectiveValue, 1e-9, "目标函数值应该为0");
    }
}