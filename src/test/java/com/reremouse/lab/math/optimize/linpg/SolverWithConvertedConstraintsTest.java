package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.optimize.OptResult;
import com.reremouse.lab.util.Tuple3;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SolverWithConvertedConstraintsTest {

    /**
     * 测试使用转换后的约束直接调用求解器
     */
    @Test
    public void testSolverWithConvertedConstraints() {
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
        
        // 使用单纯形法求解器
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        
        // 直接求解转换后的问题
        OptResult result = solver.solveWithNonNegativeEqualConstraints(converted._1, converted._2, converted._3);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertNotNull(result.getOptimalPoint(), "解向量不应为null");
        
        // 验证解向量的长度（应该是2个变量：x1和s1）
        assertEquals(2, result.getOptimalPoint().length(), "解向量应该包含2个变量");
        
        // 验证最优解
        double x1 = (Double)result.getOptimalPoint().get(0);
        double s1 = (Double)result.getOptimalPoint().get(1);
        
        // 验证约束是否满足
        double constraintValue = x1 + s1;
        assertEquals(1.0, constraintValue, 1e-9, "约束 x1 + s1 = 1 应该满足");
        
        // 验证变量非负
        assertTrue(x1 >= -1e-9, "x1应非负");
        assertTrue(s1 >= -1e-9, "s1应非负");
        
        // 验证目标函数值
        double objectiveValue = x1;
        assertEquals(x1, result.getOptimalValue(), 1e-9, "目标函数值应该等于x1");
        
        // 最重要的是验证最优性：x1应该为0，因为我们要最小化x1
        assertEquals(0.0, x1, 1e-9, "x1应该为0（因为我们要最小化x1且x1>=0）");
        assertEquals(1.0, s1, 1e-9, "s1应该为1（因为x1+s1=1且x1=0）");
        assertEquals(0.0, objectiveValue, 1e-9, "目标函数值应该为0");
    }
}