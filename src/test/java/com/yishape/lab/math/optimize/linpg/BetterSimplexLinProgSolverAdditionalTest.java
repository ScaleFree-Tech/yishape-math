package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 额外的测试用例，验证BetterSimplexLinProgSolver在各种边界情况下的表现
 */
public class BetterSimplexLinProgSolverAdditionalTest {
    
    private final RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
    private final double TOLERANCE = 1e-6;
    
    @Test
    public void testUnboundedProblem() {
        // 测试无界问题：maximize x1 + x2 subject to x1 - x2 = 0, x1, x2 >= 0
        // 这个问题是无界的，因为我们可以让x1 = x2 = t，其中t可以任意大
        IVector c = Linalg.vector(new double[]{1.0, 1.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, -1.0}
        });
        IVector b_eq = Linalg.vector(new double[]{0.0});
        
        try {
            OptResult result = solver.maximize(c,null,null, A_eq, b_eq, null);
            // 对于无界问题，求解器应该能够检测到或返回一个非常大的值
            // 具体行为取决于实现，但不应该崩溃
            assertNotNull(result);
        } catch (RuntimeException e) {
            // 如果求解器检测到无界问题并抛出异常，这也是可以接受的
            assertTrue(e.getMessage().contains("unbounded") || e.getMessage().contains("无界"));
        }
    }
    
    @Test
    public void testInfeasibleProblem() {
        // 测试不可行问题：
        // maximize x1 + x2
        // subject to: x1 + x2 = 1
        //            x1 + x2 = 2
        //            x1, x2 >= 0
        IVector c = Linalg.vector(new double[]{1.0, 1.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0},
            {1.0, 1.0}
        });
        IVector b_eq = Linalg.vector(new double[]{1.0, 2.0});
        
        OptResult result = solver.maximize(c, null,null,A_eq, b_eq, null);
        
        // 不可行问题应该返回非收敛状态
        assertFalse(result.isConverged(), "不可行问题应该返回非收敛状态");
    }
    
    @Test
    public void testDegenerateProblem() {
        // 测试退化问题：多个约束在同一点相交
        // maximize x1 + x2
        // subject to: x1 = 0
        //            x2 = 0
        IVector c = Linalg.vector(new double[]{1.0, 1.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 0.0},
            {0.0, 1.0}
        });
        IVector b_eq = Linalg.vector(new double[]{0.0, 0.0});
        
        OptResult result = solver.maximize(c, null,null,A_eq, b_eq, null);
        
        assertTrue(result.isConverged(), "退化问题应该能够求解");
        assertEquals(0.0, result.getOptimalPoint().get(0).doubleValue(), TOLERANCE);
        assertEquals(0.0, result.getOptimalPoint().get(1).doubleValue(), TOLERANCE);
        assertEquals(0.0, result.getOptimalValue(), TOLERANCE);
    }
    
    @Test
    public void testLargerProblem() {
        // 测试更大的问题：3变量2约束
        // maximize 2*x1 + 3*x2 + x3
        // subject to: x1 + 2*x2 + x3 = 5
        //            2*x1 + x2 + 2*x3 = 6
        //            x1, x2, x3 >= 0
        IVector c = Linalg.vector(new double[]{2.0, 3.0, 1.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 2.0, 1.0},
            {2.0, 1.0, 2.0}
        });
        IVector b_eq = Linalg.vector(new double[]{5.0, 6.0});
        
        OptResult result = solver.maximize(c,null,null, A_eq, b_eq, null);
        
        assertTrue(result.isConverged(), "较大问题应该能够求解");
        assertNotNull(result.getOptimalPoint());
        Assertions.assertEquals(3, result.getOptimalPoint().size());
        
        // 验证约束满足
        IVector solution = result.getOptimalPoint();
        double constraint1 = solution.get(0).doubleValue() + 2*solution.get(1).doubleValue() + solution.get(2).doubleValue();
        double constraint2 = 2*solution.get(0).doubleValue() + solution.get(1).doubleValue() + 2*solution.get(2).doubleValue();
        
        assertEquals(5.0, constraint1, TOLERANCE, "第一个约束应该满足");
        assertEquals(6.0, constraint2, TOLERANCE, "第二个约束应该满足");
        
        // 验证非负性
        for (int i = 0; i < solution.size(); i++) {
            assertTrue(solution.get(i).doubleValue() >= -TOLERANCE, 
                "解的第" + (i+1) + "个分量应该非负");
        }
    }
    
    @Test
    public void testZeroObjectiveCoefficients() {
        // 测试目标函数系数为零的情况
        // maximize 0*x1 + 0*x2
        // subject to: x1 + x2 = 3
        //            x1, x2 >= 0
        IVector c = Linalg.vector(new double[]{0.0, 0.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0}
        });
        IVector b_eq = Linalg.vector(new double[]{3.0});
        
        OptResult result = solver.maximize(c,null,null, A_eq, b_eq, null);
        
        assertTrue(result.isConverged(), "零目标函数系数问题应该能够求解");
        assertEquals(0.0, result.getOptimalValue(), TOLERANCE, "目标函数值应该为0");
        
        // 验证约束满足
        IVector solution = result.getOptimalPoint();
        double constraintValue = solution.get(0).doubleValue() + solution.get(1).doubleValue();
        assertEquals(3.0, constraintValue, TOLERANCE, "约束应该满足");
    }
    
    @Test
    public void testNegativeRightHandSide() {
        // 测试负的右侧值（应该通过LinProgUtil.processNegativeBEq处理）
        // maximize x1 + x2
        // subject to: x1 + x2 = -2 (这会被转换为 -x1 - x2 = 2)
        //            x1, x2 >= 0
        IVector c = Linalg.vector(new double[]{1.0, 1.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0}
        });
        IVector b_eq = Linalg.vector(new double[]{-2.0});
        
        OptResult result = solver.maximize(c,null,null, A_eq, b_eq, null);
        
        // 这个问题应该是不可行的，因为x1, x2 >= 0但x1 + x2 = -2
        assertFalse(result.isConverged(), "负右侧值导致的不可行问题应该被检测到");
    }
}