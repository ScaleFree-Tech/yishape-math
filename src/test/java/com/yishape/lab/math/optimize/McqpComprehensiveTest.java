package com.yishape.lab.math.optimize;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.mcqp.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;

/**
 * MCQP模块综合测试
 * Multi-Criteria Quadratic Programming Comprehensive Test
 */
public class McqpComprehensiveTest {

    // ==================== 基础功能测试 ====================

    @Test
    public void testWeightedSumQpBasic() {
        // 简单无约束QP测试
        // minimize 1/2*x1^2 + 1/2*x2^2
        IMatrix[] Q = new IMatrix[]{
            Linalg.matrix(new double[][]{{1, 0}, {0, 1}})
        };
        IVector[] c = new IVector[]{
            Linalg.vector(new double[]{0, 0})
        };

        IMcqpSolver solver = IMcqpSolver.of(McqpSolverType.WeightedSum);
        McqpResult result = solver.solve(Q, c, null, null);

        assertNotNull(result, "结果不应为空");
        System.out.println("testWeightedSumQpBasic: " + result.getConvergenceReason());
        if (result.getNumSolutions() > 0) {
            IVector sol = result.getSolution(0);
            System.out.println("Solution: [" + sol.get(0) + ", " + sol.get(1) + "]");
        }
        // 无约束QP理论上应该收敛到0
    }

    @Test
    public void testWeightedSumQpWithWeights() {
        // 带权重的单目标最小化
        // minimize (x1-1)^2 + (x2-2)^2
        IMatrix Q = Linalg.matrix(new double[][]{{2, 0}, {0, 2}});
        IVector c = Linalg.vector(new double[]{-2, -4});

        double[] weights = new double[]{1.0};
        IQpSolver qpSolver = new InteriorPointQpSolver().withVerbose(true);
        OptResult result = qpSolver.solve(Q, c, null, null, null, null, null);

        System.out.println("testWeightedSumQpWithWeights: " + result.getConvergenceReason());
        assertTrue(result.isConverged(), "QP应该收敛: " + result.getConvergenceReason());
        System.out.println("Optimal value: " + result.getOptimalValue());
    }

    @Test
    public void testLexicographicQp() {
        // 测试字典序法
        IMatrix[] Q = new IMatrix[]{
            Linalg.matrix(new double[][]{{1, 0}, {0, 1}}),
            Linalg.matrix(new double[][]{{2, 0}, {0, 2}})
        };
        IVector[] c = new IVector[]{
            Linalg.vector(new double[]{1, 1}),
            Linalg.vector(new double[]{2, 1})
        };

        int[] priority = new int[]{0, 1}; // 先优化第一个目标
        RereLexicographicQp solver = new RereLexicographicQp(priority);
        McqpResult result = solver.solve(Q, c, null, null);

        assertNotNull(result);
        System.out.println("testLexicographicQp: " + result.getConvergenceReason());

        System.out.println("testLexicographicQp: PASSED");
    }

    @Test
    public void testGoalProgrammingQp() {
        // 测试目标规划法
        IMatrix[] Q = new IMatrix[]{
            Linalg.matrix(new double[][]{{1, 0}, {0, 1}}),
            Linalg.matrix(new double[][]{{1, 0}, {0, 1}})
        };
        IVector[] c = new IVector[]{
            Linalg.vector(new double[]{1, 2}),
            Linalg.vector(new double[]{2, 1})
        };

        double[] goals = new double[]{5.0, 5.0};
        double[] weights = new double[]{0.5, 0.5};

        RereGoalProgrammingQp solver = new RereGoalProgrammingQp(goals, weights);
        McqpResult result = solver.solve(Q, c, null, null);

        assertNotNull(result);
        assertNotNull(result.getGoals());
        assertArrayEquals(goals, result.getGoals(), 1e-6);

        System.out.println("testGoalProgrammingQp: PASSED");
    }

    @Test
    public void testParetoQp() {
        // 测试Pareto最优解法
        IMatrix[] Q = new IMatrix[]{
            Linalg.matrix(new double[][]{{1, 0}, {0, 2}}),
            Linalg.matrix(new double[][]{{2, 0}, {0, 1}})
        };
        IVector[] c = new IVector[]{
            Linalg.vector(new double[]{0, 0}),
            Linalg.vector(new double[]{0, 0})
        };

        RereParetoQp solver = new RereParetoQp(20);
        McqpResult result = solver.solve(Q, c, null, null);

        assertNotNull(result);
        System.out.println("testParetoQp: 生成了 " + result.getNumSolutions() + " 个解");

        System.out.println("testParetoQp: PASSED");
    }

    @Test
    public void testAhpQp() {
        // 测试AHP法
        IMatrix[] Q = new IMatrix[]{
            Linalg.matrix(new double[][]{{1, 0}, {0, 1}}),
            Linalg.matrix(new double[][]{{2, 0}, {0, 2}})
        };
        IVector[] c = new IVector[]{
            Linalg.vector(new double[]{1, 2}),
            Linalg.vector(new double[]{2, 1})
        };

        // 成对比较矩阵：目标1比目标2稍微重要
        IMatrix comparisonMatrix = Linalg.matrix(new double[][]{
            {1, 3},
            {1.0/3, 1}
        });

        RereAhpQp solver = new RereAhpQp(comparisonMatrix);
        McqpResult result = solver.solve(Q, c, null, null);

        assertNotNull(result);
        System.out.println("testAhpQp: " + result.getConvergenceReason());
        if (result.getWeights() != null) {
            System.out.println("计算得到的权重: " + Arrays.toString(result.getWeights()));
        }

        System.out.println("testAhpQp: PASSED");
    }

    @Test
    public void testTopsisQp() {
        // 测试TOPSIS法
        IMatrix[] Q = new IMatrix[]{
            Linalg.matrix(new double[][]{{1, 0}, {0, 1}}),
            Linalg.matrix(new double[][]{{1, 0}, {0, 1}})
        };
        IVector[] c = new IVector[]{
            Linalg.vector(new double[]{1, 2}),
            Linalg.vector(new double[]{2, 1})
        };

        double[] weights = new double[]{0.5, 0.5};
        RereTopsisQp solver = new RereTopsisQp(weights);
        McqpResult result = solver.solve(Q, c, null, null);

        assertNotNull(result);
        System.out.println("testTopsisQp: " + result.getConvergenceReason());

        System.out.println("testTopsisQp: PASSED");
    }

    // ==================== 工具类测试 ====================

    @Test
    public void testMcqpUtil() {
        // 测试McqpUtil工具类
        IMatrix[] Q = new IMatrix[]{
            Linalg.matrix(new double[][]{{1, 0}, {0, 1}}),
            Linalg.matrix(new double[][]{{1, 0}, {0, 1}})
        };
        IVector[] c = new IVector[]{
            Linalg.vector(new double[]{1, 1}),
            Linalg.vector(new double[]{1, 1})
        };

        IQpSolver baseSolver = IQpSolver.of();

        // 测试理想点计算
        double[] ideal = McqpUtil.computeIdealPoint(Q, c, null, null, null, null, baseSolver);
        assertNotNull(ideal);
        assertEquals(2, ideal.length);
        System.out.println("Ideal point: [" + ideal[0] + ", " + ideal[1] + "]");

        // 测试二次目标函数计算
        IVector x = Linalg.vector(new double[]{1, 1});
        double obj = McqpUtil.computeQuadraticObjective(Q[0], c[0], x);
        System.out.println("Objective at (1,1): " + obj);

        System.out.println("testMcqpUtil: PASSED");
    }

    @Test
    public void testMcqpResultParetoFront() {
        // 测试Pareto前沿提取
        IMatrix[] Q = new IMatrix[]{
            Linalg.matrix(new double[][]{{1, 0}, {0, 1}})
        };
        IVector[] c = new IVector[]{
            Linalg.vector(new double[]{1, 1})
        };

        RereWeightedSumQp solver = new RereWeightedSumQp(new double[]{1});
        McqpResult result = solver.solve(Q, c, null, null);

        assertNotNull(result);
        var paretoFront = result.getParetoFront();
        assertNotNull(paretoFront);

        System.out.println("testMcqpResultParetoFront: PASSED");
    }

    // ==================== 性能测试 ====================

    @Test
    public void testPerformanceSmall() {
        // 小规模问题性能测试
        IMatrix Q = Linalg.matrix(new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
        IVector c = Linalg.vector(new double[]{1, 2, 3});

        long startTime = System.currentTimeMillis();

        InteriorPointQpSolver solver = new InteriorPointQpSolver();
        for (int i = 0; i < 100; i++) {
            solver.solve(Q, c, null, null, null, null, null);
        }

        long elapsed = System.currentTimeMillis() - startTime;
        double avgTime = elapsed / 100.0;

        System.out.println("testPerformanceSmall: 100次迭代平均耗时 " + avgTime + " ms");
        assertTrue(avgTime < 100, "平均耗时应该小于100ms");
    }

    // ==================== 边界情况测试 ====================

    @Test
    public void testSingleObjective() {
        // 单目标QP测试
        IMatrix[] Q = new IMatrix[]{
            Linalg.matrix(new double[][]{{1, 0}, {0, 2}})
        };
        IVector[] c = new IVector[]{
            Linalg.vector(new double[]{-2, -3})
        };

        RereWeightedSumQp solver = new RereWeightedSumQp();
        McqpResult result = solver.solve(Q, c, null, null);

        assertNotNull(result);
        System.out.println("testSingleObjective: " + result.getConvergenceReason());

        System.out.println("testSingleObjective: PASSED");
    }

    @Test
    public void testMcqpWrapper() {
        // 测试McqpWrapper门面类
        double[] weights = new double[]{0.5, 0.5};

        IMcqpSolver solver1 = McqpWrapper.weightedSumQp(weights);
        assertNotNull(solver1);

        IMcqpSolver solver2 = McqpWrapper.lexicographicQp(new int[]{0, 1});
        assertNotNull(solver2);

        IMcqpSolver solver3 = McqpWrapper.topsisQp(weights);
        assertNotNull(solver3);

        System.out.println("testMcqpWrapper: PASSED");
    }

    // ==================== 约束测试 ====================

    @Test
    public void testWithInequalityConstraints() {
        // 测试不等式约束 x1 + x2 <= 5
        IMatrix[] Q = new IMatrix[]{
            Linalg.matrix(new double[][]{{1, 0}, {0, 1}})
        };
        IVector[] c = new IVector[]{
            Linalg.vector(new double[]{1, 1})
        };

        // x1 + x2 <= 5
        IMatrix A_ub = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_ub = Linalg.vector(new double[]{5});

        RereWeightedSumQp solver = new RereWeightedSumQp();
        McqpResult result = solver.solve(Q, c, A_ub, b_ub);

        assertNotNull(result);
        System.out.println("testWithInequalityConstraints: " + result.getConvergenceReason());
        if (result.getNumSolutions() > 0) {
            IVector sol = result.getSolution(0);
            double x1 = ((Number) sol.get(0)).doubleValue();
            double x2 = ((Number) sol.get(1)).doubleValue();
            System.out.println("Solution: [" + x1 + ", " + x2 + "]");
            System.out.println("Constraint check: x1 + x2 = " + (x1 + x2) + " <= 5? " + (x1 + x2 <= 5 + 1e-6));
        }

        System.out.println("testWithInequalityConstraints: PASSED");
    }

    public static void main(String[] args) {
        McqpComprehensiveTest test = new McqpComprehensiveTest();

        System.out.println("========== MCQP 综合测试 ==========\n");

        try {
            System.out.println("\n--- 基础功能测试 ---");
            test.testWeightedSumQpBasic();
            test.testWeightedSumQpWithWeights();
            test.testLexicographicQp();
            test.testGoalProgrammingQp();
            test.testParetoQp();
            test.testAhpQp();
            test.testTopsisQp();

            System.out.println("\n--- 工具类测试 ---");
            test.testMcqpUtil();
            test.testMcqpResultParetoFront();

            System.out.println("\n--- 约束测试 ---");
            test.testWithInequalityConstraints();

            System.out.println("\n--- 性能测试 ---");
            test.testPerformanceSmall();

            System.out.println("\n--- 边界情况测试 ---");
            test.testSingleObjective();
            test.testMcqpWrapper();

            System.out.println("\n========== 所有测试通过 ==========");
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
