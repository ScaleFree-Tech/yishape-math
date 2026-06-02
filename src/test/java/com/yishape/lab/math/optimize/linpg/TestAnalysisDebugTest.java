package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import org.junit.jupiter.api.Test;

/**
 * 专门用于分析测试本身是否有问题
 */
public class TestAnalysisDebugTest {
    
    private static final double TOLERANCE = 1e-6;
    
    @Test
    public void analyzeTestLargeNumbersProblem() {
        System.out.println("=== 分析 testLargeNumbersProblem ===");
        
        // 原始测试设置
        IVector c = Linalg.vector(new double[]{-1000.0, -2000.0, 0.0, 0.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {100.0, 200.0, 1.0, 0.0},  // 100*x1 + 200*x2 + s1 = 10000
            {300.0, 100.0, 0.0, 1.0}   // 300*x1 + 100*x2 + s2 = 15000
        });
        IVector b_eq = Linalg.vector(new double[]{10000.0, 15000.0});
        
        System.out.println("问题设定分析：");
        System.out.println("目标函数：min -1000*x1 - 2000*x2（即 max 1000*x1 + 2000*x2）");
        System.out.println("约束1：100*x1 + 200*x2 + s1 = 10000");
        System.out.println("约束2：300*x1 + 100*x2 + s2 = 15000");
        System.out.println("非负性：x1, x2, s1, s2 >= 0");
        
        System.out.println("\n这是一个线性方程组求解问题，有2个方程4个未知数");
        System.out.println("理论上有无穷多解，但单纯形法会找到基本解");
        
        // 手工分析解
        System.out.println("\n手工分析：");
        System.out.println("如果s1=s2=0，则需要求解：");
        System.out.println("100*x1 + 200*x2 = 10000  ... (1)");
        System.out.println("300*x1 + 100*x2 = 15000  ... (2)");
        
        // 求解线性方程组
        // 100*x1 + 200*x2 = 10000
        // 300*x1 + 100*x2 = 15000
        // 
        // 将方程(1)乘以3：300*x1 + 600*x2 = 30000
        // 减去方程(2)：500*x2 = 15000
        // 所以 x2 = 30
        // 代入方程(1)：100*x1 + 200*30 = 10000
        // 100*x1 = 10000 - 6000 = 4000
        // 所以 x1 = 40
        
        double x1_manual = 40.0;
        double x2_manual = 30.0;
        System.out.println("手工解：x1 = " + x1_manual + ", x2 = " + x2_manual);
        
        // 验证手工解
        double constraint1 = 100*x1_manual + 200*x2_manual;
        double constraint2 = 300*x1_manual + 100*x2_manual;
        System.out.println("验证约束1：100*40 + 200*30 = " + constraint1 + " (期望: 10000)");
        System.out.println("验证约束2：300*40 + 100*30 = " + constraint2 + " (期望: 15000)");
        
        // 现在用求解器求解
        System.out.println("\n使用求解器求解：");
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true);
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        if (result.isConverged()) {
            IVector solution = result.getOptimalPoint();
            double x1_solver = solution.get(0);
            double x2_solver = solution.get(1);
            double s1 = solution.get(2);
            double s2 = solution.get(3);
            
            System.out.println("求解器解：x1=" + x1_solver + ", x2=" + x2_solver + ", s1=" + s1 + ", s2=" + s2);
            
            // 验证约束
            double check1 = 100*x1_solver + 200*x2_solver + s1;
            double check2 = 300*x1_solver + 100*x2_solver + s2;
            System.out.println("约束验证1：" + check1 + " (期望: 10000.0)");
            System.out.println("约束验证2：" + check2 + " (期望: 15000.0)");
            
            // 测试的问题在于它期望不等式约束，但设置的是等式约束！
            System.out.println("\n===== 测试错误分析 =====");
            System.out.println("原测试验证：100*x1 + 200*x2 <= 10000");
            System.out.println("实际值：" + (100*x1_solver + 200*x2_solver));
            System.out.println("这是错误的！因为约束设置的是等式，不是不等式！");
        } else {
            System.out.println("求解器未收敛：" + result.getConvergenceReason());
        }
    }
    
    @Test 
    public void analyzeTestStandardMinimizationProblem() {
        System.out.println("\n=== 分析 testStandardMinimizationProblem ===");
        
        // 原始测试设置 - 大M法
        double bigM = 10000.0;
        IVector c = Linalg.vector(new double[]{3.0, 2.0, 0.0, 0.0, bigM, bigM});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, -1.0, 0.0, 1.0, 0.0},   // x1 + x2 - s1 + a1 = 4
            {2.0, 1.0, 0.0, -1.0, 0.0, 1.0}    // 2*x1 + x2 - s2 + a2 = 6
        });
        IVector b_eq = Linalg.vector(new double[]{4.0, 6.0});
        
        System.out.println("原始问题：min 3*x1 + 2*x2");
        System.out.println("约束：x1 + x2 >= 4, 2*x1 + x2 >= 6, x1,x2 >= 0");
        
        System.out.println("\n大M法转换后：");
        System.out.println("min 3*x1 + 2*x2 + " + bigM + "*a1 + " + bigM + "*a2");
        System.out.println("约束1：x1 + x2 - s1 + a1 = 4");
        System.out.println("约束2：2*x1 + x2 - s2 + a2 = 6");
        System.out.println("所有变量 >= 0");
        
        // 手工分析最优解
        System.out.println("\n手工分析最优解：");
        System.out.println("图解法：两条约束线的交点就是最优解");
        System.out.println("x1 + x2 = 4  ... (1)");
        System.out.println("2*x1 + x2 = 6  ... (2)");
        System.out.println("(2) - (1): x1 = 2");
        System.out.println("代入(1): 2 + x2 = 4, 所以 x2 = 2");
        System.out.println("最优解：x1 = 2, x2 = 2, 目标值 = 3*2 + 2*2 = 10");
        
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true);
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        System.out.println("\n求解器结果：");
        System.out.println("是否收敛：" + result.isConverged());
        System.out.println("收敛原因：" + result.getConvergenceReason());
        
        if (result.isConverged()) {
            IVector solution = result.getOptimalPoint();
            System.out.println("解向量：" + solution);
            double x1 = solution.get(0);
            double x2 = solution.get(1);
            double a1 = solution.get(4);
            double a2 = solution.get(5);
            System.out.println("x1=" + x1 + ", x2=" + x2);
            System.out.println("人工变量：a1=" + a1 + ", a2=" + a2);
            System.out.println("真实目标值：" + (3*x1 + 2*x2));
        }
    }
    
    @Test
    public void correctLargeNumbersTest() {
        System.out.println("\n=== 正确的大数值稳定性测试 ===");
        
        // 正确的不等式约束形式
        // max 1000*x1 + 2000*x2
        // s.t. 100*x1 + 200*x2 <= 10000
        //     300*x1 + 100*x2 <= 15000
        //     x1, x2 >= 0
        
        IVector c = Linalg.vector(new double[]{-1000.0, -2000.0, 0.0, 0.0}); // 最小化负目标函数
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {100.0, 200.0, 1.0, 0.0},  // 100*x1 + 200*x2 + s1 = 10000
            {300.0, 100.0, 0.0, 1.0}   // 300*x1 + 100*x2 + s2 = 15000
        });
        IVector b_eq = Linalg.vector(new double[]{10000.0, 15000.0});
        
        System.out.println("这个测试如果用不等式约束的话，应该这样验证：");
        System.out.println("约束1：100*x1 + 200*x2 <= 10000");
        System.out.println("约束2：300*x1 + 100*x2 <= 15000");
        
        System.out.println("\n但是我们的求解器处理的是等式约束形式");
        System.out.println("所以应该验证：100*x1 + 200*x2 + s1 = 10000");
        System.out.println("所以应该验证：300*x1 + 100*x2 + s2 = 15000");
        
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        if (result.isConverged()) {
            IVector solution = result.getOptimalPoint();
            double x1 = solution.get(0);
            double x2 = solution.get(1);
            double s1 = solution.get(2);
            double s2 = solution.get(3);
            
            System.out.println("解：x1=" + x1 + ", x2=" + x2 + ", s1=" + s1 + ", s2=" + s2);
            
            // 正确的等式约束验证
            double constraint1 = 100*x1 + 200*x2 + s1;
            double constraint2 = 300*x1 + 100*x2 + s2;
            System.out.println("等式约束验证1：" + constraint1 + " = 10000? " + (Math.abs(constraint1 - 10000.0) < TOLERANCE));
            System.out.println("等式约束验证2：" + constraint2 + " = 15000? " + (Math.abs(constraint2 - 15000.0) < TOLERANCE));
            
            // 如果我们想验证原始不等式约束
            double ineq1 = 100*x1 + 200*x2;
            double ineq2 = 300*x1 + 100*x2;
            System.out.println("原始不等式验证1：" + ineq1 + " <= 10000? " + (ineq1 <= 10000.0 + TOLERANCE));
            System.out.println("原始不等式验证2：" + ineq2 + " <= 15000? " + (ineq2 <= 15000.0 + TOLERANCE));
        }
    }
}