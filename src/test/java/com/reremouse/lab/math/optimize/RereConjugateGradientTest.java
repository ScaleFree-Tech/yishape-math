package com.reremouse.lab.math.optimize;

import com.reremouse.lab.math.optimize.newton.RereConjugateGradient;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.OptResult;

/**
 * 共轭梯度法测试类 / Conjugate Gradient Method Test Class
 */
public class RereConjugateGradientTest {
    
    /**
     * 测试二次函数优化 / Test quadratic function optimization
     * <p>
     * 目标函数: f(x) = 0.5 * x^T * A * x - b^T * x
     * 其中 A 是正定矩阵，b 是向量
     * 最优解为 x* = A^(-1) * b
     * </p>
     */
    public static void testQuadraticOptimization() {
        System.out.println("Testing quadratic optimization with conjugate gradient method...");
        
        // 创建一个简单的二次函数
        // f(x) = 0.5 * (x1^2 + 2*x2^2) - (x1 + x2)
        // 最优解: x1* = 1, x2* = 0.5
        // 最优值: f* = -1.25
        
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double x1 = (Double) x.get(0);
                double x2 = (Double) x.get(1);
                return 0.5 * (x1 * x1 + 2 * x2 * x2) - (x1 + x2);
            }
        };
        
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double x1 = (Double) x.get(0);
                double x2 = (Double) x.get(1);
                double[] grad = {x1 - 1, 2 * x2 - 1};
                return Linalg.vector(grad);
            }
        };
        
        // 初始点 / Initial point
        IVector initX = Linalg.vector(new double[]{0.0, 0.0});
        
        // 创建共轭梯度优化器 / Create conjugate gradient optimizer
        RereConjugateGradient cg = new RereConjugateGradient(1e-8, 100, 0.1);
        
        // 执行优化 / Perform optimization
        OptResult result = cg.optimize(initX, objFun, grdFun);
        
        System.out.println("Optimization result:");
        System.out.println("Optimal value: " + result.getOptimalValue());
        System.out.println("Optimal point: [" + result.getOptimalPoint().get(0) + ", " + result.getOptimalPoint().get(1) + "]");
        
        // 验证结果 / Verify results
        double expectedValue = -1.25;
        double expectedX1 = 1.0;
        double expectedX2 = 0.5;
        
        double tolerance = 1e-6;
        assert Math.abs(result.getOptimalValue() - expectedValue) < tolerance : "Optimal value mismatch";
        assert Math.abs((Double) result.getOptimalPoint().get(0) - expectedX1) < tolerance : "x1 mismatch";
        assert Math.abs((Double) result.getOptimalPoint().get(1) - expectedX2) < tolerance : "x2 mismatch";
        
        System.out.println("Test passed!");
    }
    
    /**
     * 测试Rosenbrock函数优化 / Test Rosenbrock function optimization
     * <p>
     * 目标函数: f(x) = (1-x1)^2 + 100*(x2-x1^2)^2
     * 最优解: x* = [1, 1]
     * 最优值: f* = 0
     * </p>
     */
    public static void testRosenbrockOptimization() {
        System.out.println("\nTesting Rosenbrock function optimization with conjugate gradient method...");
        
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double x1 = (Double) x.get(0);
                double x2 = (Double) x.get(1);
                double term1 = 1 - x1;
                double term2 = x2 - x1 * x1;
                return term1 * term1 + 100 * term2 * term2;
            }
        };
        
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double x1 = (Double) x.get(0);
                double x2 = (Double) x.get(1);
                double dx1 = -2 * (1 - x1) - 400 * x1 * (x2 - x1 * x1);
                double dx2 = 200 * (x2 - x1 * x1);
                return Linalg.vector(new double[]{dx1, dx2});
            }
        };
        
        // 初始点 / Initial point (challenging point for CG)
        IVector initX = Linalg.vector(new double[]{-1.2, 1.0});
        
        // 创建共轭梯度优化器 / Create conjugate gradient optimizer
        RereConjugateGradient cg = new RereConjugateGradient(1e-8, 1000, 0.1);
        
        // 执行优化 / Perform optimization
        OptResult result = cg.optimize(initX, objFun, grdFun);
        
        System.out.println("Optimization result:");
        System.out.println("Optimal value: " + result.getOptimalValue());
        System.out.println("Optimal point: [" + result.getOptimalPoint().get(0) + ", " + result.getOptimalPoint().get(1) + "]");
        
        // 验证结果 / Verify results
        double expectedValue = 0.0;
        double expectedX1 = 1.0;
        double expectedX2 = 1.0;
        
        double tolerance = 1e-3; // Rosenbrock is more challenging
        assert Math.abs(result.getOptimalValue() - expectedValue) < tolerance : "Optimal value mismatch";
        assert Math.abs((Double) result.getOptimalPoint().get(0) - expectedX1) < tolerance : "x1 mismatch";
        assert Math.abs((Double) result.getOptimalPoint().get(1) - expectedX2) < tolerance : "x2 mismatch";
        
        System.out.println("Rosenbrock test passed!");
    }
    
    public static void main(String[] args) {
        try {
            testQuadraticOptimization();
            testRosenbrockOptimization();
            System.out.println("\nAll tests passed!");
        } catch (Exception e) {
            System.err.println("Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}