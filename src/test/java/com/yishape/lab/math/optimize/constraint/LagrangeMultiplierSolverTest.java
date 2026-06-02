package com.yishape.lab.math.optimize.constraint;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;

/**
 * LagrangeMultiplierSolver测试类
 */
public class LagrangeMultiplierSolverTest {

    public static void main(String[] args) {
        System.out.println("=== LagrangeMultiplierSolver Test ===\n");
        
        testQuadraticOptimizationWithLinearConstraint();
        System.out.println();
        testMultipleLinearConstraints();
    }

    /**
     * 测试带有线性等式约束的二次优化问题
     * <p>
     * 目标函数: f(x) = x1^2 + x2^2
     * 约束条件: x1 + x2 = 1
     * 解析解: x1 = x2 = 0.5, f(x) = 0.5
     * </p>
     */
    public static void testQuadraticOptimizationWithLinearConstraint() {
        System.out.println("Test 1: Quadratic optimization with linear constraint");
        
        // 定义约束: x1 + x2 = 1
        // A_eq = [1, 1], b_eq = [1]
        IMatrix<Double> A_eq = Linalg.matrix(new double[][]{{1.0, 1.0}});
        IVector<Double> b_eq = Linalg.vector(new double[]{1.0});
        
        // 创建拉格朗日乘子求解器
        LagrangeMultiplierSolver solver = new LagrangeMultiplierSolver(A_eq, b_eq);
        
        // 设置初始点
        IVector<Double> initX = Linalg.vector(new double[]{0.0, 0.0});
        
        // 定义目标函数: f(x) = x1^2 + x2^2
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                IVector<Double> xd = (IVector<Double>) x;
                double x1 = xd.get(0);
                double x2 = xd.get(1);
                return x1 * x1 + x2 * x2;
            }
        };
        
        // 定义梯度函数: grad(f) = [2*x1, 2*x2]
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                IVector<Double> xd = (IVector<Double>) x;
                double x1 = xd.get(0);
                double x2 = xd.get(1);
                return Linalg.vector(new double[]{2.0 * x1, 2.0 * x2});
            }
        };
        
        // 求解优化问题
        var result = solver.optimize(initX, objFun, grdFun);
        
        // 验证结果
        Double optimalValue = result.getOptimalValue();
        IVector<Double> optimalX = (IVector<Double>) result.getOptimalPoint();
        
        // 最优值应该接近0.5
        System.out.println("Optimal value (should be 0.5): " + optimalValue);
        
        // 最优解应该接近[0.5, 0.5]
        System.out.println("Optimal x (should be [0.5, 0.5]): [" + optimalX.get(0) + ", " + optimalX.get(1) + "]");
        
        // 验证约束满足: x1 + x2 = 1
        double constraintValue = optimalX.get(0) + optimalX.get(1);
        System.out.println("Constraint value (should be 1): " + constraintValue);
        
        // 简单验证结果是否合理
        boolean valueCheck = Math.abs(optimalValue - 0.5) < 1e-2;
        boolean xCheck = Math.abs(optimalX.get(0) - 0.5) < 1e-2 && Math.abs(optimalX.get(1) - 0.5) < 1e-2;
        boolean constraintCheck = Math.abs(constraintValue - 1.0) < 1e-6;
        
        System.out.println("Value check (|value - 0.5| < 1e-2): " + valueCheck);
        System.out.println("X check (|xi - 0.5| < 1e-2): " + xCheck);
        System.out.println("Constraint check (|constraint - 1| < 1e-6): " + constraintCheck);
        System.out.println("Test 1 result: " + (valueCheck && xCheck && constraintCheck ? "PASS" : "FAIL"));
    }
    
    /**
     * 测试带有多个线性等式约束的优化问题
     * <p>
     * 目标函数: f(x) = x1^2 + x2^2 + x3^2
     * 约束条件: 
     *   x1 + x2 + x3 = 3
     *   x1 - x2 = 0
     * 解析解: x1 = x2 = x3 = 1, f(x) = 3
     * </p>
     */
    public static void testMultipleLinearConstraints() {
        System.out.println("Test 2: Multiple linear constraints");
        
        // 定义约束:
        // x1 + x2 + x3 = 3  -> [1, 1, 1] * x = 3
        // x1 - x2 = 0       -> [1, -1, 0] * x = 0
        // A_eq = [[1, 1, 1], [1, -1, 0]], b_eq = [3, 0]
        IMatrix<Double> A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, 1.0},
            {1.0, -1.0, 0.0}
        });
        IVector<Double> b_eq = Linalg.vector(new double[]{3.0, 0.0});
        
        // 创建拉格朗日乘子求解器
        LagrangeMultiplierSolver solver = new LagrangeMultiplierSolver(A_eq, b_eq);
        
        // 设置初始点
        IVector<Double> initX = Linalg.vector(new double[]{0.0, 0.0, 0.0});
        
        // 定义目标函数: f(x) = x1^2 + x2^2 + x3^2
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                IVector<Double> xd = (IVector<Double>) x;
                double sum = 0.0;
                for (int i = 0; i < xd.length(); i++) {
                    double xi = xd.get(i);
                    sum += xi * xi;
                }
                return sum;
            }
        };
        
        // 定义梯度函数: grad(f) = [2*x1, 2*x2, 2*x3]
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                IVector<Double> xd = (IVector<Double>) x;
                double[] gradient = new double[xd.length()];
                for (int i = 0; i < xd.length(); i++) {
                    gradient[i] = 2.0 * xd.get(i);
                }
                return Linalg.vector(gradient);
            }
        };
        
        // 求解优化问题
        var result = solver.optimize(initX, objFun, grdFun);
        
        // 验证结果
        Double optimalValue = result.getOptimalValue();
        IVector<Double> optimalX = (IVector<Double>) result.getOptimalPoint();
        
        // 最优值应该接近3
        System.out.println("Optimal value (should be 3): " + optimalValue);
        
        // 最优解应该接近[1, 1, 1]
        System.out.print("Optimal x (should be [1, 1, 1]): [");
        for (int i = 0; i < optimalX.length(); i++) {
            System.out.print(optimalX.get(i));
            if (i < optimalX.length() - 1) System.out.print(", ");
        }
        System.out.println("]");
        
        // 验证约束满足
        // x1 + x2 + x3 = 3
        double constraint1 = optimalX.get(0) + optimalX.get(1) + optimalX.get(2);
        System.out.println("Constraint 1 (x1+x2+x3=3): " + constraint1);
        
        // x1 - x2 = 0
        double constraint2 = optimalX.get(0) - optimalX.get(1);
        System.out.println("Constraint 2 (x1-x2=0): " + constraint2);
        
        // 简单验证结果是否合理
        boolean valueCheck = Math.abs(optimalValue - 3.0) < 1e-2;
        boolean xCheck = true;
        for (int i = 0; i < optimalX.length(); i++) {
            if (Math.abs(optimalX.get(i) - 1.0) > 1e-2) {
                xCheck = false;
                break;
            }
        }
        boolean constraint1Check = Math.abs(constraint1 - 3.0) < 1e-6;
        boolean constraint2Check = Math.abs(constraint2 - 0.0) < 1e-6;
        
        System.out.println("Value check (|value - 3| < 1e-2): " + valueCheck);
        System.out.println("X check (|xi - 1| < 1e-2): " + xCheck);
        System.out.println("Constraint 1 check (|constraint1 - 3| < 1e-6): " + constraint1Check);
        System.out.println("Constraint 2 check (|constraint2 - 0| < 1e-6): " + constraint2Check);
        System.out.println("Test 2 result: " + (valueCheck && xCheck && constraint1Check && constraint2Check ? "PASS" : "FAIL"));
    }
}