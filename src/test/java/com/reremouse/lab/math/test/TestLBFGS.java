package com.reremouse.lab.math.test;

import com.reremouse.lab.math.optimize.*;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.math.IVector;

/**
 * LBFGS算法测试类
 * 
 * 测试LBFGS优化器在不同目标函数上的表现，包括：
 * 1. 简单的二次函数
 * 2. Rosenbrock函数（经典的优化测试函数）
 * 3. 多维二次函数
 */
public class TestLBFGS {
    
    public static void main(String[] args) {
        System.out.println("=== LBFGS算法测试 ===");
        
        // 测试1：简单的二次函数 f(x) = (x-2)^2
        System.out.println("\n1. 测试简单二次函数 f(x) = (x-2)^2");
        testQuadraticFunction();
        
        // 测试2：二维Rosenbrock函数 f(x,y) = (1-x)^2 + 100(y-x^2)^2
        System.out.println("\n2. 测试Rosenbrock函数 f(x,y) = (1-x)^2 + 100(y-x^2)^2");
        testRosenbrockFunction();
        
        // 测试3：多维二次函数 f(x) = ||x - a||^2
        System.out.println("\n3. 测试多维二次函数 f(x) = ||x - [1,2,3]||^2");
        testMultiDimensionalQuadratic();
        
        // 测试4：自定义参数的LBFGS
        System.out.println("\n4. 测试自定义参数的LBFGS");
        testCustomParameters();
    }
    
    /**
     * 测试简单的一维二次函数
     */
    private static void testQuadraticFunction() {
        // 定义目标函数：f(x) = (x-2)^2
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public float computeObjective(IVector x) {
                float val = x.get(0) - 2.0f;
                return val * val;
            }
        };
        
        // 定义梯度函数：f'(x) = 2(x-2)
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                float grad = 2 * (x.get(0) - 2.0f);
                return IVector.of(new float[]{grad});
            }
        };
        
        // 初始点
        IVector initX = IVector.of(new float[]{10.0f});
        
        // 创建LBFGS优化器
        RereLBFGS optimizer = new RereLBFGS();
        
        // 执行优化
        Tuple2<Float, IVector> result = optimizer.optimize(initX, objFun, grdFun);
        
        System.out.println("初始点: x = " + initX.get(0));
        System.out.println("最优值: " + result._1);
        System.out.println("最优点: x = " + result._2.get(0));
        System.out.println("理论最优解: x = 2.0, f = 0.0");
        System.out.println("误差: " + Math.abs(result._2.get(0) - 2.0f));
    }
    
    /**
     * 测试Rosenbrock函数
     */
    private static void testRosenbrockFunction() {
        // 定义Rosenbrock函数：f(x,y) = (1-x)^2 + 100(y-x^2)^2
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public float computeObjective(IVector x) {
                float x1 = x.get(0);
                float x2 = x.get(1);
                float term1 = (1 - x1) * (1 - x1);
                float term2 = 100 * (x2 - x1 * x1) * (x2 - x1 * x1);
                return term1 + term2;
            }
        };
        
        // 定义梯度函数
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                float x1 = x.get(0);
                float x2 = x.get(1);
                
                // ∂f/∂x1 = -2(1-x1) - 400x1(x2-x1^2)
                float grad1 = -2 * (1 - x1) - 400 * x1 * (x2 - x1 * x1);
                
                // ∂f/∂x2 = 200(x2-x1^2)
                float grad2 = 200 * (x2 - x1 * x1);
                
                return IVector.of(new float[]{grad1, grad2});
            }
        };
        
        // 初始点
        IVector initX = IVector.of(new float[]{-1.0f, 2.0f});
        
        // 创建LBFGS优化器
        RereLBFGS optimizer = new RereLBFGS();
        
        // 执行优化
        Tuple2<Float, IVector> result = optimizer.optimize(initX, objFun, grdFun);
        
        System.out.println("初始点: x = [" + initX.get(0) + ", " + initX.get(1) + "]");
        System.out.println("最优值: " + result._1);
        System.out.println("最优点: x = [" + result._2.get(0) + ", " + result._2.get(1) + "]");
        System.out.println("理论最优解: x = [1.0, 1.0], f = 0.0");
        System.out.println("误差: " + Math.sqrt(Math.pow(result._2.get(0) - 1.0f, 2) + Math.pow(result._2.get(1) - 1.0f, 2)));
    }
    
    /**
     * 测试多维二次函数
     */
    private static void testMultiDimensionalQuadratic() {
        // 目标向量
        IVector target = IVector.of(new float[]{1.0f, 2.0f, 3.0f});
        
        // 定义目标函数：f(x) = ||x - target||^2
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public float computeObjective(IVector x) {
                IVector diff = x.sub(target);
                return diff.innerProduct(diff);
            }
        };
        
        // 定义梯度函数：∇f(x) = 2(x - target)
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                return x.sub(target).multiplyScalar(2.0f);
            }
        };
        
        // 初始点
        IVector initX = IVector.of(new float[]{10.0f, -5.0f, 8.0f});
        
        // 创建LBFGS优化器
        RereLBFGS optimizer = new RereLBFGS();
        
        // 执行优化
        Tuple2<Float, IVector> result = optimizer.optimize(initX, objFun, grdFun);
        
        System.out.println("初始点: x = [" + initX.get(0) + ", " + initX.get(1) + ", " + initX.get(2) + "]");
        System.out.println("最优值: " + result._1);
        System.out.println("最优点: x = [" + result._2.get(0) + ", " + result._2.get(1) + ", " + result._2.get(2) + "]");
        System.out.println("理论最优解: x = [1.0, 2.0, 3.0], f = 0.0");
        
        // 计算误差
        IVector error = result._2.sub(target);
        System.out.println("误差范数: " + error.norm2());
    }
    
    /**
     * 测试自定义参数的LBFGS
     */
    private static void testCustomParameters() {
        // 定义目标函数：f(x,y) = x^2 + 2y^2 + 3xy
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public float computeObjective(IVector x) {
                float x1 = x.get(0);
                float x2 = x.get(1);
                return x1 * x1 + 2 * x2 * x2 + 3 * x1 * x2;
            }
        };
        
        // 定义梯度函数
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                float x1 = x.get(0);
                float x2 = x.get(1);
                
                // ∂f/∂x1 = 2x1 + 3x2
                float grad1 = 2 * x1 + 3 * x2;
                
                // ∂f/∂x2 = 4x2 + 3x1
                float grad2 = 4 * x2 + 3 * x1;
                
                return IVector.of(new float[]{grad1, grad2});
            }
        };
        
        // 初始点
        IVector initX = IVector.of(new float[]{5.0f, -3.0f});
        
        // 创建自定义参数的LBFGS优化器
        RereLBFGS optimizer = new RereLBFGS(5, 1e-8f, 500);
        
        System.out.println("使用自定义参数：");
        System.out.println("  历史信息对数: " + optimizer.getM());
        System.out.println("  收敛容差: " + optimizer.getTolerance());
        System.out.println("  最大迭代次数: " + optimizer.getMaxIterations());
        
        // 执行优化
        Tuple2<Float, IVector> result = optimizer.optimize(initX, objFun, grdFun);
        
        System.out.println("初始点: x = [" + initX.get(0) + ", " + initX.get(1) + "]");
        System.out.println("最优值: " + result._1);
        System.out.println("最优点: x = [" + result._2.get(0) + ", " + result._2.get(1) + "]");
        
        // 理论最优解通过求解线性方程组得到
        // 2x1 + 3x2 = 0
        // 3x1 + 4x2 = 0
        // 解为 x1 = 0, x2 = 0
        System.out.println("理论最优解: x = [0.0, 0.0], f = 0.0");
        System.out.println("误差范数: " + result._2.norm2());
    }
} 