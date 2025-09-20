package com.reremouse.lab.math.test;

import com.reremouse.lab.math.optimize.*;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;

/**
 * RereLBFGS算法正确性与性能测试类
 * Correctness and Performance Test Class for RereLBFGS Algorithm
 * 
 * 该测试类验证RereLBFGS优化器的正确性和性能表现，包括：
 * This test class verifies the correctness and performance of the RereLBFGS optimizer, including:
 * 1. 基本功能测试 / Basic functionality tests
 * 2. 收敛性测试 / Convergence tests
 * 3. 性能基准测试 / Performance benchmark tests
 * 4. 边界条件测试 / Boundary condition tests
 */
public class RereLBFGSTest {
    
    public static void main(String[] args) {
        System.out.println("=== RereLBFGS算法正确性与性能测试 ===");
        System.out.println("=== RereLBFGS Algorithm Correctness and Performance Tests ===\n");
        
        RereLBFGSTest test = new RereLBFGSTest();
        
        // 运行所有测试 / Run all tests
        test.runAllTests();
    }
    
    /**
     * 运行所有测试 / Run all tests
     */
    public void runAllTests() {
        System.out.println("1. 测试简单二次函数 / Testing simple quadratic function");
        testQuadraticFunction();
        
        System.out.println("\n2. 测试Rosenbrock函数 / Testing Rosenbrock function");
        testRosenbrockFunction();
        
        System.out.println("\n3. 测试多维二次函数 / Testing multi-dimensional quadratic function");
        testMultiDimensionalQuadratic();
        
        System.out.println("\n4. 测试自定义参数 / Testing custom parameters");
        testCustomParameters();
        
        System.out.println("\n5. 测试收敛性 / Testing convergence");
        testConvergence();
        
        System.out.println("\n6. 测试边界条件 / Testing boundary conditions");
        testBoundaryConditions();
        
        System.out.println("\n7. 性能基准测试 / Performance benchmark test");
        performanceBenchmark();
    }
    
    /**
     * 测试简单的一维二次函数
     * Test simple one-dimensional quadratic function
     */
    private void testQuadraticFunction() {
        try {
            // 定义目标函数：f(x) = (x-2)^2
            IObjectiveFunction objFun = new IObjectiveFunction() {
                @Override
                public double computeObjective(IVector x) {
                    double val = (Double) x.get(0) - 2.0;
                    return val * val;
                }
            };
            
            // 定义梯度函数：f'(x) = 2(x-2)
            IGradientFunction grdFun = new IGradientFunction() {
                @Override
                public IVector computeGradient(IVector x) {
                    double grad = 2 * ((Double) x.get(0) - 2.0);
                    return Linalg.vector(new double[]{grad});
                }
            };
            
            // 初始点
            IVector initX = Linalg.vector(new double[]{10.0});
            
            // 创建LBFGS优化器
            RereLBFGS optimizer = new RereLBFGS();
            
            // 执行优化
            long startTime = System.nanoTime();
            Tuple2<Double, IVector> result = optimizer.optimize(initX, objFun, grdFun);
            long endTime = System.nanoTime();
            
            double executionTime = (endTime - startTime) / 1_000_000.0; // 转换为毫秒 / Convert to milliseconds
            
            System.out.println("  初始点: x = " + initX.get(0));
            System.out.println("  最优值: " + result._1);
            System.out.println("  最优点: x = " + result._2.get(0));
            System.out.println("  理论最优解: x = 2.0, f = 0.0");
            System.out.println("  误差: " + Math.abs((Double) result._2.get(0) - 2.0));
            System.out.println("  执行时间: " + executionTime + " ms");
            
            // 验证结果
            assert Math.abs((Double) result._2.get(0) - 2.0) < 1e-5 : "Quadratic function test failed";
            assert Math.abs(result._1) < 1e-10 : "Quadratic function test failed";
            System.out.println("  ✓ 测试通过 / Test passed");
        } catch (Exception e) {
            System.err.println("  ✗ 测试失败 / Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试Rosenbrock函数
     * Test Rosenbrock function
     */
    private void testRosenbrockFunction() {
        try {
            // 定义Rosenbrock函数：f(x,y) = (1-x)^2 + 100(y-x^2)^2
            IObjectiveFunction objFun = new IObjectiveFunction() {
                @Override
                public double computeObjective(IVector x) {
                    double x1 = (Double) x.get(0);
                    double x2 = (Double) x.get(1);
                    double term1 = (1 - x1) * (1 - x1);
                    double term2 = 100 * (x2 - x1 * x1) * (x2 - x1 * x1);
                    return term1 + term2;
                }
            };
            
            // 定义梯度函数
            IGradientFunction grdFun = new IGradientFunction() {
                @Override
                public IVector computeGradient(IVector x) {
                    double x1 = (Double) x.get(0);
                    double x2 = (Double) x.get(1);
                    
                    // ∂f/∂x1 = -2(1-x1) - 400x1(x2-x1^2)
                    double grad1 = -2 * (1 - x1) - 400 * x1 * (x2 - x1 * x1);
                    
                    // ∂f/∂x2 = 200(x2-x1^2)
                    double grad2 = 200 * (x2 - x1 * x1);
                    
                    return Linalg.vector(new double[]{grad1, grad2});
                }
            };
            
            // 初始点
            IVector initX = Linalg.vector(new double[]{-1.0, 2.0});
            
            // 创建LBFGS优化器
            RereLBFGS optimizer = new RereLBFGS();
            
            // 执行优化
            long startTime = System.nanoTime();
            Tuple2<Double, IVector> result = optimizer.optimize(initX, objFun, grdFun);
            long endTime = System.nanoTime();
            
            double executionTime = (endTime - startTime) / 1_000_000.0; // 转换为毫秒 / Convert to milliseconds
            
            System.out.println("  初始点: x = [" + initX.get(0) + ", " + initX.get(1) + "]");
            System.out.println("  最优值: " + result._1);
            System.out.println("  最优点: x = [" + result._2.get(0) + ", " + result._2.get(1) + "]");
            System.out.println("  理论最优解: x = [1.0, 1.0], f = 0.0");
            System.out.println("  误差: " + Math.sqrt(Math.pow((Double) result._2.get(0) - 1.0, 2) + Math.pow((Double) result._2.get(1) - 1.0, 2)));
            System.out.println("  执行时间: " + executionTime + " ms");
            
            // 验证结果
            double error = Math.sqrt(Math.pow((Double) result._2.get(0) - 1.0, 2) + Math.pow((Double) result._2.get(1) - 1.0, 2));
            assert error < 1e-4 : "Rosenbrock function test failed";
            assert result._1 < 1e-6 : "Rosenbrock function test failed";
            System.out.println("  ✓ 测试通过 / Test passed");
        } catch (Exception e) {
            System.err.println("  ✗ 测试失败 / Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试多维二次函数
     * Test multi-dimensional quadratic function
     */
    private void testMultiDimensionalQuadratic() {
        try {
            // 目标向量
            IVector target = Linalg.vector(new double[]{1.0, 2.0, 3.0});
            
            // 定义目标函数：f(x) = ||x - target||^2
            IObjectiveFunction objFun = new IObjectiveFunction() {
                @Override
                public double computeObjective(IVector x) {
                    IVector diff = x.sub(target);
                    return (Double) diff.innerProduct(diff);
                }
            };
            
            // 定义梯度函数：∇f(x) = 2(x - target)
            IGradientFunction grdFun = new IGradientFunction() {
                @Override
                public IVector computeGradient(IVector x) {
                    return x.sub(target).multiplyScalar(2.0);
                }
            };
            
            // 初始点
            IVector initX = Linalg.vector(new double[]{10.0, -5.0, 8.0});
            
            // 创建LBFGS优化器
            RereLBFGS optimizer = new RereLBFGS();
            
            // 执行优化
            long startTime = System.nanoTime();
            Tuple2<Double, IVector> result = optimizer.optimize(initX, objFun, grdFun);
            long endTime = System.nanoTime();
            
            double executionTime = (endTime - startTime) / 1_000_000.0; // 转换为毫秒 / Convert to milliseconds
            
            System.out.println("  初始点: x = [" + initX.get(0) + ", " + initX.get(1) + ", " + initX.get(2) + "]");
            System.out.println("  最优值: " + result._1);
            System.out.println("  最优点: x = [" + result._2.get(0) + ", " + result._2.get(1) + ", " + result._2.get(2) + "]");
            System.out.println("  理论最优解: x = [1.0, 2.0, 3.0], f = 0.0");
            
            // 计算误差
            IVector error = result._2.sub(target);
            double errorNorm = (Double) error.norm2();
            System.out.println("  误差范数: " + errorNorm);
            System.out.println("  执行时间: " + executionTime + " ms");
            
            // 验证结果
            assert errorNorm < 1e-5 : "Multi-dimensional quadratic function test failed";
            assert Math.abs(result._1) < 1e-10 : "Multi-dimensional quadratic function test failed";
            System.out.println("  ✓ 测试通过 / Test passed");
        } catch (Exception e) {
            System.err.println("  ✗ 测试失败 / Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试自定义参数
     * Test custom parameters
     */
    private void testCustomParameters() {
        try {
            // 定义目标函数：f(x,y) = (x-1)^2 + 2(y-2)^2 (a convex quadratic function with minimum at (1,2))
            IObjectiveFunction objFun = new IObjectiveFunction() {
                @Override
                public double computeObjective(IVector x) {
                    double x1 = (Double) x.get(0);
                    double x2 = (Double) x.get(1);
                    double term1 = (x1 - 1.0) * (x1 - 1.0);
                    double term2 = 2.0 * (x2 - 2.0) * (x2 - 2.0);
                    return term1 + term2;
                }
            };
            
            // 定义梯度函数
            IGradientFunction grdFun = new IGradientFunction() {
                @Override
                public IVector computeGradient(IVector x) {
                    double x1 = (Double) x.get(0);
                    double x2 = (Double) x.get(1);
                    
                    // ∂f/∂x1 = 2(x1-1)
                    double grad1 = 2 * (x1 - 1.0);
                    
                    // ∂f/∂x2 = 4(x2-2)
                    double grad2 = 4 * (x2 - 2.0);
                    
                    return Linalg.vector(new double[]{grad1, grad2});
                }
            };
            
            // 初始点
            IVector initX = Linalg.vector(new double[]{5.0, -3.0});
            
            // 创建自定义参数的LBFGS优化器
            RereLBFGS optimizer = new RereLBFGS(5, 1e-8, 500);
            
            System.out.println("  使用自定义参数：");
            System.out.println("    历史信息对数: " + optimizer.getM());
            System.out.println("    收敛容差: " + optimizer.getTolerance());
            System.out.println("    最大迭代次数: " + optimizer.getMaxIterations());
            
            // 执行优化
            long startTime = System.nanoTime();
            Tuple2<Double, IVector> result = optimizer.optimize(initX, objFun, grdFun);
            long endTime = System.nanoTime();
            
            double executionTime = (endTime - startTime) / 1_000_000.0; // 转换为毫秒 / Convert to milliseconds
            
            System.out.println("  初始点: x = [" + initX.get(0) + ", " + initX.get(1) + "]");
            System.out.println("  最优值: " + result._1);
            System.out.println("  最优点: x = [" + result._2.get(0) + ", " + result._2.get(1) + "]");
            
            // 理论最优解: minimum at (1, 2)
            System.out.println("  理论最优解: x = [1.0, 2.0], f = 0.0");
            System.out.println("  误差范数: " + Math.sqrt(Math.pow((Double) result._2.get(0) - 1.0, 2) + Math.pow((Double) result._2.get(1) - 2.0, 2)));
            System.out.println("  执行时间: " + executionTime + " ms");
            
            // 验证结果
            double error = Math.sqrt(Math.pow((Double) result._2.get(0) - 1.0, 2) + Math.pow((Double) result._2.get(1) - 2.0, 2));
            assert error < 1e-6 : "Custom parameters test failed";
            assert Math.abs(result._1) < 1e-10 : "Custom parameters test failed";
            System.out.println("  ✓ 测试通过 / Test passed");
        } catch (Exception e) {
            System.err.println("  ✗ 测试失败 / Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试收敛性
     * Test convergence
     */
    private void testConvergence() {
        try {
            // 定义目标函数：f(x) = x^4 - 2x^2 + 1
            IObjectiveFunction objFun = new IObjectiveFunction() {
                @Override
                public double computeObjective(IVector x) {
                    double val = (Double) x.get(0);
                    double val2 = val * val;
                    return val2 * val2 - 2 * val2 + 1;
                }
            };
            
            // 定义梯度函数：f'(x) = 4x^3 - 4x
            IGradientFunction grdFun = new IGradientFunction() {
                @Override
                public IVector computeGradient(IVector x) {
                    double val = (Double) x.get(0);
                    double grad = 4 * val * val * val - 4 * val;
                    return Linalg.vector(new double[]{grad});
                }
            };
            
            // 初始点
            IVector initX = Linalg.vector(new double[]{2.0});
            
            // 创建LBFGS优化器
            RereLBFGS optimizer = new RereLBFGS();
            
            // 执行优化
            Tuple2<Double, IVector> result = optimizer.optimize(initX, objFun, grdFun);
            
            System.out.println("  初始点: x = " + initX.get(0));
            System.out.println("  最优值: " + result._1);
            System.out.println("  最优点: x = " + result._2.get(0));
            System.out.println("  理论最优解: x = ±1.0, f = 0.0");
            
            // 验证结果
            double xOpt = (Double) result._2.get(0);
            boolean convergedToRoot = Math.abs(xOpt - 1.0) < 1e-3 || Math.abs(xOpt + 1.0) < 1e-3;
            assert convergedToRoot : "Convergence test failed";
            assert result._1 < 1e-6 : "Convergence test failed";
            System.out.println("  ✓ 测试通过 / Test passed");
        } catch (Exception e) {
            System.err.println("  ✗ 测试失败 / Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试边界条件
     * Test boundary conditions
     */
    private void testBoundaryConditions() {
        try {
            // 定义目标函数：f(x) = x^2
            IObjectiveFunction objFun = new IObjectiveFunction() {
                @Override
                public double computeObjective(IVector x) {
                    double val = (Double) x.get(0);
                    return val * val;
                }
            };
            
            // 定义梯度函数：f'(x) = 2x
            IGradientFunction grdFun = new IGradientFunction() {
                @Override
                public IVector computeGradient(IVector x) {
                    double grad = 2 * (Double) x.get(0);
                    return Linalg.vector(new double[]{grad});
                }
            };
            
            // 初始点在最优解上
            IVector initX = Linalg.vector(new double[]{0.0});
            
            // 创建LBFGS优化器
            RereLBFGS optimizer = new RereLBFGS();
            
            // 执行优化
            Tuple2<Double, IVector> result = optimizer.optimize(initX, objFun, grdFun);
            
            System.out.println("  初始点: x = " + initX.get(0));
            System.out.println("  最优值: " + result._1);
            System.out.println("  最优点: x = " + result._2.get(0));
            System.out.println("  理论最优解: x = 0.0, f = 0.0");
            
            // 验证结果
            assert Math.abs((Double) result._2.get(0)) < 1e-10 : "Boundary conditions test failed";
            assert Math.abs(result._1) < 1e-15 : "Boundary conditions test failed";
            System.out.println("  ✓ 测试通过 / Test passed");
        } catch (Exception e) {
            System.err.println("  ✗ 测试失败 / Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 性能基准测试
     * Performance benchmark test
     */
    private void performanceBenchmark() {
        try {
            // 定义高维目标函数：f(x) = sum(xi^2)
            int dimension = 100;
            IVector target = Linalg.vector(new double[dimension]);
            
            IObjectiveFunction objFun = new IObjectiveFunction() {
                @Override
                public double computeObjective(IVector x) {
                    IVector diff = x.sub(target);
                    return (Double) diff.innerProduct(diff);
                }
            };
            
            IGradientFunction grdFun = new IGradientFunction() {
                @Override
                public IVector computeGradient(IVector x) {
                    return x.sub(target).multiplyScalar(2.0);
                }
            };
            
            // 初始点
            double[] initData = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                initData[i] = Math.random() * 10 - 5; // -5 to 5
            }
            IVector initX = Linalg.vector(initData);
            
            // 创建LBFGS优化器
            RereLBFGS optimizer = new RereLBFGS(10, 1e-8, 1000);
            
            // 执行优化并测量时间
            long startTime = System.nanoTime();
            Tuple2<Double, IVector> result = optimizer.optimize(initX, objFun, grdFun);
            long endTime = System.nanoTime();
            
            double executionTime = (endTime - startTime) / 1_000_000.0; // 转换为毫秒 / Convert to milliseconds
            
            System.out.println("  问题维度: " + dimension);
            System.out.println("  最优值: " + result._1);
            System.out.println("  执行时间: " + executionTime + " ms");
            System.out.println("  迭代次数: " + optimizer.getMaxIterations() + " (max)");
            
            // 验证结果
            assert result._1 < 1e-6 : "Performance benchmark test failed";
            System.out.println("  ✓ 测试通过 / Test passed");
        } catch (Exception e) {
            System.err.println("  ✗ 测试失败 / Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}