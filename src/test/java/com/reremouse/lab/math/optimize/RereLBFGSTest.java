package com.reremouse.lab.math.optimize;

import com.reremouse.lab.math.optimize.newton.RereLBFGS;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

/**
 * RereLBFGS算法JUnit测试类
 * JUnit Test Class for RereLBFGS Algorithm
 */
@DisplayName("RereLBFGS算法测试 / RereLBFGS Algorithm Tests")
public class RereLBFGSTest {
    
    @Test
    @DisplayName("测试简单二次函数优化 / Test simple quadratic function optimization")
    public void testQuadraticFunction() {
        // 定义目标函数：f(x) = (x-2)^2
        IObjectiveFunction objFun = x -> {
            double val = (Double) x.get(0) - 2.0;
            return val * val;
        };
        
        // 定义梯度函数：f'(x) = 2(x-2)
        IGradientFunction grdFun = x -> {
            double grad = 2 * ((Double) x.get(0) - 2.0);
            return Linalg.vector(new double[]{grad});
        };
        
        // 初始点
        IVector initX = Linalg.vector(new double[]{10.0});
        
        // 创建LBFGS优化器
        RereLBFGS optimizer = new RereLBFGS();
        
        // 执行优化
        OptResult result = optimizer.optimize(initX, objFun, grdFun);
        
        // 验证结果
        assertEquals(2.0, (Double) result.getOptimalPoint().get(0), 1e-5, "最优点应该接近2.0");
        assertEquals(0.0, result.getOptimalValue(), 1e-10, "最优值应该接近0.0");
    }
    
    @Test
    @DisplayName("测试Rosenbrock函数优化 / Test Rosenbrock function optimization")
    public void testRosenbrockFunction() {
        // 定义Rosenbrock函数：f(x,y) = (1-x)^2 + 100(y-x^2)^2
        IObjectiveFunction objFun = x -> {
            double x1 = (Double) x.get(0);
            double x2 = (Double) x.get(1);
            double term1 = (1 - x1) * (1 - x1);
            double term2 = 100 * (x2 - x1 * x1) * (x2 - x1 * x1);
            return term1 + term2;
        };
        
        // 定义梯度函数
        IGradientFunction grdFun = x -> {
            double x1 = (Double) x.get(0);
            double x2 = (Double) x.get(1);
            
            // ∂f/∂x1 = -2(1-x1) - 400x1(x2-x1^2)
            double grad1 = -2 * (1 - x1) - 400 * x1 * (x2 - x1 * x1);
            
            // ∂f/∂x2 = 200(x2-x1^2)
            double grad2 = 200 * (x2 - x1 * x1);
            
            return Linalg.vector(new double[]{grad1, grad2});
        };
        
        // 初始点
        IVector initX = Linalg.vector(new double[]{-1.0, 2.0});
        
        // 创建LBFGS优化器
        RereLBFGS optimizer = new RereLBFGS();
        
        // 执行优化
        OptResult result = optimizer.optimize(initX, objFun, grdFun);
        
        // 验证结果
        assertEquals(1.0, (Double) result.getOptimalPoint().get(0), 1e-3, "第一个变量最优点应该接近1.0");
        assertEquals(1.0, (Double) result.getOptimalPoint().get(1), 1e-3, "第二个变量最优点应该接近1.0");
        assertEquals(0.0, result.getOptimalValue(), 1e-6, "最优值应该接近0.0");
    }
    
    @Test
    @DisplayName("测试多维二次函数优化 / Test multi-dimensional quadratic function optimization")
    public void testMultiDimensionalQuadratic() {
        // 目标向量
        IVector target = Linalg.vector(new double[]{1.0, 2.0, 3.0});
        
        // 定义目标函数：f(x) = ||x - target||^2
        IObjectiveFunction objFun = x -> {
            IVector diff = x.sub(target);
            return (Double) diff.innerProduct(diff);
        };
        
        // 定义梯度函数：∇f(x) = 2(x - target)
        IGradientFunction grdFun = x -> x.sub(target).multiplyScalar(2.0);
        
        // 初始点
        IVector initX = Linalg.vector(new double[]{10.0, -5.0, 8.0});
        
        // 创建LBFGS优化器
        RereLBFGS optimizer = new RereLBFGS();
        
        // 执行优化
        OptResult result = optimizer.optimize(initX, objFun, grdFun);
        
        // 验证结果
        IVector error = result.getOptimalPoint().sub(target);
        double errorNorm = (Double) error.norm2();
        assertTrue(errorNorm < 1e-5, "误差范数应该小于1e-5");
        assertEquals(0.0, result.getOptimalValue(), 1e-10, "最优值应该接近0.0");
    }
    
    @Test
    @DisplayName("测试自定义参数 / Test custom parameters")
    public void testCustomParameters() {
        // 定义目标函数：f(x,y) = (x-1)^2 + 2(y-2)^2 (a convex quadratic function with minimum at (1,2))
        IObjectiveFunction objFun = x -> {
            double x1 = (Double) x.get(0);
            double x2 = (Double) x.get(1);
            double term1 = (x1 - 1.0) * (x1 - 1.0);
            double term2 = 2.0 * (x2 - 2.0) * (x2 - 2.0);
            return term1 + term2;
        };
        
        // 定义梯度函数
        IGradientFunction grdFun = x -> {
            double x1 = (Double) x.get(0);
            double x2 = (Double) x.get(1);
            
            // ∂f/∂x1 = 2(x1-1)
            double grad1 = 2 * (x1 - 1.0);
            
            // ∂f/∂x2 = 4(x2-2)
            double grad2 = 4 * (x2 - 2.0);
            
            return Linalg.vector(new double[]{grad1, grad2});
        };
        
        // 初始点
        IVector initX = Linalg.vector(new double[]{5.0, -3.0});
        
        // 创建自定义参数的LBFGS优化器
        RereLBFGS optimizer = new RereLBFGS(5, 1e-8, 500);
        
        // 验证参数设置
        assertEquals(5, optimizer.getM(), "历史信息对数应该为5");
        assertEquals(1e-8, optimizer.getTolerance(), 1e-15, "收敛容差应该为1e-8");
        assertEquals(500, optimizer.getMaxIterations(), "最大迭代次数应该为500");
        
        // 执行优化
        OptResult result = optimizer.optimize(initX, objFun, grdFun);
        
        // 验证结果
        double error = Math.sqrt(Math.pow((Double) result.getOptimalPoint().get(0) - 1.0, 2) + Math.pow((Double) result.getOptimalPoint().get(1) - 2.0, 2));
        assertTrue(error < 1e-6, "解应该接近(1, 2)");
        assertEquals(0.0, result.getOptimalValue(), 1e-10, "最优值应该接近0.0");
    }
    
    @Test
    @DisplayName("测试边界条件 / Test boundary conditions")
    public void testBoundaryConditions() {
        // 定义目标函数：f(x) = x^2
        IObjectiveFunction objFun = x -> {
            double val = (Double) x.get(0);
            return val * val;
        };
        
        // 定义梯度函数：f'(x) = 2x
        IGradientFunction grdFun = x -> {
            double grad = 2 * (Double) x.get(0);
            return Linalg.vector(new double[]{grad});
        };
        
        // 初始点在最优解上
        IVector initX = Linalg.vector(new double[]{0.0});
        
        // 创建LBFGS优化器
        RereLBFGS optimizer = new RereLBFGS();
        
        // 执行优化
        OptResult result = optimizer.optimize(initX, objFun, grdFun);
        
        // 验证结果
        assertEquals(0.0, (Double) result.getOptimalPoint().get(0), 1e-10, "最优点应该为0.0");
        assertEquals(0.0, result.getOptimalValue(), 1e-15, "最优值应该为0.0");
    }
    
    @Test
    @DisplayName("测试参数设置 / Test parameter setting")
    public void testParameterSetting() {
        RereLBFGS optimizer = new RereLBFGS();
        
        // 测试默认参数
        assertEquals(10, optimizer.getM(), "默认历史信息对数应该为10");
        assertEquals(1e-6, optimizer.getTolerance(), 1e-15, "默认收敛容差应该为1e-6");
        assertEquals(1000, optimizer.getMaxIterations(), "默认最大迭代次数应该为1000");
        
        // 测试参数设置
        optimizer.setM(5);
        optimizer.setTolerance(1e-8);
        optimizer.setMaxIterations(500);
        
        assertEquals(5, optimizer.getM(), "历史信息对数应该为5");
        assertEquals(1e-8, optimizer.getTolerance(), 1e-15, "收敛容差应该为1e-8");
        assertEquals(500, optimizer.getMaxIterations(), "最大迭代次数应该为500");
        
        // 测试参数边界检查
        optimizer.setM(-1);
        assertEquals(1, optimizer.getM(), "历史信息对数不能小于1");
        
        optimizer.setTolerance(-1e-6);
        assertEquals(1e-12, optimizer.getTolerance(), 1e-15, "收敛容差不能小于1e-12");
        
        optimizer.setMaxIterations(-1);
        assertEquals(1, optimizer.getMaxIterations(), "最大迭代次数不能小于1");
    }
    
    @Test
    @DisplayName("测试异常处理 / Test exception handling")
    public void testExceptionHandling() {
        RereLBFGS optimizer = new RereLBFGS();
        
        // 测试空初始点异常
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            optimizer.optimize(null, null, null);
        });
        assertTrue(exception.getMessage().contains("初始点不能为空") || exception.getMessage().contains("Initial point cannot be null"));
        
        // 测试空目标函数异常
        IVector initX = Linalg.vector(new double[]{0.0});
        exception = assertThrows(IllegalArgumentException.class, () -> {
            optimizer.optimize(initX, null, null);
        });
        assertTrue(exception.getMessage().contains("目标函数不能为空") || exception.getMessage().contains("Objective function cannot be null"));
        
        // 测试空梯度函数异常
        IObjectiveFunction objFun = x -> 0.0;
        exception = assertThrows(IllegalArgumentException.class, () -> {
            optimizer.optimize(initX, objFun, null);
        });
        assertTrue(exception.getMessage().contains("梯度函数不能为空") || exception.getMessage().contains("Gradient function cannot be null"));
    }
}