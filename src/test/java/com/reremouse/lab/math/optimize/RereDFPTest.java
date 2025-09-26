package com.reremouse.lab.math.optimize;

import com.reremouse.lab.math.optimize.newton.RereDFP;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.util.Tuple2;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * DFP优化算法测试类
 */
public class RereDFPTest {

    /**
     * 测试DFP算法求解简单的二次函数优化问题
     * 目标函数: f(x) = (x1-2)^2 + (x2-3)^2
     * 最优解: x* = [2, 3], f(x*) = 0
     */
    @Test
    public void testQuadraticOptimization() {
        // 定义目标函数: f(x) = (x1-2)^2 + (x2-3)^2
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double x1 = (Double) x.get(0);
                double x2 = (Double) x.get(1);
                return Math.pow(x1 - 2, 2) + Math.pow(x2 - 3, 2);
            }
        };
        
        // 定义梯度函数: grad(f) = [2*(x1-2), 2*(x2-3)]
        IGradientFunction gradFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double x1 = (Double) x.get(0);
                double x2 = (Double) x.get(1);
                double[] grad = {2 * (x1 - 2), 2 * (x2 - 3)};
                return Linalg.vector(grad);
            }
        };
        
        // 初始点
        IVector initX = Linalg.vector(new double[]{0.0, 0.0});
        
        // 创建DFP优化器
        RereDFP dfp = new RereDFP(1e-6, 100);
        
        // 执行优化
        Tuple2<Double, IVector> result = dfp.optimize(initX, objFun, gradFun);
        
        // 验证结果
        Double optimalValue = result._1;
        IVector optimalPoint = result._2;
        
        // 最优值应该接近0
        assertEquals(0.0, optimalValue, 1e-4);
        
        // 最优点应该接近[2, 3]
        assertEquals(2.0, (Double) optimalPoint.get(0), 1e-4);
        assertEquals(3.0, (Double) optimalPoint.get(1), 1e-4);
    }
    
    /**
     * 测试DFP算法求解Rosenbrock函数
     * 目标函数: f(x) = 100*(x2-x1^2)^2 + (1-x1)^2
     * 最优解: x* = [1, 1], f(x*) = 0
     */
    @Test
    public void testRosenbrockOptimization() {
        // 定义Rosenbrock目标函数
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double x1 = (Double) x.get(0);
                double x2 = (Double) x.get(1);
                return 100 * Math.pow(x2 - x1 * x1, 2) + Math.pow(1 - x1, 2);
            }
        };
        
        // 定义Rosenbrock梯度函数
        IGradientFunction gradFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double x1 = (Double) x.get(0);
                double x2 = (Double) x.get(1);
                double[] grad = {
                    -400 * (x2 - x1 * x1) * x1 - 2 * (1 - x1),
                    200 * (x2 - x1 * x1)
                };
                return Linalg.vector(grad);
            }
        };
        
        // 初始点
        IVector initX = Linalg.vector(new double[]{-1.0, 1.0});
        
        // 创建DFP优化器
        RereDFP dfp = new RereDFP(1e-6, 1000);
        
        // 执行优化
        Tuple2<Double, IVector> result = dfp.optimize(initX, objFun, gradFun);
        
        // 验证结果
        Double optimalValue = result._1;
        IVector optimalPoint = result._2;
        
        // 最优值应该接近0
        assertEquals(0.0, optimalValue, 1e-3);
        
        // 最优点应该接近[1, 1]
        assertEquals(1.0, (Double) optimalPoint.get(0), 1e-2);
        assertEquals(1.0, (Double) optimalPoint.get(1), 1e-2);
    }
}