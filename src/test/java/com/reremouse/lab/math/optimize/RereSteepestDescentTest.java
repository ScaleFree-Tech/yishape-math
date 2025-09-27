package com.reremouse.lab.math.optimize;

import com.reremouse.lab.math.optimize.newton.RereSteepestDescent;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.util.Tuple2;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * RereSteepestDescent测试类
 */
public class RereSteepestDescentTest {

    /**
     * 测试目标函数: f(x) = x1^2 + x2^2
     * 全局最小值在 (0, 0) 处，最小值为 0
     */
    static class QuadraticObjectiveFunction implements IObjectiveFunction {
        @Override
        public double computeObjective(IVector x) {
            // f(x) = x1^2 + x2^2
            double x1 = (Double) x.get(0);
            double x2 = (Double) x.get(1);
            return x1 * x1 + x2 * x2;
        }
    }

    /**
     * 测试梯度函数: grad(f) = [2*x1, 2*x2]
     */
    static class QuadraticGradientFunction implements IGradientFunction {
        @Override
        public IVector computeGradient(IVector x) {
            // grad(f) = [2*x1, 2*x2]
            double x1 = (Double) x.get(0);
            double x2 = (Double) x.get(1);
            return Linalg.vector(new double[]{2 * x1, 2 * x2});
        }
    }

    @Test
    public void testOptimize() {
        // 创建最速下降法优化器
        RereSteepestDescent optimizer = new RereSteepestDescent(1e-6, 1000, 1.0);
        
        // 设置初始点 (5, 5)
        IVector initialPoint = Linalg.vector(new double[]{5.0, 5.0});
        
        // 创建目标函数和梯度函数
        IObjectiveFunction objectiveFunction = new QuadraticObjectiveFunction();
        IGradientFunction gradientFunction = new QuadraticGradientFunction();
        
        // 执行优化
        var result = optimizer.optimize(initialPoint, objectiveFunction, gradientFunction);
        
        // 验证结果
        Double optimalValue = result.getOptimalValue();
        IVector optimalPoint = result.getOptimalPoint();
        
        // 最优值应该接近0
        assertTrue(Math.abs(optimalValue) < 1e-4, "最优值应该接近0");
        
        // 最优点应该接近(0, 0)
        assertEquals(0.0, (Double) optimalPoint.get(0), 1e-3, "最优解的第一个分量应该接近0");
        assertEquals(0.0, (Double) optimalPoint.get(1), 1e-3, "最优解的第二个分量应该接近0");
        
        System.out.println("最优值: " + optimalValue);
        System.out.println("最优点: [" + optimalPoint.get(0) + ", " + optimalPoint.get(1) + "]");
    }
}