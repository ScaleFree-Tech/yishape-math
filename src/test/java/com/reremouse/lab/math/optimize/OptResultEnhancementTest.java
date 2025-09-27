package com.reremouse.lab.math.optimize;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.newton.RereLBFGS;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 测试OptResult增强功能
 * Test OptResult enhancement features
 */
@DisplayName("OptResult增强功能测试 / OptResult Enhancement Tests")
public class OptResultEnhancementTest {
    
    @Test
    @DisplayName("测试LBFGS优化器返回丰富的OptResult信息 / Test LBFGS optimizer returns rich OptResult information")
    public void testRichOptResultInformation() {
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
        
        // 验证基本结果
        assertEquals(2.0, (Double) result.getOptimalPoint().get(0), 1e-5, "最优点应该接近2.0");
        assertEquals(0.0, result.getOptimalValue(), 1e-10, "最优值应该接近0.0");
        
        // 验证增强信息
        assertNotNull(result.getInitialPoint(), "应该有初始点信息");
        assertEquals(10.0, (Double) result.getInitialPoint().get(0), 1e-10, "初始点应该为10.0");
        
        assertNotNull(result.getFunctionValueHistory(), "应该有函数值历史");
        assertFalse(result.getFunctionValueHistory().isEmpty(), "函数值历史不应该为空");
        
        assertNotNull(result.getGradientNormHistory(), "应该有梯度范数历史");
        assertFalse(result.getGradientNormHistory().isEmpty(), "梯度范数历史不应该为空");
        
        assertNotNull(result.getParameterHistory(), "应该有参数历史");
        assertFalse(result.getParameterHistory().isEmpty(), "参数历史不应该为空");
        
        assertTrue(result.getIterations() > 0, "迭代次数应该大于0");
        assertTrue(result.getMaxIterations() > 0, "最大迭代次数应该大于0");
        
        assertTrue(result.getFunctionEvaluations() > 0, "函数评估次数应该大于0");
        assertTrue(result.getGradientEvaluations() > 0, "梯度评估次数应该大于0");
        
        assertTrue(result.getExecutionTimeMs() >= 0, "执行时间应该非负");
        
        assertTrue(result.getFinalGradientNorm() >= 0, "最终梯度范数应该非负");
        
        assertNotNull(result.getConvergenceReason(), "应该有收敛原因");
    }
}