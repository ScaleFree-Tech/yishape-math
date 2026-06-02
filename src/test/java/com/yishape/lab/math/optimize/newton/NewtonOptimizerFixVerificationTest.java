package com.yishape.lab.math.optimize.newton;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.OptResult;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 验证 newton 包修复的专项测试 / Regression tests for newton package fixes
 */
@DisplayName("Newton 包修复验证 / Newton package fix verification")
public class NewtonOptimizerFixVerificationTest {

    // ============================================================
    // 1. RereLBFGS: 曲率条件修复 (curvature condition)
    // ============================================================

    /**
     * 验证曲率条件 s^T y > 0 的小值现在能被正确保留。
     * 使用一个病态二次函数（条件数差），在优化过程中 s^T y 可能很小但为正。
     * 旧代码使用 compareTo(sTy, 1e-10, tolerance) 时，这些小正 sTy 会被错误丢弃。
     */
    @Test
    @DisplayName("LBFGS 曲率条件保留小正 sTy / LBFGS keeps small positive sTy")
    public void testLBFGSCurvatureConditionKeepsSmallPositiveSty() {
        // f(x) = 0.5 * (100*x1^2 + x2^2), ill-conditioned quadratic
        // Hessian diag(100, 1), condition number = 100
        IObjectiveFunction obj = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            return 0.5 * (100 * v1 * v1 + v2 * v2);
        };
        IGradientFunction grad = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            return Linalg.vector(new double[]{100 * v1, v2});
        };

        RereLBFGS optimizer = new RereLBFGS(10, 1e-8, 500);
        IVector initX = Linalg.vector(new double[]{1.0, 1.0});
        OptResult result = optimizer.optimize(initX, obj, grad);

        assertTrue(result.isConverged(), "应该在容差内收敛");
        assertEquals(0.0, result.getOptimalPoint().get(0), 1e-5, "x1 应接近 0");
        assertEquals(0.0, result.getOptimalPoint().get(1), 1e-5, "x2 应接近 0");
    }

    /**
     * 验证 LBFGS 在 Rosenbrock 上的收敛不受 ArrayList 改动影响。
     */
    @Test
    @DisplayName("LBFGS ArrayList 等价性 / LBFGS ArrayList equivalence")
    public void testLBFGSArrayListEquivalence() {
        IObjectiveFunction obj = x -> {
            double x1 = x.get(0);
            double x2 = x.get(1);
            return (1 - x1) * (1 - x1) + 100 * (x2 - x1 * x1) * (x2 - x1 * x1);
        };
        IGradientFunction grad = x -> {
            double x1 = x.get(0);
            double x2 = x.get(1);
            double g1 = -2 * (1 - x1) - 400 * x1 * (x2 - x1 * x1);
            double g2 = 200 * (x2 - x1 * x1);
            return Linalg.vector(new double[]{g1, g2});
        };

        IVector initX = Linalg.vector(new double[]{-1.2, 1.0});

        // trackHistory = true (default)
        RereLBFGS opt1 = new RereLBFGS(10, 1e-6, 500);
        OptResult r1 = opt1.optimize(initX.copy(), obj, grad);

        // trackHistory = false
        RereLBFGS opt2 = new RereLBFGS(10, 1e-6, 500);
        opt2.setTrackHistory(false);
        OptResult r2 = opt2.optimize(initX.copy(), obj, grad);

        assertTrue(r1.isConverged(), "trackHistory=true 应收敛");
        assertTrue(r2.isConverged(), "trackHistory=false 应收敛");
        assertEquals(r1.getOptimalPoint().get(0),
                     r2.getOptimalPoint().get(0), 1e-10,
                     "trackHistory 开关不应影响最优点");
        assertEquals(r1.getOptimalPoint().get(1),
                     r2.getOptimalPoint().get(1), 1e-10,
                     "trackHistory 开关不应影响最优点");
        assertTrue(r1.getFunctionValueHistory() != null && !r1.getFunctionValueHistory().isEmpty(),
            "trackHistory=true 时应记录历史");
        assertTrue(r2.getFunctionValueHistory() == null || r2.getFunctionValueHistory().isEmpty(),
            "trackHistory=false 时不应记录历史");
    }

    // ============================================================
    // 2. RereConjugateGradient: 停滞检测后状态推进
    // ============================================================

    /**
     * 验证 CG 停滞检测后不会原地踏步导致无限循环。
     * 原始代码在停滞检测后 continue 未更新 x/grad，修改后应能正确推进。
     * 这里不强制要求收敛，只验证：函数值下降 + 迭代合理推进。
     */
    @Test
    @DisplayName("CG 停滞检测后正确推进 / CG stagnation advances state")
    public void testCGStagnationAdvancesState() {
        // f(x) = (x1-1)^2 + (x2-2)^2, simple quadratic
        IObjectiveFunction obj = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            return (v1 - 1) * (v1 - 1) + (v2 - 2) * (v2 - 2);
        };
        IGradientFunction grad = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            return Linalg.vector(new double[]{2 * (v1 - 1), 2 * (v2 - 2)});
        };

        RereConjugateGradient optimizer = new RereConjugateGradient(1e-8, 200, 0.5);
        IVector initX = Linalg.vector(new double[]{10.0, -5.0});
        OptResult result = optimizer.optimize(initX, obj, grad);

        // 验证函数值显著下降（从 130 降到接近 0）
        assertTrue(result.getOptimalValue() < 1e-4,
            "函数值应显著下降，实际=" + result.getOptimalValue());
        // 确保迭代次数合理（不应因停滞检测在 1-2 次就 break）
        assertTrue(result.getIterations() >= 2,
            "迭代次数应 >= 2，实际=" + result.getIterations());
        // 验证最优点接近 (1, 2)
        assertEquals(1.0, result.getOptimalPoint().get(0), 1e-3, "x1 应接近 1");
        assertEquals(2.0, result.getOptimalPoint().get(1), 1e-3, "x2 应接近 2");
    }

    // ============================================================
    // 3. RereDFP: 曲率条件修复 + yTHy 除零保护
    // ============================================================

    @Test
    @DisplayName("DFP 小曲率条件保留 / DFP keeps small curvature")
    public void testDFPCurvatureCondition() {
        // 病态二次函数：Hessian diag(100, 1)，sTy 在某些迭代中可能很小
        IObjectiveFunction obj = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            return 0.5 * (100 * v1 * v1 + v2 * v2);
        };
        IGradientFunction grad = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            return Linalg.vector(new double[]{100 * v1, v2});
        };

        RereDFP optimizer = new RereDFP(1e-8, 500);
        IVector initX = Linalg.vector(new double[]{1.0, 1.0});
        OptResult result = optimizer.optimize(initX, obj, grad);

        assertTrue(result.isConverged(), "DFP 应收敛");
        assertEquals(0.0, result.getOptimalPoint().get(0), 1e-5, "x1 应接近 0");
        assertEquals(0.0, result.getOptimalPoint().get(1), 1e-5, "x2 应接近 0");
    }

    // ============================================================
    // 4. RereOnlineAdam: skipGradientValidation
    // ============================================================

    @Test
    @DisplayName("Adam skipGradientValidation 开关 / Adam skipGradientValidation toggle")
    public void testAdamSkipGradientValidation() {
        RereOnlineAdam adam = new RereOnlineAdam();
        IVector params = Linalg.vector(new double[]{1.0, 2.0});
        adam.initialize(params);

        // 默认不跳过，NaN 梯度应抛异常
        IVector nanGrad = Linalg.vector(new double[]{Double.NaN, 0.0});
        assertThrows(IllegalArgumentException.class, () -> adam.step(nanGrad),
            "默认应检查 NaN 并抛异常");

        // 重置并启用跳过
        adam.reset();
        adam.initialize(params);
        adam.setSkipGradientValidation(true);
        assertDoesNotThrow(() -> adam.step(nanGrad),
            "skipGradientValidation=true 时不应抛异常");
    }

    // ============================================================
    // 5. RereOnlineSGD: skipGradientValidation
    // ============================================================

    @Test
    @DisplayName("SGD skipGradientValidation 开关 / SGD skipGradientValidation toggle")
    public void testSGDSkipGradientValidation() {
        RereOnlineSGD sgd = new RereOnlineSGD();
        IVector params = Linalg.vector(new double[]{1.0, 2.0});
        sgd.initialize(params);

        IVector nanGrad = Linalg.vector(new double[]{0.0, Double.NaN});
        assertThrows(IllegalArgumentException.class, () -> sgd.step(nanGrad),
            "默认应检查 NaN 并抛异常");

        sgd.reset();
        sgd.initialize(params);
        sgd.setSkipGradientValidation(true);
        assertDoesNotThrow(() -> sgd.step(nanGrad),
            "skipGradientValidation=true 时不应抛异常");
    }

    // ============================================================
    // 6. RustLBFGS: 双参构造 m 默认值为 10
    // ============================================================

    @Test
    @DisplayName("RustLBFGS 双参构造 m 默认值 / RustLBFGS 2-arg ctor m default")
    public void testRustLBFGSTwoArgCtorMDefault() {
        RustLBFGS optimizer = new RustLBFGS(1e-6, 1000);
        // m is private, but we can verify fallback behavior indirectly
        // by checking it doesn't crash when HPC is unavailable
        IObjectiveFunction obj = x -> {
            double v = x.get(0);
            return v * v;
        };
        IGradientFunction grad = x -> {
            double v = x.get(0);
            return Linalg.vector(new double[]{2 * v});
        };
        IVector initX = Linalg.vector(new double[]{5.0});
        OptResult result = optimizer.optimize(initX, obj, grad);
        assertTrue(result.isConverged(), "应收敛");
        assertEquals(0.0, result.getOptimalPoint().get(0), 1e-5, "应接近 0");
    }
}
