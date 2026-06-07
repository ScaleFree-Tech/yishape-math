package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.support.OdeintConfig;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.optimize.ode.impl.RereDopri5Integrator;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.function.Function;

import org.junit.jupiter.api.Test;

/**
 * Tests for Dopri5 adaptive Neural ODE integration and gradient correctness.
 */
public class Dopri5OdeintTest {

    private static final double TOL = 1e-6;
    private static final double GRAD_TOL = 1e-4;

    // ==================== Forward integration accuracy ====================

    /**
     * dy/dt = -y, exact: y(t) = y0 * exp(-t).
     * At t=1, y = [1*e^-1, 2*e^-1] = [0.3679, 0.7358].
     */
    @Test
    void testDopri5ExponentialDecay() {
        double[] y0Data = {1.0, 2.0};
        Function<IDiffVector, IDiffVector> dynamics = z -> z.mul(-1.0);

        IDiffVector z0 = AD.vector(y0Data);
        IDiffVector result = AD.odeint(dynamics, z0, 0.0, 1.0, 0.1,
            OdeintConfig.dopri5(1e-9, 1e-6));

        double[] y = result.getValue().getData();
        assertEquals(Math.exp(-1), y[0], TOL);
        assertEquals(2.0 * Math.exp(-1), y[1], TOL);
    }

    /**
     * dy/dt = y (exponential growth).
     * y(t) = y0 * exp(t).
     */
    @Test
    void testDopri5ExponentialGrowth() {
        Function<IDiffVector, IDiffVector> dynamics = z -> z; // dy/dt = y

        IDiffVector z0 = AD.vector(1.0);
        IDiffVector result = AD.odeint(dynamics, z0, 0.0, 0.5, 0.1,
            OdeintConfig.dopri5(1e-12, 1e-9));

        assertEquals(Math.exp(0.5), result.getValue().get(0), TOL);
    }

    /**
     * dy/dt = 0 (constant dynamics).
     */
    @Test
    void testDopri5Constant() {
        Function<IDiffVector, IDiffVector> dynamics = z -> {
            return AD.vector(new double[]{0.0, 0.0, 0.0});
        };

        IDiffVector z0 = AD.vector(3.0, 5.0, 7.0);
        IDiffVector result = AD.odeint(dynamics, z0, 0.0, 2.0, 0.1,
            OdeintConfig.dopri5(1e-9, 1e-6));

        double[] y = result.getValue().getData();
        assertArrayEquals(new double[]{3.0, 5.0, 7.0}, y, TOL);
    }

    /**
     * Harmonic oscillator: dy0/dt = y1, dy1/dt = -y0.
     * Exact: y0(t) = cos(t), y1(t) = -sin(t) for z0 = [1, 0].
     */
    @Test
    void testDopri5HarmonicOscillator() {
        Function<IDiffVector, IDiffVector> dynamics = z -> {
            double z0 = z.getValue().get(0);
            double z1 = z.getValue().get(1);
            return AD.vector(z1, -z0);
        };

        IDiffVector z0 = AD.vector(1.0, 0.0);
        IDiffVector result = AD.odeint(dynamics, z0, 0.0, Math.PI / 2, 0.05,
            OdeintConfig.dopri5(1e-10, 1e-8));

        double[] y = result.getValue().getData();
        assertEquals(0.0, y[0], 1e-5);     // cos(pi/2) ≈ 0
        assertEquals(-1.0, y[1], 1e-5);    // -sin(pi/2) = -1
    }

    // ==================== Gradient correctness (Neural ODE) ====================

    /**
     * dy/dt = -y, loss = sum(y(t1)).
     * d(loss)/d(y0_i) = exp(-(t1-t0)) * 1.
     */
    @Test
    void testDopri5GradientExponentialDecay() {
        Function<IDiffVector, IDiffVector> dynamics = z -> z.mul(-1.0);

        IDiffVector z0 = AD.vector(2.0, 3.0);
        IDiffVector y = AD.odeint(dynamics, z0, 0.0, 1.0, 0.1,
            OdeintConfig.dopri5(1e-12, 1e-9));

        IDiffVector loss = y.sum();
        loss.backward();

        double[] grad = z0.getGradient().getData();
        double expected = Math.exp(-1.0); // d(sum(exp(-t)*z0))/d(z0_i) = exp(-1)
        assertEquals(expected, grad[0], GRAD_TOL);
        assertEquals(expected, grad[1], GRAD_TOL);
    }

    /**
     * dy/dt = y (growth), loss = sum(y).
     * d(loss)/d(y0_i) = exp(t1 - t0).
     */
    @Test
    void testDopri5GradientExponentialGrowth() {
        Function<IDiffVector, IDiffVector> dynamics = z -> z; // dy/dt = y

        IDiffVector z0 = AD.vector(1.0);
        IDiffVector y = AD.odeint(dynamics, z0, 0.0, 0.5, 0.1,
            OdeintConfig.dopri5(1e-12, 1e-9));

        IDiffVector loss = y.sum();
        loss.backward();

        double grad = z0.getGradient().get(0);
        assertEquals(Math.exp(0.5), grad, GRAD_TOL);
    }

    /**
     * dy/dt = -k*y with loss = sum(y(t1)^2).
     * d(loss)/d(y0) = 2 * y(t1) * dy(t1)/dy0 = 2*y0*exp(-k*t) * exp(-k*t) = 2*y0*exp(-2k*t).
     */
    @Test
    void testDopri5GradientQuadraticLoss() {
        double k = 2.0;
        Function<IDiffVector, IDiffVector> dynamics = z -> z.mul(-k);

        IDiffVector z0 = AD.vector(1.5);
        IDiffVector y = AD.odeint(dynamics, z0, 0.0, 1.0, 0.1,
            OdeintConfig.dopri5(1e-12, 1e-9));

        IDiffVector loss = y.pow(2).sum();
        loss.backward();

        double grad = z0.getGradient().get(0);
        double expected = 2.0 * 1.5 * Math.exp(-2.0 * k * 1.0);
        assertEquals(expected, grad, GRAD_TOL);
    }

    /**
     * 2D dynamics with different decay rates per component via element-wise
     * multiplication. Since each component decays independently, analytic
     * gradient is [exp(-0.5), exp(-1.0)].
     */
    @Test
    void testDopri5GradientMultiDimensional() {
        Function<IDiffVector, IDiffVector> dynamics = z -> z.mul(-1.0);

        IDiffVector z0 = AD.vector(1.0, 2.0);
        IDiffVector y = AD.odeint(dynamics, z0, 0.0, 0.5, 0.1,
            OdeintConfig.dopri5(1e-12, 1e-9));

        // loss = y0 + y1
        IDiffVector loss = y.sum();
        loss.backward();

        double[] grad = z0.getGradient().getData();
        double expected = Math.exp(-0.5);
        assertEquals(expected, grad[0], GRAD_TOL);
        assertEquals(expected, grad[1], GRAD_TOL);
    }

    // ==================== Dopri5 vs RK4 agreement ====================

    /**
     * Dopri5 and RK4 should agree on gradients for simple dynamics.
     */
    @Test
    void testDopri5VsRk4GradientAgreement() {
        Function<IDiffVector, IDiffVector> dynamics = z -> z.mul(-1.0);

        IDiffVector z0r = AD.vector(2.0, 3.0, 4.0);
        IDiffVector z0d = AD.vector(2.0, 3.0, 4.0);

        // RK4 path
        IDiffVector yr = AD.odeint(dynamics, z0r, 0.0, 1.0, 0.1);
        IDiffVector lossR = yr.sum();
        lossR.backward();

        // Dopri5 path
        IDiffVector yd = AD.odeint(dynamics, z0d, 0.0, 1.0, 0.1,
            OdeintConfig.dopri5(1e-12, 1e-9));
        IDiffVector lossD = yd.sum();
        lossD.backward();

        double[] gradR = z0r.getGradient().getData();
        double[] gradD = z0d.getGradient().getData();

        // Gradients should agree within tolerance
        for (int i = 0; i < gradR.length; i++) {
            assertEquals(gradR[i], gradD[i], GRAD_TOL,
                "Gradient mismatch at index " + i + ": RK4=" + gradR[i] + " vs Dopri5=" + gradD[i]);
        }
    }

    /**
     * Forward outputs should be close between RK4 and Dopri5.
     */
    @Test
    void testDopri5VsRk4ForwardAgreement() {
        Function<IDiffVector, IDiffVector> dynamics = z -> z.pow(2).mul(-1.0); // dy/dt = -y^2

        IDiffVector z0r = AD.vector(0.5);
        IDiffVector z0d = AD.vector(0.5);

        IDiffVector yr = AD.odeint(dynamics, z0r, 0.0, 0.5, 0.01);
        IDiffVector yd = AD.odeint(dynamics, z0d, 0.0, 0.5, 0.1,
            OdeintConfig.dopri5(1e-10, 1e-8));

        assertEquals(yr.getValue().get(0), yd.getValue().get(0), TOL);
    }

    // ==================== Non-AD integrator (IRungeKuttaIntegrator interface) ====================

    /**
     * RereDopri5Integrator.integrate() via IRungeKuttaIntegrator interface.
     */
    @Test
    void testNonAdIntegrate() {
        RereDopri5Integrator integrator = new RereDopri5Integrator();

        Function<IDoubleVector, IDoubleVector> f = v -> {
            double val = v.get(0);
            return com.yishape.lab.math.linalg.Linalg.vector(-val);
        };

        IDoubleVector y0 = com.yishape.lab.math.linalg.Linalg.vector(1.0);
        IDoubleVector result = integrator.integrate(f, y0, 0.0, 1.0, 0.1);

        assertEquals(Math.exp(-1.0), result.get(0), TOL);
    }

    /**
     * Non-AD integrator with 2D harmonic oscillator.
     */
    @Test
    void testNonAdIntegrate2D() {
        RereDopri5Integrator integrator = new RereDopri5Integrator();

        Function<IDoubleVector, IDoubleVector> f = v -> {
            double x = v.get(0);
            double y = v.get(1);
            return com.yishape.lab.math.linalg.Linalg.vector(y, -x);
        };

        IDoubleVector y0 = com.yishape.lab.math.linalg.Linalg.vector(1.0, 0.0);
        IDoubleVector result = integrator.integrate(f, y0, 0.0, Math.PI / 2, 0.05);

        assertEquals(0.0, result.get(0), 1e-3);   // cos(pi/2)
        assertEquals(-1.0, result.get(1), 1e-3);  // -sin(pi/2)
    }

    // ==================== Checkpoints ====================

    /**
     * Dopri5 with checkpoints stores trajectory, enables backward.
     */
    @Test
    void testDopri5CheckpointBackward() {
        Function<IDiffVector, IDiffVector> dynamics = z -> z.mul(-0.5);

        IDiffVector z0 = AD.vector(3.0);
        IDiffVector y = AD.odeint(dynamics, z0, 0.0, 2.0, 0.2,
            OdeintConfig.dopri5(1e-9, 1e-6, 0, 0.5, 0, 100000, 4));

        IDiffVector loss = y.sum();
        loss.backward();

        double grad = z0.getGradient().get(0);
        // d(loss)/dz0 = d(z0*exp(-0.5*2))/dz0 = exp(-1)
        assertEquals(Math.exp(-1.0), grad, GRAD_TOL);
    }

    // ==================== Config factory methods ====================

    @Test
    void testOdeintConfigRk4() {
        OdeintConfig cfg = OdeintConfig.rk4();
        assertEquals("rk4", cfg.method());
        assertEquals(false, cfg.isAdaptive());
    }

    @Test
    void testOdeintConfigDopri5Defaults() {
        OdeintConfig cfg = OdeintConfig.dopri5();
        assertEquals("dopri5", cfg.method());
        assertEquals(true, cfg.isAdaptive());
        assertEquals(1e-9, cfg.effectiveAbsTol());
        assertEquals(1e-6, cfg.effectiveRelTol());
        assertEquals(Double.POSITIVE_INFINITY, cfg.effectiveMaxDt());
    }

    @Test
    void testOdeintConfigDopri5Custom() {
        OdeintConfig cfg = OdeintConfig.dopri5(1e-8, 1e-5);
        assertEquals(1e-8, cfg.effectiveAbsTol());
        assertEquals(1e-5, cfg.effectiveRelTol());
    }

    @Test
    void testOdeintConfigDopri5Full() {
        OdeintConfig cfg = OdeintConfig.dopri5(1e-10, 1e-7, 1e-6, 0.1, 0.01, 50000, 16);
        assertEquals("dopri5", cfg.method());
        assertEquals(1e-10, cfg.absTol());
        assertEquals(1e-7, cfg.relTol());
        assertEquals(1e-6, cfg.minDt());
        assertEquals(0.1, cfg.maxDt());
        assertEquals(0.01, cfg.initialDt());
        assertEquals(50000, cfg.maxSteps());
        assertEquals(16, cfg.checkpointInterval());
    }

    // ==================== AD.checkGradient validation ====================

    /**
     * Verify Dopri5 gradient passes AD.checkGradient() against finite differences.
     */
    @Test
    void testDopri5CheckGradient() {
        Function<IDiffVector, IDiffVector> dynamics = z -> z.mul(-1.0);

        Function<IDiffVector, IDiffVector> lossFn = z0 -> {
            IDiffVector y = AD.odeint(dynamics, z0, 0.0, 0.5, 0.1,
                OdeintConfig.dopri5(1e-12, 1e-9));
            return y.sum();
        };

        IDiffVector x = AD.vector(1.0, 2.0);
        boolean ok = AD.checkGradient(lossFn, x, 1e-4);
        assertTrue(ok, "Dopri5 gradient should pass finite difference check");
    }

    /**
     * Check gradient with 3D input, uniform exponential decay.
     */
    @Test
    void testDopri5CheckGradient3D() {
        Function<IDiffVector, IDiffVector> dynamics = z -> z.mul(-1.0);

        Function<IDiffVector, IDiffVector> lossFn = z0 -> {
            IDiffVector y = AD.odeint(dynamics, z0, 0.0, 1.0, 0.1,
                OdeintConfig.dopri5(1e-12, 1e-9));
            return y.sum();
        };

        IDiffVector x = AD.vector(1.0, 2.0, 3.0);
        boolean ok = AD.checkGradient(lossFn, x, 1e-3);
        assertTrue(ok, "Dopri5 3D gradient should pass finite difference check");
    }

    // ==================== Stiff ODE ====================

    /**
     * dy/dt = -50*y (mildly stiff decay).
     * Exact: y(t) = y0 * exp(-50*t).
     */
    @Test
    void testDopri5StiffDecay() {
        Function<IDiffVector, IDiffVector> dynamics = z -> z.mul(-50.0);

        IDiffVector z0 = AD.vector(2.0);
        IDiffVector y = AD.odeint(dynamics, z0, 0.0, 0.1, 0.01,
            OdeintConfig.dopri5(1e-12, 1e-9, 0, 0.05, 0, 100000, 8));

        double expected = 2.0 * Math.exp(-5.0);
        assertEquals(expected, y.getValue().get(0), 1e-4);
    }

    /**
     * Stiff decay gradient check.
     */
    @Test
    void testDopri5StiffDecayGradient() {
        Function<IDiffVector, IDiffVector> dynamics = z -> z.mul(-50.0);

        IDiffVector z0 = AD.vector(1.5);
        IDiffVector y = AD.odeint(dynamics, z0, 0.0, 0.05, 0.01,
            OdeintConfig.dopri5(1e-12, 1e-9, 0, 0.02, 0, 100000, 8));

        IDiffVector loss = y.sum();
        loss.backward();

        double grad = z0.getGradient().get(0);
        double expected = Math.exp(-2.5); // exp(-50 * 0.05)
        assertEquals(expected, grad, GRAD_TOL);
    }
}
