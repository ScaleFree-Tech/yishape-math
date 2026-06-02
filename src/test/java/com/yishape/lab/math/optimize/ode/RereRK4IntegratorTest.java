package com.yishape.lab.math.optimize.ode;

import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.ode.impl.RereRK4Integrator;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link RereRK4Integrator}.
 */
class RereRK4IntegratorTest {

    // ==================== Simple ODEs ====================

    @Test
    void exponentialGrowth() {
        // dy/dt = y, solution: y(t) = y0 * e^t
        RereRK4Integrator integrator = new RereRK4Integrator();
        IDoubleVector y0 = Linalg.vector(new double[]{1.0});
        double t0 = 0, t1 = 1.0, dt = 0.01;

        IDoubleVector result = integrator.integrate(
            y -> y, // f(y) = y (time-independent)
            y0, t0, t1, dt
        );

        // y(1) = e^1 ≈ 2.71828
        assertEquals(Math.E, result.get(0), 0.01);
    }

    @Test
    void constantDerivative() {
        // dy/dt = 1, solution: y(t) = y0 + t
        RereRK4Integrator integrator = new RereRK4Integrator();
        IDoubleVector y0 = Linalg.vector(new double[]{0.0});

        IDoubleVector result = integrator.integrate(
            y -> Linalg.vector(new double[]{1.0}),
            y0, 0, 5.0, 0.1
        );

        assertEquals(5.0, result.get(0), 0.01);
    }

    @Test
    void harmonicOscillator() {
        // dy/dt = [v, -y] for harmonic oscillator
        RereRK4Integrator integrator = new RereRK4Integrator();
        IDoubleVector y0 = Linalg.vector(new double[]{1.0, 0.0}); // y=1, v=0

        IDoubleVector result = integrator.integrate(
            y -> Linalg.vector(new double[]{y.get(1), -y.get(0)}),
            y0, 0, 2 * Math.PI, 0.01
        );

        // After one full period, should return to approximately (1, 0)
        assertEquals(1.0, result.get(0), 0.05);
        assertEquals(0.0, result.get(1), 0.05);
    }

    // ==================== Edge Cases ====================

    @Test
    void zeroTimeSpan() {
        RereRK4Integrator integrator = new RereRK4Integrator();
        IDoubleVector y0 = Linalg.vector(new double[]{5.0});

        IDoubleVector result = integrator.integrate(
            y -> y,
            y0, 1.0, 1.0, 0.01
        );

        assertEquals(5.0, result.get(0), 1e-10);
    }

    @Test
    void multipleDimensions() {
        RereRK4Integrator integrator = new RereRK4Integrator();
        IDoubleVector y0 = Linalg.vector(new double[]{1.0, 2.0, 3.0});

        IDoubleVector result = integrator.integrate(
            y -> Linalg.vector(new double[]{
                y.get(0), y.get(1), y.get(2)
            }),
            y0, 0, 1.0, 0.01
        );

        // Each component grows as e^t
        assertEquals(Math.E, result.get(0), 0.01);
        assertEquals(2 * Math.E, result.get(1), 0.02);
        assertEquals(3 * Math.E, result.get(2), 0.03);
    }

    @Test
    void smallStepSize_moreAccurate() {
        RereRK4Integrator integrator = new RereRK4Integrator();
        IDoubleVector y0 = Linalg.vector(new double[]{1.0});

        IDoubleVector coarse = integrator.integrate(y -> y, y0, 0, 1.0, 0.1);
        IDoubleVector fine = integrator.integrate(y -> y, y0, 0, 1.0, 0.001);

        double errorCoarse = Math.abs(coarse.get(0) - Math.E);
        double errorFine = Math.abs(fine.get(0) - Math.E);

        assertTrue(errorFine < errorCoarse,
            "Finer step should be more accurate");
    }
}
