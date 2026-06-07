package com.yishape.lab.math.optimize.ode.impl;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import com.yishape.lab.math.autodiff.support.OdeintConfig;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.ode.IRungeKuttaIntegrator;

/**
 * Dormand-Prince 5(4) adaptive ODE integrator.
 *
 * <p>Implements the DOPRI5(4) embedded Runge-Kutta pair with 7 stages,
 * providing both 5th-order and 4th-order estimates. The error estimate
 * (difference between the two orders) drives adaptive step size control
 * via a PI (Proportional-Integral) controller.</p>
 *
 * <p>Reference: Dormand &amp; Prince (1980), "A family of embedded
 * Runge-Kutta formulae", J. Comp. Appl. Math. 6(1):19-26.</p>
 */
public class RereDopri5Integrator implements IRungeKuttaIntegrator {

    // ==================== Butcher Tableau (Dormand-Prince 5(4)) ====================

    private static final double[] C  = { 0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0 };
    private static final double[][] A = {
        {},
        { 1.0/5.0 },
        { 3.0/40.0,        9.0/40.0 },
        { 44.0/45.0,      -56.0/15.0,        32.0/9.0 },
        { 19372.0/6561.0, -25360.0/2187.0,   64448.0/6561.0,   -212.0/729.0 },
        { 9017.0/3168.0,  -355.0/33.0,       46732.0/5247.0,    49.0/176.0,    -5103.0/18656.0 },
        { 35.0/384.0,      0.0,               500.0/1113.0,     125.0/192.0,    -2187.0/6784.0,   11.0/84.0 }
    };
    /** 5th-order coefficients (used for the next step). */
    private static final double[] B5 = {
        35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0
    };
    /** 4th-order coefficients (used for error estimation). */
    private static final double[] B4 = {
        5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0,
        -92097.0/339200.0, 187.0/2100.0, 1.0/40.0
    };

    // PI controller safety factors
    private static final double SAFETY = 0.9;
    private static final double MIN_FACTOR = 0.2;
    private static final double MAX_FACTOR = 10.0;

    // ==================== Result types ====================

    /** Result of Dopri5 forward integration. */
    public record Dopri5Result(IDoubleVector y, List<Checkpoint> checkpoints) {}

    /** Checkpoint: state at a saved time point. */
    public record Checkpoint(IDoubleVector y) {}

    // ==================== IRungeKuttaIntegrator (standard ODE interface) ====================

    @Override
    public IDoubleVector integrate(Function<IDoubleVector, IDoubleVector> f, IDoubleVector y0,
            double t0, double t1, double dt) {
        double t = t0;
        IDoubleVector y = y0.copy();
        while (t < t1) {
            double h = Math.min(dt, t1 - t);
            y = dopri5Step(f, y, h);
            t += h;
        }
        return y;
    }

    // ==================== Static: AD-compatible forward integration ====================

    /**
     * Forward integration with adaptive step size, saving checkpoints.
     *
     * @param dynamics the ODE dynamics function (must produce differentiable output)
     * @param y0       initial state
     * @param t0       start time
     * @param t1       end time
     * @param config   ODE integration configuration
     * @return final state and list of checkpoints
     */
    public static Dopri5Result integrateForward(
            Function<IDiffVector, IDiffVector> dynamics,
            IDoubleVector y0, double t0, double t1, OdeintConfig config) {
        List<Checkpoint> checkpoints = new ArrayList<>();
        IDoubleVector y = y0.copy();
        double t = t0;
        double h = config.initialDt();
        double minDt = Math.max(config.minDt(), 1e-16); // safety floor
        double maxDt = config.maxDt();
        int steps = 0;

        // Compute initial step size if not provided
        if (h <= 0) {
            h = Math.max(0.01 * (t1 - t0), minDt);
        }
        if (h > maxDt) h = maxDt;

        // Save initial checkpoint
        checkpoints.add(new Checkpoint(y.copy()));
        int checkpointCounter = 1;

        while (t < t1 && steps < config.maxSteps()) {
            // Clamp last step
            if (t + h > t1) h = t1 - t;
            if (h < minDt) h = minDt;

            // Attempt a step
            Dopri5StepResult result = adaptiveStep(dynamics, y, h,
                    config.absTol(), config.relTol());

            if (result.needsReject && h > minDt) {
                // Rejected — retry with smaller step
                h = Math.max(result.newH, minDt);
                continue;
            }

            y = result.yNext;
            t += h;
            steps++;

            // Save checkpoint every checkpointInterval steps
            if (checkpointCounter++ >= config.checkpointInterval()) {
                checkpoints.add(new Checkpoint(y.copy()));
                checkpointCounter = 0;
            }

            // Update step size
            h = Math.min(Math.max(result.newH, minDt), maxDt);
        }

        // Ensure final checkpoint
        checkpoints.add(new Checkpoint(y.copy()));

        return new Dopri5Result(y, checkpoints);
    }

    // ==================== Internal: adaptive step ====================

    private static class Dopri5StepResult {
        IDoubleVector yNext;
        boolean needsReject;
        double newH;
    }

    @SuppressWarnings("unchecked")
    private static Dopri5StepResult adaptiveStep(
            Function<IDiffVector, IDiffVector> dynamics, IDoubleVector y, double h,
            double absTol, double relTol) {
        Dopri5StepResult res = new Dopri5StepResult();

        // Compute 7 stages
        List<IDoubleVector> k = new ArrayList<>();
        k.add(evalDynamics(dynamics, y)); // k0

        for (int s = 1; s < 7; s++) {
            IDoubleVector sum = y.copy();
            double[] row = A[s];
            for (int j = 0; j < row.length; j++) {
                sum = (IDoubleVector) sum.add(k.get(j).scale(h * row[j]));
            }
            k.add(evalDynamics(dynamics, sum));
        }

        // 5th-order estimate (next step)
        IDoubleVector y5 = y.copy();
        for (int j = 0; j < 7; j++) {
            y5 = (IDoubleVector) y5.add(k.get(j).scale(h * B5[j]));
        }

        // Error estimate = 5th - 4th order
        IDoubleVector err = IDoubleVector.zeros(y.size());
        for (int j = 0; j < 7; j++) {
            double diff = B5[j] - B4[j];
            if (diff != 0) {
                err = (IDoubleVector) err.add(k.get(j).scale(h * diff));
            }
        }

        // Compute error norm using config tolerances
        double[] yd = y5.getData();
        double[] ed = err.getData();
        double[] y0Data = y.getData();
        double errNorm = 0;
        for (int i = 0; i < yd.length; i++) {
            double scale = absTol + relTol * Math.max(Math.abs(yd[i]), Math.abs(y0Data[i]));
            double q = ed[i] / scale;
            errNorm += q * q;
        }
        errNorm = Math.sqrt(errNorm / yd.length);

        // PI step size controller
        double hNew;
        if (errNorm < 1e-15) {
            hNew = h * MAX_FACTOR;
        } else {
            hNew = SAFETY * h * Math.pow(1.0 / errNorm, 1.0 / 5.0);
        }
        hNew = Math.min(Math.max(hNew, h * MIN_FACTOR), h * MAX_FACTOR);

        res.yNext = y5;
        res.needsReject = errNorm > 1.0;
        res.newH = hNew;
        return res;
    }

    // ==================== Internal: fixed-step Dopri5 for non-adaptive calls ====================

    @SuppressWarnings("unchecked")
    private static IDoubleVector dopri5Step(
            Function<IDoubleVector, IDoubleVector> f, IDoubleVector y, double h) {
        List<IDoubleVector> k = new ArrayList<>();
        k.add(f.apply(y));

        for (int s = 1; s < 7; s++) {
            IDoubleVector sum = y.copy();
            double[] row = A[s];
            for (int j = 0; j < row.length; j++) {
                sum = (IDoubleVector) sum.add(k.get(j).scale(h * row[j]));
            }
            k.add(f.apply(sum));
        }

        IDoubleVector yNext = y.copy();
        for (int j = 0; j < 7; j++) {
            yNext = (IDoubleVector) yNext.add(k.get(j).scale(h * B5[j]));
        }
        return yNext;
    }

    @SuppressWarnings("unchecked")
    private static IDoubleVector evalDynamics(
            Function<IDiffVector, IDiffVector> dynamics, IVector<Double> z) {
        IDiffVector zv = new RereDiffVector((IDoubleVector) z);
        return ((RereDiffVector) dynamics.apply(zv)).getValue();
    }
}
