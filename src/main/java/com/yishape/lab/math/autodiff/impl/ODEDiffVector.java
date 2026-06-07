package com.yishape.lab.math.autodiff.impl;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.support.OdeintConfig;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.ode.IRungeKuttaIntegrator;
import com.yishape.lab.math.optimize.ode.impl.RereDopri5Integrator;
import com.yishape.lab.math.optimize.ode.impl.RereRK4Integrator;

/**
 * Differentiable Neural ODE layer: supports RK4 (fixed-step) and Dopri5 (adaptive)
 * forward integration and adjoint backward.
 * 可微 Neural ODE 层：支持 RK4（固定步长）和 Dopri5（自适应）前向积分与伴随法反向。
 *
 * <p>Stores the state trajectory during forward; backward walks adjoint equations in reverse time.
 * 前向保存状态轨迹；反向沿伴随方程倒序积分。</p>
 */
public class ODEDiffVector extends RereDiffVector {

    private static final long serialVersionUID = 1L;

    /** RK4 constructor (backward-compatible). */
    public ODEDiffVector(Function<IDiffVector, IDiffVector> dynamics, IDiffVector z0,
            double t0, double t1, double dt) {
        this(dynamics, z0, t0, t1, dt, OdeintConfig.rk4());
    }

    /** Configurable constructor supporting RK4 and Dopri5. */
    public ODEDiffVector(Function<IDiffVector, IDiffVector> dynamics, IDiffVector z0,
            double t0, double t1, double dt, OdeintConfig config) {
        super(IDoubleVector.zeros(1)); // placeholder, will overwrite
        RereDiffVector z0r = (RereDiffVector) z0;
        IDoubleVector y0 = z0r.getValue().copy();
        IDoubleVector y;
        List<IDoubleVector> trajectory = new ArrayList<>();

        if (config.isAdaptive()) {
            // Dopri5 adaptive integration
            RereDopri5Integrator.Dopri5Result result =
                RereDopri5Integrator.integrateForward(dynamics, y0, t0, t1, config);
            y = result.y();
            for (RereDopri5Integrator.Checkpoint cp : result.checkpoints()) {
                trajectory.add(cp.y().copy());
            }
        } else {
            // RK4 fixed-step integration (original path)
            trajectory.add(y0.copy());
            y = integrateForward(dynamics, y0, t0, t1, dt, trajectory);
        }

        // Overwrite the placeholder tensor data
        this.tensor.setValue(new com.yishape.lab.math.linalg.tensor.RereDoubleTensor(
            y.getData(), new int[]{y.size()}));
        this.tensor.setIsLeaf(false);
        this.tensor.inputs().add(z0r.tensor);

        // Wrap the adjoint backward for the tensor graph
        Consumer<IDoubleVector> vecBackwardFn;
        if (config.isAdaptive()) {
            vecBackwardFn = buildAdjointBackwardAdaptive(dynamics, z0r, trajectory, t0, t1, config);
        } else {
            vecBackwardFn = buildAdjointBackward(dynamics, z0r, trajectory, t0, t1, dt);
        }
        this.tensor.setBackwardFn((selfTensor) -> {
            vecBackwardFn.accept(IDoubleVector.of(selfTensor.gradData()));
        });
    }

    private static IDoubleVector integrateForward(Function<IDiffVector, IDiffVector> dynamics,
            IDoubleVector y0, double t0, double t1, double dt, List<IDoubleVector> trajectory) {
        IDoubleVector y = y0;
        double t = t0;
        while (t < t1) {
            double h = Math.min(dt, t1 - t);
            y = rk4Step(dynamics, y, h);
            trajectory.add(y.copy());
            t += h;
        }
        return y;
    }

    private static IDoubleVector rk4Step(Function<IDiffVector, IDiffVector> dynamics,
            IDoubleVector y, double h) {
        IDoubleVector k1 = evalDynamics(dynamics, y);
        IDoubleVector k2 = evalDynamics(dynamics, y.add(k1.scale(h / 2.0)));
        IDoubleVector k3 = evalDynamics(dynamics, y.add(k2.scale(h / 2.0)));
        IDoubleVector k4 = evalDynamics(dynamics, y.add(k3.scale(h)));
        return y.add(k1.add(k2.scale(2.0)).add(k3.scale(2.0)).add(k4).scale(h / 6.0));
    }

    @SuppressWarnings("unchecked")
    private static IDoubleVector evalDynamics(Function<IDiffVector, IDiffVector> dynamics,
            IVector<Double> z) {
        IDiffVector zv = new RereDiffVector((IDoubleVector) z);
        return ((RereDiffVector) dynamics.apply(zv)).getValue();
    }

    private static Consumer<IDoubleVector> buildAdjointBackward(
            Function<IDiffVector, IDiffVector> dynamics, RereDiffVector z0r,
            List<IDoubleVector> trajectory, double t0, double t1, double dt) {
        return (gradOut) -> {
            IDoubleVector a = gradOut.copy();
            for (int i = trajectory.size() - 1; i > 0; i--) {
                double h = (i == trajectory.size() - 1) ? t1 - (t0 + (trajectory.size() - 2) * dt) : dt;
                IDoubleVector zi = trajectory.get(i);
                a = adjointRk4Step(dynamics, zi, a, -Math.abs(h));
            }
            z0r.accGrad(a);
        };
    }

    /**
     * Single RK4 adjoint step: integrates da/dt = -(∂f/∂z)ᵀ·a.
     * The adjoint equation is da/dt = -J(z)ᵀ·a, so we negate each VJP.
     */
    @SuppressWarnings("unchecked")
    private static IDoubleVector adjointRk4Step(Function<IDiffVector, IDiffVector> dynamics,
            IDoubleVector z, IDoubleVector a, double h) {
        IDoubleVector k1 = computeVJP(dynamics, z, a);
        if (k1 == null) return a;
        k1 = (IDoubleVector) k1.scale(-1.0);
        IDoubleVector k2 = computeVJP(dynamics, z, (IDoubleVector) a.add(k1.scale(h / 2.0)));
        if (k2 == null) return (IDoubleVector) a.add(k1.scale(h));
        k2 = (IDoubleVector) k2.scale(-1.0);
        IDoubleVector k3 = computeVJP(dynamics, z, (IDoubleVector) a.add(k2.scale(h / 2.0)));
        if (k3 == null) return (IDoubleVector) a.add(k2.scale(h));
        k3 = (IDoubleVector) k3.scale(-1.0);
        IDoubleVector k4 = computeVJP(dynamics, z, (IDoubleVector) a.add(k3.scale(h)));
        if (k4 == null) return (IDoubleVector) a.add(k3.scale(h));
        k4 = (IDoubleVector) k4.scale(-1.0);
        return (IDoubleVector) a.add(
            ((IDoubleVector) k1.add(k2.scale(2.0)))
                .add(k3.scale(2.0))
                .add(k4)
                .scale(h / 6.0));
    }

    /** Vector-Jacobian product J(z)ᵀ·a = ∂(dynamics(z))ᵀ·a via one local backward pass.
     *  Returns null if the gradient does not flow back to the input. */
    private static IDoubleVector computeVJP(Function<IDiffVector, IDiffVector> dynamics,
            IDoubleVector z, IDoubleVector a) {
        RereDiffVector zv = new RereDiffVector(z.copy());
        RereDiffVector fz = (RereDiffVector) dynamics.apply(zv);
        fz.backward(a.copy());
        return zv.getGradient();
    }

    // ==================== Adaptive (Dopri5) adjoint backward ====================

    /**
     * Build adjoint backward for adaptive integration (Dopri5 checkpoints).
     *
     * <p>Algorithm: for each interval [z_i, z_{i+1}], re-integrate forward from
     * checkpoint z_i using fine-grained RK4 sub-steps to get dense intermediate
     * states, then reverse the adjoint through those states. This recompute-from-
     * checkpoint strategy trades O(steps/k) compute for O(k) memory.</p>
     */
    private static Consumer<IDoubleVector> buildAdjointBackwardAdaptive(
            Function<IDiffVector, IDiffVector> dynamics, RereDiffVector z0r,
            List<IDoubleVector> trajectory, double t0, double t1,
            OdeintConfig config) {
        return (gradOut) -> {
            IDoubleVector a = gradOut.copy();
            int n = trajectory.size();
            double dt = (t1 - t0) / Math.max(n - 1, 1);
            int subSteps = config.checkpointInterval() > 0 ? config.checkpointInterval() : 4;
            double h = dt / subSteps;

            // Walk backward through checkpoints
            for (int i = n - 2; i >= 0; i--) {
                IDoubleVector zi = trajectory.get(i);

                // Re-integrate forward from checkpoint zi to zi+1 with sub-steps
                List<IDoubleVector> dense = new ArrayList<>();
                IDoubleVector y = zi.copy();
                dense.add(y.copy());
                for (int k = 0; k < subSteps; k++) {
                    y = rk4Step(dynamics, y, h);
                    dense.add(y.copy());
                }

                // Reverse adjoint through dense intermediate states
                for (int k = dense.size() - 1; k > 0; k--) {
                    a = adjointRk4Step(dynamics, dense.get(k - 1), a, -h);
                }
            }

            z0r.accGrad(a);
        };
    }
}
