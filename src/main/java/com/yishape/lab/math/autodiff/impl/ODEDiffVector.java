package com.yishape.lab.math.autodiff.impl;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.ode.IRungeKuttaIntegrator;
import com.yishape.lab.math.optimize.ode.impl.RereRK4Integrator;
import com.yishape.lab.math.autodiff.IDiffVector;

/**
 * Differentiable Neural ODE layer: RK4 forward integration and adjoint backward.
 * 可微 Neural ODE 层：RK4 前向积分与伴随法反向。
 *
 * <p>Stores the state trajectory during forward; backward walks adjoint equations in reverse time.
 * 前向保存状态轨迹；反向沿伴随方程倒序积分。</p>
 */
public class ODEDiffVector extends RereDiffVector {

    private static final long serialVersionUID = 1L;

    public ODEDiffVector(Function<IDiffVector, IDiffVector> dynamics, IDiffVector z0,
            double t0, double t1, double dt) {
        super(IDoubleVector.zeros(1)); // placeholder, will overwrite
        RereDiffVector z0r = (RereDiffVector) z0;
        List<IDoubleVector> trajectory = new ArrayList<>();
        IDoubleVector y0 = z0r.getValue().copy();
        trajectory.add(y0.copy());
        IDoubleVector y = integrateForward(dynamics, y0, t0, t1, dt, trajectory);
        // Overwrite the placeholder tensor data
        this.tensor.value = new com.yishape.lab.math.linalg.tensor.RereDoubleTensor(
            y.getData(), new int[]{y.size()});
        this.tensor.isLeaf = false;
        this.tensor.inputs.add(z0r.tensor);
        // Wrap the adjoint backward for the tensor graph
        Consumer<IDoubleVector> vecBackwardFn = buildAdjointBackward(dynamics, z0r, trajectory, t0, t1, dt);
        this.tensor.backwardFn = (selfTensor) -> {
            vecBackwardFn.accept(IDoubleVector.of(selfTensor.grad));
        };
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

    private static IDoubleVector adjointRk4Step(Function<IDiffVector, IDiffVector> dynamics,
            IDoubleVector z, IDoubleVector a, double h) {
        IDoubleVector k1 = computeVJP(dynamics, z, a);
        IDoubleVector k2 = computeVJP(dynamics, z, a.add(k1.scale(h / 2.0)));
        IDoubleVector k3 = computeVJP(dynamics, z, a.add(k2.scale(h / 2.0)));
        IDoubleVector k4 = computeVJP(dynamics, z, a.add(k3.scale(h)));
        return a.add(k1.add(k2.scale(2.0)).add(k3.scale(2.0)).add(k4).scale(h / 6.0));
    }

    /** Vector-Jacobian product ∂(dynamics(z))ᵀ·a via one local backward pass. / 单次局部反向求 VJP。 */
    private static IDoubleVector computeVJP(Function<IDiffVector, IDiffVector> dynamics,
            IDoubleVector z, IDoubleVector a) {
        RereDiffVector zv = new RereDiffVector(z.copy());
        RereDiffVector fz = (RereDiffVector) dynamics.apply(zv);
        fz.backward(a.copy());
        return zv.getGradient();
    }
}
