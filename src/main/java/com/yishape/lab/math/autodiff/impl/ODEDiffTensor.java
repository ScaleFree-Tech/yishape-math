package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.ITensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Differentiable Neural ODE layer (tensor version): RK4 forward integration and adjoint backward.
 * 可微 Neural ODE 层（张量版）：RK4 前向积分与伴随法反向。
 *
 * <p>Stores the state trajectory during forward; backward walks adjoint equations in reverse time.
 * Supports any N-D tensor primal type (batches of features, images, etc.).
 */
public class ODEDiffTensor extends RereDiffTensor {

    public ODEDiffTensor(Function<IDiffTensor, IDiffTensor> dynamics, IDiffTensor z0,
                         double t0, double t1, double dt) {
        super(new double[1], 1); // placeholder, will overwrite in body
        RereDiffTensor z0r = (RereDiffTensor) z0;
        List<IDoubleTensor> trajectory = new ArrayList<>();
        IDoubleTensor y0 = z0r.value.copy();
        trajectory.add(y0.copy());
        IDoubleTensor y = integrateForward(dynamics, y0, t0, t1, dt, trajectory);
        this.value = (RereDoubleTensor) y;
        this.isLeaf = false;
        this.inputs = new ArrayList<>();
        this.inputs.add(z0r);
        this.backwardFn = buildAdjointBackward(dynamics, z0r, trajectory, t0, t1, dt);
    }

    private static IDoubleTensor integrateForward(Function<IDiffTensor, IDiffTensor> dynamics,
                                                  IDoubleTensor y0, double t0, double t1, double dt,
                                                  List<IDoubleTensor> trajectory) {
        IDoubleTensor y = y0;
        double t = t0;
        while (t < t1) {
            double h = Math.min(dt, t1 - t);
            y = rk4Step(dynamics, y, h);
            trajectory.add(y.copy());
            t += h;
        }
        return y;
    }

    private static IDoubleTensor rk4Step(Function<IDiffTensor, IDiffTensor> dynamics,
                                         IDoubleTensor y, double h) {
        IDoubleTensor k1 = evalDynamics(dynamics, y);
        IDoubleTensor k2 = evalDynamics(dynamics, y.add(k1.mul(h / 2.0)));
        IDoubleTensor k3 = evalDynamics(dynamics, y.add(k2.mul(h / 2.0)));
        IDoubleTensor k4 = evalDynamics(dynamics, y.add(k3.mul(h)));
        return y.add(k1.add(k2.mul(2.0)).add(k3.mul(2.0)).add(k4).mul(h / 6.0));
    }

    /** Evaluate dynamics as a pure tensor function (no gradient tracking). */
    private static IDoubleTensor evalDynamics(Function<IDiffTensor, IDiffTensor> dynamics,
                                              IDoubleTensor z) {
        RereDiffTensor zv = new RereDiffTensor(z.toDoubleArray(), z.shape());
        return ((RereDiffTensor) dynamics.apply(zv)).value;
    }

    private static Consumer<RereDiffTensor> buildAdjointBackward(
            Function<IDiffTensor, IDiffTensor> dynamics, RereDiffTensor z0r,
            List<IDoubleTensor> trajectory, double t0, double t1, double dt) {
        return (self) -> {
            IDoubleTensor a = ITensor.tensor(self.grad.clone(), trajectory.get(0).shape());
            for (int i = trajectory.size() - 1; i > 0; i--) {
                double h = (i == trajectory.size() - 1)
                    ? t1 - (t0 + (trajectory.size() - 2) * dt) : dt;
                IDoubleTensor zi = trajectory.get(i);
                a = adjointRk4Step(dynamics, zi, a, -Math.abs(h));
            }
            z0r.accGrad(a.toDoubleArray());
        };
    }

    private static IDoubleTensor adjointRk4Step(Function<IDiffTensor, IDiffTensor> dynamics,
                                                IDoubleTensor z, IDoubleTensor a, double h) {
        // Adjoint equation: da/dt = -J^T*a = -VJP.  The RK4 assumes dz/dt = f(z),
        // so we negate h to account for the extra negative in the adjoint.
        IDoubleTensor k1 = computeVJP(dynamics, z, a);
        IDoubleTensor k2 = computeVJP(dynamics, z, a.add(k1.mul(-h / 2.0)));
        IDoubleTensor k3 = computeVJP(dynamics, z, a.add(k2.mul(-h / 2.0)));
        IDoubleTensor k4 = computeVJP(dynamics, z, a.add(k3.mul(-h)));
        return a.add(k1.add(k2.mul(2.0)).add(k3.mul(2.0)).add(k4).mul(-h / 6.0));
    }

    /** Vector-Jacobian product ∂(dynamics(z))ᵀ·a via one local backward pass. / 单次局部反向求 VJP。 */
    private static IDoubleTensor computeVJP(Function<IDiffTensor, IDiffTensor> dynamics,
                                            IDoubleTensor z, IDoubleTensor a) {
        RereDiffTensor zv = new RereDiffTensor(z.toDoubleArray(), z.shape());
        RereDiffTensor fz = (RereDiffTensor) dynamics.apply(zv);
        fz.grad = a.toDoubleArray();  // set gradient directly (no ThreadLocal state)
        fz.propagateGrad();           // backward with local collections (safe for nested calls)
        double[] grad = zv.grad;
        zv.grad = null;               // clean up to avoid memory leak
        zv.requiresGrad = false;      // detach local graph
        return ITensor.tensor(grad, z.shape());
    }
}
