package com.yishape.lab.math.optimize.ode.impl;

import java.util.function.Function;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.optimize.ode.IRungeKuttaIntegrator;

/**
 * Classical fourth-order Runge-Kutta (RK4) ODE integrator.
 * 经典四阶 Runge-Kutta（RK4）常微分方程积分器。
 *
 * <p>Fixed-step explicit RK4: {@code y_{n+1} = y_n + h/6·(k1 + 2k2 + 2k3 + k4)}.
 * 固定步长显式 RK4 公式。</p>
 */
public class RereRK4Integrator implements IRungeKuttaIntegrator {

    @Override
    public IDoubleVector integrate(Function<IDoubleVector, IDoubleVector> f, IDoubleVector y0,
            double t0, double t1, double dt) {
        double t = t0;
        IDoubleVector y = y0.copy();
        while (t < t1) {
            double h = Math.min(dt, t1 - t); // shrink last step to land exactly on t1 / 末步缩至 t1
            IDoubleVector k1 = f.apply(y);
            IDoubleVector k2 = f.apply(y.add(k1.scale(h / 2.0)));
            IDoubleVector k3 = f.apply(y.add(k2.scale(h / 2.0)));
            IDoubleVector k4 = f.apply(y.add(k3.scale(h)));
            y = y.add(k1.add(k2.scale(2.0)).add(k3.scale(2.0)).add(k4).scale(h / 6.0));
            t += h;
        }
        return y;
    }
}
