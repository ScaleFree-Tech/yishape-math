package com.yishape.lab.math.optimize.ode;

import java.util.function.Function;

import com.yishape.lab.math.linalg.IDoubleVector;

/**
 * Ordinary differential equation (ODE) integrator.
 * 常微分方程（ODE）数值积分器。
 *
 * <p>Solves {@code dy/dt = f(t, y)} from {@code t0} to {@code t1} with fixed step {@code dt}
 * (last step may be shorter). Implementations typically use classical Runge-Kutta (RK4).
 *
 * <p>从 {@code t0} 积分到 {@code t1}，步长为 {@code dt}（末步可能更短）。
 * 实现类通常采用经典四阶 Runge-Kutta（RK4）。</p>
 */
public interface IRungeKuttaIntegrator {

    /**
     * Integrates the ODE and returns the state at {@code t1}.
     * 积分 ODE 并返回 {@code t1} 时刻的状态。
     *
     * @param f dynamics {@code dy/dt = f(y)} (time-independent) / 动力学函数
     * @param y0 initial state / 初值
     * @param t0 start time / 起始时刻
     * @param t1 end time / 终止时刻
     * @param dt step size / 步长
     * @return state at {@code t1} / {@code t1} 时刻状态
     */
    IDoubleVector integrate(Function<IDoubleVector, IDoubleVector> f, IDoubleVector y0,
            double t0, double t1, double dt);
}
