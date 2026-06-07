package com.yishape.lab.math.autodiff.support;

/**
 * Configuration for Neural ODE integration (RK4 or adaptive Dopri5).
 *
 * @param method            "rk4" or "dopri5"
 * @param absTol            absolute tolerance (Dopri5 only)
 * @param relTol            relative tolerance (Dopri5 only)
 * @param minDt             minimum step size (Dopri5 only)
 * @param maxDt             maximum step size (Dopri5 only)
 * @param initialDt         initial step size (Dopri5 only)
 * @param maxSteps          maximum number of steps (Dopri5 only)
 * @param checkpointInterval how many steps between saved checkpoints (1 = every step)
 */
public record OdeintConfig(String method, double absTol, double relTol,
        double minDt, double maxDt, double initialDt, int maxSteps, int checkpointInterval) {

    /** Default RK4 configuration (non-adaptive). */
    public static OdeintConfig rk4() {
        return new OdeintConfig("rk4", 0, 0, 0, 0, 0, 0, 1);
    }

    /** Dopri5 with default tolerances: absTol=1e-9, relTol=1e-6. */
    public static OdeintConfig dopri5() {
        return new OdeintConfig("dopri5", 1e-9, 1e-6,
                1e-8, Double.POSITIVE_INFINITY, 0.01, 10_000, 8);
    }

    /** Dopri5 with custom tolerances. */
    public static OdeintConfig dopri5(double absTol, double relTol) {
        return new OdeintConfig("dopri5", absTol, relTol,
                1e-8, Double.POSITIVE_INFINITY, 0.01, 10_000, 8);
    }

    /** Full control Dopri5 constructor. */
    public static OdeintConfig dopri5(double absTol, double relTol,
            double minDt, double maxDt, double initialDt, int maxSteps, int checkpointInterval) {
        return new OdeintConfig("dopri5", absTol, relTol,
                minDt, maxDt, initialDt, maxSteps, checkpointInterval);
    }

    public boolean isAdaptive() { return "dopri5".equals(method); }

    /** Effective absolute tolerance (alias for absTol). */
    public double effectiveAbsTol() { return absTol; }
    /** Effective relative tolerance (alias for relTol). */
    public double effectiveRelTol() { return relTol; }
    /** Effective maximum step size (alias for maxDt). */
    public double effectiveMaxDt() { return maxDt; }
}
