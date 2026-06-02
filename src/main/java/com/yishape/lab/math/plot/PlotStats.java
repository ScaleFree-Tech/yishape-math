package com.yishape.lab.math.plot;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * 绘图统计辅助类 / Plot Statistical Helpers
 * <p>
 * 提供绘图层的统计辅助方法（OLS、正态分位数等）。
 * Small statistical helpers for plot layers (OLS, normal quantiles).
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public final class PlotStats {

    private PlotStats() {
    }

    /**
     * OLS mean-response confidence band at observed x (95%, two-sided t on residual df).
     */
    public static final class OlsMeanBand {
        public final IVector<Double> x;
        public final IVector<Double> yHat;
        public final IVector<Double> yLow;
        public final IVector<Double> yHigh;

        public OlsMeanBand(IVector<Double> x, IVector<Double> yHat,
                          IVector<Double> yLow, IVector<Double> yHigh) {
            this.x = x;
            this.yHat = yHat;
            this.yLow = yLow;
            this.yHigh = yHigh;
        }
    }

    /**
     * Two-sided t critical for 95% CI: t_{0.975, df}. Uses a short table + normal tail for large df.
     */
    public static double studentTCrit975(int df) {
        if (df < 1) {
            return Double.NaN;
        }
        if (df >= 240) {
            return normalPpf(0.975);
        }
        final double[] table = {
            0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
            2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
            2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
            2.040, 2.037, 2.035, 2.032, 2.030, 2.028, 2.026, 2.024, 2.023, 2.021,
            2.020, 2.018, 2.017, 2.015, 2.014, 2.013, 2.012, 2.010, 2.009, 2.008,
            2.007, 2.006, 2.005, 2.004, 2.003, 2.002, 2.001, 2.000, 1.999, 1.998,
            1.997, 1.996, 1.995, 1.995, 1.994, 1.993, 1.993, 1.992, 1.991, 1.990,
            1.990, 1.989, 1.988, 1.988, 1.987, 1.986, 1.986, 1.985, 1.985, 1.984,
            1.984, 1.983, 1.983, 1.982, 1.982, 1.981, 1.981, 1.980, 1.980, 1.980,
            1.979, 1.979, 1.978, 1.978, 1.978, 1.977, 1.977, 1.977, 1.976, 1.976,
            1.976, 1.975, 1.975, 1.975, 1.974, 1.974, 1.974, 1.974, 1.973, 1.973,
            1.973, 1.973, 1.972, 1.972, 1.972, 1.972, 1.972, 1.971, 1.971, 1.971,
            1.971, 1.971, 1.970, 1.970, 1.970, 1.970, 1.970, 1.970, 1.969, 1.969,
            1.969, 1.969, 1.969, 1.969, 1.969, 1.968, 1.968, 1.968, 1.968, 1.968,
            1.968, 1.968, 1.968, 1.967, 1.967, 1.967, 1.967, 1.967, 1.967, 1.967,
            1.967, 1.967, 1.967, 1.967, 1.966, 1.966, 1.966, 1.966, 1.966, 1.966,
            1.966, 1.966, 1.966, 1.966, 1.966, 1.966, 1.966, 1.966, 1.965, 1.965,
            1.965, 1.965, 1.965, 1.965, 1.965, 1.965, 1.965, 1.965, 1.965, 1.965,
            1.965, 1.965, 1.965, 1.965, 1.965, 1.965, 1.964, 1.964, 1.964, 1.964,
            1.964, 1.964, 1.964, 1.964, 1.964, 1.964, 1.964, 1.964, 1.964, 1.964,
            1.964, 1.964, 1.964, 1.964, 1.964, 1.963, 1.963, 1.963, 1.963, 1.963
        };
        return df < table.length ? table[df] : normalPpf(0.975);
    }

    /**
     * Simple least-squares regression y ≈ slope * x + intercept.
     *
     * @return { slope, intercept }
     */
    public static double[] ols(IVector<Double> x, IVector<Double> y) {
        int n = x.length();
        if (n != y.length() || n < 2) {
            return new double[] { 0.0, y.length() > 0 ? y.get(0) : 0.0 };
        }
        double mx = 0.0;
        double my = 0.0;
        for (int i = 0; i < n; i++) {
            mx += x.get(i);
            my += y.get(i);
        }
        mx /= n;
        my /= n;
        double sxy = 0.0;
        double sxx = 0.0;
        for (int i = 0; i < n; i++) {
            double dx = x.get(i) - mx;
            sxy += dx * (y.get(i) - my);
            sxx += dx * dx;
        }
        if (sxx == 0.0) {
            return new double[] { 0.0, my };
        }
        double slope = sxy / sxx;
        double intercept = my - slope * mx;
        return new double[] { slope, intercept };
    }

    /**
     * 95% confidence band for E[y|x] at each {@code x} point (requires n ≥ 3).
     * For n &lt; 3 returns empty band (zero-width intervals at fitted line).
     */
    public static OlsMeanBand olsMeanResponseBand95(IVector<Double> x, IVector<Double> y) {
        int n = x.length();
        if (x == null || y == null || n != y.length() || n < 3) {
            IVector<Double> empty = Linalg.vector(new double[0]);
            return new OlsMeanBand(empty, empty, empty, empty);
        }
        double[] coef = ols(x, y);
        double slope = coef[0];
        double icept = coef[1];
        double mx = 0.0;
        for (int i = 0; i < n; i++) {
            mx += x.get(i);
        }
        mx /= n;
        double sxx = 0.0;
        for (int i = 0; i < n; i++) {
            double dx = x.get(i) - mx;
            sxx += dx * dx;
        }
        double[] yHat = new double[n];
        double sse = 0.0;
        for (int i = 0; i < n; i++) {
            yHat[i] = slope * x.get(i) + icept;
            double r = y.get(i) - yHat[i];
            sse += r * r;
        }
        double[] lo = new double[n];
        double[] hi = new double[n];
        if (n < 3 || sxx <= 0) {
            for (int i = 0; i < n; i++) {
                lo[i] = yHat[i];
                hi[i] = yHat[i];
            }
            return new OlsMeanBand(x, Linalg.vector(yHat), Linalg.vector(lo), Linalg.vector(hi));
        }
        int df = n - 2;
        double mse = sse / df;
        double tc = studentTCrit975(df);
        for (int i = 0; i < n; i++) {
            double xi = x.get(i);
            double se = Math.sqrt(mse * (1.0 / n + (xi - mx) * (xi - mx) / sxx));
            lo[i] = yHat[i] - tc * se;
            hi[i] = yHat[i] + tc * se;
        }
        return new OlsMeanBand(x, Linalg.vector(yHat), Linalg.vector(lo), Linalg.vector(hi));
    }

    /**
     * Inverse of standard normal CDF (quantile function), 0 &lt; p &lt; 1.
     * Uses Acklam's rational approximation.
     */
    public static double normalPpf(double p) {
        if (p <= 0.0 || p >= 1.0) {
            throw new IllegalArgumentException("p must be in (0,1), got " + p);
        }
        final double[] a = {
            -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
            1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00
        };
        final double[] b = {
            -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
            6.680131188771972e+01, -1.328068155288572e+01
        };
        final double[] c = {
            -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
            -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00
        };
        final double[] d = {
            7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
            3.754408661907416e+00
        };
        double q = p - 0.5;
        double r;
        double x;
        if (Math.abs(q) <= 0.425) {
            r = 0.180625 - q * q;
            x = q * (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
        } else {
            r = q < 0 ? p : 1.0 - p;
            r = Math.sqrt(-Math.log(r));
            x = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5])
                / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0);
            if (q < 0) {
                x = -x;
            }
        }
        return x;
    }
}
