package com.yishape.lab.math.plot;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.ArrayList;
import java.util.List;

/**
 * 高斯核密度估计 / Gaussian Kernel Density Estimation
 * <p>
 * 在规则网格上计算高斯KDE（用于kdeplot和边缘分布）。
 * Gaussian KDE on a regular grid (for kdeplot / marginals).
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public final class PlotKde {

    private PlotKde() {
    }

    /**
     * Scott's rule bandwidth: 1.06 σ n^(-1/5) with sample std (ddof=1).
     */
    public static double scottBandwidth(IVector<Double> data) {
        int n = data.length();
        if (n < 2) {
            return 1.0;
        }
        double std = data.std(1).doubleValue();
        double iqr = data.q3().doubleValue() - data.q1().doubleValue();
        double sigma = Math.min(std, iqr / 1.34);
        if (sigma <= 0 || Double.isNaN(sigma)) {
            sigma = (data.max().doubleValue() - data.min().doubleValue()) / 10;
        }
        if (sigma <= 0) {
            sigma = 1e-6;
        }
        return 1.06 * sigma * Math.pow(n, -0.2);
    }

    /**
     * Evaluate KDE at grid points; returns list of (x, density).
     */
    public static List<double[]> evaluate(IVector<Double> data, double bandwidth, int gridPoints) {
        List<double[]> points = new ArrayList<>();
        double min = data.min().doubleValue();
        double max = data.max().doubleValue();
        double range = max - min;
        int n = data.length();
        if (range <= 0 || n < 2) {
            points.add(new double[] { min, 1.0 });
            return points;
        }
        int g = Math.max(32, gridPoints);
        double lo = min - 0.15 * range;
        double hi = max + 0.15 * range;
        double step = (hi - lo) / (g - 1);
        double h = bandwidth <= 0 ? scottBandwidth(data) : bandwidth;
        for (int k = 0; k < g; k++) {
            double x = lo + k * step;
            double density = 0;
            for (int i = 0; i < n; i++) {
                double u = (x - data.get(i)) / h;
                density += Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI);
            }
            density /= (n * h);
            points.add(new double[] { x, density });
        }
        return points;
    }

    public static IVector<Double>[] toVectors(List<double[]> pts) {
        int m = pts.size();
        double[] xs = new double[m];
        double[] ys = new double[m];
        for (int i = 0; i < m; i++) {
            xs[i] = pts.get(i)[0];
            ys[i] = pts.get(i)[1];
        }
        @SuppressWarnings("unchecked")
        IVector<Double>[] out = new IVector[] { Linalg.vector(xs), Linalg.vector(ys) };
        return out;
    }
}
