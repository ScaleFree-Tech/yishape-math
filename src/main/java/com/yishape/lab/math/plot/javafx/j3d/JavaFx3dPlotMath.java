package com.yishape.lab.math.plot.javafx.j3d;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

import java.util.List;
import java.util.Map;

/**
 * 三维绘图用到的向量、矩阵与布局静态工具（与 JavaFX 解耦）。
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public final class JavaFx3dPlotMath {

    private JavaFx3dPlotMath() {
    }

    public static int len(IVector<?> v) {
        return v == null ? 0 : v.length();
    }

    public static void requireSameLength(int n, IVector<?>... vs) {
        for (IVector<?> v : vs) {
            if (v == null || v.length() != n) {
                throw new IllegalArgumentException("向量长度须一致 / vector lengths must match");
            }
        }
    }

    public static double minVec(IVector<?> v) {
        double m = Double.POSITIVE_INFINITY;
        for (int i = 0; i < v.length(); i++) {
            m = Math.min(m, v.get(i));
        }
        return m;
    }

    public static double maxVec(IVector<?> v) {
        double m = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < v.length(); i++) {
            m = Math.max(m, v.get(i));
        }
        return m;
    }

    public static void padRange(double[] minmax) {
        double lo = minmax[0];
        double hi = minmax[1];
        double p = 0.05 * (hi - lo + 1e-9);
        minmax[0] = lo - p;
        minmax[1] = hi + p;
    }

    public static int clampBin(double t, int g) {
        int i = (int) Math.floor(t);
        return clampInt(i, 0, g - 1);
    }

    public static int clampInt(int v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    public static int asInt(Object o, int d) {
        if (o instanceof Number) {
            return ((Number) o).intValue();
        }
        return d;
    }

    public static double[] idwVel(double px, double py, double pz,
            IVector<?> xs, IVector<?> ys, IVector<?> zs,
            IVector<?> us, IVector<?> vs, IVector<?> ws) {
        double sx = 0, sy = 0, sz = 0, sw = 0;
        int n = xs.length();
        for (int i = 0; i < n; i++) {
            double dx = px - xs.get(i);
            double dy = py - ys.get(i);
            double dz = pz - zs.get(i);
            double d2 = dx * dx + dy * dy + dz * dz + 1e-9;
            double wgt = 1.0 / d2;
            sx += wgt * us.get(i);
            sy += wgt * vs.get(i);
            sz += wgt * ws.get(i);
            sw += wgt;
        }
        return new double[] {sx / sw, sy / sw, sz / sw};
    }

    public static void springLayout3d(double[] px, double[] py, double[] pz, List<Map<String, Object>> links, int n, double area) {
        double k = Math.sqrt(area / Math.max(1, n));
        int iter = Math.min(300, 80 + n * 3);
        for (int it = 0; it < iter; it++) {
            double[] fx = new double[n];
            double[] fy = new double[n];
            double[] fz = new double[n];
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    double dx = px[j] - px[i];
                    double dy = py[j] - py[i];
                    double dz = pz[j] - pz[i];
                    double dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.01;
                    double f = k * k / dist;
                    fx[i] -= f * dx / dist;
                    fy[i] -= f * dy / dist;
                    fz[i] -= f * dz / dist;
                    fx[j] += f * dx / dist;
                    fy[j] += f * dy / dist;
                    fz[j] += f * dz / dist;
                }
            }
            if (links != null) {
                for (Map<String, Object> L : links) {
                    int i = asInt(L.get("source"), -1);
                    int j = asInt(L.get("target"), -1);
                    if (i < 0 || j < 0 || i >= n || j >= n) {
                        continue;
                    }
                    double dx = px[j] - px[i];
                    double dy = py[j] - py[i];
                    double dz = pz[j] - pz[i];
                    double dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.01;
                    double f = (dist - k) * 0.04;
                    double ux = dx / dist, uy = dy / dist, uz = dz / dist;
                    fx[i] += f * ux;
                    fy[i] += f * uy;
                    fz[i] += f * uz;
                    fx[j] -= f * ux;
                    fy[j] -= f * uy;
                    fz[j] -= f * uz;
                }
            }
            double t = 0.12 / (1 + it * 0.01);
            for (int i = 0; i < n; i++) {
                px[i] += t * fx[i];
                py[i] += t * fy[i];
                pz[i] += t * fz[i];
            }
        }
    }

    public static double zMinMatrix(IMatrix<?> z) {
        double m = Double.POSITIVE_INFINITY;
        for (int i = 0; i < z.rows(); i++) {
            for (int j = 0; j < z.cols(); j++) {
                m = Math.min(m, z.get(i, j));
            }
        }
        return m;
    }

    public static double zMaxMatrix(IMatrix<?> z) {
        double m = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < z.rows(); i++) {
            for (int j = 0; j < z.cols(); j++) {
                m = Math.max(m, z.get(i, j));
            }
        }
        return m;
    }
}
