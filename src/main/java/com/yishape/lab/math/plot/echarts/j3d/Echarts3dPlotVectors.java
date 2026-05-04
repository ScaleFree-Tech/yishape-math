package com.yishape.lab.math.plot.echarts.j3d;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

import java.util.ArrayList;
import java.util.List;

/**
 * 与 JavaFX {@link com.yishape.lab.math.plot.javafx.j3d.JavaFx3dPlotMath} 类似，
 * 集中向量/矩阵到 double 数组的转换与小几何工具。
 *
 * @author lteb2
 */
final class Echarts3dPlotVectors {

    private Echarts3dPlotVectors() {
    }

    static double toDouble(Object obj) {
        if (obj instanceof Number) return ((Number) obj).doubleValue();
        return 0;
    }

    static double[][] extractXYZ(IVector x, IVector y, IVector z, int n) {
        double[][] data = new double[n][3];
        for (int i = 0; i < n; i++) {
            data[i][0] = toDouble(x.get(i));
            data[i][1] = toDouble(y.get(i));
            data[i][2] = toDouble(z.get(i));
        }
        return data;
    }

    static double[] range(IVector v) {
        double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < v.length(); i++) {
            double val = toDouble(v.get(i));
            min = Math.min(min, val);
            max = Math.max(max, val);
        }
        if (!Double.isFinite(min)) min = 0;
        if (!Double.isFinite(max) || max <= min) max = min + 1;
        return new double[]{min, max};
    }

    static double maxValue(IVector v) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < v.length(); i++) {
            max = Math.max(max, toDouble(v.get(i)));
        }
        return Double.isFinite(max) ? max : 0;
    }

    static double minValue(IVector v) {
        double min = Double.POSITIVE_INFINITY;
        for (int i = 0; i < v.length(); i++) {
            min = Math.min(min, toDouble(v.get(i)));
        }
        return Double.isFinite(min) ? min : 0;
    }

    static double maxValue(IMatrix m) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.cols(); j++) {
                max = Math.max(max, toDouble(m.get(i, j)));
            }
        }
        return Double.isFinite(max) ? max : 0;
    }

    static double[] matrixZBounds(IMatrix z) {
        double minZ = Double.POSITIVE_INFINITY, maxZ = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < z.rows(); i++) {
            for (int j = 0; j < z.cols(); j++) {
                double v = toDouble(z.get(i, j));
                minZ = Math.min(minZ, v);
                maxZ = Math.max(maxZ, v);
            }
        }
        return new double[]{minZ, maxZ};
    }

    static int clamp(int val, int min, int max) {
        return Math.max(min, Math.min(max, val));
    }

    static double[][] convertToPointGrid(IVector x, IVector y, double[][] zGrid) {
        int nx = x.length();
        int ny = y.length();
        List<double[]> points = new ArrayList<>();
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                points.add(new double[]{toDouble(x.get(i)), toDouble(y.get(j)), zGrid[i][j]});
            }
        }
        return points.toArray(new double[0][]);
    }

    static double[][] generateContourLinesSimple(IVector x, IVector y, double[][] z) {
        List<double[]> lines = new ArrayList<>();
        int nx = x.length();
        int ny = y.length();

        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny - 1; j++) {
                lines.add(new double[]{toDouble(x.get(i)), toDouble(y.get(j)), z[i][j]});
                lines.add(new double[]{toDouble(x.get(i)), toDouble(y.get(j + 1)), z[i][j + 1]});
            }
        }
        return lines.toArray(new double[0][]);
    }

    /** 逆向距离加权矢量（流线种子步进）。 */
    static double[] idwVelocity(double px, double py, double pz,
                                 IVector xs, IVector ys, IVector zs,
                                 IVector us, IVector vs, IVector ws) {
        int n = xs.length();
        double[] vel = new double[3];
        double wSum = 0;

        for (int i = 0; i < n; i++) {
            double dx = px - toDouble(xs.get(i));
            double dy = py - toDouble(ys.get(i));
            double dz = pz - toDouble(zs.get(i));
            double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < 1e-18) {
                return new double[]{toDouble(us.get(i)), toDouble(vs.get(i)), toDouble(ws.get(i))};
            }
            double w = 1.0 / d2;
            vel[0] += w * toDouble(us.get(i));
            vel[1] += w * toDouble(vs.get(i));
            vel[2] += w * toDouble(ws.get(i));
            wSum += w;
        }

        if (wSum > 0) {
            vel[0] /= wSum;
            vel[1] /= wSum;
            vel[2] /= wSum;
        }
        return vel;
    }

    static int asInt(Object obj, int defaultVal) {
        if (obj instanceof Number) return ((Number) obj).intValue();
        try {
            return Integer.parseInt(String.valueOf(obj));
        } catch (NumberFormatException e) {
            return defaultVal;
        }
    }
}
