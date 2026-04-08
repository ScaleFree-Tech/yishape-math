package com.yishape.lab.math.linalg;

import com.yishape.lab.math.util.NpyArrayIO;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.SplittableRandom;

import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.RereFFT;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 稠密数组工具：{@link IMatrix}/{@link IVector} 为广播与一维静态工具的首选入口；{@link IDoubleMatrix}/{@link IFloatMatrix}、{@link IDoubleVector}/{@link IFloatVector} 为同名委托。另测 {@link NpyArrayIO} 与 {@link Linalg} 工厂。
 */
class DenseDoubleArrayUtilitiesTest {

    private static double normal01(SplittableRandom rnd) {
        double u1 = rnd.nextDouble();
        double u2 = rnd.nextDouble();
        return Math.sqrt(-2.0 * Math.log(u1 + 1e-300)) * Math.cos(2.0 * Math.PI * u2);
    }

    @Test
    void broadcastAdd() {
        double[][] a = {{1}, {2}, {3}};
        double[][] b = {{10, 20, 30}};
        double[][] c = IMatrix.broadcastElementWise(a, b, Double::sum);
        assertEquals(3, c.length);
        assertEquals(11, c[0][0], 1e-12);
        assertEquals(22, c[1][1], 1e-12);
    }

    @Test
    void broadcastIncompatibleThrows() {
        assertThrows(IllegalArgumentException.class, () ->
                IMatrix.broadcastShape(2, 2, 3, 3));
    }

    @Test
    void matrixMulTransposeAndRowSumsViaIMatrix() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
        IMatrix<Double> b = Linalg.matrix(new double[][]{{2, 0}, {1, 2}});
        IMatrix<Double> prod = a.mmul(b);
        assertEquals(4, prod.get(0, 0), 1e-12);
        assertEquals(10, prod.get(1, 0), 1e-12);

        IMatrix<Double> t = Linalg.matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        IMatrix<Double> tr = t.transpose();
        assertEquals(1, tr.get(0, 0), 1e-12);
        assertEquals(4, tr.get(0, 1), 1e-12);

        IVector<Double> rs = t.rowSums();
        assertEquals(6, rs.get(0), 1e-12);
        assertEquals(15, rs.get(1), 1e-12);
    }

    @Test
    void npyRoundTrip2D() throws IOException {
        double[][] data = {{1, 2, 3}, {4, 5, 6}};
        byte[] raw = NpyArrayIO.toByteArrayDouble2D(data);
        double[][] back = NpyArrayIO.readDouble2D(new ByteArrayInputStream(raw));
        assertArrayEquals(data[0], back[0], 1e-15);
    }

    @Test
    void npyMatrixApi() throws IOException {
        IMatrix<Double> mx = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
        IMatrix<Double> roundTrip = NpyArrayIO.fromByteArray(NpyArrayIO.toByteArray(mx));
        assertEquals(1, roundTrip.get(0, 0), 1e-15);
        assertEquals(4, roundTrip.get(1, 1), 1e-15);
    }

    @Test
    void rfftIrfft() {
        double[] x = {1, 2, 3, 4};
        Complex[] y = RereFFT.rfft(x);
        double[] z = RereFFT.irfft(y, x.length);
        for (int i = 0; i < x.length; i++) {
            assertEquals(x[i], z[i], 1e-10);
        }
    }

    @Test
    void splittableNormalReproducible() {
        assertEquals(normal01(new SplittableRandom(42L)), normal01(new SplittableRandom(42L)), 1e-15);
    }

    @Test
    void polyfitDigitizeHistogramWhere() {
        double[] coef = IVector.polyfit(new double[]{0, 1, 2, 3}, new double[]{1, 3, 5, 7}, 1);
        assertEquals(2, coef[0], 1e-10);
        assertEquals(1, coef[1], 1e-10);

        int[] d = IVector.digitize(new double[]{-1, 0.5, 1.5, 2.5}, new double[]{0, 1, 2});
        assertEquals(0, d[0]);
        assertEquals(3, d[3]);

        var h = IVector.histogram(new double[]{0, 0.5, 1, 1.5, 2}, 2);
        assertEquals(5, h.counts[0] + h.counts[1]);

        boolean[] c = {true, false, true};
        assertArrayEquals(new double[]{1, 0, 1}, IVector.where(c, 1.0, 0.0));
    }

    @Test
    void vectorIndexingUsesIVector() {
        IVector<Double> v = Linalg.vector(new double[]{10, 20, 30, 40});
        assertArrayEquals(new double[]{10, 30},
                v.booleanGet(new boolean[]{true, false, true, false}).toDoubleArray());
        assertEquals(40, v.fancyGet(new int[]{-1}).get(0), 1e-12);
        assertArrayEquals(new double[]{1, 1, 2, 2},
                Linalg.vector(new double[]{1, 2}).repeat(2).toDoubleArray());
    }

    @Test
    void linalgBroadcastMatrixApi() {
        IMatrix<Double> ma = Linalg.matrix(new double[][]{{1}, {2}, {3}});
        IMatrix<Double> mb = Linalg.matrix(new double[][]{{10, 20, 30}});
        assertArrayEquals(new int[]{3, 3}, IMatrix.broadcastShape(ma, mb));
        IMatrix<Double> mc = IMatrix.broadcastElementWise(ma, mb, Double::sum);
        assertEquals(11, mc.get(0, 0), 1e-12);

        IVector<Double> coef = (IVector<Double>) IVector.polyfit(
                Linalg.vector(new double[]{0, 1, 2, 3}),
                Linalg.vector(new double[]{1, 3, 5, 7}), 1);
        assertEquals(2, coef.get(0), 1e-10);
    }

    @Test
    void floatBroadcastAndVectorToolsDelegateToDouble() {
        float[][] a = {{1}, {2}, {3}};
        float[][] b = {{10, 20, 30}};
        float[][] c = IMatrix.broadcastElementWise(a, b, Double::sum);
        assertEquals(3, c.length);
        assertEquals(11f, c[0][0], 1e-5f);
        assertEquals(22f, c[1][1], 1e-5f);

        assertThrows(IllegalArgumentException.class, () ->
                IMatrix.broadcastShape(2, 2, 3, 3));

        IMatrix<Float> ma = Linalg.matrix(new float[][]{{1}, {2}, {3}});
        IMatrix<Float> mb = Linalg.matrix(new float[][]{{10, 20, 30}});
        assertArrayEquals(new int[]{3, 3}, IMatrix.broadcastShape(ma, mb));
        IMatrix<Float> mc = IFloatMatrix.broadcastElementWise(ma, mb, Double::sum);
        assertEquals(11f, mc.get(0, 0), 1e-5f);

        float[] coef = IVector.polyfit(new float[]{0, 1, 2, 3}, new float[]{1, 3, 5, 7}, 1);
        assertEquals(2f, coef[0], 1e-5f);
        assertEquals(1f, coef[1], 1e-5f);

        int[] d = IVector.digitize(new float[]{-1, 0.5f, 1.5f, 2.5f}, new float[]{0, 1, 2});
        assertEquals(0, d[0]);
        assertEquals(3, d[3]);

        var h = IVector.histogram(new float[]{0, 0.5f, 1, 1.5f, 2}, 2);
        assertEquals(5, h.counts[0] + h.counts[1]);

        boolean[] cond = {true, false, true};
        assertArrayEquals(new float[]{1, 0, 1}, IVector.where(cond, 1f, 0f), 1e-6f);

        IVector<Float> p = (IVector<Float>) IVector.polyfit(
                Linalg.vector(new float[]{0, 1, 2, 3}),
                Linalg.vector(new float[]{1, 3, 5, 7}), 1);
        assertEquals(2f, p.get(0), 1e-5f);
    }
}
