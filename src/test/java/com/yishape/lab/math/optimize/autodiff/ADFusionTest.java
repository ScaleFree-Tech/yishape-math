package com.yishape.lab.math.optimize.autodiff;

import com.yishape.lab.math.autodiff.IDiffMatrix;
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffVector;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

public class ADFusionTest {

    private static final double TOL = 1e-10;

    // ================ scalar-parameter fusion ================

    @Test
    void testFusedPow() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector y = AD.fuse(x).pow(3).compute();
        IDiffVector loss = y.sum();
        loss.backward();

        double[] result = y.getValue().getData();
        assertArrayEquals(new double[] { 1, 8, 27 }, result, TOL);
        assertArrayEquals(new double[] { 3, 12, 27 }, x.getGradient().getData(), TOL);
    }

    @Test
    void testFusedAddScalar() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector y = AD.fuse(x).add(5).compute();
        IDiffVector loss = y.sum();
        loss.backward();

        assertArrayEquals(new double[] { 6, 7, 8 }, y.getValue().getData(), TOL);
        assertArrayEquals(new double[] { 1, 1, 1 }, x.getGradient().getData(), TOL);
    }

    @Test
    void testFusedSubScalar() {
        IDiffVector x = AD.vector(new double[] { 5.0, 7.0, 9.0 });
        IDiffVector y = AD.fuse(x).sub(2).compute();
        IDiffVector loss = y.sum();
        loss.backward();

        assertArrayEquals(new double[] { 3, 5, 7 }, y.getValue().getData(), TOL);
        assertArrayEquals(new double[] { 1, 1, 1 }, x.getGradient().getData(), TOL);
    }

    @Test
    void testFusedMulScalar() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector y = AD.fuse(x).mul(3).compute();
        IDiffVector loss = y.sum();
        loss.backward();

        assertArrayEquals(new double[] { 3, 6, 9 }, y.getValue().getData(), TOL);
        assertArrayEquals(new double[] { 3, 3, 3 }, x.getGradient().getData(), TOL);
    }

    @Test
    void testFusedDivScalar() {
        IDiffVector x = AD.vector(new double[] { 6.0, 8.0, 10.0 });
        IDiffVector y = AD.fuse(x).div(2).compute();
        IDiffVector loss = y.sum();
        loss.backward();

        assertArrayEquals(new double[] { 3, 4, 5 }, y.getValue().getData(), TOL);
        assertArrayEquals(new double[] { 0.5, 0.5, 0.5 }, x.getGradient().getData(), TOL);
    }

    @Test
    void testFusedMixedScalarAndElemOps() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector y = AD.fuse(x).mul(2).add(1).exp().div(3).compute();
        IDiffVector loss = y.sum();
        loss.backward();

        double[] autoGrad = x.getGradient().getData();
        double eps = 1e-6;
        double[] xData = { 1.0, 2.0, 3.0 };
        for (int i = 0; i < xData.length; i++) {
            xData[i] += eps;
            double fp = mixedScalarEval(xData);
            xData[i] -= 2 * eps;
            double fm = mixedScalarEval(xData);
            xData[i] += eps;
            double numGrad = (fp - fm) / (2 * eps);
            assertEquals(numGrad, autoGrad[i], 1e-5);
        }
    }

    private double mixedScalarEval(double[] x) {
        double s = 0;
        for (double v : x) {
            s += Math.exp(v * 2 + 1) / 3;
        }
        return s;
    }

    @Test
    void testFusedScalarOpsVsUnfused() {
        double[] xData = { 1.0, 2.0, 3.0 };

        IDiffVector x1 = AD.vector(xData.clone());
        IDiffVector y1 = AD.fuse(x1).mul(2).add(1).pow(3).div(4).compute();
        IDiffVector loss1 = y1.sum();
        loss1.backward();
        double[] fusedGrad = x1.getGradient().getData().clone();

        IDiffVector x2 = AD.vector(xData.clone());
        IDiffVector y2 = x2.mul(2).add(1).pow(3).div(4);
        IDiffVector loss2 = y2.sum();
        loss2.backward();
        double[] unfusedGrad = x2.getGradient().getData().clone();

        assertArrayEquals(unfusedGrad, fusedGrad, 1e-10);
    }

    @Test
    void testFusedPowNumericalGradient() {
        IDiffVector x = AD.vector(new double[] { 0.5, 1.0, 1.5, 2.0 });
        IDiffVector y = AD.fuse(x).pow(2.5).exp().compute();
        IDiffVector loss = y.sum();
        loss.backward();

        double[] autoGrad = x.getGradient().getData();
        double eps = 1e-6;
        double[] xData = { 0.5, 1.0, 1.5, 2.0 };
        for (int i = 0; i < xData.length; i++) {
            xData[i] += eps;
            double fp = powExpEval(xData);
            xData[i] -= 2 * eps;
            double fm = powExpEval(xData);
            xData[i] += eps;
            double numGrad = (fp - fm) / (2 * eps);
            assertEquals(numGrad, autoGrad[i], 1e-5);
        }
    }

    private double powExpEval(double[] x) {
        double s = 0;
        for (double v : x) {
            s += Math.exp(Math.pow(v, 2.5));
        }
        return s;
    }

    // ================ matrix element-wise fusion ================

    @Test
    void testMatrixFusedExpLog() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        IDiffMatrix Z = AD.fuseMatrix(A).exp().log().compute();
        double[][] result = Z.getValue().getData();

        assertEquals(1.0, result[0][0], TOL);
        assertEquals(2.0, result[0][1], TOL);
        assertEquals(3.0, result[1][0], TOL);
        assertEquals(4.0, result[1][1], TOL);
    }

    @Test
    void testMatrixFusedSqrtSquare() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, 4 }, { 9, 16 } });
        IDiffMatrix Z = AD.fuseMatrix(A).sqrt().square().compute();
        double[][] result = Z.getValue().getData();

        assertEquals(1.0, result[0][0], TOL);
        assertEquals(4.0, result[0][1], TOL);
        assertEquals(9.0, result[1][0], TOL);
        assertEquals(16.0, result[1][1], TOL);
    }

    @Test
    void testMatrixFusedEmpty() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        IDiffMatrix Z = AD.fuseMatrix(A).compute();

        assertEquals(1.0, Z.getValue().get(0, 0), TOL);
        assertEquals(2.0, Z.getValue().get(0, 1), TOL);
        assertEquals(3.0, Z.getValue().get(1, 0), TOL);
        assertEquals(4.0, Z.getValue().get(1, 1), TOL);
    }

    @Test
    void testMatrixFusedGradient() {
        IDiffMatrix A = AD.matrix(new double[][] { { 0.5, 1.0 }, { 1.5, 2.0 } });
        IDiffMatrix Z = AD.fuseMatrix(A).square().exp().compute();
        IDiffMatrix loss = Z.sum();
        loss.backward();

        double[][] autoGrad = A.getGradient().getData();
        double eps = 1e-6;
        double[][] aData = { { 0.5, 1.0 }, { 1.5, 2.0 } };
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                aData[i][j] += eps;
                double fp = matrixFusedEval(aData);
                aData[i][j] -= 2 * eps;
                double fm = matrixFusedEval(aData);
                aData[i][j] += eps;
                double numGrad = (fp - fm) / (2 * eps);
                assertEquals(numGrad, autoGrad[i][j], 1e-5);
            }
        }
    }

    private double matrixFusedEval(double[][] a) {
        double s = 0;
        for (double[] row : a) {
            for (double v : row) {
                double r = v * v;
                r = Math.exp(r);
                s += r;
            }
        }
        return s;
    }

    @Test
    void testMatrixFusedVsUnfused() {
        double[][] aData = { { 1, 2 }, { 3, 4 } };

        IDiffMatrix A1 = AD.matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        IDiffMatrix Z1 = AD.fuseMatrix(A1).exp().log().sqrt().square().compute();
        IDiffMatrix loss1 = Z1.sum();
        loss1.backward();
        double[][] fusedGrad = new double[2][2];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                fusedGrad[i][j] = A1.getGradient().get(i, j);

        IDiffMatrix A2 = AD.matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        IDiffMatrix Z2 = A2.exp().log().sqrt().square();
        IDiffMatrix loss2 = Z2.sum();
        loss2.backward();
        double[][] unfusedGrad = new double[2][2];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                unfusedGrad[i][j] = A2.getGradient().get(i, j);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(unfusedGrad[i][j], fusedGrad[i][j], 1e-10);
            }
        }
    }

    @Test
    void testMatrixFusedRelu() {
        IDiffMatrix A = AD.matrix(new double[][] { { -2, 0 }, { 1, 3 } });
        IDiffMatrix Z = AD.fuseMatrix(A).relu().compute();
        IDiffMatrix loss = Z.sum();
        loss.backward();

        double[][] result = Z.getValue().getData();
        assertEquals(0.0, result[0][0], TOL);
        assertEquals(0.0, result[0][1], TOL);
        assertEquals(1.0, result[1][0], TOL);
        assertEquals(3.0, result[1][1], TOL);

        double[][] gradA = A.getGradient().getData();
        assertEquals(0.0, gradA[0][0], TOL);
        assertEquals(0.0, gradA[0][1], TOL);
        assertEquals(1.0, gradA[1][0], TOL);
        assertEquals(1.0, gradA[1][1], TOL);
    }

    @Test
    void testMatrixFusedSigmoid() {
        IDiffMatrix A = AD.matrix(new double[][] { { 0, 1 }, { -1, 2 } });
        IDiffMatrix Z = AD.fuseMatrix(A).sigmoid().compute();
        IDiffMatrix loss = Z.sum();
        loss.backward();

        double[][] gradA = A.getGradient().getData();
        double s0 = 1.0 / (1.0 + 1.0);
        double s1 = 1.0 / (1.0 + Math.exp(-1));
        double s_1 = 1.0 / (1.0 + Math.exp(1));
        double s2 = 1.0 / (1.0 + Math.exp(-2));
        assertEquals(s0 * (1 - s0), gradA[0][0], TOL);
        assertEquals(s1 * (1 - s1), gradA[0][1], TOL);
        assertEquals(s_1 * (1 - s_1), gradA[1][0], TOL);
        assertEquals(s2 * (1 - s2), gradA[1][1], TOL);
    }

    @Test
    void testMatrixFusedPow() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        IDiffMatrix Z = AD.fuseMatrix(A).pow(3).compute();
        IDiffMatrix loss = Z.sum();
        loss.backward();

        double[][] result = Z.getValue().getData();
        assertEquals(1.0, result[0][0], TOL);
        assertEquals(8.0, result[0][1], TOL);
        assertEquals(27.0, result[1][0], TOL);
        assertEquals(64.0, result[1][1], TOL);

        double[][] gradA = A.getGradient().getData();
        assertEquals(3.0, gradA[0][0], TOL);
        assertEquals(12.0, gradA[0][1], TOL);
        assertEquals(27.0, gradA[1][0], TOL);
        assertEquals(48.0, gradA[1][1], TOL);
    }

    @Test
    void testMatrixFusedAddScalar() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        IDiffMatrix Z = AD.fuseMatrix(A).add(10).compute();
        IDiffMatrix loss = Z.sum();
        loss.backward();

        double[][] result = Z.getValue().getData();
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                assertEquals(A.getValue().get(i, j) + 10, result[i][j], TOL);

        double[][] gradA = A.getGradient().getData();
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                assertEquals(1.0, gradA[i][j], TOL);
    }

    @Test
    void testMatrixFusedMulScalar() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        IDiffMatrix Z = AD.fuseMatrix(A).mul(5).compute();
        IDiffMatrix loss = Z.sum();
        loss.backward();

        double[][] result = Z.getValue().getData();
        assertEquals(5.0, result[0][0], TOL);
        assertEquals(10.0, result[0][1], TOL);
        assertEquals(15.0, result[1][0], TOL);
        assertEquals(20.0, result[1][1], TOL);

        double[][] gradA = A.getGradient().getData();
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                assertEquals(5.0, gradA[i][j], TOL);
    }

    @Test
    void testMatrixFusedTanh() {
        IDiffMatrix A = AD.matrix(new double[][] { { 0, 1 }, { -1, 2 } });
        IDiffMatrix Z = AD.fuseMatrix(A).tanh().compute();
        IDiffMatrix loss = Z.sum();
        loss.backward();

        double[][] gradA = A.getGradient().getData();
        double t0 = Math.tanh(0), t1 = Math.tanh(1), t_1 = Math.tanh(-1), t2 = Math.tanh(2);
        assertEquals(1 - t0 * t0, gradA[0][0], TOL);
        assertEquals(1 - t1 * t1, gradA[0][1], TOL);
        assertEquals(1 - t_1 * t_1, gradA[1][0], TOL);
        assertEquals(1 - t2 * t2, gradA[1][1], TOL);
    }

    @Test
    void testMatrixFusedLongChain() {
        IDiffMatrix A = AD.matrix(new double[][] { { 0.5, 1.0 }, { 1.5, 2.0 } });
        IDiffMatrix Z = AD.fuseMatrix(A)
                .mul(2).add(1).exp().sqrt().square().sigmoid()
                .compute();
        IDiffMatrix loss = Z.sum();
        loss.backward();

        double[][] autoGrad = A.getGradient().getData();
        double eps = 1e-6;
        double[][] aData = { { 0.5, 1.0 }, { 1.5, 2.0 } };
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                aData[i][j] += eps;
                double fp = matrixLongChainEval(aData);
                aData[i][j] -= 2 * eps;
                double fm = matrixLongChainEval(aData);
                aData[i][j] += eps;
                double numGrad = (fp - fm) / (2 * eps);
                assertEquals(numGrad, autoGrad[i][j], 1e-5);
            }
        }
    }

    private double matrixLongChainEval(double[][] a) {
        double s = 0;
        for (double[] row : a) {
            for (double v : row) {
                double r = v * 2 + 1;
                r = Math.exp(r);
                r = Math.sqrt(r);
                r = r * r;
                r = 1.0 / (1.0 + Math.exp(-r));
                s += r;
            }
        }
        return s;
    }

    @Test
    void testMatrixFusedNegAbs() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, -2 }, { -3, 4 } });
        IDiffMatrix Z = AD.fuseMatrix(A).neg().abs().compute();
        IDiffMatrix loss = Z.sum();
        loss.backward();

        double[][] result = Z.getValue().getData();
        assertEquals(1.0, result[0][0], TOL);
        assertEquals(2.0, result[0][1], TOL);
        assertEquals(3.0, result[1][0], TOL);
        assertEquals(4.0, result[1][1], TOL);
    }

    // ================ binary variable fusion ================

    @Test
    void testFusedAddVariable() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector y = AD.vector(new double[] { 4.0, 5.0, 6.0 });
        IDiffVector z = AD.fuse(x).add(y).compute();
        IDiffVector loss = z.sum();
        loss.backward();

        assertArrayEquals(new double[] { 5, 7, 9 }, z.getValue().getData(), TOL);
        assertArrayEquals(new double[] { 1, 1, 1 }, x.getGradient().getData(), TOL);
        assertArrayEquals(new double[] { 1, 1, 1 }, y.getGradient().getData(), TOL);
    }

    @Test
    void testFusedSubVariable() {
        IDiffVector x = AD.vector(new double[] { 5.0, 7.0, 9.0 });
        IDiffVector y = AD.vector(new double[] { 2.0, 3.0, 4.0 });
        IDiffVector z = AD.fuse(x).sub(y).compute();
        IDiffVector loss = z.sum();
        loss.backward();

        assertArrayEquals(new double[] { 3, 4, 5 }, z.getValue().getData(), TOL);
        assertArrayEquals(new double[] { 1, 1, 1 }, x.getGradient().getData(), TOL);
        assertArrayEquals(new double[] { -1, -1, -1 }, y.getGradient().getData(), TOL);
    }

    @Test
    void testFusedMulVariable() {
        IDiffVector x = AD.vector(new double[] { 2.0, 3.0 });
        IDiffVector y = AD.vector(new double[] { 4.0, 5.0 });
        IDiffVector z = AD.fuse(x).mul(y).compute();
        IDiffVector loss = z.sum();
        loss.backward();

        assertArrayEquals(new double[] { 8, 15 }, z.getValue().getData(), TOL);
        assertArrayEquals(new double[] { 4, 5 }, x.getGradient().getData(), TOL);
        assertArrayEquals(new double[] { 2, 3 }, y.getGradient().getData(), TOL);
    }

    @Test
    void testFusedDivVariable() {
        IDiffVector x = AD.vector(new double[] { 6.0, 8.0 });
        IDiffVector y = AD.vector(new double[] { 2.0, 4.0 });
        IDiffVector z = AD.fuse(x).div(y).compute();
        IDiffVector loss = z.sum();
        loss.backward();

        assertArrayEquals(new double[] { 3, 2 }, z.getValue().getData(), TOL);
        assertEquals(1.0 / 2.0, x.getGradient().get(0), TOL);
        assertEquals(1.0 / 4.0, x.getGradient().get(1), TOL);
        assertEquals(-6.0 / 4.0, y.getGradient().get(0), TOL);
        assertEquals(-8.0 / 16.0, y.getGradient().get(1), TOL);
    }

    @Test
    void testFusedBinaryWithUnary() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector y = AD.vector(new double[] { 0.5, 1.0, 1.5 });
        IDiffVector z = AD.fuse(x).exp().mul(y).sqrt().compute();
        IDiffVector loss = z.sum();
        loss.backward();

        double[] autoGradX = x.getGradient().getData();
        double[] autoGradY = y.getGradient().getData();
        double eps = 1e-6;
        double[] xData = { 1.0, 2.0, 3.0 };
        double[] yData = { 0.5, 1.0, 1.5 };
        for (int i = 0; i < xData.length; i++) {
            xData[i] += eps;
            double fp = binaryUnaryEval(xData, yData);
            xData[i] -= 2 * eps;
            double fm = binaryUnaryEval(xData, yData);
            xData[i] += eps;
            double numGrad = (fp - fm) / (2 * eps);
            assertEquals(numGrad, autoGradX[i], 1e-5);
        }
        for (int i = 0; i < yData.length; i++) {
            yData[i] += eps;
            double fp = binaryUnaryEval(xData, yData);
            yData[i] -= 2 * eps;
            double fm = binaryUnaryEval(xData, yData);
            yData[i] += eps;
            double numGrad = (fp - fm) / (2 * eps);
            assertEquals(numGrad, autoGradY[i], 1e-5);
        }
    }

    private double binaryUnaryEval(double[] x, double[] y) {
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            s += Math.sqrt(Math.exp(x[i]) * y[i]);
        }
        return s;
    }

    @Test
    void testFusedTwoBinaryOps() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0 });
        IDiffVector a = AD.vector(new double[] { 3.0, 4.0 });
        IDiffVector b = AD.vector(new double[] { 5.0, 6.0 });
        IDiffVector z = AD.fuse(x).add(a).mul(b).compute();
        IDiffVector loss = z.sum();
        loss.backward();

        // z = (x + a) * b
        assertArrayEquals(new double[] { 20, 36 }, z.getValue().getData(), TOL);
        assertArrayEquals(new double[] { 5, 6 }, x.getGradient().getData(), TOL);
        assertArrayEquals(new double[] { 5, 6 }, a.getGradient().getData(), TOL);
        assertArrayEquals(new double[] { 4, 6 }, b.getGradient().getData(), TOL);
    }

    @Test
    void testFusedSameBinaryVarTwice() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0 });
        IDiffVector y = AD.vector(new double[] { 3.0, 4.0 });
        IDiffVector z = AD.fuse(x).add(y).mul(y).compute();
        IDiffVector loss = z.sum();
        loss.backward();

        // z = (x + y) * y
        assertArrayEquals(new double[] { 12, 24 }, z.getValue().getData(), TOL);
        assertArrayEquals(new double[] { 3, 4 }, x.getGradient().getData(), TOL);
        // d/dy = d/dy[(x+y)*y] = y + (x+y) = x + 2y = [7, 10]
        assertArrayEquals(new double[] { 7, 10 }, y.getGradient().getData(), TOL);
    }

    // ================ auto-fusion (elementwise) ================

    @Test
    void testElementwiseSimpleChain() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector y = AD.elementwise(x, v -> v.exp().log().sqrt().square());
        IDiffVector loss = y.sum();
        loss.backward();

        // exp(log(sqrt(x^2))) - wait, order is: sqrt(square(log(exp(x)))) = sqrt(x^2) = |x|
        assertArrayEquals(new double[] { 1, 2, 3 }, y.getValue().getData(), 1e-10);

        IDiffVector x2 = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector y2 = x2.exp().log().sqrt().square();
        IDiffVector loss2 = y2.sum();
        loss2.backward();
        assertArrayEquals(x2.getGradient().getData(), x.getGradient().getData(), 1e-10);
    }

    @Test
    void testElementwiseWithScalarOps() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector y = AD.elementwise(x, v -> v.mul(2).add(1).pow(3));
        IDiffVector loss = y.sum();
        loss.backward();

        double[] fusedGrad = x.getGradient().getData().clone();

        IDiffVector x2 = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector y2 = x2.mul(2).add(1).pow(3);
        IDiffVector loss2 = y2.sum();
        loss2.backward();

        assertArrayEquals(x2.getGradient().getData(), fusedGrad, 1e-10);
    }

    @Test
    void testElementwiseWithBinaryVar() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector b = AD.vector(new double[] { 4.0, 5.0, 6.0 });
        IDiffVector y = AD.elementwise(x, v -> v.exp().add(b).sqrt());
        IDiffVector loss = y.sum();
        loss.backward();

        double[] fusedGradX = x.getGradient().getData().clone();
        double[] fusedGradB = b.getGradient().getData().clone();

        IDiffVector x2 = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector b2 = AD.vector(new double[] { 4.0, 5.0, 6.0 });
        IDiffVector y2 = x2.exp().add(b2).sqrt();
        IDiffVector loss2 = y2.sum();
        loss2.backward();

        assertArrayEquals(x2.getGradient().getData(), fusedGradX, 1e-10);
        assertArrayEquals(b2.getGradient().getData(), fusedGradB, 1e-10);
    }

    @Test
    void testElementwiseFallbackOnReduction() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        // sum() is not fusible — should fall back to regular execution
        IDiffVector y = AD.elementwise(x, v -> v.exp().sum());
        assertEquals(Math.exp(1) + Math.exp(2) + Math.exp(3), y.getValue().get(0), TOL);
    }

    @Test
    void testElementwiseFallbackOnDot() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector w = AD.vector(new double[] { 0.5, 0.5, 0.5 });
        // dot() is not fusible — should fall back to regular execution
        IDiffVector y = AD.elementwise(x, v -> v.mul(2).dot(w));
        assertEquals((2 + 4 + 6) * 0.5, y.getValue().get(0), TOL);
    }

    @Test
    void testElementwiseEmptyChain() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        // Identity function — no element-wise ops means empty chain, triggers fallback
        IDiffVector y = AD.elementwise(x, v -> v);
        assertArrayEquals(x.getValue().getData(), y.getValue().getData(), TOL);
    }

    @Test
    void testElementwiseFusedVsExplicitFuse() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector yAuto = AD.elementwise(x, v -> v.exp().sqrt().square());
        IDiffVector lossAuto = yAuto.sum();
        lossAuto.backward();
        double[] autoGrad = x.getGradient().getData().clone();

        IDiffVector x2 = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector yExplicit = AD.fuse(x2).exp().sqrt().square().compute();
        IDiffVector lossExp = yExplicit.sum();
        lossExp.backward();
        double[] explicitGrad = x2.getGradient().getData().clone();

        assertArrayEquals(explicitGrad, autoGrad, 1e-10);
    }
}
