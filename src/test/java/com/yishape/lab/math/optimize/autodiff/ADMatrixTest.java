package com.yishape.lab.math.optimize.autodiff;

import com.yishape.lab.math.autodiff.IDiffMatrix;
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffVector;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.Opts;

public class ADMatrixTest {

    private static final double TOL = 1e-10;

    // ---- matrix multiply gradient ----

    @Test
    void testMatmulGradient() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        IDiffMatrix B = AD.matrix(new double[][] { { 5, 6 }, { 7, 8 } });
        IDiffMatrix Z = A.matmul(B);
        IDiffMatrix loss = Z.sum();
        loss.backward();

        // Z = A @ B = [[19,22],[43,50]], sum = 134
        assertEquals(134.0, loss.getValue().get(0, 0), TOL);

        // d(sum(Z))/dA = ones(2,2) @ B^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
        double[][] gradA = A.getGradient().getData();
        assertEquals(11.0, gradA[0][0], TOL);
        assertEquals(15.0, gradA[0][1], TOL);
        assertEquals(11.0, gradA[1][0], TOL);
        assertEquals(15.0, gradA[1][1], TOL);

        // d(sum(Z))/dB = A^T @ ones(2,2) = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        double[][] gradB = B.getGradient().getData();
        assertEquals(4.0, gradB[0][0], TOL);
        assertEquals(4.0, gradB[0][1], TOL);
        assertEquals(6.0, gradB[1][0], TOL);
        assertEquals(6.0, gradB[1][1], TOL);
    }

    @Test
    void testMatmulNumericalGradient() {
        double[][] aData = { { 1, 2 }, { 3, 4 }, { 5, 6 } };
        double[][] bData = { { 0.5, 1.0 }, { 1.5, 2.0 } };
        IDiffMatrix A = AD.matrix(aData);
        IDiffMatrix B = AD.matrix(bData);
        IDiffMatrix Z = A.matmul(B);
        IDiffMatrix loss = Z.sum();
        loss.backward();

        double eps = 1e-6;
        for (int i = 0; i < aData.length; i++) {
            for (int j = 0; j < aData[0].length; j++) {
                aData[i][j] += eps;
                double fp = matmulSum(aData, bData);
                aData[i][j] -= 2 * eps;
                double fm = matmulSum(aData, bData);
                aData[i][j] += eps;
                double numGrad = (fp - fm) / (2 * eps);
                assertEquals(numGrad, A.getGradient().get(i, j), 1e-5);
            }
        }
    }

    private double matmulSum(double[][] a, double[][] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                double v = 0;
                for (int k = 0; k < a[0].length; k++) {
                    v += a[i][k] * b[k][j];
                }
                sum += v;
            }
        }
        return sum;
    }

    // ---- matrix-vector multiply ----

    @Test
    void testMatmulVectorGradient() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, 2, 3 }, { 4, 5, 6 } });
        IDiffVector x = AD.vector(new double[] { 1, 2, 3 });
        IDiffVector z = A.matmul(x);
        IDiffVector loss = z.sum();
        loss.backward();

        // z = [14, 32], sum = 46
        assertEquals(46.0, loss.getValue().get(0), TOL);

        // dL/dA = gradOut outer x = 1 * [1,2,3]^T = outer([1,1], [1,2,3])
        double[][] gradA = A.getGradient().getData();
        assertEquals(1.0, gradA[0][0], TOL);
        assertEquals(2.0, gradA[0][1], TOL);
        assertEquals(3.0, gradA[0][2], TOL);
        assertEquals(1.0, gradA[1][0], TOL);
        assertEquals(2.0, gradA[1][1], TOL);
        assertEquals(3.0, gradA[1][2], TOL);

        // dL/dx = A^T @ gradOut = [1,4; 2,5; 3,6]^T @ [1,1] = [1+4, 2+5, 3+6]
        double[] gradX = x.getGradient().getData();
        assertArrayEquals(new double[] { 5, 7, 9 }, gradX, TOL);
    }

    // ---- transpose ----

    @Test
    void testTransposeGradient() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, 2, 3 }, { 4, 5, 6 } });
        IDiffMatrix Z = A.transpose();
        IDiffMatrix loss = Z.sum();
        loss.backward();

        double[][] gradA = A.getGradient().getData();
        assertEquals(2, gradA.length);
        assertEquals(3, gradA[0].length);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(1.0, gradA[i][j], TOL);
            }
        }
    }

    // ---- element-wise matrix ops ----

    @Test
    void testMatrixAddGradient() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        IDiffMatrix B = AD.matrix(new double[][] { { 5, 6 }, { 7, 8 } });
        IDiffMatrix Z = A.add(B).sum();
        Z.backward();

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(1.0, A.getGradient().get(i, j), TOL);
                assertEquals(1.0, B.getGradient().get(i, j), TOL);
            }
        }
    }

    @Test
    void testMatrixElementWiseMulGradient() {
        IDiffMatrix A = AD.matrix(new double[][] { { 2, 3 }, { 4, 5 } });
        IDiffMatrix B = AD.matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        IDiffMatrix Z = A.mul(B).sum();
        Z.backward();

        // d(sum(A⊙B))/dA = ones ⊙ B = [[1,2],[3,4]]
        double[][] gradA = A.getGradient().getData();
        assertEquals(1.0, gradA[0][0], TOL);
        assertEquals(2.0, gradA[0][1], TOL);
        assertEquals(3.0, gradA[1][0], TOL);
        assertEquals(4.0, gradA[1][1], TOL);

        // d(sum(A⊙B))/dB = A ⊙ ones = [[2,3],[4,5]]
        double[][] gradB = B.getGradient().getData();
        assertEquals(2.0, gradB[0][0], TOL);
        assertEquals(3.0, gradB[0][1], TOL);
        assertEquals(4.0, gradB[1][0], TOL);
        assertEquals(5.0, gradB[1][1], TOL);
    }

    @Test
    void testMatrixScalarOps() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        IDiffMatrix Z = A.mul(2).add(1).sum();
        Z.backward();

        double[][] gradA = A.getGradient().getData();
        assertEquals(2.0, gradA[0][0], TOL);
        assertEquals(2.0, gradA[0][1], TOL);
        assertEquals(2.0, gradA[1][0], TOL);
        assertEquals(2.0, gradA[1][1], TOL);
    }

    @Test
    void testMatrixSigmoidGradient() {
        IDiffMatrix A = AD.matrix(new double[][] { { 0, 1 }, { -1, 2 } });
        IDiffMatrix Z = A.sigmoid().sum();
        Z.backward();

        double[][] gradA = A.getGradient().getData();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                double v = (i == 0 ? (j == 0 ? 0 : (j == 1 ? 1 : 0)) : (j == 0 ? -1 : 2));
                // Actually compute the value from the matrix data
            }
        }
        // Verify gradients are non-zero and correct via numerical check
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
    void testMatrixReluGradient() {
        IDiffMatrix A = AD.matrix(new double[][] { { -2, 0 }, { 1, 3 } });
        IDiffMatrix Z = A.relu().sum();
        Z.backward();

        double[][] gradA = A.getGradient().getData();
        assertEquals(0.0, gradA[0][0], TOL);
        assertEquals(0.0, gradA[0][1], TOL);
        assertEquals(1.0, gradA[1][0], TOL);
        assertEquals(1.0, gradA[1][1], TOL);
    }

    @Test
    void testMatrixMeanGradient() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        IDiffMatrix Z = A.mean();
        Z.backward();

        double[][] gradA = A.getGradient().getData();
        assertEquals(0.25, gradA[0][0], TOL);
        assertEquals(0.25, gradA[0][1], TOL);
        assertEquals(0.25, gradA[1][0], TOL);
        assertEquals(0.25, gradA[1][1], TOL);
    }

    // ---- reshape / flatten ----

    @Test
    void testFlattenGradient() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        IDiffVector v = A.flatten();
        IDiffVector loss = v.sum();
        loss.backward();

        double[][] gradA = A.getGradient().getData();
        assertEquals(1.0, gradA[0][0], TOL);
        assertEquals(1.0, gradA[0][1], TOL);
        assertEquals(1.0, gradA[1][0], TOL);
        assertEquals(1.0, gradA[1][1], TOL);
    }

    // ---- grad() second-order ----

    @Test
    void testGradWrapsGradientAsVariable() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 3 });
        IDiffVector y = x.pow(2).sum();
        y.backward();

        // grad() wraps the gradient as a new leaf variable
        IDiffVector gradX = x.grad();
        assertArrayEquals(new double[] { 2, 4, 6 }, gradX.getValue().getData(), TOL);
        assertTrue(gradX.isLeaf());

        // Can run autodiff on the wrapped gradient independently
        IDiffVector gradSum = gradX.sum();
        gradSum.backward();
        assertArrayEquals(new double[] { 1, 1, 1 }, gradX.getGradient().getData(), TOL);

        // Original x gradient is unchanged
        assertArrayEquals(new double[] { 2, 4, 6 }, x.getGradient().getData(), TOL);
    }

    // ---- isLeaf ----

    @Test
    void testMatrixIsLeaf() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        assertTrue(A.isLeaf());
        IDiffMatrix B = A.transpose();
        assertTrue(!B.isLeaf());
    }

    // ---- zeroGradient ----

    @Test
    void testMatrixZeroGradient() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        IDiffMatrix Z = A.sum();
        Z.backward();
        assertNotNull(A.getGradient());
        A.zeroGradient();
        assertEquals(null, A.getGradient());
    }

    // ---- reshape ----

    @Test
    void testReshapeForwardShape() {
        IDiffVector v = AD.vector(new double[] { 1, 2, 3, 4, 5, 6 });
        IDiffMatrix M = v.reshape(2, 3);
        assertEquals(2, M.getValue().rows());
        assertEquals(3, M.getValue().cols());
        assertEquals(1.0, M.get(0, 0), TOL);
        assertEquals(2.0, M.get(0, 1), TOL);
        assertEquals(3.0, M.get(0, 2), TOL);
        assertEquals(6.0, M.get(1, 2), TOL);
    }

    @Test
    void testReshapeGradient() {
        IDiffVector v = AD.vector(new double[] { 1, 2, 3, 4 });
        IDiffMatrix M = v.reshape(2, 2);
        IDiffMatrix loss = M.sum();
        loss.backward();
        double[] grad = v.getGradient().getData();
        assertArrayEquals(new double[] { 1, 1, 1, 1 }, grad, TOL);
    }

    @Test
    void testReshapeFlattenRoundtrip() {
        IDiffMatrix A = AD.matrix(new double[][] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        IDiffVector flat = A.flatten();
        IDiffMatrix B = flat.reshape(3, 2);
        assertEquals(A.getValue().get(0, 0), B.getValue().get(0, 0), TOL);
        assertEquals(A.getValue().get(2, 1), B.getValue().get(2, 1), TOL);
        IDiffMatrix loss = B.sum();
        loss.backward();
        double[][] gradA = A.getGradient().getData();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(1.0, gradA[i][j], TOL);
            }
        }
    }

    @Test
    void testReshapeBadDimensions() {
        IDiffVector v = AD.vector(new double[] { 1, 2, 3 });
        assertThrows(IllegalArgumentException.class, () -> v.reshape(2, 2));
    }

    @Test
    void testReshapeNumericalGradient() {
        double[] origData = { 1, 2, 3, 4, 5, 6 };
        double eps = 1e-6;
        for (int i = 0; i < 6; i++) {
            double[] dataFp = origData.clone();
            dataFp[i] += eps;
            IDiffVector vFp = AD.vector(dataFp);
            double fp = vFp.reshape(2, 3).sum().getValue().get(0, 0);

            double[] dataFm = origData.clone();
            dataFm[i] -= eps;
            IDiffVector vFm = AD.vector(dataFm);
            double fm = vFm.reshape(2, 3).sum().getValue().get(0, 0);

            double numGrad = (fp - fm) / (2 * eps);
            IDiffVector v = AD.vector(origData);
            v.reshape(2, 3).sum().backward();
            assertEquals(numGrad, v.getGradient().get(i), 1e-5);
        }
    }

    @Test
    void testReshapeWithMatmulGradient() {
        IDiffVector w = AD.vector(new double[] { 1, 2, 3, 4, 5, 6 });
        IDiffMatrix L = w.reshape(2, 3);
        IDiffMatrix X = AD.matrix(new double[][] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 1, 0 } });
        IDiffVector y = X.matmul(L.transpose()).square().matmul(AD.ones(2));
        IDiffVector loss = y.sum();
        loss.backward();

        double eps = 1e-6;
        double[] wArr = { 1, 2, 3, 4, 5, 6 };
        for (int i = 0; i < 6; i++) {
            double[] wp = wArr.clone();
            wp[i] += eps;
            IDiffVector vp = AD.vector(wp);
            IDiffMatrix Lp = vp.reshape(2, 3);
            IDiffVector yp = X.matmul(Lp.transpose()).square().matmul(AD.ones(2));
            double fp = yp.sum().getValue().get(0);

            double[] wm = wArr.clone();
            wm[i] -= eps;
            IDiffVector vm = AD.vector(wm);
            IDiffMatrix Lm = vm.reshape(2, 3);
            IDiffVector ym = X.matmul(Lm.transpose()).square().matmul(AD.ones(2));
            double fm = ym.sum().getValue().get(0);

            double numGrad = (fp - fm) / (2 * eps);
            assertEquals(numGrad, w.getGradient().get(i), 1e-5);
        }
    }

    // ---- optimizer integration ----

    @Test
    void testOptimizeIntegration() {
        IOptimizer lbfgs = Opts.lbfgs();
        OptResult result = AD.optimize(
                IDoubleVector.of(3.0),
                x -> x.pow(2).sub(2),
                lbfgs);

        double xOpt = result.getOptimalPoint().get(0);
        double fOpt = result.getOptimalValue();
        assertEquals(0.0, xOpt, 1e-6);
        assertEquals(-2.0, fOpt, 1e-6);
    }

    // ---- multi-class LR gradient check ----

    @Test
    void testMulticlassLRGradientWithL2() {
        // Small synthetic data: 4 samples, 2 features, 3 classes
        int m = 4, d = 2, k = 3;
        double[][] Xdata = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
        double[][] Ydata = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}};

        // Build augmented feature matrix [X | 1]
        double[][] augData = new double[m][d + 1];
        for (int i = 0; i < m; i++) {
            System.arraycopy(Xdata[i], 0, augData[i], 0, d);
            augData[i][d] = 1.0;
        }
        IDiffMatrix Xa = AD.matrix(augData);
        IDiffMatrix Ya = AD.matrix(Ydata);

        double lambda2 = 0.03;
        java.util.Random rng = new java.util.Random(42);
        double[] wInit = new double[k * (d + 1)];
        for (int i = 0; i < wInit.length; i++) {
            wInit[i] = rng.nextGaussian() * 0.5;
        }

        IDiffVector w0 = AD.vector(wInit);
        var result = AD.checkGradientDetailed(w -> {
            IDiffMatrix Wa = w.reshape(k, d + 1);
            IDiffMatrix Z = Xa.matmul(Wa.transpose());
            IDiffVector ceLoss = Z.softmaxCrossEntropy(Ya);
            return ceLoss.add(w.pow(2).sum().mul(lambda2 / 2.0));
        }, w0, 1e-4);

        assertTrue(result.passed(),
            "Gradient check failed: " + result);
    }
}
