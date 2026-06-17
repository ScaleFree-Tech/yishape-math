package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffMatrix;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 4.1 Step 1.4 — regression baseline for the {@code RereDiffMatrix} thin
 * proxy over {@code RereDiffTensor(shape=[rows, cols])}.
 *
 * <p>After the matrix→tensor unification, every matrix op delegates to the
 * underlying tensor graph. This test pins the forward values and gradients of
 * the full matrix API so a regression in the delegation layer (wrong shape
 * wrap, broken backward, lost gradient) is caught immediately.
 *
 * <p>Gradients are checked both analytically (known closed form) and against
 * central-difference numerical gradients for the non-trivial ops.
 */
public class RereDiffMatrixMigrationTest {

    private static final double TOL = 1e-9;
    private static final double NUM_TOL = 1e-6;

    // ── Forward value correctness ─────────────────────────────────────

    @Test
    void matmul_forwardValue() {
        IDiffMatrix A = AD.matrix(new double[][]{{1, 2}, {3, 4}});
        IDiffMatrix B = AD.matrix(new double[][]{{5, 6}, {7, 8}});
        IDiffMatrix Z = A.matmul(B);
        double[][] z = Z.getValue().getData();
        assertEquals(19, z[0][0], TOL); assertEquals(22, z[0][1], TOL);
        assertEquals(43, z[1][0], TOL); assertEquals(50, z[1][1], TOL);
    }

    @Test
    void matmulVector_forwardValue() {
        // [2x3] @ [3] → [2]
        IDiffMatrix A = AD.matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        IDiffVector v = AD.vector(new double[]{1, 1, 1});
        IDiffVector y = A.matmul(v);
        double[] d = y.getValue().getData();
        assertArrayEquals(new double[]{6, 15}, d, TOL);
    }

    @Test
    void unaryOps_forwardValues() {
        // All-positive so log/sqrt are finite.
        IDiffMatrix A = AD.matrix(new double[][]{{1.0, Math.E}, {4.0, 9.0}});
        assertEquals(Math.E, A.exp().getValue().get(0, 0), 1e-9);      // exp(1)=e
        assertEquals(0.0, A.log().getValue().get(0, 0), TOL);          // ln(1)=0
        assertEquals(1.0, A.log().getValue().get(0, 1), TOL);          // ln(e)=1
        double s1 = 1.0 / (1.0 + Math.exp(-1.0));
        assertEquals(s1, A.sigmoid().getValue().get(0, 0), 1e-6);       // σ(1)
        assertEquals(4.0, A.relu().getValue().get(1, 0), TOL);
        assertEquals(81.0, A.square().getValue().get(1, 1), TOL);      // 9^2
        assertEquals(3.0, A.sqrt().getValue().get(1, 1), TOL);          // √9
        assertEquals(-1.0, A.neg().getValue().get(0, 0), TOL);
    }

    @Test
    void reductions_forwardValues() {
        IDiffMatrix A = AD.matrix(new double[][]{{1, 2}, {3, 4}});
        assertEquals(10.0, A.sum().getValue().get(0, 0), TOL);
        assertEquals(2.5, A.mean().getValue().get(0, 0), TOL);
        assertArrayEquals(new double[]{3, 7}, A.sum(1).getValue().getData(), TOL); // row sums
    }

    @Test
    void transposeReshape_forwardValues() {
        IDiffMatrix A = AD.matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        double[][] t = A.transpose().getValue().getData();
        assertEquals(1, t[0][0], TOL); assertEquals(4, t[0][1], TOL);
        assertEquals(3, t[2][0], TOL); assertEquals(6, t[2][1], TOL);

        double[][] r = A.reshape(3, 2).getValue().getData();
        assertEquals(1, r[0][0], TOL); assertEquals(2, r[0][1], TOL);
        assertEquals(3, r[1][0], TOL); assertEquals(4, r[1][1], TOL);
    }

    // ── Gradient correctness (analytic) ───────────────────────────────

    @Test
    void matmul_gradient() {
        IDiffMatrix A = AD.matrix(new double[][]{{1, 2}, {3, 4}});
        IDiffMatrix B = AD.matrix(new double[][]{{5, 6}, {7, 8}});
        IDiffMatrix loss = A.matmul(B).sum();
        loss.backward();
        // d(sum(A@B))/dA = ones @ B^T = [[11,15],[11,15]]
        double[][] gA = A.getGradient().getData();
        assertEquals(11, gA[0][0], TOL); assertEquals(15, gA[0][1], TOL);
        assertEquals(11, gA[1][0], TOL); assertEquals(15, gA[1][1], TOL);
        // d(sum(A@B))/dB = A^T @ ones = [[4,4],[6,6]]
        double[][] gB = B.getGradient().getData();
        assertEquals(4, gB[0][0], TOL); assertEquals(6, gB[1][0], TOL);
    }

    @Test
    void elementwise_gradient() {
        IDiffMatrix A = AD.matrix(new double[][]{{1, 2}, {3, 4}});
        IDiffMatrix B = AD.matrix(new double[][]{{5, 6}, {7, 8}});
        IDiffMatrix loss = A.mul(B).sum();  // d/dA = B
        loss.backward();
        double[][] gA = A.getGradient().getData();
        assertEquals(5, gA[0][0], TOL); assertEquals(8, gA[1][1], TOL);
    }

    @Test
    void scalarOps_gradient() {
        IDiffMatrix A = AD.matrix(new double[][]{{1, 2}, {3, 4}});
        IDiffMatrix loss = A.mul(2.0).sum();  // d/dA = 2
        loss.backward();
        double[][] g = A.getGradient().getData();
        for (double[] row : g) for (double v : row) assertEquals(2.0, v, TOL);
    }

    @Test
    void square_gradient() {
        IDiffMatrix A = AD.matrix(new double[][]{{1, 2}, {3, 4}});
        IDiffMatrix loss = A.square().sum();  // d/dA = 2A
        loss.backward();
        double[][] g = A.getGradient().getData();
        assertEquals(2, g[0][0], TOL); assertEquals(4, g[0][1], TOL);
        assertEquals(6, g[1][0], TOL); assertEquals(8, g[1][1], TOL);
    }

    @Test
    void exp_gradient() {
        IDiffMatrix A = AD.matrix(new double[][]{{0, 1}, {2, 0}});
        IDiffMatrix loss = A.exp().sum();  // d/dA = exp(A)
        loss.backward();
        double[][] g = A.getGradient().getData();
        assertEquals(1, g[0][0], TOL);
        assertEquals(Math.E, g[0][1], 1e-8);
        assertEquals(Math.E * Math.E, g[1][0], 1e-8);
    }

    @Test
    void matmulVector_gradient() {
        // loss = sum(A @ v), A=[2x3], v=[3]
        // d/dA = ones(2) outer v = [[v0,v1,v2],[v0,v1,v2]]
        // d/dv = A^T @ ones(2) = column sums of A
        IDiffMatrix A = AD.matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        IDiffVector v = AD.vector(new double[]{7, 8, 9});
        IDiffVector loss = A.matmul(v);
        loss.backward();

        double[][] gA = A.getGradient().getData();
        assertEquals(7, gA[0][0], TOL); assertEquals(9, gA[0][2], TOL);
        assertEquals(7, gA[1][0], TOL); assertEquals(9, gA[1][2], TOL);

        double[] gv = v.getGradient().getData();
        assertArrayEquals(new double[]{5, 7, 9}, gv, TOL); // col sums of A
    }

    // ── Numerical gradient cross-check (the real migration guard) ─────

    @Test
    void matmul_numericalGradient() {
        double[][] aData = {{1, 2}, {3, 4}};
        double[][] bData = {{5, 6}, {7, 8}};
        IDiffMatrix A = AD.matrix(aData);
        IDiffMatrix B = AD.matrix(bData);
        IDiffMatrix loss = A.matmul(B).sum();
        loss.backward();

        double eps = 1e-6;
        double[][] gA = A.getGradient().getData();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                aData[i][j] += eps;
                double fp = matmulSum(aData, bData);
                aData[i][j] -= 2 * eps;
                double fm = matmulSum(aData, bData);
                aData[i][j] += eps;
                double num = (fp - fm) / (2 * eps);
                assertEquals(num, gA[i][j], NUM_TOL,
                    "dA[" + i + "][" + j + "]");
            }
        }
    }

    private static double matmulSum(double[][] a, double[][] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < b[0].length; j++) {
                double v = 0;
                for (int k = 0; k < a[0].length; k++) v += a[i][k] * b[k][j];
                s += v;
            }
        return s;
    }

    @Test
    void chainedOps_numericalGradient() {
        // f = sum(relu(A @ B)^2) — exercises matmul → relu → square → sum
        double[][] aData = {{0.5, -0.3}, {0.1, 0.8}};
        double[][] bData = {{1.0, 0.2}, {-0.4, 0.6}};
        IDiffMatrix A = AD.matrix(aData);
        IDiffMatrix B = AD.matrix(bData);
        IDiffMatrix loss = A.matmul(B).relu().square().sum();
        loss.backward();

        double eps = 1e-6;
        double[][] gA = A.getGradient().getData();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                aData[i][j] += eps;
                double fp = reluSquareSum(aData, bData);
                aData[i][j] -= 2 * eps;
                double fm = reluSquareSum(aData, bData);
                aData[i][j] += eps;
                double num = (fp - fm) / (2 * eps);
                assertEquals(num, gA[i][j], NUM_TOL,
                    "chained dA[" + i + "][" + j + "]");
            }
        }
    }

    private static double reluSquareSum(double[][] a, double[][] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < b[0].length; j++) {
                double v = 0;
                for (int k = 0; k < a[0].length; k++) v += a[i][k] * b[k][j];
                double r = Math.max(0, v);
                s += r * r;
            }
        return s;
    }

    // ── Proxy-identity invariants ─────────────────────────────────────

    @Test
    void matrixIsThinProxyOverTensor() {
        IDiffMatrix A = AD.matrix(new double[][]{{1, 2}, {3, 4}});
        assertInstanceOf(RereDiffMatrix.class, A);
        // The single source-of-truth field must exist and be rank-2.
        RereDiffMatrix rm = (RereDiffMatrix) A;
        assertNotNull(rm.tensor);
        assertArrayEquals(new int[]{2, 2}, rm.tensor.shape());
    }

    @Test
    void backward_delegatesToTensor() {
        // backward() on the matrix must seed grad into the underlying tensor
        // and propagate to leaves through the single tensor graph.
        IDiffMatrix A = AD.matrix(new double[][]{{1, 2}, {3, 4}});
        IDiffMatrix loss = A.square().sum();
        loss.backward();
        assertNotNull(((RereDiffMatrix) A).tensor.gradData(),
            "matrix.backward() must populate the underlying tensor's gradient");
    }

    @Test
    void detach_blocksGradient() {
        IDiffMatrix A = AD.matrix(new double[][]{{1, 2}, {3, 4}});
        IDiffMatrix W = AD.matrix(new double[][]{{1, 1}, {1, 1}});
        IDiffMatrix loss = A.detach().mul(W).sum();
        loss.backward();
        // W gets gradient (= detached A values); A does not.
        assertNotNull(W.getGradient());
        assertNull(A.getGradient(), "detach must block gradient to A on the matrix proxy");
    }
}
