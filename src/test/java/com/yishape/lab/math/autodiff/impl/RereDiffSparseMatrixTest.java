package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.autodiff.IDiffSparseMatrix;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.linalg.sparse.ISparseMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for {@link RereDiffSparseMatrix}.
 *
 * <p>Covers all 14 operations: add, sub, mul, div, elementwiseMul, negate,
 * transpose, relu, sigmoid, tanh, abs, sum, mean, matmul.</p>
 *
 * <p>All backward tests verify gradient correctness by comparing against
 * known analytical gradients.</p>
 */
public class RereDiffSparseMatrixTest {

    private static final double TOL = 1e-10;

    // ==================== Helper factories ====================

    private static ISparseMatrix mat(double[][] data) {
        return ISparseMatrix.fromDense(data);
    }

    private static RereDiffSparseMatrix leaf(ISparseMatrix v) {
        return new RereDiffSparseMatrix(v);
    }

    // ==================== add ====================

    @Test
    void testAddForward() {
        ISparseMatrix a = mat(new double[][]{{1, 2}, {3, 4}});
        ISparseMatrix b = mat(new double[][]{{5, 6}, {7, 8}});
        RereDiffSparseMatrix x = leaf(a);
        RereDiffSparseMatrix y = leaf(b);
        IDiffSparseMatrix z = x.add(y);
        double[][] expected = {{6, 8}, {10, 12}};
        assertMatrixEquals(expected, z.getValue().toDenseArray(), TOL);
    }

    @Test
    void testAddBackward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        RereDiffSparseMatrix y = leaf(mat(new double[][]{{5, 6}, {7, 8}}));
        IDiffSparseMatrix z = x.add(y);
        z.backward(mat(new double[][]{{1, 1}, {1, 1}}));
        double[][] expectedGrad = {{1, 1}, {1, 1}};
        assertMatrixEquals(expectedGrad, x.getGradient().toDenseArray(), TOL);
        assertMatrixEquals(expectedGrad, y.getGradient().toDenseArray(), TOL);
    }

    // ==================== sub ====================

    @Test
    void testSubForward() {
        ISparseMatrix a = mat(new double[][]{{5, 6}, {7, 8}});
        ISparseMatrix b = mat(new double[][]{{1, 2}, {3, 4}});
        RereDiffSparseMatrix x = leaf(a);
        RereDiffSparseMatrix y = leaf(b);
        IDiffSparseMatrix z = x.sub(y);
        double[][] expected = {{4, 4}, {4, 4}};
        assertMatrixEquals(expected, z.getValue().toDenseArray(), TOL);
    }

    @Test
    void testSubBackward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{5, 6}, {7, 8}}));
        RereDiffSparseMatrix y = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        IDiffSparseMatrix z = x.sub(y);
        z.backward(mat(new double[][]{{1, 1}, {1, 1}}));
        assertMatrixEquals(new double[][]{{1, 1}, {1, 1}}, x.getGradient().toDenseArray(), TOL);
        assertMatrixEquals(new double[][]{{-1, -1}, {-1, -1}}, y.getGradient().toDenseArray(), TOL);
    }

    // ==================== mul (scalar) ====================

    @Test
    void testMulScalarForward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        IDiffSparseMatrix z = x.mul(3.0);
        double[][] expected = {{3, 6}, {9, 12}};
        assertMatrixEquals(expected, z.getValue().toDenseArray(), TOL);
    }

    @Test
    void testMulScalarBackward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        IDiffSparseMatrix z = x.mul(3.0);
        z.backward(mat(new double[][]{{1, 1}, {1, 1}}));
        assertMatrixEquals(new double[][]{{3, 3}, {3, 3}}, x.getGradient().toDenseArray(), TOL);
    }

    // ==================== div (scalar) ====================

    @Test
    void testDivScalarForward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{2, 4}, {6, 8}}));
        IDiffSparseMatrix z = x.div(2.0);
        double[][] expected = {{1, 2}, {3, 4}};
        assertMatrixEquals(expected, z.getValue().toDenseArray(), TOL);
    }

    @Test
    void testDivScalarBackward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{2, 4}, {6, 8}}));
        IDiffSparseMatrix z = x.div(2.0);
        z.backward(mat(new double[][]{{1, 1}, {1, 1}}));
        // d(x/2)/dx = 0.5
        assertMatrixEquals(new double[][]{{0.5, 0.5}, {0.5, 0.5}}, x.getGradient().toDenseArray(), TOL);
    }

    // ==================== elementwiseMul ====================

    @Test
    void testElementwiseMulForward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        RereDiffSparseMatrix y = leaf(mat(new double[][]{{5, 6}, {7, 8}}));
        IDiffSparseMatrix z = x.elementwiseMul(y);
        double[][] expected = {{5, 12}, {21, 32}};
        assertMatrixEquals(expected, z.getValue().toDenseArray(), TOL);
    }

    @Test
    void testElementwiseMulBackward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        RereDiffSparseMatrix y = leaf(mat(new double[][]{{5, 6}, {7, 8}}));
        IDiffSparseMatrix z = x.elementwiseMul(y);
        z.backward(mat(new double[][]{{1, 1}, {1, 1}}));
        // dA = G * B, dB = G * A
        assertMatrixEquals(new double[][]{{5, 6}, {7, 8}}, x.getGradient().toDenseArray(), TOL);
        assertMatrixEquals(new double[][]{{1, 2}, {3, 4}}, y.getGradient().toDenseArray(), TOL);
    }

    // ==================== negate ====================

    @Test
    void testNegateForward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, -2}, {3, -4}}));
        IDiffSparseMatrix z = x.negate();
        double[][] expected = {{-1, 2}, {-3, 4}};
        assertMatrixEquals(expected, z.getValue().toDenseArray(), TOL);
    }

    @Test
    void testNegateBackward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        IDiffSparseMatrix z = x.negate();
        z.backward(mat(new double[][]{{1, 1}, {1, 1}}));
        assertMatrixEquals(new double[][]{{-1, -1}, {-1, -1}}, x.getGradient().toDenseArray(), TOL);
    }

    // ==================== transpose ====================

    @Test
    void testTransposeForward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2, 3}, {4, 5, 6}}));
        IDiffSparseMatrix z = x.transpose();
        assertEquals(2, z.getValue().cols()); // original rows -> cols
        assertEquals(3, z.getValue().rows()); // original cols -> rows
        double[][] expected = {{1, 4}, {2, 5}, {3, 6}};
        assertMatrixEquals(expected, z.getValue().toDenseArray(), TOL);
    }

    @Test
    void testTransposeBackward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        IDiffSparseMatrix z = x.transpose();
        // grad for 2x2 transposed = same shape as transpose → 2x2
        z.backward(mat(new double[][]{{1, 2}, {3, 4}}));
        // transpose gradient propagates as transpose upstream
        double[][] expected = {{1, 3}, {2, 4}};
        assertMatrixEquals(expected, x.getGradient().toDenseArray(), TOL);
    }

    // ==================== relu ====================

    @Test
    void testReluForward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{-1, 2}, {-3, 4}}));
        IDiffSparseMatrix z = x.relu();
        double[][] expected = {{0, 2}, {0, 4}};
        assertMatrixEquals(expected, z.getValue().toDenseArray(), TOL);
    }

    @Test
    void testReluBackward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{-1, 2}, {-3, 4}}));
        IDiffSparseMatrix z = x.relu();
        z.backward(mat(new double[][]{{1, 1}, {1, 1}}));
        // gradient = 1 where input > 0, else 0
        double[][] expected = {{0, 1}, {0, 1}};
        assertMatrixEquals(expected, x.getGradient().toDenseArray(), TOL);
    }

    // ==================== sigmoid ====================

    @Test
    void testSigmoidForward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{0, 1}, {-1, 2}}));
        IDiffSparseMatrix z = x.sigmoid();
        double[][] val = z.getValue().toDenseArray();
        assertEquals(1.0 / (1.0 + Math.exp(-0)), val[0][0], TOL);
        assertEquals(1.0 / (1.0 + Math.exp(-1)), val[0][1], TOL);
    }

    @Test
    void testSigmoidBackward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{0, 1}, {-1, 2}}));
        IDiffSparseMatrix z = x.sigmoid();
        z.backward(mat(new double[][]{{1, 1}, {1, 1}}));
        double[][] g = x.getGradient().toDenseArray();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                double s = 1.0 / (1.0 + Math.exp(-x.getValue().toDenseArray()[i][j]));
                assertEquals(s * (1.0 - s), g[i][j], TOL);
            }
        }
    }

    // ==================== tanh ====================

    @Test
    void testTanhForward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{0, 1}, {-1, 2}}));
        IDiffSparseMatrix z = x.tanh();
        double[][] val = z.getValue().toDenseArray();
        assertEquals(Math.tanh(0), val[0][0], TOL);
        assertEquals(Math.tanh(1), val[0][1], TOL);
    }

    @Test
    void testTanhBackward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{0, 1}, {-1, 2}}));
        IDiffSparseMatrix z = x.tanh();
        z.backward(mat(new double[][]{{1, 1}, {1, 1}}));
        double[][] g = x.getGradient().toDenseArray();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                double t = Math.tanh(x.getValue().toDenseArray()[i][j]);
                assertEquals(1.0 - t * t, g[i][j], TOL);
            }
        }
    }

    // ==================== abs ====================

    @Test
    void testAbsForward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{-1, 2}, {-3, 4}}));
        IDiffSparseMatrix z = x.abs();
        double[][] expected = {{1, 2}, {3, 4}};
        assertMatrixEquals(expected, z.getValue().toDenseArray(), TOL);
    }

    @Test
    void testAbsBackward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{-1, 2}, {-3, 4}}));
        IDiffSparseMatrix z = x.abs();
        z.backward(mat(new double[][]{{1, 1}, {1, 1}}));
        // d|x|/dx = sign(x), 0 at x=0
        double[][] expected = {{-1, 1}, {-1, 1}};
        assertMatrixEquals(expected, x.getGradient().toDenseArray(), TOL);
    }

    // ==================== sum (→ IDiffVector) ====================

    @Test
    void testSumForward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        IDiffVector z = x.sum();
        // sum of [[1,2],[3,4]] = 10
        assertEquals(10.0, z.getValue().get(0), TOL);
    }

    @Test
    void testSumBackward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        IDiffVector sum = x.sum();
        // sum() returns IDiffVector; backward on vector grad propagates to sparse matrix
        sum.backward();
        // gradient of sum w.r.t. each element = 1
        double[][] expected = {{1, 1}, {1, 1}};
        assertMatrixEquals(expected, x.getGradient().toDenseArray(), TOL);
    }

    // ==================== mean (→ IDiffVector) ====================

    @Test
    void testMeanForward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        IDiffVector z = x.mean();
        assertEquals(2.5, z.getValue().get(0), TOL);
    }

    @Test
    void testMeanBackward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        IDiffVector mean = x.mean();
        mean.backward();
        // gradient of mean w.r.t. each element = 1/N = 1/4
        double[][] expected = {{0.25, 0.25}, {0.25, 0.25}};
        assertMatrixEquals(expected, x.getGradient().toDenseArray(), TOL);
    }

    // ==================== matmul (sparse @ dense vector) ====================

    @Test
    void testMatmulForward() {
        RereDiffSparseMatrix A = leaf(mat(new double[][]{{1, 0}, {0, 2}, {3, 0}}));
        double[] vecData = {2, 3};
        RereDiffVector x = new RereDiffVector(vecData);
        IDiffVector z = A.matmul(x);
        assertArrayEquals(new double[]{2, 6, 6}, z.getValue().getData(), TOL);
    }

    @Test
    void testMatmulBackwardToSparse() {
        // A (3x2) @ x (2) = y (3)
        // backward from y to A: dA = outer(grad, x)
        RereDiffSparseMatrix A = leaf(mat(new double[][]{{1, 0}, {0, 2}, {3, 0}}));
        double[] vecData = {2, 3};
        RereDiffVector x = new RereDiffVector(vecData);
        IDiffVector y = A.matmul(x);

        // backward with grad output = [1, 1, 1]
        y.backward();

        // dA = grad @ x^T = [1,1,1]^T @ [2,3] = [[2,3],[2,3],[2,3]]
        double[][] expected = {{2, 3}, {2, 3}, {2, 3}};
        ISparseMatrix gA = A.getGradient();
        assertNotNull(gA, "Sparse matrix A should receive gradient");
        assertMatrixEquals(expected, gA.toDenseArray(), TOL);
    }

    // ==================== backward twice (gradient accumulation) ====================

    @Test
    void testBackwardThenZeroThenBackward() {
        // Standard autodiff pattern: backward → read → zeroGrad → backward
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        IDiffSparseMatrix z = x.mul(2.0);
        z.backward(mat(new double[][]{{1, 0}, {0, 1}}));
        // gradient = 2 * [[1,0],[0,1]]
        assertMatrixEquals(new double[][]{{2, 0}, {0, 2}}, x.getGradient().toDenseArray(), TOL);

        x.zeroGradient();
        assertNull(x.getGradient());

        z.backward(mat(new double[][]{{1, 0}, {0, 1}}));
        assertMatrixEquals(new double[][]{{2, 0}, {0, 2}}, x.getGradient().toDenseArray(), TOL);
    }

    // ==================== zeroGradient ====================

    @Test
    void testZeroGradient() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        IDiffSparseMatrix z = x.add(x);
        z.backward(mat(new double[][]{{1, 0}, {0, 1}}));
        assertNotNull(x.getGradient());
        x.zeroGradient();
        assertNull(x.getGradient());
    }

    // ==================== grad (copy) ====================

    @Test
    void testGradCopy() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        IDiffSparseMatrix z = x.add(x);
        z.backward(mat(new double[][]{{5, 6}, {7, 8}}));
        IDiffSparseMatrix gradCopy = x.grad();
        assertNotNull(gradCopy);
        // grad copy should have the same values
        assertMatrixEquals(new double[][]{{10, 12}, {14, 16}}, gradCopy.getValue().toDenseArray(), TOL);
        // modifying original gradient should not affect copy
        x.zeroGradient();
        assertMatrixEquals(new double[][]{{10, 12}, {14, 16}}, gradCopy.getValue().toDenseArray(), TOL);
    }

    // ==================== isLeaf ====================

    @Test
    void testIsLeaf() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        assertTrue(x.isLeaf());
        IDiffSparseMatrix z = x.add(x);
        assertFalse(((RereDiffSparseMatrix) z).isLeaf());
    }

    // ==================== gradient with zero entries (sparsity preservation) ====================

    @Test
    void testReluBackwardZeroEntriesStayZero() {
        // Input with many zeros: relu should keep zero gradient for zeros
        RereDiffSparseMatrix x = leaf(mat(new double[][]{
            {-1, 0, 2},
            {0, -3, 4},
            {5, 0, -6}
        }));
        IDiffSparseMatrix z = x.relu();
        z.backward(mat(new double[][]{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}));
        double[][] g = x.getGradient().toDenseArray();
        double[][] expected = {
            {0, 0, 1},
            {0, 0, 1},
            {1, 0, 0}
        };
        assertMatrixEquals(expected, g, TOL);
    }

    // ==================== chain operations ====================

    @Test
    void testChainAddMulRelu() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{-1, 2}, {3, -4}}));
        RereDiffSparseMatrix y = leaf(mat(new double[][]{{1, 1}, {1, 1}}));
        // z = relu(x * 0.5 + y)
        IDiffSparseMatrix z = x.mul(0.5).add(y).relu();
        double[][] fwd = z.getValue().toDenseArray();
        // x*0.5 = [[-0.5, 1], [1.5, -2]]
        // +y     = [[0.5, 2], [2.5, -1]]
        // relu   = [[0.5, 2], [2.5, 0]]
        double[][] expectedFwd = {{0.5, 2}, {2.5, 0}};
        assertMatrixEquals(expectedFwd, fwd, TOL);

        z.backward(mat(new double[][]{{1, 1}, {1, 1}}));
        // gradient for x: chain rule: relu'(x*0.5+y) * 0.5
        // relu' = 1 where > 0, 0 where <= 0
        double[][] expectedGradX = {{0.5, 0.5}, {0.5, 0}};
        assertMatrixEquals(expectedGradX, x.getGradient().toDenseArray(), TOL);
    }

    // ==================== A2: Mixed sparse↔dense graph integration ====================

    @Test
    void testAsDenseDiffTensorForward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        IDiffTensor dense = x.asDenseDiffTensor();
        assertArrayEquals(new int[]{2, 2}, dense.shape());
        assertArrayEquals(new double[]{1, 2, 3, 4}, dense.toDoubleArray(), TOL);
    }

    @Test
    void testAsDenseDiffTensorBackward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        IDiffTensor dense = x.asDenseDiffTensor();
        // dense graph: sum(dense) = 1+2+3+4 = 10
        IDiffTensor loss = dense.sum();
        loss.backward();
        // gradient of sum w.r.t. each element = 1
        assertMatrixEquals(new double[][]{{1, 1}, {1, 1}}, x.getGradient().toDenseArray(), TOL);
    }

    @Test
    void testAsDenseDiffTensorBackwardWithComplexGrad() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{1, 2}, {3, 4}}));
        IDiffTensor dense = x.asDenseDiffTensor();
        // dense graph: dense.mul(2).relu().sum()
        IDiffTensor loss = dense.mul(2.0).relu().sum();
        loss.backward();
        // gradient chain: d(sum(relu(2*x)))/dx = relu'(2x) * 2
        // relu'(2x) = 1 for 2x>0 i.e., every element since all positive
        // So gradient = 2 for all elements
        assertMatrixEquals(new double[][]{{2, 2}, {2, 2}}, x.getGradient().toDenseArray(), TOL);
    }

    @Test
    void testSparseMatmulDenseThenSumBackward() {
        // A (2x2) @ v (2) → dense → sum → backward
        // Full chain: sparse → dense → backward flows back to sparse
        RereDiffSparseMatrix A = leaf(mat(new double[][]{{1, 0}, {0, 2}}));
        RereDiffVector v = new RereDiffVector(new double[]{3, 4});

        // sparse @ dense → dense vector
        IDiffVector y = A.matmul(v); // y = [3, 8]
        IDiffVector loss = y.sum();  // scalar loss = 11

        loss.backward();

        // A should get gradient: d(sum(A@v))/dA = outer(ones, v) = [[3,4],[3,4]]
        assertMatrixEquals(new double[][]{{3, 4}, {3, 4}}, A.getGradient().toDenseArray(), TOL);

        // v should get gradient: A^T @ ones = [1, 2]
        assertArrayEquals(new double[]{1, 2}, v.getGradient().getData(), TOL);
    }

    @Test
    void testSparseChainThroughDenseBridge() {
        // Sparse chain: (A.add(B)).asDenseDiffTensor() + dense_C → sum → backward
        // Verifies that upward sparse graph propagation works through asDenseDiffTensor
        RereDiffSparseMatrix A = leaf(mat(new double[][]{{1, 0}, {0, 2}}));
        RereDiffSparseMatrix B = leaf(mat(new double[][]{{3, 0}, {0, 4}}));
        IDiffSparseMatrix C = A.add(B); // [[4,0],[0,6]]
        IDiffTensor denseC = C.asDenseDiffTensor();

        // Build dense loss: sum(denseC) = 10
        IDiffTensor loss = denseC.sum();
        loss.backward();

        // Gradient = 1 for each element in both A and B
        assertMatrixEquals(new double[][]{{1, 1}, {1, 1}}, A.getGradient().toDenseArray(), TOL);
        assertMatrixEquals(new double[][]{{1, 1}, {1, 1}}, B.getGradient().toDenseArray(), TOL);
    }

    @Test
    void testSparseReluThenDenseBridgeBackward() {
        RereDiffSparseMatrix x = leaf(mat(new double[][]{{-2, 3}, {-1, 5}}));
        IDiffSparseMatrix r = x.relu(); // [[0, 3], [0, 5]]
        IDiffTensor denseR = r.asDenseDiffTensor();

        IDiffTensor loss = denseR.sum(); // loss = 8
        loss.backward();

        // relu gradient: 1 for x>0, else 0
        assertMatrixEquals(new double[][]{{0, 1}, {0, 1}}, x.getGradient().toDenseArray(), TOL);
    }

    // ==================== Utility assertion ====================

    private static void assertMatrixEquals(double[][] expected, double[][] actual, double tol) {
        assertEquals(expected.length, actual.length, "Row count mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], actual[i], tol,
                "Row " + i + " mismatch");
        }
    }
}
