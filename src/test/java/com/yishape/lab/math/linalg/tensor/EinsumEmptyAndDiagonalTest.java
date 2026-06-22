package com.yishape.lab.math.linalg.tensor;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the two EinsumParser limitations fixed after the C2 batch-shift work:
 * <ol>
 *   <li>Empty (scalar) output — {@code "ij->"} (full reduction), {@code "ii->"} (trace),
 *       and the 2-input full reduction {@code "ij,jk->"}.</li>
 *   <li>Repeated labels within one input — {@code "ii->i"} (diagonal) and
 *       {@code "ii->"} (trace), across non-diff double, float, and differentiable paths.</li>
 * </ol>
 */
public class EinsumEmptyAndDiagonalTest {

    private static final double SQ3 = Math.sqrt(3);

    // ==================== empty output: single-input full reduction ====================

    @Test
    void testDoubleEmptyOutputSum() {
        IDoubleTensor a = new RereDoubleTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDoubleTensor s = a.einsum("ij->");
        assertEquals(1, s.totalSize(), "scalar output");
        assertEquals(21.0, s.toDoubleArray()[0], 1e-12);
    }

    @Test
    void testFloatEmptyOutputSum() {
        IFloatTensor a = new RereFloatTensor(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IFloatTensor s = a.einsum("ij->");
        assertEquals(1, s.totalSize());
        assertEquals(21f, s.toFloatArray()[0], 1e-5f);
    }

    @Test
    void testDiffEmptyOutputSum() {
        IDiffTensor a = AD.tensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor s = a.einsum("ij->");
        assertEquals(1, s.totalSize());
        assertEquals(21.0, s.toDoubleArray()[0], 1e-12);
        // gradient of sum w.r.t. every element is 1
        s.backward();
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, a.grad().toDoubleArray(), 1e-12);
    }

    // ==================== empty output: 2-input full reduction ====================

    @Test
    void testDoubleTwoInputEmptyOutput() {
        // "ij,jk->" = sum of all entries of A@B
        IDoubleTensor a = new RereDoubleTensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDoubleTensor b = new RereDoubleTensor(new double[]{1, 0, 0, 1}, 2, 2); // identity
        IDoubleTensor s = a.einsum("ij,jk->");
        // A@I = A, sum = 1+2+3+4 = 10
        assertEquals(1, s.totalSize());
        assertEquals(10.0, s.toDoubleArray()[0], 1e-12);
    }

    @Test
    void testDiffTwoInputEmptyOutputGradient() {
        IDiffTensor a = AD.tensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDiffTensor b = AD.tensor(new double[]{1, 1, 1, 1}, 2, 2);
        IDiffTensor s = a.einsum("ij,jk->");
        // A@B with B all-ones: C[i,j] = sum_k A[i,k]*1 = row sum; total = sum A = 10
        assertEquals(10.0, s.toDoubleArray()[0], 1e-12);
        s.backward();
        // d/dA[i,j] (sum_ij sum_k A[i,k] B[k,j]) = sum_j B[j,j]... = sum over j of B[j,?].
        // Actually d(total)/dA[i,k] = sum_j B[k,j] = row sum of B = 2 for each (i,k).
        assertArrayEquals(new double[]{2, 2, 2, 2}, a.grad().toDoubleArray(), 1e-12);
        // d/dB[k,j] = sum_i A[i,k] = column sum of A = [4,6]
        assertArrayEquals(new double[]{4, 6, 4, 6}, b.grad().toDoubleArray(), 1e-12);
    }

    // ==================== diagonal: "ii->i" ====================

    @Test
    void testDoubleDiagonal() {
        IDoubleTensor a = new RereDoubleTensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
        IDoubleTensor d = a.einsum("ii->i");
        assertArrayEquals(new int[]{3}, d.shape());
        assertArrayEquals(new double[]{1, 5, 9}, d.toDoubleArray(), 1e-12);
    }

    @Test
    void testFloatDiagonal() {
        IFloatTensor a = new RereFloatTensor(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
        IFloatTensor d = a.einsum("ii->i");
        assertArrayEquals(new int[]{3}, d.shape());
        assertArrayEquals(new float[]{1, 5, 9}, d.toFloatArray(), 1e-5f);
    }

    @Test
    void testDiffDiagonalValueAndGradient() {
        IDiffTensor a = AD.tensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
        IDiffTensor d = a.einsum("ii->i");
        assertArrayEquals(new int[]{3}, d.shape());
        assertArrayEquals(new double[]{1, 5, 9}, d.toDoubleArray(), 1e-12);
        // d.sum().backward() → gradient 1 on each diagonal position, 0 elsewhere
        d.sum().backward();
        assertArrayEquals(new double[]{1, 0, 0, 0, 1, 0, 0, 0, 1}, a.grad().toDoubleArray(), 1e-12);
    }

    @Test
    void testDiffDiagonalFiniteDifference() {
        double[] aData = {1.5, 0.2, -0.4, 0.3, 2.1, 0.7, -1.0, 0.5, 3.0};
        IDiffTensor a = AD.tensor(aData, 3, 3);
        // loss = sum of squares of diagonal
        IDiffTensor d = a.einsum("ii->i");
        IDiffTensor loss = d.square().sum();
        loss.backward();
        double[] analytic = a.grad().toDoubleArray();

        double eps = 1e-6;
        for (int idx = 0; idx < 9; idx++) {
            double[] p = aData.clone(); p[idx] += eps;
            double[] m = aData.clone(); p[idx] -= eps; m[idx] -= eps;
            // recompute p with fresh tensor
            double[] pp = aData.clone(); pp[idx] += eps;
            double lp = diagSqSum(pp);
            double lm = diagSqSum(m);
            double numeric = (lp - lm) / (2 * eps);
            // Only diagonal positions contribute; off-diagonal gradient is 0.
            assertEquals(analytic[idx], numeric, 1e-5, "idx " + idx);
        }
    }

    private static double diagSqSum(double[] data) {
        IDiffTensor a = AD.tensor(data, 3, 3);
        return a.einsum("ii->i").square().sum().toDoubleArray()[0];
    }

    // ==================== trace: "ii->" ====================

    @Test
    void testDoubleTrace() {
        IDoubleTensor a = new RereDoubleTensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
        IDoubleTensor t = a.einsum("ii->");
        assertEquals(1, t.totalSize());
        assertEquals(15.0, t.toDoubleArray()[0], 1e-12);
    }

    @Test
    void testFloatTrace() {
        IFloatTensor a = new RereFloatTensor(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
        IFloatTensor t = a.einsum("ii->");
        assertEquals(1, t.totalSize());
        assertEquals(15f, t.toFloatArray()[0], 1e-5f);
    }

    @Test
    void testDiffTraceGradient() {
        IDiffTensor a = AD.tensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
        IDiffTensor t = a.einsum("ii->");
        assertEquals(15.0, t.toDoubleArray()[0], 1e-12);
        t.backward();
        assertArrayEquals(new double[]{1, 0, 0, 0, 1, 0, 0, 0, 1}, a.grad().toDoubleArray(), 1e-12);
    }

    // ==================== error cases ====================

    @Test
    void testDiagonalRejectsNonSquare() {
        IDoubleTensor a = new RereDoubleTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        assertThrows(IllegalArgumentException.class, () -> a.einsum("ii->i"));
    }

    @Test
    void testDiagonalRejectsHigherRankRepeat() {
        IDoubleTensor a = new RereDoubleTensor(new double[8], 2, 2, 2);
        assertThrows(UnsupportedOperationException.class, () -> a.einsum("iij->ij"));
    }

    @Test
    void testDiagonalRejectsBadOutputLabel() {
        IDoubleTensor a = new RereDoubleTensor(new double[]{1, 2, 3, 4}, 2, 2);
        // output 'j' is not the repeated label 'i'
        assertThrows(UnsupportedOperationException.class, () -> a.einsum("ii->j"));
    }
}
