package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Verification tests for Phase 0 new ops: triu, logSumExp, diag, diagonal, trace, split, chunk, unbind.
 */
public class Phase0OpsTest {

    // ==================== triu ====================

    @Test
    public void testTriuMirrorsTril() {
        IDiffTensor x = AD.leafTensor(new double[]{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        }, 3, 3);

        IDiffTensor upper = x.triu(0);
        IDiffTensor lower = x.tril(-1);
        // triu(0) + tril(-1) should recover all elements (triu includes diagonal, tril excludes it)
        IDiffTensor sum = upper.add(lower);
        double[] expected = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        assertArrayEquals(expected, sum.toDoubleArray(), 1e-12);
    }

    @Test
    public void testTriuGradient() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
        IDiffTensor y = x.triu(0).sum();
        y.backward();
        IDoubleTensor g = x.grad();
        // Upper triangle (incl diagonal) has gradient 1, lower has 0
        double[] expected = {1, 1, 1, 0, 1, 1, 0, 0, 1};
        assertArrayEquals(expected, g.toDoubleArray(), 1e-12);
    }

    @Test
    public void testTriuWithDiagonalOffset() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
        IDiffTensor y = x.triu(1); // exclude main diagonal
        double[] expected = {0, 2, 3, 0, 0, 6, 0, 0, 0};
        assertArrayEquals(expected, y.toDoubleArray(), 1e-12);
    }

    @Test
    public void testTriuScalarNoop() {
        IDiffTensor x = AD.leafTensor(new double[]{5}, 1);
        IDiffTensor y = x.triu(0);
        assertArrayEquals(new double[]{5}, y.toDoubleArray(), 1e-12);
    }

    // ==================== logSumExp ====================

    @Test
    public void testLogSumExpForward() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor y = x.logSumExp(1, false); // shape [2]
        double[] expected = new double[2];
        for (int i = 0; i < 2; i++) {
            double max = Math.max(x.toDoubleArray()[i * 3], Math.max(x.toDoubleArray()[i * 3 + 1], x.toDoubleArray()[i * 3 + 2]));
            double sumExp = 0;
            for (int j = 0; j < 3; j++) sumExp += Math.exp(x.toDoubleArray()[i * 3 + j] - max);
            expected[i] = Math.log(sumExp) + max;
        }
        double[] actual = y.toDoubleArray();
        assertArrayEquals(expected, actual, 1e-10);
    }

    @Test
    public void testLogSumExpAllDim() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor y = x.logSumExp(0, false); // shape [3]
        double[] expected = {
            Math.log(Math.exp(1) + Math.exp(4)),
            Math.log(Math.exp(2) + Math.exp(5)),
            Math.log(Math.exp(3) + Math.exp(6))
        };
        assertArrayEquals(expected, y.toDoubleArray(), 1e-10);
    }

    @Test
    public void testLogSumExpNumericalStability() {
        // Large values should not overflow
        IDiffTensor x = AD.leafTensor(new double[]{1000, 1001, 1002, -1000, -1001, -1002}, 2, 3);
        IDiffTensor y = x.logSumExp(1, false);
        assertFalse(Double.isInfinite(y.toDoubleArray()[0]));
        assertFalse(Double.isNaN(y.toDoubleArray()[0]));
        // log(exp(1000) + exp(1001) + exp(1002)) ≈ 1002 + log(1 + exp(-1) + exp(-2))
        double expected0 = 1002 + Math.log(1 + Math.exp(-1) + Math.exp(-2));
        assertEquals(expected0, y.toDoubleArray()[0], 1e-6);
        // For negative values: log(exp(-1000) + exp(-1001) + exp(-1002)) ≈ -1000 + log(1 + exp(-1) + exp(-2))
        double expected1 = -1000 + Math.log(1 + Math.exp(-1) + Math.exp(-2));
        assertEquals(expected1, y.toDoubleArray()[1], 1e-6);
    }

    @Test
    public void testLogSumExpGradient() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor y = x.logSumExp(1, false);
        y.backward();
        IDoubleTensor g = x.grad();
        // d(lse)/dx_i = exp(x_i) / sum(exp(x))
        double[] vals = x.toDoubleArray();
        for (int i = 0; i < 2; i++) {
            double sumExp = 0;
            for (int j = 0; j < 3; j++) sumExp += Math.exp(vals[i * 3 + j]);
            for (int j = 0; j < 3; j++) {
                double expectedGrad = Math.exp(vals[i * 3 + j]) / sumExp;
                assertEquals(expectedGrad, g.toDoubleArray()[i * 3 + j], 1e-10);
            }
        }
    }

    // ==================== diag / diagonal / trace ====================

    @Test
    public void testDiagForward() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDiffTensor d = x.diag();
        assertArrayEquals(new double[]{1, 4}, d.toDoubleArray(), 1e-12);
        assertArrayEquals(new int[]{2}, d.shape());
    }

    @Test
    public void testDiagGradient() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDiffTensor y = x.diag().sum();
        y.backward();
        IDoubleTensor g = x.grad();
        // Gradient: 1 on diagonal, 0 off-diagonal
        assertArrayEquals(new double[]{1, 0, 0, 1}, g.toDoubleArray(), 1e-12);
    }

    @Test
    public void testDiagonalOffset() {
        IDiffTensor x = AD.leafTensor(new double[]{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        }, 3, 3);
        IDiffTensor d1 = x.diagonal(1, 0, 1); // super-diagonal: [2, 6]
        assertArrayEquals(new double[]{2, 6}, d1.toDoubleArray(), 1e-12);
        IDiffTensor dm1 = x.diagonal(-1, 0, 1); // sub-diagonal: [4, 8]
        assertArrayEquals(new double[]{4, 8}, dm1.toDoubleArray(), 1e-12);
    }

    @Test
    public void testTraceForward() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDiffTensor t = x.trace();
        assertEquals(5.0, t.toDoubleArray()[0], 1e-12);
    }

    @Test
    public void testTraceGradient() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
        IDiffTensor y = x.trace();
        y.backward();
        IDoubleTensor g = x.grad();
        // Gradient: 1 on diagonal
        double[] expected = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        assertArrayEquals(expected, g.toDoubleArray(), 1e-12);
    }

    // ==================== split / chunk / unbind ====================

    @Test
    public void testSplitRoundTrip() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor[] pieces = x.split(2, 0); // split into [2,3] and [0,3] — but dim 0 size is 2, so 1 piece
        assertEquals(1, pieces.length);
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, pieces[0].toDoubleArray(), 1e-12);
    }

    @Test
    public void testSplitMultiPiece() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 4, 3);
        IDiffTensor[] pieces = x.split(2, 0); // 2 pieces: [0:2,:] and [2:4,:]
        assertEquals(2, pieces.length);
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, pieces[0].toDoubleArray(), 1e-12);
        assertArrayEquals(new double[]{7, 8, 9, 10, 11, 12}, pieces[1].toDoubleArray(), 1e-12);
    }

    @Test
    public void testSplitExplicitSizes() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 4, 3);
        IDiffTensor[] pieces = x.split(new int[]{1, 3}, 0);
        assertEquals(2, pieces.length);
        assertArrayEquals(new double[]{1, 2, 3}, pieces[0].toDoubleArray(), 1e-12);
        // narrow(0, 1, 3) on [4,3] gives shape [3,3] = 9 elements
        assertArrayEquals(new double[]{4, 5, 6, 7, 8, 9, 10, 11, 12}, pieces[1].toDoubleArray(), 1e-12);
    }

    @Test
    public void testSplitGradient() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor[] pieces = x.split(new int[]{1, 1}, 0);
        IDiffTensor y = pieces[0].sum().add(pieces[1].sum());
        y.backward();
        IDoubleTensor g = x.grad();
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, g.toDoubleArray(), 1e-12);
    }

    @Test
    public void testChunk() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 5, 2);
        IDiffTensor[] chunks = x.chunk(3, 0); // expected: 2, 2, 1
        assertEquals(3, chunks.length);
        assertEquals(2, chunks[0].dim(0));
        assertEquals(2, chunks[1].dim(0));
        assertEquals(1, chunks[2].dim(0));
    }

    @Test
    public void testUnbind() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor[] slices = x.unbind(0);
        assertEquals(2, slices.length);
        assertEquals(1, slices[0].rank());
        assertArrayEquals(new double[]{1, 2, 3}, slices[0].toDoubleArray(), 1e-12);
        assertArrayEquals(new double[]{4, 5, 6}, slices[1].toDoubleArray(), 1e-12);
    }

    @Test
    public void testUnbindGradient() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor[] slices = x.unbind(0);
        IDiffTensor y = slices[0].sum().add(slices[1].sum());
        y.backward();
        IDoubleTensor g = x.grad();
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, g.toDoubleArray(), 1e-12);
    }

    // ==================== constant tensor coverage ====================

    @Test
    public void testConstantTriu() {
        IDiffTensor x = AD.constantTensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
        IDiffTensor y = x.triu(0);
        assertArrayEquals(new double[]{1, 2, 3, 0, 5, 6, 0, 0, 9}, y.toDoubleArray(), 1e-12);
    }

    @Test
    public void testConstantDiag() {
        IDiffTensor x = AD.constantTensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDiffTensor d = x.diag();
        assertArrayEquals(new double[]{1, 4}, d.toDoubleArray(), 1e-12);
    }

    @Test
    public void testConstantLogSumExp() {
        IDiffTensor x = AD.constantTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor y = x.logSumExp(1, false);
        assertEquals(2, y.shape().length == 2 ? y.shape()[0] : y.toDoubleArray().length);
    }

    @Test
    public void testConstantSplit() {
        IDiffTensor x = AD.constantTensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 4, 3);
        IDiffTensor[] pieces = x.split(2, 0);
        assertEquals(2, pieces.length);
    }

    @Test
    public void testConstantUnbind() {
        IDiffTensor x = AD.constantTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor[] slices = x.unbind(0);
        assertEquals(2, slices.length);
    }
}
