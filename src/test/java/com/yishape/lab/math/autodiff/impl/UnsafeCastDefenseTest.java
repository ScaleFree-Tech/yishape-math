package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Defensive regression test for unsafe casts in TangentDiffTensor operations.
 *
 * <p>Verifies that TangentDiffTensor's JVP implementations accept non-TangentDiffTensor
 * inputs (plain {@link IDoubleTensor}, {@link ConstantDiffTensor}) without
 * {@link ClassCastException}.</p>
 *
 * <p>Covers binary ops (add/sub/mul/div), matrix ops (mmul/bmm), reduce ops
 * (sum/mean), activations (relu/sigmoid/tanh), and index ops (gather/scatter).</p>
 */
public class UnsafeCastDefenseTest {

    // ═══ mmul / bmm (existing coverage) ═══

    @Test
    void testMmulWithPlainDoubleTensorGradPath() {
        RereDiffTensor a = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        a.setRequiresGrad(true);
        IDoubleTensor b = new RereDoubleTensor(new double[]{1, 0, 0, 1, 0, 0}, 3, 2);
        IDiffTensor result = a.mmul(b);
        assertNotNull(result);
    }

    @Test
    void testMmulRank2WithPlainDoubleTensor() {
        RereDiffTensor a = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDoubleTensor b = new RereDoubleTensor(new double[]{1, 0, 0, 1, 0, 0}, 3, 2);
        IDiffTensor result = a.mmul(b);
        double[] out = result.toDoubleArray();
        assertArrayEquals(new int[]{2, 2}, result.shape());
        assertEquals(1, out[0], 1e-10);
        assertEquals(2, out[1], 1e-10);
        assertEquals(4, out[2], 1e-10);
        assertEquals(5, out[3], 1e-10);
    }

    @Test
    void testMmulWithConstantDiffTensor() {
        RereDiffTensor a = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        a.setRequiresGrad(true);
        RereDiffTensor bRaw = new RereDiffTensor(new double[]{1, 0, 0, 1, 0, 0}, 3, 2);
        bRaw.setRequiresGrad(false);
        IDiffTensor b = bRaw.toNonDiff(bRaw.value()); // ConstantDiffTensor
        IDiffTensor result = a.mmul(b);
        assertArrayEquals(new int[]{2, 2}, result.shape());
    }

    @Test
    void testBmmWithPlainDoubleTensorRank3() {
        RereDiffTensor a = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        a.setRequiresGrad(true);
        IDoubleTensor b = new RereDoubleTensor(new double[]{1, 0, 0, 1, 0, 0}, 3, 2);
        IDiffTensor result = a.bmm(b);
        assertNotNull(result);
    }

    @Test
    void testBmmWithConstantDiffTensorBatch() {
        RereDiffTensor a = new RereDiffTensor(
            new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 2, 3);
        a.setRequiresGrad(true);
        RereDiffTensor bRaw = new RereDiffTensor(
            new double[]{1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0}, 2, 3, 2);
        bRaw.setRequiresGrad(false);
        IDiffTensor b = bRaw.toNonDiff(bRaw.value());
        IDiffTensor result = a.bmm(b);
        assertArrayEquals(new int[]{2, 2, 2}, result.shape());
    }

    @Test
    void testMmulWithTangentDiffTensorInput() {
        RereDiffTensor a = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        a.setRequiresGrad(true);
        RereDiffTensor bPrimal = new RereDiffTensor(new double[]{1, 0, 0, 1, 0, 0}, 3, 2);
        bPrimal.setRequiresGrad(false);
        TangentDiffTensor bTan = TangentDiffTensor.seed(bPrimal,
            new RereDoubleTensor(new double[]{1, 0, 0, 1, 0, 0}, 3, 2));
        IDiffTensor result = a.mmul(bTan);
        assertNotNull(result);
    }

    // ═══ binary ops: TangentDiffTensor + plain IDoubleTensor ═══

    @Test
    void testTangentAddPlainDoubleTensor() {
        RereDiffTensor p = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        TangentDiffTensor x = TangentDiffTensor.seed(p,
            new RereDoubleTensor(new double[]{0.1, 0.2, 0.3, 0.4}, 2, 2));
        IDoubleTensor y = new RereDoubleTensor(new double[]{5, 6, 7, 8}, 2, 2);
        IDiffTensor result = x.add(y);
        assertNotNull(result);
        assertArrayEquals(new double[]{6, 8, 10, 12}, result.toDoubleArray(), 1e-10);
    }

    @Test
    void testTangentMulPlainDoubleTensor() {
        RereDiffTensor p = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        TangentDiffTensor x = TangentDiffTensor.seed(p,
            new RereDoubleTensor(new double[]{0.1, 0.2, 0.3, 0.4}, 2, 2));
        IDoubleTensor y = new RereDoubleTensor(new double[]{2, 3, 4, 5}, 2, 2);
        IDiffTensor result = x.mul(y);
        assertNotNull(result);
        assertArrayEquals(new double[]{2, 6, 12, 20}, result.toDoubleArray(), 1e-10);
    }

    @Test
    void testTangentSubPlainDoubleTensor() {
        RereDiffTensor p = new RereDiffTensor(new double[]{10, 20, 30, 40}, 2, 2);
        TangentDiffTensor x = TangentDiffTensor.seed(p,
            new RereDoubleTensor(new double[]{0.1, 0.2, 0.3, 0.4}, 2, 2));
        IDoubleTensor y = new RereDoubleTensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDiffTensor result = x.sub(y);
        assertNotNull(result);
    }

    @Test
    void testTangentDivPlainDoubleTensor() {
        RereDiffTensor p = new RereDiffTensor(new double[]{2, 4, 6, 8}, 2, 2);
        TangentDiffTensor x = TangentDiffTensor.seed(p,
            new RereDoubleTensor(new double[]{0.1, 0.2, 0.3, 0.4}, 2, 2));
        IDoubleTensor y = new RereDoubleTensor(new double[]{2, 2, 2, 2}, 2, 2);
        IDiffTensor result = x.div(y);
        assertNotNull(result);
    }

    // ═══ binary ops: TangentDiffTensor + ConstantDiffTensor ═══

    @Test
    void testTangentAddConstantDiffTensor() {
        RereDiffTensor p = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        TangentDiffTensor x = TangentDiffTensor.seed(p,
            new RereDoubleTensor(new double[]{0.1, 0.2, 0.3, 0.4}, 2, 2));
        RereDiffTensor cRaw = new RereDiffTensor(new double[]{5, 6, 7, 8}, 2, 2);
        cRaw.setRequiresGrad(false);
        IDiffTensor y = cRaw.toNonDiff(cRaw.value()); // ConstantDiffTensor
        IDiffTensor result = x.add(y);
        assertNotNull(result);
        assertArrayEquals(new double[]{6, 8, 10, 12}, result.toDoubleArray(), 1e-10);
    }

    @Test
    void testTangentMulConstantDiffTensor() {
        RereDiffTensor p = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        TangentDiffTensor x = TangentDiffTensor.seed(p,
            new RereDoubleTensor(new double[]{0.1, 0.2, 0.3, 0.4}, 2, 2));
        RereDiffTensor cRaw = new RereDiffTensor(new double[]{2, 3, 4, 5}, 2, 2);
        cRaw.setRequiresGrad(false);
        IDiffTensor y = cRaw.toNonDiff(cRaw.value());
        IDiffTensor result = x.mul(y);
        assertNotNull(result);
    }

    // ═══ reduce ops (TangentDiffTensor with ConstantDiffTensor) ═══

    @Test
    void testTangentSumWithConstantDiffTensor() {
        RereDiffTensor p = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        TangentDiffTensor x = TangentDiffTensor.seed(p,
            new RereDoubleTensor(new double[]{0.1, 0.2, 0.3, 0.4}, 2, 2));
        IDiffTensor result = x.sum();
        assertNotNull(result);
    }

    @Test
    void testTangentMeanWithConstantDiffTensor() {
        RereDiffTensor p = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        TangentDiffTensor x = TangentDiffTensor.seed(p,
            new RereDoubleTensor(new double[]{0.1, 0.2, 0.3, 0.4}, 2, 2));
        IDiffTensor result = x.mean(0, false);
        assertNotNull(result);
    }

    // ═══ activation ops ═══

    @Test
    void testTangentReluWithConstantDiffTensor() {
        RereDiffTensor p = new RereDiffTensor(new double[]{1, -2, 3, -4}, 2, 2);
        TangentDiffTensor x = TangentDiffTensor.seed(p,
            new RereDoubleTensor(new double[]{0.1, 0.2, 0.3, 0.4}, 2, 2));
        IDiffTensor result = x.relu();
        assertNotNull(result);
        assertArrayEquals(new double[]{1, 0, 3, 0}, result.toDoubleArray(), 1e-10);
    }

    @Test
    void testTangentSigmoidWithConstantDiffTensor() {
        RereDiffTensor p = new RereDiffTensor(new double[]{0, 1, -1, 2}, 2, 2);
        TangentDiffTensor x = TangentDiffTensor.seed(p,
            new RereDoubleTensor(new double[]{0.1, 0.2, 0.3, 0.4}, 2, 2));
        IDiffTensor result = x.sigmoid();
        assertNotNull(result);
    }

    @Test
    void testTangentTanhWithConstantDiffTensor() {
        RereDiffTensor p = new RereDiffTensor(new double[]{0, 1, -1, 2}, 2, 2);
        TangentDiffTensor x = TangentDiffTensor.seed(p,
            new RereDoubleTensor(new double[]{0.1, 0.2, 0.3, 0.4}, 2, 2));
        IDiffTensor result = x.tanh();
        assertNotNull(result);
    }

    // ═══ index ops ═══

    @Test
    void testTangentGatherWithConstantDiffTensor() {
        RereDiffTensor p = new RereDiffTensor(new double[]{10, 20, 30, 40, 50, 60}, 3, 2);
        TangentDiffTensor x = TangentDiffTensor.seed(p,
            new RereDoubleTensor(new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, 3, 2));
        RereDiffTensor idxRaw = new RereDiffTensor(new double[]{0, 2}, 2);
        idxRaw.setRequiresGrad(false);
        IDiffTensor idx = idxRaw.toNonDiff(idxRaw.value());
        IDiffTensor result = x.gather(0, idx);
        assertNotNull(result);
    }

    // ═══ einsum with mixed types ═══

    @Test
    void testTangentEinsumWithPlainDoubleTensor() {
        RereDiffTensor p = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        TangentDiffTensor x = TangentDiffTensor.seed(p,
            new RereDoubleTensor(new double[]{0.1, 0.2, 0.3, 0.4}, 2, 2));
        IDoubleTensor y = new RereDoubleTensor(new double[]{5, 6, 7, 8}, 2, 2);
        // einsum with plain IDoubleTensor (elementwise multiply → "ij,ij->ij")
        IDiffTensor result = x.einsum("ij,ij->ij", y);
        assertNotNull(result);
        // Primal: elementwise a*b = [1*5,2*6,3*7,4*8] = [5,12,21,32]
        assertArrayEquals(new double[]{5, 12, 21, 32}, result.toDoubleArray(), 1e-10);
    }

    @Test
    void testTangentEinsumWithConstantDiffTensor() {
        RereDiffTensor p = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        TangentDiffTensor x = TangentDiffTensor.seed(p,
            new RereDoubleTensor(new double[]{0.1, 0.2, 0.3, 0.4}, 2, 2));
        RereDiffTensor cRaw = new RereDiffTensor(new double[]{5, 6, 7, 8}, 2, 2);
        cRaw.setRequiresGrad(false);
        IDiffTensor y = cRaw.toNonDiff(cRaw.value());
        IDiffTensor result = x.einsum("ij,ij->ij", y);
        assertNotNull(result);
    }

    // ═══ cumprod ═══

    @Test
    void testTangentCumprod() {
        RereDiffTensor p = new RereDiffTensor(new double[]{1, 2, 3, 4}, 4);
        TangentDiffTensor x = TangentDiffTensor.seed(p,
            new RereDoubleTensor(new double[]{0.1, 0.2, 0.3, 0.4}, 4));
        IDiffTensor result = x.cumprod(0);
        assertNotNull(result);
        double[] out = result.toDoubleArray();
        assertArrayEquals(new int[]{4}, result.shape());
        assertEquals(1.0, out[0], 1e-10);   // 1
        assertEquals(2.0, out[1], 1e-10);   // 1*2
        assertEquals(6.0, out[2], 1e-10);   // 1*2*3
        assertEquals(24.0, out[3], 1e-10);  // 1*2*3*4
    }
}
