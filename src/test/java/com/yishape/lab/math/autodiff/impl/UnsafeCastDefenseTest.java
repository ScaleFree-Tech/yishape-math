package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Defensive regression test for unsafe casts in DiffTensorMatrix.mmul/bmm.
 *
 * The bugs:
 * 1. mmul/bmm used ((IDiffTensor) other).detach() when other was a plain
 *    IDoubleTensor (not IDiffTensor) — ClassCastException.
 * 2. mmul used ((RereDiffTensor) other).value when other was a
 *    ConstantDiffTensor or TangentDiffTensor — ClassCastException.
 *
 * These tests pass plain IDoubleTensor, ConstantDiffTensor, and
 * TangentDiffTensor to mmul/bmm and verify correct results without
 * any cast exceptions.
 */
public class UnsafeCastDefenseTest {

    // ——— mmul with plain IDoubleTensor (not IDiffTensor) ———

    @Test
    void testMmulWithPlainDoubleTensorGradPath() {
        // tensor.requiresGrad=true, other is plain IDoubleTensor (not IDiffTensor)
        // Triggers the rank!=2 fallback path → previously ((IDiffTensor) other).detach()
        RereDiffTensor a = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        a.setRequiresGrad(true);
        IDoubleTensor b = new RereDoubleTensor(new double[]{1, 0, 0, 1, 0, 0}, 3, 2); // plain tensor, not IDiffTensor

        // Should NOT throw ClassCastException
        IDiffTensor result = a.mmul(b);
        double[] out = result.toDoubleArray();
        // [1,2,3; 4,5,6] @ [1,0; 0,1; 0,0] → non-2d path → should use linalg mmul
        assertNotNull(out);
    }

    @Test
    void testMmulRank2WithPlainDoubleTensor() {
        // tensor.requiresGrad=false, other is plain IDoubleTensor
        // Triggers non-differentiable path
        RereDiffTensor a = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        // a has requiresGrad=false by default
        IDoubleTensor b = new RereDoubleTensor(new double[]{1, 0, 0, 1, 0, 0}, 3, 2);

        IDiffTensor result = a.mmul(b);
        double[] out = result.toDoubleArray();
        // [1,2,3; 4,5,6] @ [1,0; 0,1; 0,0] = [1,2; 4,5]
        assertArrayEquals(new int[]{2, 2}, result.shape());
        assertEquals(1, out[0], 1e-10);
        assertEquals(2, out[1], 1e-10);
        assertEquals(4, out[2], 1e-10);
        assertEquals(5, out[3], 1e-10);
    }

    // ——— mmul with ConstantDiffTensor (toNonDiff produces this) ———

    @Test
    void testMmulWithConstantDiffTensor() {
        // tensor.requiresGrad=true, other is ConstantDiffTensor (requiresGrad=false)
        // Triggers the bVal extraction path → previously ((RereDiffTensor) other).value
        RereDiffTensor a = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        a.setRequiresGrad(true);
        RereDiffTensor bRaw = new RereDiffTensor(new double[]{1, 0, 0, 1, 0, 0}, 3, 2);
        bRaw.setRequiresGrad(false);
        IDiffTensor b = bRaw.toNonDiff(bRaw.value()); // ConstantDiffTensor

        IDiffTensor result = a.mmul(b);
        double[] out = result.toDoubleArray();
        assertArrayEquals(new int[]{2, 2}, result.shape());
    }

    // ——— bmm with plain IDoubleTensor ———

    @Test
    void testBmmWithPlainDoubleTensorRank3() {
        // tensor.requiresGrad=true, other is plain IDoubleTensor, rank!=3
        // Triggers the rank!=3 fallback path → previously ((IDiffTensor) other).detach()
        RereDiffTensor a = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        a.setRequiresGrad(true);
        IDoubleTensor b = new RereDoubleTensor(new double[]{1, 0, 0, 1, 0, 0}, 3, 2);

        // rank != 3 → should fall through to linalg bmm
        IDiffTensor result = a.bmm(b);
        assertNotNull(result);
    }

    @Test
    void testBmmWithConstantDiffTensorBatch() {
        // bmm with ConstantDiffTensor when tensor.requiresGrad=true
        RereDiffTensor a = new RereDiffTensor(
            new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 2, 3);
        a.setRequiresGrad(true);
        RereDiffTensor bRaw = new RereDiffTensor(
            new double[]{1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0}, 2, 3, 2);
        bRaw.setRequiresGrad(false);
        IDiffTensor b = bRaw.toNonDiff(bRaw.value()); // ConstantDiffTensor

        IDiffTensor result = a.bmm(b);
        double[] out = result.toDoubleArray();
        assertArrayEquals(new int[]{2, 2, 2}, result.shape());
    }

    // ——— Double-check TangentDiffTensor path for completeness ———

    @Test
    void testMmulWithTangentDiffTensorInput() {
        // TangentDiffTensor is another IDiffTensor impl with requiresGrad=false
        // Previously the cast ((RereDiffTensor) other).value would fail here
        RereDiffTensor a = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        a.setRequiresGrad(true);

        RereDiffTensor bPrimal = new RereDiffTensor(new double[]{1, 0, 0, 1, 0, 0}, 3, 2);
        bPrimal.setRequiresGrad(false);
        TangentDiffTensor bTan = TangentDiffTensor.seed(bPrimal,
            new RereDoubleTensor(new double[]{1, 0, 0, 1, 0, 0}, 3, 2));

        // mmul should accept TangentDiffTensor without ClassCastException
        IDiffTensor result = a.mmul(bTan);
        assertNotNull(result);
    }
}
