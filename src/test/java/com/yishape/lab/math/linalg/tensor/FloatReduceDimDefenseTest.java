package com.yishape.lab.math.linalg.tensor;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Defensive regression test for RereFloatTensor.reduceDim() C-order stride bug.
 *
 * The bug: reduceDim used right-to-left sequential multiplication
 * {@code iIdx = iIdx * dim(j) + outPos[outJ]} to compute inner-dim flat
 * indices. This is wrong for C-order when innerCount >= 2.
 *
 * Fix matches RereDoubleTensor.reduceDim(): pre-compute C-order strides
 * and use {@code iIdx += mult[p] * outPos[srcIdx]}.
 *
 * max/min/prod delegate to reduceDim, so they all exercise this path.
 */
public class FloatReduceDimDefenseTest {

    // ——— Rank 3: shape [2, 3, 4], data = 1..24 ———
    // position (i,j,k) = i*12 + j*4 + k + 1

    private RereFloatTensor rank3Tensor() {
        float[] data = new float[24];
        for (int i = 0; i < 24; i++) data[i] = i + 1f;
        return new RereFloatTensor(data, 2, 3, 4);
    }

    @Test
    void testRank3MaxDim0KeepdimFalse() {
        // max over dim 0: result[j,k] = max(data[0,j,k], data[1,j,k])
        RereFloatTensor t = rank3Tensor();
        IFloatTensor m = t.max(0, false);
        assertArrayEquals(new int[]{3, 4}, m.shape());
        float[] out = m.toFloatArray();
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 4; k++) {
                float v0 = 1f + j * 4f + k;           // i=0: j*4+k+1
                float v1 = 1f + 12f + j * 4f + k;     // i=1: 12+j*4+k+1
                assertEquals(Math.max(v0, v1), out[j * 4 + k], 1e-6f,
                    "at (" + j + "," + k + ")");
            }
        }
    }

    @Test
    void testRank3MinDim1KeepdimFalse() {
        // min over dim 1: result[i,k] = min(data[i,0,k], data[i,1,k], data[i,2,k])
        RereFloatTensor t = rank3Tensor();
        IFloatTensor m = t.min(1, false);
        assertArrayEquals(new int[]{2, 4}, m.shape());
        float[] out = m.toFloatArray();
        for (int i = 0; i < 2; i++) {
            for (int k = 0; k < 4; k++) {
                // values at (i,0,k), (i,1,k), (i,2,k)
                float v0 = 1f + i * 12f + 0f * 4f + k;
                float v1 = 1f + i * 12f + 1f * 4f + k;
                float v2 = 1f + i * 12f + 2f * 4f + k;
                float expected = Math.min(v0, Math.min(v1, v2));
                assertEquals(expected, out[i * 4 + k], 1e-6f,
                    "at (" + i + "," + k + ")");
            }
        }
    }

    @Test
    void testRank3ProdDim2KeepdimFalse() {
        // prod over dim 2 (last dim): result[i,j] = prod(data[i,j,0..3])
        RereFloatTensor t = rank3Tensor();
        IFloatTensor p = t.prod(2, false);
        assertArrayEquals(new int[]{2, 3}, p.shape());
        float[] out = p.toFloatArray();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                float expected = 1f;
                for (int k = 0; k < 4; k++) {
                    expected *= (1f + i * 12f + j * 4f + k);
                }
                assertEquals(expected, out[i * 3 + j], 1e-3f,
                    "at (" + i + "," + j + ")");
            }
        }
    }

    @Test
    void testRank3MaxDim2KeepdimTrue() {
        // max over last dim with keepdim=true — same inner-dims regression risk
        RereFloatTensor t = rank3Tensor();
        IFloatTensor m = t.max(2, true);
        assertArrayEquals(new int[]{2, 3, 1}, m.shape());
        float[] out = m.toFloatArray();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                float expected = 1f + i * 12f + j * 4f + 3f; // k=3 is max
                assertEquals(expected, out[(i * 3 + j) * 1], 1e-6f,
                    "at (" + i + "," + j + ",0)");
            }
        }
    }

    // ——— Rank 4 non-uniform: shape [2, 3, 4, 5], data = 1..120 ———
    // position (i,j,k,l) = i*60 + j*20 + k*5 + l + 1

    private RereFloatTensor rank4Tensor() {
        float[] data = new float[120];
        for (int i = 0; i < 120; i++) data[i] = i + 1f;
        return new RereFloatTensor(data, 2, 3, 4, 5);
    }

    @Test
    void testRank4MaxDim0KeepdimFalse() {
        // max over dim 0, innerCount = 3 — STRESS CASE for old stride bug
        RereFloatTensor t = rank4Tensor();
        IFloatTensor m = t.max(0, false);
        assertArrayEquals(new int[]{3, 4, 5}, m.shape());
        float[] out = m.toFloatArray();
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 4; k++) {
                for (int l = 0; l < 5; l++) {
                    float v0 = 1f + 0f * 60f + j * 20f + k * 5f + l;
                    float v1 = 1f + 1f * 60f + j * 20f + k * 5f + l;
                    float expected = Math.max(v0, v1);
                    assertEquals(expected, out[j * 20 + k * 5 + l], 1e-6f,
                        "at (" + j + "," + k + "," + l + ")");
                }
            }
        }
    }

    @Test
    void testRank4MinDim1KeepdimFalse() {
        // min over dim 1 (innerCount = 2)
        RereFloatTensor t = rank4Tensor();
        IFloatTensor m = t.min(1, false);
        assertArrayEquals(new int[]{2, 4, 5}, m.shape());
        float[] out = m.toFloatArray();
        for (int i = 0; i < 2; i++) {
            for (int k = 0; k < 4; k++) {
                for (int l = 0; l < 5; l++) {
                    float expected = Float.POSITIVE_INFINITY;
                    for (int j = 0; j < 3; j++) {
                        float v = 1f + i * 60f + j * 20f + k * 5f + l;
                        if (v < expected) expected = v;
                    }
                    assertEquals(expected, out[i * 20 + k * 5 + l], 1e-6f,
                        "at (" + i + "," + k + "," + l + ")");
                }
            }
        }
    }

    @Test
    void testRank4ProdDim2KeepdimFalse() {
        // prod over dim 2 (innerCount = 1)
        RereFloatTensor t = rank4Tensor();
        IFloatTensor p = t.prod(2, false);
        assertArrayEquals(new int[]{2, 3, 5}, p.shape());
        float[] out = p.toFloatArray();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int l = 0; l < 5; l++) {
                    float expected = 1f;
                    for (int k = 0; k < 4; k++) {
                        expected *= (1f + i * 60f + j * 20f + k * 5f + l);
                    }
                    assertEquals(expected, out[i * 15 + j * 5 + l], 1e-3f,
                        "at (" + i + "," + j + "," + l + ")");
                }
            }
        }
    }

    @Test
    void testRank4MaxLastDimKeepdimFalse() {
        // max over last dim with keepdim=false — OOB regression for dim==rank-1
        RereFloatTensor t = rank4Tensor();
        IFloatTensor m = t.max(3, false);
        assertArrayEquals(new int[]{2, 3, 4}, m.shape());
        float[] out = m.toFloatArray();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 4; k++) {
                    float expected = 1f + i * 60f + j * 20f + k * 5f + 4f;
                    assertEquals(expected, out[i * 12 + j * 4 + k], 1e-6f,
                        "at (" + i + "," + j + "," + k + ")");
                }
            }
        }
    }

    @Test
    void testRank4ProdAllKeepdimPreservesShape() {
        // prod with keepdim=true should preserve rank with size-1 at reduced dims
        RereFloatTensor t = rank4Tensor();
        IFloatTensor p = t.prod(0, true);
        assertArrayEquals(new int[]{1, 3, 4, 5}, p.shape());
    }

    @Test
    void testSyncWithDoubleReduceDim() {
        // Cross-validate: RereFloatTensor.max() and RereDoubleTensor.max() should agree
        int D0 = 2, D1 = 3, D2 = 4;
        double[] dData = new double[24];
        float[] fData = new float[24];
        for (int i = 0; i < 24; i++) {
            dData[i] = i + 1.0;
            fData[i] = i + 1f;
        }
        RereDoubleTensor dt = new RereDoubleTensor(dData, D0, D1, D2);
        RereFloatTensor ft = new RereFloatTensor(fData, D0, D1, D2);

        for (int dim = 0; dim < 3; dim++) {
            double[] dOut = dt.max(dim, false).toDoubleArray();
            float[] fOut = ft.max(dim, false).toFloatArray();
            assertEquals(dOut.length, fOut.length, "length mismatch at dim=" + dim);
            for (int i = 0; i < dOut.length; i++) {
                assertEquals(dOut[i], (double) fOut[i], 1e-5,
                    "dim=" + dim + " pos=" + i);
            }
        }
    }
}
