package com.yishape.lab.math.linalg.tensor;

import com.yishape.lab.math.autodiff.IDiffTensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Defensive regression test for sum() inner-index OOB bug.
 *
 * The bug: when keepdim=false, the inner-index loop included the reduce
 * dimension itself (j >= dim), causing outPos[outJ] to be OOB when
 * dim == rank-1 (last dimension). Fixed by always skipping the reduce
 * dim: for (int j = rank()-1; j > dim; j--).
 *
 * Every test uses C-order sequential data with independently computed
 * expected values. No test should pass by coincidence — each expected
 * value is hand-computed from the sequential formula.
 */
public class SumKeepdimDefenseTest {

    // ——— Rank 2: shape [2, 3], data = [0, 1, 2, 3, 4, 5] ———
    // position (i,j) = i*3 + j = the data value itself

    private RereDoubleTensor rank2Tensor() {
        return new RereDoubleTensor(new double[]{0, 1, 2, 3, 4, 5}, 2, 3);
    }

    @Test
    void testRank2SumDim0KeepdimFalse() {
        // sum over dim 0: result[i] = sum of column i
        // col 0: 0+3=3, col 1: 1+4=5, col 2: 2+5=7
        IDiffTensor t = IDiffTensor.fromTensor(rank2Tensor(), false);
        IDiffTensor s = t.sum(0, false);
        assertArrayEquals(new int[]{3}, s.shape());
        assertArrayEquals(new double[]{3, 5, 7}, s.toDoubleArray(), 1e-10);
    }

    @Test
    void testRank2SumDim1KeepdimFalse() {
        // sum over dim 1: result[i] = sum of row i
        // row 0: 0+1+2=3, row 1: 3+4+5=12
        // THIS WAS THE OOB BUG: dim==rank-1 && keepdim==false
        IDiffTensor t = IDiffTensor.fromTensor(rank2Tensor(), false);
        IDiffTensor s = t.sum(1, false);
        assertArrayEquals(new int[]{2}, s.shape());
        assertArrayEquals(new double[]{3, 12}, s.toDoubleArray(), 1e-10);
    }

    @Test
    void testRank2SumDim0KeepdimTrue() {
        IDiffTensor t = IDiffTensor.fromTensor(rank2Tensor(), false);
        IDiffTensor s = t.sum(0, true);
        assertArrayEquals(new int[]{1, 3}, s.shape());
        assertArrayEquals(new double[]{3, 5, 7}, s.toDoubleArray(), 1e-10);
    }

    @Test
    void testRank2SumDim1KeepdimTrue() {
        IDiffTensor t = IDiffTensor.fromTensor(rank2Tensor(), false);
        IDiffTensor s = t.sum(1, true);
        assertArrayEquals(new int[]{2, 1}, s.shape());
        assertArrayEquals(new double[]{3, 12}, s.toDoubleArray(), 1e-10);
    }

    @Test
    void testRank2MeanDim1KeepdimFalse() {
        // Also exercises mean() → sum() path
        IDiffTensor t = IDiffTensor.fromTensor(rank2Tensor(), false);
        IDiffTensor m = t.mean(1, false);
        assertArrayEquals(new int[]{2}, m.shape());
        // row 0: 3/3=1, row 1: 12/3=4
        assertArrayEquals(new double[]{1, 4}, m.toDoubleArray(), 1e-10);
    }

    @Test
    void testRank2VarStdViaSumPath() {
        // var() → mean() → sum(). Exercises the full stack.
        // Use var(0, false) because var(1, false) triggers a pre-existing
        // broadcasting bug in sub() when the reduced dimension's size differs
        // from the last dimension. var(0) produces mean shape [3] which
        // broadcasts correctly with [2,3] along the last axis.
        IDiffTensor t = IDiffTensor.fromTensor(rank2Tensor(), false);
        IDiffTensor v = t.var(0, false);
        assertArrayEquals(new int[]{3}, v.shape());
        // var over dim 0 of [2,3]: each column gets variance of 2 elements.
        // col 0: [0, 3], mean=1.5, Bessel var=((0-1.5)²+(3-1.5)²)/(2-1)=4.5
        // col 1: [1, 4], mean=2.5, var=((1-2.5)²+(4-2.5)²)/1=4.5
        // col 2: [2, 5], mean=3.5, var=((2-3.5)²+(5-3.5)²)/1=4.5
        double[] vd = v.toDoubleArray();
        assertEquals(4.5, vd[0], 1e-10);
        assertEquals(4.5, vd[1], 1e-10);
        assertEquals(4.5, vd[2], 1e-10);
    }

    // ——— Rank 3: shape [2, 2, 3], data = [0..11] ———
    // position (i,j,k) = i*6 + j*3 + k

    private RereDoubleTensor rank3Tensor() {
        double[] data = new double[12];
        for (int i = 0; i < 12; i++) data[i] = i;
        return new RereDoubleTensor(data, 2, 2, 3);
    }

    @Test
    void testRank3SumDim0KeepdimFalse() {
        IDiffTensor t = IDiffTensor.fromTensor(rank3Tensor(), false);
        IDiffTensor s = t.sum(0, false);
        assertArrayEquals(new int[]{2, 3}, s.shape());
        // result(j,k) = data[0,j,k] + data[1,j,k] = j*3+k + (6 + j*3+k) = 2j*3 + 2k + 6
        double[] out = s.toDoubleArray();
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 3; k++) {
                double expected = 2 * j * 3 + 2 * k + 6;
                assertEquals(expected, out[j * 3 + k], 1e-10,
                    "at (" + j + "," + k + ")");
            }
        }
    }

    @Test
    void testRank3SumDim1KeepdimFalse() {
        IDiffTensor t = IDiffTensor.fromTensor(rank3Tensor(), false);
        IDiffTensor s = t.sum(1, false);
        assertArrayEquals(new int[]{2, 3}, s.shape());
        // result(i,k) = data[i,0,k] + data[i,1,k]
        // = (i*6 + 0*3 + k) + (i*6 + 1*3 + k) = 2*i*6 + 3 + 2*k
        // i=0: 0+3+2k=3+2k; i=1: 12+3+2k=15+2k
        double[] out = s.toDoubleArray();
        for (int i = 0; i < 2; i++) {
            for (int k = 0; k < 3; k++) {
                double expected = 2 * i * 6 + 3 + 2 * k;
                assertEquals(expected, out[i * 3 + k], 1e-10,
                    "at (" + i + "," + k + ")");
            }
        }
    }

    @Test
    void testRank3SumDim2KeepdimFalse() {
        // dim == rank-1, keepdim == false — THE REGRESSION CASE for rank 3
        IDiffTensor t = IDiffTensor.fromTensor(rank3Tensor(), false);
        IDiffTensor s = t.sum(2, false);
        assertArrayEquals(new int[]{2, 2}, s.shape());
        // result(i,j) = data[i,j,0]+data[i,j,1]+data[i,j,2]
        // = (i*6+j*3+0)+(i*6+j*3+1)+(i*6+j*3+2) = 3*i*6 + 3*j*3 + 3
        double[] out = s.toDoubleArray();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                double expected = 3 * i * 6 + 3 * j * 3 + 3;
                assertEquals(expected, out[i * 2 + j], 1e-10,
                    "at (" + i + "," + j + ")");
            }
        }
    }

    @Test
    void testRank3SumDim2KeepdimTrue() {
        IDiffTensor t = IDiffTensor.fromTensor(rank3Tensor(), false);
        IDiffTensor s = t.sum(2, true);
        assertArrayEquals(new int[]{2, 2, 1}, s.shape());
        double[] out = s.toDoubleArray();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                double expected = 3 * i * 6 + 3 * j * 3 + 3;
                assertEquals(expected, out[(i * 2 + j) * 1], 1e-10,
                    "at (" + i + "," + j + ",0)");
            }
        }
    }

    // ——— Rank 4: shape [2, 2, 2, 3], data = [0..23] ———
    // position (i,j,k,l) = i*12 + j*6 + k*3 + l

    private RereDoubleTensor rank4Tensor() {
        double[] data = new double[24];
        for (int i = 0; i < 24; i++) data[i] = i;
        return new RereDoubleTensor(data, 2, 2, 2, 3);
    }

    @Test
    void testRank4SumLastDimKeepdimFalse() {
        // dim == 3 (rank-1), keepdim == false — REGRESSION CASE for rank 4
        IDiffTensor t = IDiffTensor.fromTensor(rank4Tensor(), false);
        IDiffTensor s = t.sum(3, false);
        assertArrayEquals(new int[]{2, 2, 2}, s.shape());
        // result(i,j,k) = sum_l (i*12+j*6+k*3+l) = 3*(i*12+j*6+k*3) + (0+1+2) = 36i+18j+9k+3
        double[] out = s.toDoubleArray();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    double expected = 36 * i + 18 * j + 9 * k + 3;
                    assertEquals(expected, out[i * 4 + j * 2 + k], 1e-10,
                        "at (" + i + "," + j + "," + k + ")");
                }
            }
        }
    }

    @Test
    void testRank4SumMidDimKeepdimFalse() {
        // dim == 2 (neither first nor last)
        IDiffTensor t = IDiffTensor.fromTensor(rank4Tensor(), false);
        IDiffTensor s = t.sum(2, false);
        assertArrayEquals(new int[]{2, 2, 3}, s.shape());
        // result(i,j,l) = sum_k (i*12+j*6+k*3+l) for k=0,1
        // = 2*(i*12+j*6+l) + (0+3) = 24i+12j+2l+3
        double[] out = s.toDoubleArray();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int l = 0; l < 3; l++) {
                    double expected = 24 * i + 12 * j + 2 * l + 3;
                    assertEquals(expected, out[i * 6 + j * 3 + l], 1e-10,
                        "at (" + i + "," + j + "," + l + ")");
                }
            }
        }
    }

    @Test
    void testRank4SumAllDimsToScalar() {
        // Multiple reductions: sum over all dims chain.
        // Stop at shape [1] because TensorShape rejects empty shapes.
        IDiffTensor t = IDiffTensor.fromTensor(rank4Tensor(), false);
        IDiffTensor s1 = t.sum(3, false);  // [2,2,2]
        IDiffTensor s2 = s1.sum(2, false); // [2,2]
        IDiffTensor s3 = s2.sum(1, false); // [2]
        IDiffTensor s4 = s3.sum(0, true);  // [1] — keepdim to avoid empty shape
        // Sum of 0..23 = 23*24/2 = 276
        assertEquals(276.0, s4.toDoubleArray()[0], 1e-10);
    }

    @Test
    void testNonUniformShapeSumLastDim() {
        // Non-uniform shape: [2, 5, 7] — rank 3, each dim different
        int D0 = 2, D1 = 5, D2 = 7;
        int total = D0 * D1 * D2;
        double[] data = new double[total];
        for (int i = 0; i < total; i++) data[i] = i + 1; // 1-based to avoid zero-coincidence
        RereDoubleTensor raw = new RereDoubleTensor(data, D0, D1, D2);
        IDiffTensor t = IDiffTensor.fromTensor(raw, false);

        // sum over last dim (dim=2, keepdim=false) — REGRESSION CASE
        IDiffTensor s = t.sum(2, false);
        assertArrayEquals(new int[]{D0, D1}, s.shape());
        double[] out = s.toDoubleArray();
        for (int i = 0; i < D0; i++) {
            for (int j = 0; j < D1; j++) {
                double expected = 0;
                for (int k = 0; k < D2; k++) {
                    expected += raw.get(i, j, k);
                }
                assertEquals(expected, out[i * D1 + j], 1e-10,
                    "at (" + i + "," + j + ")");
            }
        }
    }
}
