package com.yishape.lab.math.linalg.tensor;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;

/**
 * 多维张量层综合测试.
 */
class TensorTest {

    // ==================== 基本工厂 + 访问 ====================

    @Test
    void testZeros() {
        IDoubleTensor t = ITensor.zeros(2, 3);
        assertArrayEquals(new int[]{2, 3}, t.shape());
        assertEquals(2, t.rank());
        assertEquals(6, t.totalSize());
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                assertEquals(0.0, t.get(i, j), 1e-15);
    }

    @Test
    void testOnes() {
        IDoubleTensor t = ITensor.ones(4);
        assertArrayEquals(new int[]{4}, t.shape());
        for (int i = 0; i < 4; i++)
            assertEquals(1.0, t.get(i), 1e-15);
    }

    @Test
    void testTensorFromArray() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        assertEquals(1.0, t.get(0, 0), 1e-15);
        assertEquals(6.0, t.get(1, 2), 1e-15);
    }

    @Test
    void testScalar() {
        IDoubleTensor s = ITensor.scalar(42.0);
        assertEquals(1, s.rank());
        assertEquals(42.0, s.item(), 1e-15);
    }

    @Test
    void testArange() {
        IDoubleTensor t = ITensor.arange(0, 5, 1);
        assertArrayEquals(new int[]{5}, t.shape());
        for (int i = 0; i < 5; i++)
            assertEquals(i, t.get(i), 1e-15);
    }

    // ==================== 视图操作 ====================

    @Test
    void testPermute() {
        IDoubleTensor t = ITensor.tensor(new double[]{1,2,3,4,5,6}, 2, 3);
        IDoubleTensor p = t.permute(1, 0);
        assertArrayEquals(new int[]{3, 2}, p.shape());
        assertEquals(1.0, p.get(0, 0), 1e-15);
        assertEquals(4.0, p.get(0, 1), 1e-15);
        assertEquals(2.0, p.get(1, 0), 1e-15);
        assertEquals(3.0, p.get(2, 0), 1e-15);
    }

    @Test
    void testTranspose2D() {
        IDoubleTensor t = ITensor.tensor(new double[]{1,2,3,4}, 2, 2);
        IDoubleTensor tt = t.transpose();
        assertEquals(t.get(0, 1), tt.get(1, 0), 1e-15);
        assertEquals(t.get(1, 0), tt.get(0, 1), 1e-15);
    }

    @Test
    void testSqueezeUnsqueeze() {
        IDoubleTensor t = ITensor.tensor(new double[]{1,2,3,4,5,6}, 1, 2, 1, 3);
        assertEquals(4, t.rank());
        IDoubleTensor sq = t.squeeze();
        assertArrayEquals(new int[]{2, 3}, sq.shape());
        IDoubleTensor us = sq.unsqueeze(0);
        assertArrayEquals(new int[]{1, 2, 3}, us.shape());
    }

    @Test
    void testSlice() {
        IDoubleTensor t = ITensor.tensor(new double[]{1,2,3,4,5,6,7,8,9}, 3, 3);
        IDoubleTensor s = t.slice(0, 0, 2);
        assertArrayEquals(new int[]{2, 3}, s.shape());
        assertEquals(1.0, s.get(0, 0), 1e-15);
        assertEquals(6.0, s.get(1, 2), 1e-15);
    }

    @Test
    void testSelect() {
        IDoubleTensor t = ITensor.tensor(new double[]{1,2,3,4,5,6}, 2, 3);
        IDoubleTensor row = t.select(0, 1);
        assertArrayEquals(new int[]{3}, row.shape());
        assertEquals(4.0, row.get(0), 1e-15);
        assertEquals(5.0, row.get(1), 1e-15);
        assertEquals(6.0, row.get(2), 1e-15);
    }

    @Test
    void testExpand() {
        IDoubleTensor t = ITensor.tensor(new double[]{1,2,3}, 1, 3);
        IDoubleTensor e = t.expand(2, 3);
        assertArrayEquals(new int[]{2, 3}, e.shape());
        assertEquals(1.0, e.get(0, 0), 1e-15);
        assertEquals(1.0, e.get(1, 0), 1e-15);
        assertEquals(3.0, e.get(0, 2), 1e-15);
        assertEquals(3.0, e.get(1, 2), 1e-15);
    }

    @Test
    void testFlatten() {
        IDoubleTensor t = ITensor.ones(2, 3, 4);
        IDoubleTensor f = t.flatten(0, 1);
        assertArrayEquals(new int[]{6, 4}, f.shape());
        IDoubleTensor f2 = t.flatten(1, 2);
        assertArrayEquals(new int[]{2, 12}, f2.shape());
    }

    @Test
    void testReshape() {
        IDoubleTensor t = ITensor.tensor(new double[]{1,2,3,4,5,6}, 2, 3);
        IDoubleTensor r = t.reshape(3, 2);
        assertArrayEquals(new int[]{3, 2}, r.shape());
        assertEquals(1.0, r.get(0, 0), 1e-15);
        assertEquals(6.0, r.get(2, 1), 1e-15);

        // -1 auto infer
        IDoubleTensor r2 = t.reshape(6, -1);
        assertArrayEquals(new int[]{6, 1}, r2.shape());
    }

    @Test
    void testContiguous() {
        IDoubleTensor t = ITensor.tensor(new double[]{1,2,3,4}, 2, 2);
        IDoubleTensor p = t.permute(1, 0);
        assertFalse(p.isContiguous());
        IDoubleTensor c = p.contiguous();
        assertTrue(c.isContiguous());
        assertEquals(1.0, c.get(0, 0), 1e-15);
        assertEquals(2.0, c.get(1, 0), 1e-15);
    }

    @Test
    void testTile() {
        IDoubleTensor t = ITensor.tensor(new double[]{1,2}, 1, 2);
        IDoubleTensor r = t.tile(2, 3);
        assertArrayEquals(new int[]{2, 6}, r.shape());
        assertEquals(1.0, r.get(0, 0), 1e-15);
        assertEquals(2.0, r.get(0, 1), 1e-15);
        assertEquals(1.0, r.get(1, 0), 1e-15);
        assertEquals(2.0, r.get(1, 3), 1e-15);
    }

    // ==================== 逐元素运算 ====================

    @Test
    void testUnaryOps() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 4, 9}, 3);
        assertArrayEquals(new double[]{1, 2, 3}, t.sqrt().toDoubleArray(), 1e-10);
        assertArrayEquals(new double[]{Math.E, Math.pow(Math.E, 4), Math.pow(Math.E, 9)},
            t.exp().toDoubleArray(), 1e-10);
        assertArrayEquals(new double[]{0, Math.log(4), Math.log(9)},
            t.log().toDoubleArray(), 1e-10);
        assertArrayEquals(new double[]{-1, -4, -9}, t.neg().toDoubleArray(), 1e-10);
        assertArrayEquals(new double[]{1, 16, 81}, t.pow(2).toDoubleArray(), 1e-10);
    }

    @Test
    void testBinaryOps() {
        IDoubleTensor a = ITensor.tensor(new double[]{1, 2, 3}, 3);
        IDoubleTensor b = ITensor.tensor(new double[]{10, 20, 30}, 3);
        assertArrayEquals(new double[]{11, 22, 33}, a.add(b).toDoubleArray(), 1e-10);
        assertArrayEquals(new double[]{-9, -18, -27}, a.sub(b).toDoubleArray(), 1e-10);
        assertArrayEquals(new double[]{10, 40, 90}, a.mul(b).toDoubleArray(), 1e-10);
        assertArrayEquals(new double[]{0.1, 0.1, 0.1}, a.div(b).toDoubleArray(), 1e-10);
    }

    @Test
    void testScalarOps() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 2, 3}, 3);
        assertArrayEquals(new double[]{3, 4, 5}, t.add(2).toDoubleArray(), 1e-10);
        assertArrayEquals(new double[]{2, 4, 6}, t.mul(2).toDoubleArray(), 1e-10);
    }

    @Test
    void testBroadcasting() {
        IDoubleTensor a = ITensor.tensor(new double[]{1, 2, 3}, 1, 3);
        IDoubleTensor b = ITensor.tensor(new double[]{10, 20, 30}, 3, 1);
        IDoubleTensor r = a.add(b);
        assertArrayEquals(new int[]{3, 3}, r.shape());
        assertEquals(11.0, r.get(0, 0), 1e-10);
        assertEquals(22.0, r.get(1, 1), 1e-10);
        assertEquals(33.0, r.get(2, 2), 1e-10);
    }

    @Test
    void testBroadcastDifferentRank() {
        IDoubleTensor a = ITensor.tensor(new double[]{1, 2, 3}, 3);       // (3,)
        IDoubleTensor b = ITensor.tensor(new double[]{10, 20, 30}, 1, 3);  // (1,3)
        IDoubleTensor r = a.add(b);
        assertArrayEquals(new int[]{1, 3}, r.shape());
        assertEquals(11.0, r.get(0, 0), 1e-10);
    }

    @Test
    void testRelu() {
        IDoubleTensor t = ITensor.tensor(new double[]{-2, -1, 0, 1, 2}, 5);
        assertArrayEquals(new double[]{0, 0, 0, 1, 2}, t.relu().toDoubleArray(), 1e-10);
    }

    @Test
    void testClamp() {
        IDoubleTensor t = ITensor.tensor(new double[]{-5, 0, 5, 10}, 4);
        assertArrayEquals(new double[]{-2, 0, 5, 7}, t.clamp(-2, 7).toDoubleArray(), 1e-10);
    }

    // ==================== 归约 ====================

    @Test
    void testSumKeepdim() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDoubleTensor s = t.sum(0, true);
        assertArrayEquals(new int[]{1, 3}, s.shape());
        assertArrayEquals(new double[]{5, 7, 9}, s.toDoubleArray(), 1e-10);
        IDoubleTensor s2 = t.sum(1, true);
        assertArrayEquals(new int[]{2, 1}, s2.shape());
    }

    @Test
    void testSumNoKeepdim() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDoubleTensor s = t.sum(0, false);
        assertArrayEquals(new int[]{3}, s.shape());
        assertArrayEquals(new double[]{5, 7, 9}, s.toDoubleArray(), 1e-10);
    }

    @Test
    void testSumAll() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 2, 3, 4}, 2, 2);
        assertEquals(10.0, t.sumAll(), 1e-10);
    }

    @Test
    void testMeanKeepdim() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDoubleTensor m = t.mean(0, true);
        assertArrayEquals(new int[]{1, 3}, m.shape());
        assertArrayEquals(new double[]{2.5, 3.5, 4.5}, m.toDoubleArray(), 1e-10);
    }

    @Test
    void testMaxMinDim() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 5, 3, 9, 2, 4}, 2, 3);
        IDoubleTensor mx = t.max(1, false);
        assertArrayEquals(new double[]{5, 9}, mx.toDoubleArray(), 1e-10);
        IDoubleTensor mn = t.min(1, false);
        assertArrayEquals(new double[]{1, 2}, mn.toDoubleArray(), 1e-10);
    }

    @Test
    void testArgmax() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 5, 3, 9, 2, 4}, 2, 3);
        IDoubleTensor am = t.argmax(1);
        assertArrayEquals(new double[]{1, 0}, am.toDoubleArray(), 1e-10);
    }

    @Test
    void testProd() {
        IDoubleTensor t = ITensor.tensor(new double[]{2, 3, 4, 5}, 2, 2);
        assertEquals(120.0, t.prodAll(), 1e-10);
        IDoubleTensor p = t.prod(0, false);
        assertArrayEquals(new double[]{8, 15}, p.toDoubleArray(), 1e-10);
    }

    // ==================== Softmax ====================

    @Test
    void testSoftmax() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDoubleTensor s = t.softmax(1);
        assertArrayEquals(new int[]{2, 2}, s.shape());
        // each row sums to 1
        for (int i = 0; i < 2; i++) {
            double rowSum = s.get(i, 0) + s.get(i, 1);
            assertEquals(1.0, rowSum, 1e-10);
        }
    }

    @Test
    void testLogSoftmax() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDoubleTensor ls = t.logSoftmax(1);
        IDoubleTensor s = t.softmax(1);
        assertArrayEquals(s.log().toDoubleArray(), ls.toDoubleArray(), 1e-10);
    }

    // ==================== 线性代数 ====================

    @Test
    void testMmul() {
        IDoubleTensor a = ITensor.tensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDoubleTensor b = ITensor.tensor(new double[]{5, 6, 7, 8}, 2, 2);
        IDoubleTensor c = a.mmul(b);
        // [1 2; 3 4] @ [5 6; 7 8] = [19 22; 43 50]
        assertEquals(19.0, c.get(0, 0), 1e-10);
        assertEquals(22.0, c.get(0, 1), 1e-10);
        assertEquals(43.0, c.get(1, 0), 1e-10);
        assertEquals(50.0, c.get(1, 1), 1e-10);
    }

    @Test
    void testBmm() {
        IDoubleTensor a = ITensor.tensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2);
        IDoubleTensor b = ITensor.tensor(new double[]{1, 0, 0, 1, 1, 0, 0, 1}, 2, 2, 2);
        IDoubleTensor c = a.bmm(b);
        assertArrayEquals(new int[]{2, 2, 2}, c.shape());
        assertEquals(1.0, c.get(0, 0, 0), 1e-10);
        assertEquals(2.0, c.get(0, 0, 1), 1e-10);
        assertEquals(5.0, c.get(1, 0, 0), 1e-10);
        assertEquals(6.0, c.get(1, 0, 1), 1e-10);
    }

    // ==================== 高级操作 ====================

    @Test
    void testGatherScatter() {
        IDoubleTensor t = ITensor.tensor(new double[]{10, 20, 30, 40}, 2, 2);
        // gather along dim=1: for idx[i][j], out[i][j] = t[i][idx[i][j]]
        IDoubleTensor idx = ITensor.tensor(new double[]{0, 1, 0, 1}, 2, 2);
        IDoubleTensor g = t.gather(1, idx);
        assertArrayEquals(new double[]{10, 20, 30, 40}, g.toDoubleArray(), 1e-10);
    }

    @Test
    void testCat() {
        IDoubleTensor a = ITensor.tensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDoubleTensor b = ITensor.tensor(new double[]{5, 6, 7, 8}, 2, 2);
        IDoubleTensor c = a.cat(0, b);
        assertArrayEquals(new int[]{4, 2}, c.shape());
        assertEquals(5.0, c.get(2, 0), 1e-10);
        IDoubleTensor c1 = a.cat(1, b);
        assertArrayEquals(new int[]{2, 4}, c1.shape());
    }

    @Test
    void testStack() {
        IDoubleTensor a = ITensor.tensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDoubleTensor b = ITensor.tensor(new double[]{5, 6, 7, 8}, 2, 2);
        IDoubleTensor s = a.stack(0, b);
        assertArrayEquals(new int[]{2, 2, 2}, s.shape());
        assertEquals(1.0, s.get(0, 0, 0), 1e-10);
        assertEquals(5.0, s.get(1, 0, 0), 1e-10);
    }

    @Test
    void testUnstack() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        List<IDoubleTensor> rows = t.unstack(0);
        assertEquals(2, rows.size());
        assertArrayEquals(new int[]{3}, rows.get(0).shape());
        assertEquals(1.0, rows.get(0).get(0), 1e-10);
        assertEquals(4.0, rows.get(1).get(0), 1e-10);
    }

    @Test
    void testWhere() {
        IDoubleTensor cond = ITensor.tensor(new double[]{1, 0, 1, 0}, 2, 2);
        IDoubleTensor a = ITensor.tensor(new double[]{10, 20, 30, 40}, 2, 2);
        IDoubleTensor b = ITensor.tensor(new double[]{100, 200, 300, 400}, 2, 2);
        IDoubleTensor w = a.where(cond, b);
        assertEquals(10.0, w.get(0, 0), 1e-10);
        assertEquals(200.0, w.get(0, 1), 1e-10);
        assertEquals(30.0, w.get(1, 0), 1e-10);
        assertEquals(400.0, w.get(1, 1), 1e-10);
    }

    @Test
    void testPad() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDoubleTensor p = t.pad(new int[][]{{1, 1}, {1, 1}}, "constant", 0.0);
        assertArrayEquals(new int[]{4, 4}, p.shape());
        assertEquals(0.0, p.get(0, 0), 1e-10);
        assertEquals(1.0, p.get(1, 1), 1e-10);
        assertEquals(0.0, p.get(3, 3), 1e-10);
    }

    @Test
    void testTopk() {
        IDoubleTensor t = ITensor.tensor(new double[]{5, 1, 3, 9, 2, 7}, 2, 3);
        IDoubleTensor tk = t.topk(2, 1, true);
        assertArrayEquals(new int[]{2, 2}, tk.shape());
        assertEquals(5.0, tk.get(0, 0), 1e-10);
        assertEquals(3.0, tk.get(0, 1), 1e-10);
        assertEquals(9.0, tk.get(1, 0), 1e-10);
        assertEquals(7.0, tk.get(1, 1), 1e-10);
    }

    @Test
    void testUnfold() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
        IDoubleTensor u = t.unfold(1, 2, 1, 1);
        assertArrayEquals(new int[]{3, 2, 2}, u.shape());
        assertEquals(1.0, u.get(0, 0, 0), 1e-10);
        assertEquals(2.0, u.get(0, 0, 1), 1e-10);
    }

    @Test
    void testNonzero() {
        IDoubleTensor t = ITensor.tensor(new double[]{0, 1, 0, 2, 0, 3}, 2, 3);
        IDoubleTensor nz = t.nonzero();
        assertArrayEquals(new int[]{3, 2}, nz.shape());
    }

    @Test
    void testEye() {
        IDoubleTensor eye = ITensor.eye(3);
        assertArrayEquals(new int[]{3, 3}, eye.shape());
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(i == j ? 1.0 : 0.0, eye.get(i, j), 1e-15);
            }
        }
    }

    // ==================== 视图链 ====================

    @Test
    void testViewChain() {
        // permute → slice → contiguous → reshape
        IDoubleTensor t = ITensor.arange(0, 24, 1).reshape(2, 3, 4);
        IDoubleTensor v = t.permute(2, 0, 1)   // (4, 2, 3)
            .slice(0, 0, 2)                      // (2, 2, 3)
            .contiguous();                        // materialize
        assertTrue(v.isContiguous());
        assertArrayEquals(new int[]{2, 2, 3}, v.shape());
    }

    @Test
    void testExpandThenAdd() {
        IDoubleTensor a = ITensor.tensor(new double[]{1, 2, 3}, 1, 3);
        IDoubleTensor b = ITensor.tensor(new double[]{10, 20, 30}, 1, 3);
        IDoubleTensor e = a.expand(2, 3);
        IDoubleTensor r = e.add(b);
        assertArrayEquals(new int[]{2, 3}, r.shape());
        assertEquals(11.0, r.get(0, 0), 1e-10);
        assertEquals(33.0, r.get(0, 2), 1e-10);
        assertEquals(11.0, r.get(1, 0), 1e-10);
    }

    // ==================== 转换 ====================

    @Test
    void testToVector() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDoubleVector v = t.toVector();
        assertEquals(6, v.length());
        assertEquals(1.0, v.get(0), 1e-10);
    }

    @Test
    void testToMatrix() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 2, 3, 4}, 2, 2);
        IMatrix m = t.toMatrix();
        assertEquals(2, m.getRowNum());
        assertEquals(2, m.getColNum());
        assertEquals(1.0, m.get(0, 0), 1e-10);
    }

    @Test
    void testBroadcastTo() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 2, 3}, 1, 3);
        IDoubleTensor b = t.broadcastTo(2, 3);
        assertArrayEquals(new int[]{2, 3}, b.shape());
        assertEquals(1.0, b.get(1, 0), 1e-10);
        assertEquals(3.0, b.get(1, 2), 1e-10);
    }

    @Test
    void testCumsum() {
        IDoubleTensor t = ITensor.tensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDoubleTensor cs = t.cumsum(0);
        assertArrayEquals(new double[]{1, 2, 4, 6}, cs.toDoubleArray(), 1e-10);
    }

    @Test
    void testNormalize() {
        IDoubleTensor t = ITensor.tensor(new double[]{3, 4}, 2);
        IDoubleTensor n = t.normalize(2, 0);
        // should be [0.6, 0.8]
        assertEquals(0.6, n.get(0), 1e-10);
        assertEquals(0.8, n.get(1), 1e-10);
    }

    @Test
    void testEinsumMmul() {
        IDoubleTensor a = ITensor.tensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDoubleTensor b = ITensor.tensor(new double[]{5, 6, 7, 8}, 2, 2);
        IDoubleTensor c = a.einsum("ij,jk->ik", b);
        assertEquals(19.0, c.get(0, 0), 1e-10);
        assertEquals(50.0, c.get(1, 1), 1e-10);
    }

    @Test
    void testVarStd() {
        IDoubleTensor t = ITensor.tensor(new double[]{1.0, 2.0, 3.0, 4.0}, 2, 2);
        IDoubleTensor v = t.var(0, false);
        IDoubleTensor s = t.std(0, false);
        // var = sum((x-mean)^2)/(n-1) = [1^2+1^2]/1 = 2 for each dim
        assertEquals(2.0, v.get(0), 1e-10);
        assertEquals(2.0, v.get(1), 1e-10);
        assertEquals(Math.sqrt(2.0), s.get(0), 1e-10);
    }

    @Test
    void testLargeTensor() {
        IDoubleTensor t = ITensor.rand(100, 100);
        assertEquals(10000, t.totalSize());
        IDoubleTensor s = t.sum(0, false);
        assertArrayEquals(new int[]{100}, s.shape());
        IDoubleTensor m = t.mean(1, true);
        assertArrayEquals(new int[]{100, 1}, m.shape());
    }

    @Test
    void testInPlaceOps() {
        IDoubleTensor a = ITensor.tensor(new double[]{1, 2, 3}, 3);
        IDoubleTensor b = ITensor.tensor(new double[]{10, 20, 30}, 3);
        a.add_(b);
        assertArrayEquals(new double[]{11, 22, 33}, a.toDoubleArray(), 1e-10);
    }
}
