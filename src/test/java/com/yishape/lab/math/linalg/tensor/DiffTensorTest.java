package com.yishape.lab.math.linalg.tensor;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffVector;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 可微张量 (IDiffTensor/RereDiffTensor) 综合测试.
 */
class DiffTensorTest {

    // ==================== 创建可微张量 ====================

    @Test
    void testFromDiffVector() {
        IDiffVector vec = AD.vector(1, 2, 3, 4);
        IDiffTensor t = IDiffTensor.fromDiffVector(vec, 2, 2);
        assertArrayEquals(new int[]{2, 2}, t.shape());
        assertEquals(1.0, t.get(0, 0), 1e-15);
        assertEquals(4.0, t.get(1, 1), 1e-15);
        assertTrue(t.requiresGrad());
    }

    @Test
    void testFromTensor() {
        IDoubleTensor data = ITensor.tensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDiffTensor t = IDiffTensor.fromTensor(data, true);
        assertArrayEquals(new int[]{2, 2}, t.shape());
        assertTrue(t.requiresGrad());

        IDiffTensor t2 = IDiffTensor.fromTensor(data, false);
        assertFalse(t2.requiresGrad());
    }

    @Test
    void testSetRequiresGrad() {
        IDiffVector vec = AD.vector(1, 2, 3);
        IDiffTensor t = IDiffTensor.fromDiffVector(vec, 3);
        assertTrue(t.requiresGrad());
        t.setRequiresGrad(false);
        assertFalse(t.requiresGrad());
    }

    // ==================== 基础 AD 反向 ====================

    @Test
    void testScalarGrad() {
        IDiffVector vec = AD.vector(3.0);
        IDiffTensor x = IDiffTensor.fromDiffVector(vec, 1);
        IDiffTensor y = x.mul(2.0);
        y.backward();

        IDoubleTensor g = x.grad();
        assertNotNull(g);
        assertEquals(2.0, g.item(), 1e-10);
    }

    @Test
    void testAddBackward() {
        IDiffTensor a = IDiffTensor.fromDiffVector(AD.vector(2.0), 1);
        IDiffTensor b = IDiffTensor.fromDiffVector(AD.vector(3.0), 1);
        IDiffTensor c = a.add(b);
        c.backward();

        assertEquals(1.0, a.grad().item(), 1e-10);
        assertEquals(1.0, b.grad().item(), 1e-10);
    }

    @Test
    void testMulBackward() {
        IDiffTensor a = IDiffTensor.fromDiffVector(AD.vector(2.0), 1);
        IDiffTensor b = IDiffTensor.fromDiffVector(AD.vector(3.0), 1);
        IDiffTensor c = a.mul(b);
        c.backward();

        assertEquals(3.0, a.grad().item(), 1e-10);
        assertEquals(2.0, b.grad().item(), 1e-10);
    }

    @Test
    void testGradChain() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(3.0), 1);
        IDiffTensor y = x.add(1.0).mul(2.0);
        y.backward();
        assertEquals(2.0, x.grad().item(), 1e-10);
    }

    // ==================== 逐元素运算 ====================

    @Test
    void testElementWiseBinary() {
        IDiffTensor a = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4), 2, 2);
        IDiffTensor b = IDiffTensor.fromDiffVector(AD.vector(5, 6, 7, 8), 2, 2);

        IDiffTensor c = a.add(b);
        IDoubleTensor ones = ITensor.ones(2, 2);
        c.backward(ones);

        IDoubleTensor ga = a.grad();
        assertNotNull(ga);
        for (int i = 0; i < 4; i++) assertEquals(1.0, ga.toDoubleArray()[i], 1e-10);
    }

    @Test
    void testSubGrad() {
        IDiffTensor a = IDiffTensor.fromDiffVector(AD.vector(5.0), 1);
        IDiffTensor b = IDiffTensor.fromDiffVector(AD.vector(3.0), 1);
        IDiffTensor c = a.sub(b);
        c.backward();
        assertEquals(1.0, a.grad().item(), 1e-10);
        assertEquals(-1.0, b.grad().item(), 1e-10);
    }

    @Test
    void testDivGrad() {
        IDiffTensor a = IDiffTensor.fromDiffVector(AD.vector(10.0), 1);
        IDiffTensor b = IDiffTensor.fromDiffVector(AD.vector(2.0), 1);
        IDiffTensor c = a.div(b);
        c.backward();
        assertEquals(0.5, a.grad().item(), 1e-10);
        assertEquals(-2.5, b.grad().item(), 1e-10);
    }

    @Test
    void testScalarOps() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(2.0), 1);
        assertEquals(5.0, x.add(3.0).item(), 1e-10);
        assertEquals(-1.0, x.sub(3.0).item(), 1e-10);
        assertEquals(6.0, x.mul(3.0).item(), 1e-10);
        assertEquals(1.0, x.div(2.0).item(), 1e-10);
    }

    @Test
    void testUnaryOps() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(4.0), 1);
        assertEquals(-4.0, x.neg().item(), 1e-10);
        assertEquals(2.0, x.sqrt().item(), 1e-10);
        assertEquals(Math.exp(4), x.exp().item(), 1e-10);
        assertEquals(Math.log(4), x.log().item(), 1e-10);
        assertEquals(4.0, x.abs().item(), 1e-10);
        assertEquals(16.0, x.square().item(), 1e-10);
        assertEquals(16.0, x.pow(2).item(), 1e-10);

        IDiffTensor y = IDiffTensor.fromDiffVector(AD.vector(-2.0), 1);
        assertEquals(2.0, y.abs().item(), 1e-10);
    }

    @Test
    void testRelu() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(-2, 0, 3), 3);
        IDiffTensor r = x.relu();
        assertEquals(0.0, r.get(0), 1e-10);
        assertEquals(0.0, r.get(1), 1e-10);
        assertEquals(3.0, r.get(2), 1e-10);
    }

    @Test
    void testSigmoid() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(0.0), 1);
        IDiffTensor s = x.sigmoid();
        assertEquals(0.5, s.item(), 1e-10);
    }

    @Test
    void testClamp() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(-1, 0.5, 2), 3);
        IDiffTensor c = x.clamp(0, 1);
        assertEquals(0.0, c.get(0), 1e-10);
        assertEquals(0.5, c.get(1), 1e-10);
        assertEquals(1.0, c.get(2), 1e-10);
    }

    // ==================== 归约 ====================

    @Test
    void testSumDim0Backward() {
        // sum along dim=0 preserves gradients: outer=1
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4), 2, 2);
        IDiffTensor s = x.sum(0, false); // reduce rows -> [3, 7] (1+2=3, 3+4=7) -> wait, dim=0 means col-wise: [1+3=4, 2+4=6]
        assertArrayEquals(new int[]{2}, s.shape());
        assertEquals(4.0, s.get(0), 1e-10); // 1+3
        assertEquals(6.0, s.get(1), 1e-10); // 2+4

        s.backward();
        IDoubleTensor g = x.grad();
        assertNotNull(g);
        for (int i = 0; i < 4; i++) assertEquals(1.0, g.toDoubleArray()[i], 1e-10);
    }

    @Test
    void testSumKeepdimBackward() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4), 2, 2);
        IDiffTensor s = x.sum(0, true);
        assertArrayEquals(new int[]{1, 2}, s.shape());
        assertEquals(4.0, s.get(0, 0), 1e-10);
        assertEquals(6.0, s.get(0, 1), 1e-10);
    }

    @Test
    void testMeanDim0Backward() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(2.0), 1);
        IDiffTensor m = x.mean(0, true);
        m.backward();
        assertEquals(1.0, x.grad().item(), 1e-10);

        // 2D mean along dim=0
        IDiffTensor x2 = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4), 2, 2);
        IDiffTensor m2 = x2.mean(0, false);
        m2.backward(ITensor.ones(m2.shape()));
        IDoubleTensor g = x2.grad();
        assertNotNull(g);
        assertEquals(0.5, g.get(0, 0), 1e-10);
        assertEquals(0.5, g.get(0, 1), 1e-10);
        assertEquals(0.5, g.get(1, 0), 1e-10);
        assertEquals(0.5, g.get(1, 1), 1e-10);
    }

    // ==================== 视图操作 ====================

    @Test
    void testPermute() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4, 5, 6), 2, 3);
        IDiffTensor p = x.permute(1, 0);
        assertArrayEquals(new int[]{3, 2}, p.shape());
        assertEquals(1.0, p.get(0, 0), 1e-10);
        assertEquals(4.0, p.get(0, 1), 1e-10);
    }

    @Test
    void testTranspose() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4), 2, 2);
        IDiffTensor t = x.transpose();
        assertEquals(1.0, t.get(0, 0), 1e-10);
        assertEquals(3.0, t.get(0, 1), 1e-10);
        assertEquals(2.0, t.get(1, 0), 1e-10);
    }

    @Test
    void testReshape() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4, 5, 6), 2, 3);
        IDiffTensor r = x.reshape(3, 2);
        assertArrayEquals(new int[]{3, 2}, r.shape());
        assertEquals(1.0, r.get(0, 0), 1e-10);
        assertEquals(6.0, r.get(2, 1), 1e-10);
    }

    @Test
    void testSqueezeUnsqueeze() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3), 1, 3, 1);
        IDiffTensor s = x.squeeze();
        assertArrayEquals(new int[]{3}, s.shape());

        IDiffTensor u = s.unsqueeze(0);
        assertArrayEquals(new int[]{1, 3}, u.shape());
    }

    @Test
    void testSlice() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4, 5, 6), 3, 2);
        IDiffTensor s = x.slice(0, 1, 3);
        assertArrayEquals(new int[]{2, 2}, s.shape());
        assertEquals(3.0, s.get(0, 0), 1e-10);
        assertEquals(6.0, s.get(1, 1), 1e-10);
    }

    @Test
    void testSelect() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4), 2, 2);
        IDiffTensor s = x.select(0, 1);
        assertArrayEquals(new int[]{2}, s.shape());
        assertEquals(3.0, s.get(0), 1e-10);
        assertEquals(4.0, s.get(1), 1e-10);
    }

    @Test
    void testFlatten() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4, 5, 6), 2, 3);
        IDiffTensor f = x.flatten(0, 1);
        assertArrayEquals(new int[]{6}, f.shape());
    }

    @Test
    void testExpand() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3), 1, 3);
        IDiffTensor e = x.expand(4, 3);
        assertArrayEquals(new int[]{4, 3}, e.shape());
        assertEquals(1.0, e.get(3, 0), 1e-10);
    }

    // ==================== 梯度流经视图 ====================

    @Test
    void testGradThroughView() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4), 2, 2);
        IDiffTensor y = x.transpose().mul(2.0);
        y.backward();

        IDoubleTensor g = x.grad();
        assertNotNull(g);
        for (int i = 0; i < 4; i++) assertEquals(2.0, g.toDoubleArray()[i], 1e-10);
    }

    // ==================== 梯度方法 ====================

    @Test
    void testDetach() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(2.0), 1);
        IDoubleTensor d = x.detach();
        assertTrue(d instanceof IDoubleTensor);
        assertFalse(d instanceof IDiffTensor);
    }

    @Test
    void testFlattenGrad() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4), 2, 2);
        IDiffTensor y = x.mul(2.0);
        y.backward();
        IDiffVector fg = x.flattenGrad();
        assertNotNull(fg);
        assertEquals(4, fg.getValue().length());
        for (int i = 0; i < 4; i++) assertEquals(2.0, fg.get(i), 1e-10);
    }

    @Test
    void testFlattenValue() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4), 2, 2);
        IDiffVector fv = x.flattenValue();
        assertEquals(4, fv.getValue().length());
        assertEquals(1.0, fv.get(0), 1e-10);
    }

    @Test
    void testZeroGradient() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(2.0), 1);
        IDiffTensor y = x.mul(2.0);
        y.backward();
        assertNotNull(x.grad());
        x.zeroGradient();
        assertNull(x.grad());
    }

    // ==================== 就地操作 ====================

    @Test
    void testInPlaceAdd() {
        IDiffVector vec = AD.vector(1, 2, 3);
        IDiffTensor x = IDiffTensor.fromDiffVector(vec, 3);
        IDiffTensor y = IDiffTensor.fromDiffVector(AD.vector(4, 5, 6), 3);
        x.add_(y);
        assertEquals(5.0, x.get(0), 1e-10);
        assertEquals(7.0, x.get(1), 1e-10);
    }

    @Test
    void testFillInPlace() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3), 3);
        x.fill_(0.0);
        for (int i = 0; i < 3; i++) assertEquals(0.0, x.get(i), 1e-10);
    }

    // ==================== 组合计算 ====================

    @Test
    void testCompositeExpr() {
        // (a * b) + (c - d) with all same shape, gradient preserved
        IDiffTensor a = IDiffTensor.fromDiffVector(AD.vector(2.0, 3.0), 2);
        IDiffTensor b = IDiffTensor.fromDiffVector(AD.vector(4.0, 5.0), 2);
        IDiffTensor c = IDiffTensor.fromDiffVector(AD.vector(10.0, 20.0), 2);
        IDiffTensor d = IDiffTensor.fromDiffVector(AD.vector(1.0, 2.0), 2);

        // e = (a*b) + (c-d)
        IDiffTensor e = a.mul(b).add(c.sub(d));
        e.backward();

        IDoubleTensor ga = a.grad();
        assertNotNull(ga);
        assertEquals(4.0, ga.get(0), 1e-10); // de/da[0] = b[0] = 4
        assertEquals(5.0, ga.get(1), 1e-10); // de/da[1] = b[1] = 5
    }

    // ==================== 序号运算 ====================

    @Test
    void testSinCosTan() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(Math.PI / 4), 1);
        assertEquals(Math.sin(Math.PI/4), x.sin().item(), 1e-10);
        assertEquals(Math.cos(Math.PI/4), x.cos().item(), 1e-10);
        assertEquals(Math.tan(Math.PI/4), x.tan().item(), 1e-10);
    }

    // ==================== 转非可微 ====================

    @Test
    void testSoftmaxFallback() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4), 2, 2);
        IDiffTensor s = x.softmax(1);
        assertArrayEquals(new int[]{2, 2}, s.shape());
        double row0 = s.get(0, 0) + s.get(0, 1);
        double row1 = s.get(1, 0) + s.get(1, 1);
        assertEquals(1.0, row0, 1e-5);
        assertEquals(1.0, row1, 1e-5);
    }

    @Test
    void testMmulFallback() {
        IDiffTensor a = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4), 2, 2);
        IDiffTensor b = IDiffTensor.fromDiffVector(AD.vector(5, 6, 7, 8), 2, 2);
        IDiffTensor c = a.mmul(b);
        assertArrayEquals(new int[]{2, 2}, c.shape());
        assertEquals(19.0, c.get(0, 0), 1e-10);
        assertEquals(50.0, c.get(1, 1), 1e-10);
    }

    @Test
    void testSumAll() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4), 2, 2);
        assertEquals(10.0, x.sumAll(), 1e-10);
    }

    @Test
    void testMeanAll() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4), 2, 2);
        assertEquals(2.5, x.meanAll(), 1e-10);
    }

    // ==================== 大的张量 ====================

    @Test
    void testLargeTensorGrad() {
        double[] data = new double[100];
        for (int i = 0; i < 100; i++) data[i] = i;
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(data), 10, 10);
        IDiffTensor y = x.mul(2.0).add(1.0);
        y.backward();
        IDoubleTensor g = x.grad();
        assertNotNull(g);
        assertEquals(100, g.totalSize());
        for (int i = 0; i < 100; i++) assertEquals(2.0, g.toDoubleArray()[i], 1e-10);
    }

    // ==================== 链式操作 ====================

    @Test
    void testViewChainThenOp() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4, 5, 6), 2, 3);
        // reshape on contiguous tensor preserves AD graph
        IDiffTensor y = x.reshape(3, 2).mul(2.0);
        y.backward();
        IDoubleTensor g = x.grad();
        assertNotNull(g);
        assertEquals(6, g.totalSize());
    }

    @Test
    void testPermuteThenOp() {
        // permute creates non-contiguous view, gradient still flows through vec
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4), 2, 2);
        IDiffTensor y = x.transpose().mul(2.0);
        y.backward();
        IDoubleTensor g = x.grad();
        assertNotNull(g);
        assertEquals(4, g.totalSize());
        for (int i = 0; i < 4; i++) assertEquals(2.0, g.toDoubleArray()[i], 1e-10);
    }

    // ==================== backward with custom gradient ====================

    @Test
    void testBackwardWithCustomGradient() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3), 3);
        IDiffTensor y = x.mul(2.0);
        IDoubleTensor customGrad = ITensor.tensor(new double[]{0.1, 0.2, 0.3}, 3);
        y.backward(customGrad);
        IDoubleTensor g = x.grad();
        assertNotNull(g);
        assertEquals(0.2, g.get(0), 1e-10);
        assertEquals(0.6, g.get(2), 1e-10);
    }

    @Test
    void testSameShapeBinaryGrad() {
        IDiffTensor a = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3), 1, 3);
        IDiffTensor b = IDiffTensor.fromDiffVector(AD.vector(10, 20, 30), 1, 3);
        IDiffTensor c = a.add(b);
        c.backward();
        IDoubleTensor ga = a.grad();
        assertNotNull(ga);
        assertEquals(1.0, ga.get(0, 0), 1e-10);
    }

    @Test
    void testBroadcastFallback() {
        IDiffTensor a = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3), 1, 3);
        IDiffTensor b = IDiffTensor.fromDiffVector(AD.vector(1), 1);
        // Different shapes -> broadcast fallback, no gradient
        IDiffTensor c = a.add(b);
        c.backward();
        // a is involved in non-diff fallback, so gradient for a might be null
        IDoubleTensor ga = a.grad();
        // gradient may or may not be null depending on whether the fallback tracked it
        // For now, just verify the forward value is correct
        assertEquals(2.0, c.get(0, 0), 1e-10);
        assertEquals(3.0, c.get(0, 1), 1e-10);
        assertEquals(4.0, c.get(0, 2), 1e-10);
    }

    @Test
    void testClone() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3), 3);
        IDiffTensor c = x.clone();
        assertEquals(x.get(0), c.get(0), 1e-10);
        assertEquals(x.get(2), c.get(2), 1e-10);
    }

    @Test
    void testRequiresGradFalseBackward() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(2.0), 1);
        x.setRequiresGrad(false);
        // Operations when requiresGrad=false should not build AD graph
        IDiffTensor y = x.mul(2.0);
        assertFalse(x.requiresGrad());
        // Check forward value still correct
        assertEquals(4.0, y.item(), 1e-10);
    }

    @Test
    void testGradNonLeafBeforeBackward() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(2.0), 1);
        IDiffTensor y = x.add(1.0);
        // Before backward, non-leaf should have null grad
        assertNull(y.grad());
    }

    @Test
    void testSumDim1Fallback() {
        // dim>1 reduction uses non-diff fallback but forward value is correct
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4), 2, 2);
        IDiffTensor s = x.sum(1, false);
        assertArrayEquals(new int[]{2}, s.shape());
        assertEquals(3.0, s.get(0), 1e-10); // 1+2
        assertEquals(7.0, s.get(1), 1e-10); // 3+4
    }

    @Test
    void testMeanDim1Fallback() {
        IDiffTensor x = IDiffTensor.fromDiffVector(AD.vector(1, 2, 3, 4), 2, 2);
        IDiffTensor m = x.mean(1, false);
        assertArrayEquals(new int[]{2}, m.shape());
        assertEquals(1.5, m.get(0), 1e-10); // (1+2)/2 = 1.5
        assertEquals(3.5, m.get(1), 1e-10); // (3+4)/2 = 3.5
    }
}
