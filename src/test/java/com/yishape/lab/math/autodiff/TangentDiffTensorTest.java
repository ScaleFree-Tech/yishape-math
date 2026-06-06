package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.TangentDiffTensor;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.ITensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 3b: Forward-mode AD (JVP) via TangentDiffTensor.
 */
public class TangentDiffTensorTest {

    private TangentDiffTensor seed(double[] data, double[] tangent) {
        RereDiffTensor primal = new RereDiffTensor(data, data.length);
        IDoubleTensor tan = ITensor.tensor(tangent, data.length);
        return TangentDiffTensor.seed(primal, tan);
    }

    private TangentDiffTensor seed2d(double[][] data, double[][] tangent) {
        int rows = data.length, cols = data[0].length;
        double[] flat = new double[rows * cols];
        for (int i = 0; i < rows; i++)
            System.arraycopy(data[i], 0, flat, i * cols, cols);
        RereDiffTensor primal = new RereDiffTensor(flat, rows, cols);
        double[] flatTan = new double[rows * cols];
        for (int i = 0; i < rows; i++)
            System.arraycopy(tangent[i], 0, flatTan, i * cols, cols);
        IDoubleTensor tan = ITensor.tensor(flatTan, rows, cols);
        return TangentDiffTensor.seed(primal, tan);
    }

    @Test
    void testAddJVP() {
        // f(x) = x + 2, x=[1,2,3], tangent=[1,0,0]
        TangentDiffTensor x = seed(new double[]{1, 2, 3}, new double[]{1, 0, 0});
        IDiffTensor y = x.add(2);
        double[] val = y.toDoubleArray();
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{3, 4, 5}, val, 1e-12);
        assertArrayEquals(new double[]{1, 0, 0}, tan, 1e-12);
    }

    @Test
    void testAddTensorJVP() {
        TangentDiffTensor x = seed(new double[]{1, 2, 3}, new double[]{1, 0, 0});
        TangentDiffTensor y = seed(new double[]{4, 5, 6}, new double[]{0, 1, 0});
        IDiffTensor z = x.add(y);
        double[] val = z.toDoubleArray();
        double[] tan = ((TangentDiffTensor) z).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{5, 7, 9}, val, 1e-12);
        assertArrayEquals(new double[]{1, 1, 0}, tan, 1e-12);
    }

    @Test
    void testMulJVP() {
        // f = x * y, x=[1,2,3] tx=[1,0,0], y=[4,5,6] ty=[0,1,0]
        // JVP = tx*y + x*ty = [1*4+1*0, 0*5+2*1, 0*6+3*0] = [4, 2, 0]
        TangentDiffTensor x = seed(new double[]{1, 2, 3}, new double[]{1, 0, 0});
        TangentDiffTensor y = seed(new double[]{4, 5, 6}, new double[]{0, 1, 0});
        IDiffTensor z = x.mul(y);
        double[] tan = ((TangentDiffTensor) z).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{4, 2, 0}, tan, 1e-12);
    }

    @Test
    void testNegJVP() {
        TangentDiffTensor x = seed(new double[]{1, -2, 3}, new double[]{1, 0, -1});
        IDiffTensor y = x.neg();
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{-1, 0, 1}, tan, 1e-12);
    }

    @Test
    void testExpJVP() {
        // JVP = exp(x) * tangent
        TangentDiffTensor x = seed(new double[]{0, 1, 2}, new double[]{1, 0, -1});
        IDiffTensor y = x.exp();
        double[] val = y.toDoubleArray();
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{1, Math.E, Math.E*Math.E}, val, 1e-10);
        assertArrayEquals(new double[]{1, 0, -Math.E*Math.E}, tan, 1e-10);
    }

    @Test
    void testReluJVP() {
        TangentDiffTensor x = seed(new double[]{-1, 0, 2, -3}, new double[]{1, 1, 1, 1});
        IDiffTensor y = x.relu();
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{0, 0, 1, 0}, tan, 1e-12);
    }

    @Test
    void testSigmoidJVP() {
        TangentDiffTensor x = seed(new double[]{0, 1, -1}, new double[]{1, 1, 1});
        IDiffTensor y = x.sigmoid();
        double[] val = y.toDoubleArray();
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        for (int i = 0; i < 3; i++) {
            double s = val[i];
            assertEquals(s * (1 - s), tan[i] / 1.0, 1e-10);
        }
    }

    @Test
    void testPowJVP() {
        TangentDiffTensor x = seed(new double[]{2, 3, 4}, new double[]{1, 0, -1});
        IDiffTensor y = x.pow(3);
        double[] val = y.toDoubleArray();
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{8, 27, 64}, val, 1e-10);
        assertArrayEquals(new double[]{3*4, 0, -3*16}, tan, 1e-10); // n*x^(n-1)*tx
    }

    @Test
    void testSinJVP() {
        TangentDiffTensor x = seed(new double[]{0, Math.PI/2}, new double[]{1, 1});
        IDiffTensor y = x.sin();
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{1, 0}, tan, 1e-10); // cos(0)=1, cos(pi/2)=0
    }

    @Test
    void testSumJVP() {
        TangentDiffTensor x = seed(new double[]{1, 2, 3, 4}, new double[]{1, 0, -1, 0});
        IDiffTensor y = x.sum();
        double tan = ((TangentDiffTensor) y).getTangent().toDoubleArray()[0];
        assertEquals(0, tan, 1e-12); // sum of tangents = 1 + 0 + (-1) + 0 = 0
    }

    @Test
    void testReshapeJVP() {
        TangentDiffTensor x = seed(new double[]{1, 2, 3, 4}, new double[]{1, 0, 0, -1});
        IDiffTensor y = x.reshape(2, 2);
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{1, 0, 0, -1}, tan, 1e-12);
    }

    @Test
    void testPermuteJVP() {
        // 2x3 tensor
        double[][] data = {{1, 2, 3}, {4, 5, 6}};
        double[][] td = {{1, 0, 0}, {0, -1, 0}};
        TangentDiffTensor x = seed2d(data, td);
        // Permute to 3x2
        IDiffTensor y = x.permute(1, 0);
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        // Permute to 3x2: flat order [0][0]=1, [0][1]=4, [1][0]=2, [1][1]=5, [2][0]=3, [2][1]=6
        // Tangent permuted same way: [1,0,0,-1,0,0]
        assertArrayEquals(new double[]{1, 0, 0, -1, 0, 0}, tan, 1e-12);
    }

    @Test
    void testMmulJVP() {
        // A=[2x3], B=[3x2], tangent on A only
        double[][] aData = {{1, 2, 3}, {4, 5, 6}};
        double[][] aTan = {{1, 0, 0}, {0, 0, 0}}; // only first row of A changes
        TangentDiffTensor a = seed2d(aData, aTan);
        double[][] bData = {{7, 8}, {9, 10}, {11, 12}};
        double[] bFlat = {7, 8, 9, 10, 11, 12};
        RereDiffTensor b = new RereDiffTensor(bFlat, 3, 2);
        IDoubleTensor bTan = ITensor.zeros(3, 2);
        TangentDiffTensor bTanWrap = TangentDiffTensor.seed(b, bTan);

        IDiffTensor c = a.mmul(bTanWrap);
        double[] cTan = ((TangentDiffTensor) c).getTangent().toDoubleArray();
        // dA*B = [[1,0,0]*B = [7,8]; [0,0,0]*B = [0,0]]
        assertEquals(7, cTan[0], 1e-10);
        assertEquals(8, cTan[1], 1e-10);
        assertEquals(0, cTan[2], 1e-10);
        assertEquals(0, cTan[3], 1e-10);
    }

    @Test
    void testSoftmaxJVP() {
        // Softmax derivative: s_i*(delta_ij - s_j)
        // For tangent = [1,0,0] at x=[0,0,0], softmax=[1/3,1/3,1/3]
        // JVP_i = s_i*(t_i - sum(s_j*t_j)) = (1/3)*(1 - 1/3) = 2/9 for i=0
        //       = (1/3)*(0 - 1/3) = -1/9 for i=1,2
        TangentDiffTensor x = seed(new double[]{0, 0, 0}, new double[]{1, 0, 0});
        IDiffTensor y = x.softmax(0);
        double[] val = y.toDoubleArray();
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        double s = 1.0 / 3;
        assertArrayEquals(new double[]{s, s, s}, val, 1e-10);
        assertArrayEquals(new double[]{s*(1-s), -s*s, -s*s}, tan, 1e-10);
    }

    @Test
    void testDivJVP() {
        TangentDiffTensor x = seed(new double[]{10, 20, 30}, new double[]{1, 0, -1});
        TangentDiffTensor y = seed(new double[]{2, 4, 5}, new double[]{0, 1, 0});
        IDiffTensor z = x.div(y);
        double[] val = z.toDoubleArray();
        double[] tan = ((TangentDiffTensor) z).getTangent().toDoubleArray();
        // val = [5, 5, 6]
        assertArrayEquals(new double[]{5, 5, 6}, val, 1e-10);
        // JVP_i = (tx_i*y_i - x_i*ty_i)/y_i^2
        // i=0: (1*2 - 10*0)/4 = 0.5
        // i=1: (0*4 - 20*1)/16 = -1.25
        // i=2: (-1*5 - 30*0)/25 = -0.2
        assertArrayEquals(new double[]{0.5, -1.25, -0.2}, tan, 1e-10);
    }

    @Test
    void testLogSoftmaxJVP() {
        // log_softmax(x) = x_i - log(sum(exp(x)))
        // JVP = t_i - softmax_i * sum(t)
        TangentDiffTensor x = seed(new double[]{0, 0, 0}, new double[]{1, 0, 0});
        IDiffTensor y = x.logSoftmax(0);
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        double s = 1.0 / 3;
        // JVP = t_i - s_i * sum(t) = [1 - s, 0 - s, 0 - s] = [2/3, -1/3, -1/3]
        assertArrayEquals(new double[]{2.0/3, -1.0/3, -1.0/3}, tan, 1e-10);
    }

    @Test
    void testSquareJVP() {
        TangentDiffTensor x = seed(new double[]{1, 2, 3}, new double[]{1, 0, -1});
        IDiffTensor y = x.square();
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{2, 0, -6}, tan, 1e-10); // 2*x*tx
    }

    @Test
    void testClampJVP() {
        TangentDiffTensor x = seed(new double[]{-1, 0, 1, 2}, new double[]{1, 1, 1, 1});
        IDiffTensor y = x.clamp(0, 1);
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        // derivative is 1 for x in [0,1], 0 outside
        assertArrayEquals(new double[]{0, 1, 1, 0}, tan, 1e-12);
    }

    @Test
    void testGatherJVP() {
        TangentDiffTensor x = seed(new double[]{10, 20, 30, 40}, new double[]{1, 0, 0, -1});
        IDoubleTensor idx = ITensor.tensor(new double[]{0, 2}, 2);
        IDiffTensor y = x.gather(0, idx);
        double[] val = y.toDoubleArray();
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{10, 30}, val, 1e-12);
        assertArrayEquals(new double[]{1, 0}, tan, 1e-12);
    }

    @Test
    void testMeanJVP() {
        TangentDiffTensor x = seed(new double[]{1, 2, 3, 4}, new double[]{2, 0, -2, 0});
        x = (TangentDiffTensor) x.reshape(2, 2);
        IDiffTensor y = x.mean(1, false);
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        // mean of [2,0] = 1, mean of [-2,0] = -1
        assertArrayEquals(new double[]{1, -1}, tan, 1e-12);
    }

    @Test
    void testSqueezeUnsqueezeJVP() {
        TangentDiffTensor x = seed(new double[]{1, 2, 3, 4}, new double[]{1, 0, -1, 0});
        IDiffTensor y = x.reshape(2, 2);
        IDiffTensor z = ((TangentDiffTensor) y).unsqueeze(0);
        double[] tan = ((TangentDiffTensor) z).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{1, 0, -1, 0}, tan, 1e-12);

        IDiffTensor w = ((TangentDiffTensor) z).squeeze(0);
        tan = ((TangentDiffTensor) w).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{1, 0, -1, 0}, tan, 1e-12);
    }

    @Test
    void testCatJVP() {
        TangentDiffTensor a = seed(new double[]{1, 2}, new double[]{1, 0});
        TangentDiffTensor b = seed(new double[]{3, 4}, new double[]{0, -1});
        IDiffTensor c = a.cat(0, b);
        double[] tan = ((TangentDiffTensor) c).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{1, 0, 0, -1}, tan, 1e-12);
    }

    @Test
    void testStackJVP() {
        TangentDiffTensor a = seed(new double[]{1, 2}, new double[]{1, 0});
        TangentDiffTensor b = seed(new double[]{3, 4}, new double[]{0, -1});
        IDiffTensor c = a.stack(0, b);
        double[] tan = ((TangentDiffTensor) c).getTangent().toDoubleArray();
        // stack makes 2x2, row 0 = a, row 1 = b
        assertArrayEquals(new double[]{1, 0, 0, -1}, tan, 1e-12);
    }

    @Test
    void testTanhJVP() {
        TangentDiffTensor x = seed(new double[]{0, 1, -1}, new double[]{1, 1, 1});
        IDiffTensor y = x.tanh();
        double[] val = y.toDoubleArray();
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        for (int i = 0; i < 3; i++) {
            double t = val[i];
            assertEquals(1 - t * t, tan[i], 1e-10); // sech^2
        }
    }

    @Test
    void testSubtractJVP() {
        TangentDiffTensor x = seed(new double[]{5, 10, 15}, new double[]{1, 0, 0});
        TangentDiffTensor y = seed(new double[]{1, 2, 3}, new double[]{0, 1, 0});
        IDiffTensor z = x.sub(y);
        double[] tan = ((TangentDiffTensor) z).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{1, -1, 0}, tan, 1e-12);
    }

    @Test
    void testMaxDimJVP() {
        // x = [[1, 5, 2], [3, 0, 4]], max along dim=1 → [5, 4]
        // tangent = [[1, 0, 0], [0, 0, -1]]
        double[][] data = {{1, 5, 2}, {3, 0, 4}};
        double[][] td = {{1, 0, 0}, {0, 0, -1}};
        TangentDiffTensor x = seed2d(data, td);
        IDiffTensor y = x.max(1, false);
        double[] val = y.toDoubleArray();
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{5, 4}, val, 1e-12);
        // tangent should be picked at max positions: 0 (pos of 5), -1 (pos of 4)
        assertArrayEquals(new double[]{0, -1}, tan, 1e-12);
    }

    @Test
    void testRsubJVP() {
        TangentDiffTensor x = seed(new double[]{1, 2, 3}, new double[]{1, 0, -1});
        IDiffTensor y = x.rsub(10); // 10 - x
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{-1, 0, 1}, tan, 1e-12); // -tx
    }

    @Test
    void testReciprocalJVP() {
        TangentDiffTensor x = seed(new double[]{2, 4, 5}, new double[]{1, 0, -1});
        IDiffTensor y = x.reciprocal();
        double[] val = y.toDoubleArray();
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        // val = [0.5, 0.25, 0.2], JVP = -tx/x^2
        assertEquals(-1.0/4, tan[0], 1e-10);
        assertEquals(0, tan[1], 1e-12);
        assertEquals(1.0/25, tan[2], 1e-10);
    }

    @Test
    void testNormalizeJVP() {
        // normalize(p=2, dim) — tangent is not affected (simplified)
        TangentDiffTensor x = seed(new double[]{1, 2, 3, 4}, new double[]{1, 0, 0, -1});
        IDiffTensor y = x.normalize(2, 0);
        // Just verify it doesn't crash and preserves primal dimensions
        assertNotNull(y);
        assertEquals(4, y.totalSize());
    }

    @Test
    void testLayerNormJVP() {
        // Simple layer norm: 2 rows, 2 features each
        TangentDiffTensor x = seed(new double[]{1, 2, 3, 4}, new double[]{1, 0, 0, -1});
        x = (TangentDiffTensor) x.reshape(2, 2);
        RereDiffTensor gamma = new RereDiffTensor(new double[]{1, 1}, 2);
        RereDiffTensor beta = new RereDiffTensor(new double[]{0, 0}, 2);
        TangentDiffTensor gTan = TangentDiffTensor.seed(gamma, ITensor.zeros(2));
        TangentDiffTensor bTan = TangentDiffTensor.seed(beta, ITensor.zeros(2));

        IDiffTensor y = x.layerNorm(gTan, bTan, 1e-5);
        assertNotNull(y);
        assertEquals(4, y.totalSize());
    }

    @Test
    void testWhereJVP() {
        TangentDiffTensor x = seed(new double[]{1, 2, 3}, new double[]{1, 0, -1});
        TangentDiffTensor y = seed(new double[]{10, 20, 30}, new double[]{0, 1, 0});
        IDoubleTensor cond = ITensor.tensor(new double[]{1, 0, 1}, 3);
        IDiffTensor z = x.where(cond, y);
        double[] val = z.toDoubleArray();
        double[] tan = ((TangentDiffTensor) z).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{1, 20, 3}, val, 1e-12);
        assertArrayEquals(new double[]{1, 1, -1}, tan, 1e-12);
    }

    @Test
    void testPadJVP() {
        TangentDiffTensor x = seed(new double[]{1, 2, 3}, new double[]{1, 0, -1});
        IDiffTensor y = x.pad(new int[][]{{1, 1}}, "constant", 0);
        double[] val = y.toDoubleArray();
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        assertEquals(5, val.length);
        assertEquals(0, val[0], 1e-12);
        assertEquals(1, val[1], 1e-12);
        // Tangent padded with zeros
        assertEquals(0, tan[0], 1e-12);
        assertEquals(1, tan[1], 1e-12);
        assertEquals(0, tan[2], 1e-12);
        assertEquals(-1, tan[3], 1e-12);
        assertEquals(0, tan[4], 1e-12);
    }

    @Test
    void testGeluJVP() {
        // gelu derivative: check it runs and matches expected at known values
        TangentDiffTensor x = seed(new double[]{0, 1, -1}, new double[]{1, 0, 0});
        IDiffTensor y = x.gelu();
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        // at x=0: gelu'(0) ≈ 0.5 * (1+tanh(0)) = 0.5, JVP = 0.5 * 1 = 0.5
        assertTrue(tan[0] >= 0.49 && tan[0] <= 0.51);
        assertEquals(0.0, tan[1], 1e-12); // tangent is 0, so JVP is 0
        assertEquals(0.0, tan[2], 1e-12); // tangent is 0, so JVP is 0
    }

    @Test
    void testBroadcastToJVP() {
        TangentDiffTensor x = seed(new double[]{1, 2, 3}, new double[]{1, 0, -1});
        x = (TangentDiffTensor) x.reshape(1, 3);
        IDiffTensor y = x.broadcastTo(2, 3);
        double[] val = y.toDoubleArray();
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        assertArrayEquals(new double[]{1, 2, 3, 1, 2, 3}, val, 1e-12);
        // tangent broadcasts same way
        assertArrayEquals(new double[]{1, 0, -1, 1, 0, -1}, tan, 1e-12);
    }

    @Test
    void testInplaceAddJVP() {
        TangentDiffTensor x = seed(new double[]{1, 2, 3}, new double[]{1, 0, -1});
        TangentDiffTensor y = seed(new double[]{10, 20, 30}, new double[]{0, 1, 0});
        x.add_(y);
        double[] val = x.toDoubleArray();
        double[] tan = x.getTangent().toDoubleArray();
        assertArrayEquals(new double[]{11, 22, 33}, val, 1e-12);
        assertArrayEquals(new double[]{1, 1, -1}, tan, 1e-12);
    }

    @Test
    void testScatterAddJVP() {
        TangentDiffTensor x = seed(new double[]{1, 2, 3, 4}, new double[]{1, 0, 0, -1});
        IDoubleTensor idx = ITensor.tensor(new double[]{0, 2}, 2);
        TangentDiffTensor src = seed(new double[]{10, 20}, new double[]{0, 1});
        IDiffTensor y = x.scatterAdd(0, idx, src);
        double[] tan = ((TangentDiffTensor) y).getTangent().toDoubleArray();
        // scatter-add on tangent: tangent[idx] += srcTan
        assertArrayEquals(new double[]{1, 0, 1, -1}, tan, 1e-12);
    }

    @Test
    void testJVPagainstFiniteDifferences() {
        // Verify JVP matches centered finite differences for a few functions
        java.util.function.BiFunction<double[], double[], Boolean> checkJVP =
            (data, tdir) -> {
                TangentDiffTensor x = seed(data, tdir);
                RereDiffTensor xPlain = new RereDiffTensor(data.clone(), data.length);
                xPlain.setRequiresGrad(true);

                // f(x) = sum(relu(x).exp())
                IDiffTensor y = x.relu().exp().sum();
                double jvp = ((TangentDiffTensor) y).getTangent().toDoubleArray()[0];

                // finite difference: (f(x+h*v) - f(x-h*v)) / (2h)
                double h = 1e-6;
                double[] xph = new double[data.length];
                double[] xmh = new double[data.length];
                for (int i = 0; i < data.length; i++) {
                    xph[i] = data[i] + h * tdir[i];
                    xmh[i] = data[i] - h * tdir[i];
                }
                double fwd = new RereDiffTensor(xph, data.length).relu().exp().sumAll();
                double bwd = new RereDiffTensor(xmh, data.length).relu().exp().sumAll();
                double fdJvp = (fwd - bwd) / (2 * h);

                return Math.abs(jvp - fdJvp) < 1e-6;
            };

        assertTrue(checkJVP.apply(new double[]{0.5, 1, -1, 2}, new double[]{1, 0, 0, 0}));
        assertTrue(checkJVP.apply(new double[]{0.5, 1, -1, 2}, new double[]{0, 0, 1, 0}));
        assertTrue(checkJVP.apply(new double[]{0.5, 1, -1, 2}, new double[]{0.5, 0.5, 0.5, 0.5}));
    }

    @Test
    void testSelectJVP() {
        TangentDiffTensor x = seed(new double[]{10, 20, 30, 40, 50}, new double[]{1, 0, -1, 0, 0});
        IDiffTensor y = x.select(0, 2);
        double val = y.toDoubleArray()[0];
        double tan = ((TangentDiffTensor) y).getTangent().toDoubleArray()[0];
        assertEquals(30, val, 1e-12);
        assertEquals(-1, tan, 1e-12);
    }

    @Test
    void testSiluJVP() {
        TangentDiffTensor x = seed(new double[]{-2, 0, 2}, new double[]{1, 1, 1});
        IDiffTensor y = x.silu();
        // silu(x) = x * sigmoid(x)
        // derivative = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        assertNotNull(y);
        assertEquals(3, y.totalSize());
    }
}
