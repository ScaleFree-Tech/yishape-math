package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Verification tests for Phase 1 LLM ops: rmsNorm, rope, embedding, lstmCell, gruCell.
 */
public class Phase1OpsTest {

    // ==================== rmsNorm ====================

    @Test
    public void testRmsNormForward() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor gamma = AD.leafTensor(new double[]{1, 1, 1}, 3);
        IDiffTensor y = x.rmsNorm(gamma, 1e-5);

        double meanSq0 = (1 + 4 + 9) / 3.0;
        double rms0 = Math.sqrt(meanSq0 + 1e-5);
        assertEquals(1/rms0, y.toDoubleArray()[0], 1e-5);
        assertEquals(2/rms0, y.toDoubleArray()[1], 1e-5);
        assertEquals(3/rms0, y.toDoubleArray()[2], 1e-5);
        double meanSq1 = (16 + 25 + 36) / 3.0;
        double rms1 = Math.sqrt(meanSq1 + 1e-5);
        assertEquals(4/rms1, y.toDoubleArray()[3], 1e-5);
        assertArrayEquals(new int[]{2, 3}, y.shape());
    }

    @Test
    public void testRmsNormGammaScale() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor gamma = AD.leafTensor(new double[]{2, 2, 2}, 3);
        IDiffTensor y = x.rmsNorm(gamma, 1e-5);

        double meanSq0 = (1 + 4 + 9) / 3.0;
        double rms0 = Math.sqrt(meanSq0 + 1e-5);
        assertEquals(2*1/rms0, y.toDoubleArray()[0], 1e-5);
    }

    @Test
    public void testRmsNormBackward() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor gamma = AD.leafTensor(new double[]{1, 1, 1}, 3);
        IDiffTensor y = x.rmsNorm(gamma, 1e-5);
        IDiffTensor loss = y.sum();
        loss.backward();

        IDoubleTensor gx = x.grad();
        assertNotNull(gx);
        assertTrue(gx.toDoubleArray()[0] != 0);
        IDoubleTensor gg = gamma.grad();
        assertNotNull(gg);
        assertEquals(1.252, gg.toDoubleArray()[0], 1e-2);
    }

    @Test
    public void testRmsNormHighDim() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 2, 3);
        IDiffTensor gamma = AD.leafTensor(new double[]{1, 1, 1}, 3);
        IDiffTensor y = x.rmsNorm(gamma, 1e-5);
        assertArrayEquals(new int[]{2, 2, 3}, y.shape());
        y.sum().backward();
        assertNotNull(x.grad());
    }

    @Test
    public void testRmsNormConstant() {
        IDiffTensor x = AD.constantTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor gamma = AD.constantTensor(new double[]{1, 1, 1}, 3);
        IDiffTensor y = x.rmsNorm(gamma, 1e-5);
        assertArrayEquals(new int[]{2, 3}, y.shape());
        assertFalse(Double.isNaN(y.toDoubleArray()[0]));
    }

    // ==================== rope ====================

    @Test
    public void testRopeForwardSimple() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 0, 0, 1, 0, 1, 1, 0}, 1, 2, 4);
        IDiffTensor y = x.rope(2, 10, 10000.0);
        // At pos 0: theta=0, cos=1, sin=0 => no rotation
        assertEquals(1.0, y.toDoubleArray()[0], 1e-10);
        assertEquals(0.0, y.toDoubleArray()[1], 1e-10);
        assertEquals(0.0, y.toDoubleArray()[2], 1e-10);
        assertEquals(1.0, y.toDoubleArray()[3], 1e-10);
        // At pos 1: adjacent pairs (0,1) and (2,3)
        // pair 0 (idx 4,5): angle = 1/10000^(0/2)=1, c=cos(1), s=sin(1)
        // pair 1 (idx 6,7): angle = 1/10000^(2/2)=1/10000, c=cos(1e-4), s=sin(1e-4)
        double c0 = Math.cos(1.0), s0 = Math.sin(1.0);
        // x[4,5,6,7] = [0,1,1,0]
        // pair0: y[4]=0*c0-1*s0=-s0, y[5]=0*s0+1*c0=c0
        // pair1: y[6]=1*cos(1e-4)-0*sin(1e-4)=cos(1e-4), y[7]=1*sin(1e-4)+0*cos(1e-4)=sin(1e-4)
        assertEquals(-s0, y.toDoubleArray()[4], 1e-10);
        assertEquals(c0, y.toDoubleArray()[5], 1e-10);
        assertEquals(1.0, y.toDoubleArray()[6], 1e-8);
        assertEquals(0.0, y.toDoubleArray()[7], 1e-4);
    }

    @Test
    public void testRopeBackward() {
        IDiffTensor x = AD.leafTensor(new double[]{1, 0, 0, 1, 0, 1, 1, 0}, 1, 2, 4);
        IDiffTensor y = x.rope(2, 10, 10000.0);
        y.sum().backward();
        assertNotNull(x.grad());
        assertTrue(x.grad().toDoubleArray()[0] != 0);
    }

    @Test
    public void testRopePreservesNorm() {
        IDiffTensor x = AD.leafTensor(new double[]{3, 4, 0, 0, -1, 2, -3, 4}, 1, 2, 4);
        double[] xData = x.toDoubleArray();
        IDiffTensor y = x.rope(2, 10, 10000.0);
        double[] yData = y.toDoubleArray();
        for (int pos = 0; pos < 2; pos++) {
            double xNorm = 0, yNorm = 0;
            for (int j = 0; j < 4; j++) {
                xNorm += xData[pos * 4 + j] * xData[pos * 4 + j];
                yNorm += yData[pos * 4 + j] * yData[pos * 4 + j];
            }
            assertEquals(xNorm, yNorm, 1e-10);
        }
    }

    @Test
    public void testRopeConstant() {
        IDiffTensor x = AD.constantTensor(new double[]{1, 0, 0, 1}, 1, 1, 4);
        IDiffTensor y = x.rope(2, 10, 10000.0);
        assertArrayEquals(new int[]{1, 1, 4}, y.shape());
        assertEquals(1.0, y.toDoubleArray()[0], 1e-10);
    }

    // ==================== embedding ====================

    @Test
    public void testEmbeddingForward() {
        IDiffTensor weight = AD.leafTensor(new double[]{
            1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4
        }, 4, 3);
        IDiffTensor indices = AD.constantTensor(new double[]{0, 2, 1}, 3);
        IDiffTensor result = weight.embedding(indices);
        assertArrayEquals(new int[]{3, 3}, result.shape());
        assertEquals(1, result.toDoubleArray()[0], 1e-10);
        assertEquals(3, result.toDoubleArray()[3], 1e-10);
        assertEquals(2, result.toDoubleArray()[6], 1e-10);
    }

    @Test
    public void testEmbeddingMultiDimIndices() {
        IDiffTensor weight = AD.leafTensor(new double[]{
            1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4
        }, 4, 3);
        IDiffTensor indices = AD.constantTensor(new double[]{0, 1, 2, 3}, 2, 2);
        IDiffTensor result = weight.embedding(indices);
        assertArrayEquals(new int[]{2, 2, 3}, result.shape());
        assertEquals(1, result.toDoubleArray()[0], 1e-10);
        assertEquals(2, result.toDoubleArray()[3], 1e-10);
    }

    @Test
    public void testEmbeddingBackward() {
        IDiffTensor weight = AD.leafTensor(new double[]{
            1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4
        }, 4, 3);
        IDiffTensor indices = AD.constantTensor(new double[]{0, 2, 1}, 3);
        IDiffTensor result = weight.embedding(indices);
        result.sum().backward();
        IDoubleTensor gw = weight.grad();
        assertNotNull(gw);
        assertEquals(1, gw.toDoubleArray()[0], 1e-10);
        assertEquals(1, gw.toDoubleArray()[3], 1e-10);
        assertEquals(0, gw.toDoubleArray()[9], 1e-10);
    }

    @Test
    public void testEmbeddingRepeatedIndex() {
        IDiffTensor weight = AD.leafTensor(new double[]{1, 2, 3, 4}, 4, 1);
        IDiffTensor indices = AD.constantTensor(new double[]{0, 0, 1}, 3);
        IDiffTensor result = weight.embedding(indices);
        result.sum().backward();
        IDoubleTensor gw = weight.grad();
        assertEquals(2, gw.toDoubleArray()[0], 1e-10);
        assertEquals(1, gw.toDoubleArray()[1], 1e-10);
    }

    @Test
    public void testEmbeddingConstant() {
        IDiffTensor weight = AD.constantTensor(new double[]{1, 2, 3, 4}, 4, 1);
        IDiffTensor indices = AD.constantTensor(new double[]{0, 2}, 2);
        IDiffTensor result = weight.embedding(indices);
        assertArrayEquals(new int[]{2, 1}, result.shape());
        assertEquals(1, result.toDoubleArray()[0], 1e-10);
        assertEquals(3, result.toDoubleArray()[1], 1e-10);
    }

    // ==================== LSTM cell ====================

    @Test
    public void testLstmCellForward() {
        int batch = 1, inputSize = 3, hiddenSize = 2;
        IDiffTensor x = AD.leafTensor(new double[]{1, 1, 1}, batch, inputSize);
        IDiffTensor hPrev = AD.leafTensor(new double[]{0, 0}, batch, hiddenSize);
        IDiffTensor cPrev = AD.leafTensor(new double[]{0, 0}, batch, hiddenSize);
        IDiffTensor wInput = AD.leafTensor(new double[]{
            1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0,
            1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0
        }, 4 * hiddenSize, inputSize);
        IDiffTensor wHidden = AD.leafTensor(new double[4 * hiddenSize * hiddenSize], 4 * hiddenSize, hiddenSize);

        IDiffTensor[] result = x.lstmCell(x, hPrev, cPrev, wInput, wHidden, null);
        assertEquals(2, result.length);
        // Manual: wInput * x = [1,1] per gate, then sigmoid/tanh
        double sig = 1.0 / (1.0 + Math.exp(-1));
        double tanh1 = Math.tanh(1);
        double cVal = sig * tanh1;
        double hVal = sig * Math.tanh(cVal);
        assertEquals(hVal, result[0].toDoubleArray()[0], 1e-4);
        assertEquals(cVal, result[1].toDoubleArray()[0], 1e-4);
    }

    @Test
    public void testLstmCellGradient() {
        int batch = 1, inputSize = 2, hiddenSize = 2;
        IDiffTensor x = AD.leafTensor(new double[]{1, 1}, batch, inputSize);
        IDiffTensor hPrev = AD.leafTensor(new double[]{0, 0}, batch, hiddenSize);
        IDiffTensor cPrev = AD.leafTensor(new double[]{0, 0}, batch, hiddenSize);
        double[] wData = new double[4 * hiddenSize * inputSize];
        for (int i = 0; i < 4 * hiddenSize; i++) wData[i * inputSize + (i % inputSize)] = 1.0;
        IDiffTensor wInput = AD.leafTensor(wData, 4 * hiddenSize, inputSize);
        IDiffTensor wHidden = AD.leafTensor(new double[4 * hiddenSize * hiddenSize], 4 * hiddenSize, hiddenSize);

        IDiffTensor[] result = x.lstmCell(x, hPrev, cPrev, wInput, wHidden, null);
        result[0].sum().backward();

        IDoubleTensor gx = x.grad();
        assertNotNull(gx);
        assertTrue(Math.abs(gx.toDoubleArray()[0]) > 1e-6);
    }

    // ==================== GRU cell ====================

    @Test
    public void testGruCellForward() {
        int batch = 1, inputSize = 2, hiddenSize = 2;
        IDiffTensor x = AD.leafTensor(new double[]{1, 0}, batch, inputSize);
        IDiffTensor hPrev = AD.leafTensor(new double[]{0, 0}, batch, hiddenSize);
        IDiffTensor wInput = AD.leafTensor(new double[]{
            1, 0,  0, 1, 1, 0,  0, 1, 1, 0,  0, 1
        }, 3 * hiddenSize, inputSize);
        IDiffTensor wHidden = AD.leafTensor(new double[3 * hiddenSize * hiddenSize], 3 * hiddenSize, hiddenSize);

        IDiffTensor h = x.gruCell(x, hPrev, wInput, wHidden, null);
        assertArrayEquals(new int[]{batch, hiddenSize}, h.shape());
        assertFalse(Double.isNaN(h.toDoubleArray()[0]));
    }
}
