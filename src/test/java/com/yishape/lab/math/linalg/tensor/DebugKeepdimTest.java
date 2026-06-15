package com.yishape.lab.math.linalg.tensor;

import com.yishape.lab.math.autodiff.IDiffTensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class DebugKeepdimTest {

    private static final int N = 2, C = 3, H = 4, W = 5;
    private static final int STRIDE_N = C * H * W; // 60
    private static final int STRIDE_C = H * W;     // 20
    private static final int STRIDE_H = W;         // 5
    private static final int STRIDE_W = 1;         // 1

    private RereDoubleTensor createSequentialTensor() {
        int total = N * C * H * W;
        double[] data = new double[total];
        for (int i = 0; i < total; i++) data[i] = i;
        return new RereDoubleTensor(data, N, C, H, W);
    }

    // Correct values computed independently: data is sequential, row-major C-order.
    // Position (n,c,h,w) = data[n*60 + c*20 + h*5 + w] = n*60 + c*20 + h*5 + w.
    // reduce over dim 0 (N): max over n → result depends only on (c,h,w)
    //   For each (c,h,w), max = max(data[c*20+h*5+w], data[60+c*20+h*5+w]) = 60 + c*20+h*5+w

    @Test
    public void testMaxDim0KeepdimTrue() {
        RereDoubleTensor raw = createSequentialTensor();
        IDiffTensor t = IDiffTensor.fromTensor(raw, false);
        IDiffTensor maxed = t.max(0, true);
        double[] out = maxed.toDoubleArray();
        assertArrayEquals(new int[]{1, C, H, W}, maxed.shape());

        // out[k] where k = c*H*W + h*W + w (in [1,C,H,W] layout)
        // = 60 + c*20 + h*5 + w
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int k = c * H * W + h * W + w;
                    double expected = 60.0 + c * STRIDE_C + h * STRIDE_H + w;
                    assertEquals(expected, out[k], 1e-10,
                        "at (0," + c + "," + h + "," + w + ")");
                }
            }
        }
    }

    @Test
    public void testMaxDim1KeepdimTrue() {
        RereDoubleTensor raw = createSequentialTensor();
        IDiffTensor t = IDiffTensor.fromTensor(raw, false);
        IDiffTensor maxed = t.max(1, true);
        double[] out = maxed.toDoubleArray();
        assertArrayEquals(new int[]{N, 1, H, W}, maxed.shape());

        // out[k] where k = n*H*W + h*W + w (in [2,1,4,5] layout)
        // = max over c of (n*60 + c*20 + h*5 + w) = n*60 + 2*20 + h*5 + w = n*60 + 40 + h*5 + w
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int k = n * H * W + h * W + w;
                    double expected = n * STRIDE_N + (C - 1) * STRIDE_C + h * STRIDE_H + w;
                    assertEquals(expected, out[k], 1e-10,
                        "at (" + n + ",0," + h + "," + w + ")");
                }
            }
        }
    }

    @Test
    public void testMaxDim2KeepdimTrue() {
        RereDoubleTensor raw = createSequentialTensor();
        IDiffTensor t = IDiffTensor.fromTensor(raw, false);
        IDiffTensor maxed = t.max(2, true);
        double[] out = maxed.toDoubleArray();
        assertArrayEquals(new int[]{N, C, 1, W}, maxed.shape());

        // out[k] where k = n*C*W + c*W + w (in [2,3,1,5] layout)
        // = max over h of (n*60 + c*20 + h*5 + w) = n*60 + c*20 + 3*5 + w = n*60 + c*20 + 15 + w
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int w = 0; w < W; w++) {
                    int k = n * C * W + c * W + w;
                    double expected = n * STRIDE_N + c * STRIDE_C + (H - 1) * STRIDE_H + w;
                    assertEquals(expected, out[k], 1e-10,
                        "at (" + n + "," + c + ",0," + w + ")");
                }
            }
        }
    }

    @Test
    public void testMaxDim3KeepdimTrue() {
        RereDoubleTensor raw = createSequentialTensor();
        IDiffTensor t = IDiffTensor.fromTensor(raw, false);
        IDiffTensor maxed = t.max(3, true);
        double[] out = maxed.toDoubleArray();
        assertArrayEquals(new int[]{N, C, H, 1}, maxed.shape());

        // out[k] where k = n*C*H + c*H + h (in [2,3,4,1] layout)
        // = max over w of (n*60 + c*20 + h*5 + w) = n*60 + c*20 + h*5 + 4
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    int k = n * C * H + c * H + h;
                    double expected = n * STRIDE_N + c * STRIDE_C + h * STRIDE_H + (W - 1);
                    assertEquals(expected, out[k], 1e-10,
                        "at (" + n + "," + c + "," + h + ",0)");
                }
            }
        }
    }

    @Test
    public void testMaxDim0KeepdimFalse() {
        RereDoubleTensor raw = createSequentialTensor();
        IDiffTensor t = IDiffTensor.fromTensor(raw, false);
        IDiffTensor maxed = t.max(0, false);
        double[] out = maxed.toDoubleArray();
        assertArrayEquals(new int[]{C, H, W}, maxed.shape());

        // Same as keepdim=true but squeezed: out[c,h,w] = 60 + c*20 + h*5 + w
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int k = c * H * W + h * W + w;
                    double expected = 60.0 + c * STRIDE_C + h * STRIDE_H + w;
                    assertEquals(expected, out[k], 1e-10,
                        "at (" + c + "," + h + "," + w + ")");
                }
            }
        }
    }

    @Test
    public void testMaxDim1KeepdimFalse() {
        RereDoubleTensor raw = createSequentialTensor();
        IDiffTensor t = IDiffTensor.fromTensor(raw, false);
        IDiffTensor maxed = t.max(1, false);
        double[] out = maxed.toDoubleArray();
        assertArrayEquals(new int[]{N, H, W}, maxed.shape());

        // out[n,h,w] = n*60 + 40 + h*5 + w
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int k = n * H * W + h * W + w;
                    double expected = n * STRIDE_N + (C - 1) * STRIDE_C + h * STRIDE_H + w;
                    assertEquals(expected, out[k], 1e-10,
                        "at (" + n + "," + h + "," + w + ")");
                }
            }
        }
    }

    @Test
    public void testMaxDim2KeepdimFalse() {
        RereDoubleTensor raw = createSequentialTensor();
        IDiffTensor t = IDiffTensor.fromTensor(raw, false);
        IDiffTensor maxed = t.max(2, false);
        double[] out = maxed.toDoubleArray();
        assertArrayEquals(new int[]{N, C, W}, maxed.shape());

        // out[n,c,w] = n*60 + c*20 + 15 + w
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int w = 0; w < W; w++) {
                    int k = n * C * W + c * W + w;
                    double expected = n * STRIDE_N + c * STRIDE_C + (H - 1) * STRIDE_H + w;
                    assertEquals(expected, out[k], 1e-10,
                        "at (" + n + "," + c + "," + w + ")");
                }
            }
        }
    }

    @Test
    public void testMaxDim3KeepdimFalse() {
        RereDoubleTensor raw = createSequentialTensor();
        IDiffTensor t = IDiffTensor.fromTensor(raw, false);
        IDiffTensor maxed = t.max(3, false);
        double[] out = maxed.toDoubleArray();
        assertArrayEquals(new int[]{N, C, H}, maxed.shape());

        // out[n,c,h] = n*60 + c*20 + h*5 + 4
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    int k = n * C * H + c * H + h;
                    double expected = n * STRIDE_N + c * STRIDE_C + h * STRIDE_H + (W - 1);
                    assertEquals(expected, out[k], 1e-10,
                        "at (" + n + "," + c + "," + h + ")");
                }
            }
        }
    }
}
