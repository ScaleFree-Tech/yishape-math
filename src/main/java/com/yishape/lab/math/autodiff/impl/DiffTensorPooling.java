package com.yishape.lab.math.autodiff.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.compute.DoubleFlatGemm;
import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.gpu.GpuActivation;
import com.yishape.lab.math.compute.gpu.GpuReduce;
import com.yishape.lab.math.compute.hpc.HpcIm2col;
import com.yishape.lab.math.compute.hpc.HpcLoss;
import com.yishape.lab.math.compute.hpc.HpcCross;
import com.yishape.lab.math.compute.hpc.HpcGridSample;
import com.yishape.lab.math.compute.hpc.HpcTrapezoidalScan;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.tensor.EinsumParser;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import com.yishape.lab.math.linalg.tensor.TensorShape;
import com.yishape.lab.math.compute.gpu.GpuGroupNorm;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.autodiff.AD;

/**
 * Extracted from {@link RereDiffTensor}.
 * All methods are static, taking the tensor as first parameter.
 */
public final class DiffTensorPooling {
    private DiffTensorPooling() { /* utility class */ }

// ==================== Phase 3: Pooling ====================

public static IDiffTensor maxPool2d(RereDiffTensor tensor, int kH, int kW, int stride, int padding) {
    int[] s = tensor.shape();
    if (s.length != 4) throw new IllegalArgumentException("maxPool2d requires 4D input [N,C,H,W], got rank " + s.length);
    int N = s[0], C = s[1], H = s[2], W = s[3];
    int effStride = (stride <= 0) ? kH : stride;
    int outH = (H + 2 * padding - kH) / effStride + 1;
    int outW = (W + 2 * padding - kW) / effStride + 1;
    long outElements = (long) N * C * outH * outW;
    int inElements = (int) tensor.value.totalSize();

    double[] xd = tensor.value.toDoubleArray();
    double[] y = new double[(int) outElements];
    int[] argmax = new int[(int) outElements];

    // Try HPC accelerated path first
    boolean hpcOk = com.yishape.lab.math.compute.hpc.HpcPool.tryMaxPool2dForward(
            xd, N, C, H, W, kH, kW, effStride, padding, y, argmax);

    if (!hpcOk) {
        // SISD fallback: im2col-style pooling
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        int outIdx = ((n * C + c) * outH + oh) * outW + ow;
                        double maxVal = Double.NEGATIVE_INFINITY;
                        int maxPos = -1;
                        for (int kh = 0; kh < kH; kh++) {
                            int hIdx = oh * effStride + kh - padding;
                            if (hIdx < 0 || hIdx >= H) continue;
                            for (int kw = 0; kw < kW; kw++) {
                                int wIdx = ow * effStride + kw - padding;
                                if (wIdx < 0 || wIdx >= W) continue;
                                int inIdx = ((n * C + c) * H + hIdx) * W + wIdx;
                                if (xd[inIdx] > maxVal) {
                                    maxVal = xd[inIdx];
                                    maxPos = inIdx;
                                }
                            }
                        }
                        y[outIdx] = maxVal;
                        argmax[outIdx] = maxPos;
                    }
                }
            }
        }
    }

    int[] outShape = new int[]{N, C, outH, outW};
    int[] savedArgmax = argmax;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inp = self.inputs.get(0);
        double[] g = self.grad;
        double[] dx = new double[inElements];
        if (!com.yishape.lab.math.compute.hpc.HpcPool.tryMaxPool2dBackward(
                g, savedArgmax, N, C, H, W, outH, outW, dx)) {
            // SISD fallback
            int outLen = (int) outElements;
            for (int i = 0; i < outLen; i++) {
                int maxIdx = savedArgmax[i];
                if (maxIdx >= 0) dx[maxIdx] += g[i];
            }
        }
        inp.accGrad(dx);
    };
    RereDiffTensor result = new RereDiffTensor(y, outShape, List.of(tensor), bw, "maxpool2d");
    // D8: Bit layout — scalarParam[47:32]=kH [31:16]=kW [15:0]=stride; scalarParam2[31:16]=padding
    result.scalarParam = Double.longBitsToDouble(((long) kH << 16) | ((long) kW << 8) | (long) stride);
    result.scalarParam2 = Double.longBitsToDouble(((long) padding << 16));
    // 6D exportShape lets HPC/GPU backends use actual input dims directly,
    // avoiding incorrect derivation when stride does not divide evenly.
    result.exportShape = new int[]{N, C, H, W, outH, outW};
    return result;
}

public static IDiffTensor avgPool2d(RereDiffTensor tensor, int kH, int kW, int stride, int padding) {
    int[] s = tensor.shape();
    if (s.length != 4) throw new IllegalArgumentException("avgPool2d requires 4D input [N,C,H,W], got rank " + s.length);
    int N = s[0], C = s[1], H = s[2], W = s[3];
    int effStride = (stride <= 0) ? kH : stride;
    int outH = (H + 2 * padding - kH) / effStride + 1;
    int outW = (W + 2 * padding - kW) / effStride + 1;
    long outElements = (long) N * C * outH * outW;
    int inElements = (int) tensor.value.totalSize();

    double[] xd = tensor.value.toDoubleArray();
    double[] y = new double[(int) outElements];
    // Save counts for backward SISD fallback (HPC backward recomputes internally)
    int[] counts = new int[(int) outElements];

    // Try HPC accelerated path first
    boolean hpcOk = com.yishape.lab.math.compute.hpc.HpcPool.tryAvgPool2dForward(
            xd, N, C, H, W, kH, kW, effStride, padding, y);

    if (!hpcOk) {
        // SISD fallback
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        int outIdx = ((n * C + c) * outH + oh) * outW + ow;
                        double sum = 0;
                        int count = 0;
                        for (int kh = 0; kh < kH; kh++) {
                            int hIdx = oh * effStride + kh - padding;
                            if (hIdx < 0 || hIdx >= H) continue;
                            for (int kw = 0; kw < kW; kw++) {
                                int wIdx = ow * effStride + kw - padding;
                                if (wIdx < 0 || wIdx >= W) continue;
                                sum += xd[((n * C + c) * H + hIdx) * W + wIdx];
                                count++;
                            }
                        }
                        y[outIdx] = (count > 0) ? sum / count : 0;
                        counts[outIdx] = count;
                    }
                }
            }
        }
    }

    int[] outShape = new int[]{N, C, outH, outW};
    int[] savedCounts = counts;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inp = self.inputs.get(0);
        double[] g = self.grad;
        double[] dx = new double[inElements];
        if (!com.yishape.lab.math.compute.hpc.HpcPool.tryAvgPool2dBackward(
                g, N, C, H, W, kH, kW, effStride, padding, outH, outW, dx)) {
            // SISD fallback
            int outLen = (int) outElements;
            for (int n2 = 0; n2 < N; n2++) {
                for (int c2 = 0; c2 < C; c2++) {
                    for (int oh2 = 0; oh2 < outH; oh2++) {
                        for (int ow2 = 0; ow2 < outW; ow2++) {
                            int outIdx2 = ((n2 * C + c2) * outH + oh2) * outW + ow2;
                            int cnt = savedCounts[outIdx2];
                            if (cnt == 0) continue;
                            double gradPer = g[outIdx2] / cnt;
                            for (int kh2 = 0; kh2 < kH; kh2++) {
                                int hIdx2 = oh2 * effStride + kh2 - padding;
                                if (hIdx2 < 0 || hIdx2 >= H) continue;
                                for (int kw2 = 0; kw2 < kW; kw2++) {
                                    int wIdx2 = ow2 * effStride + kw2 - padding;
                                    if (wIdx2 < 0 || wIdx2 >= W) continue;
                                    dx[((n2 * C + c2) * H + hIdx2) * W + wIdx2] += gradPer;
                                }
                            }
                        }
                    }
                }
            }
        }
        inp.accGrad(dx);
    };
    RereDiffTensor result = new RereDiffTensor(y, outShape, List.of(tensor), bw, "avgpool2d");
    // D8: Bit layout — scalarParam[47:32]=kH [31:16]=kW [15:0]=stride; scalarParam2[31:16]=padding
    result.scalarParam = Double.longBitsToDouble(((long) kH << 16) | ((long) kW << 8) | (long) stride);
    result.scalarParam2 = Double.longBitsToDouble(((long) padding << 16));
    // 6D exportShape lets HPC/GPU backends use actual input dims directly,
    // avoiding incorrect derivation when stride does not divide evenly.
    result.exportShape = new int[]{N, C, H, W, outH, outW};
    return result;
}

public static IDiffTensor adaptiveAvgPool2d(RereDiffTensor tensor, int outH, int outW) {
    int[] s = tensor.shape();
    if (s.length < 3) throw new IllegalArgumentException("adaptiveAvgPool2d requires rank >= 3, got " + s.length);
    int N, C, H, W;
    if (s.length == 3) { N = 1; C = s[0]; H = s[1]; W = s[2]; }
    else if (s.length == 4) { N = s[0]; C = s[1]; H = s[2]; W = s[3]; }
    else { N = 1; for (int i = 0; i < s.length - 3; i++) N *= s[i]; C = s[s.length - 3]; H = s[s.length - 2]; W = s[s.length - 1]; }

    // Each output position (oh, ow) maps to a contiguous block in input space
    double[] xd = tensor.value.toDoubleArray();
    double[] y = new double[N * C * outH * outW];
    // Save start/end indices for backward (SISD fallback)
    int[] startH = new int[outH + 1];
    int[] startW = new int[outW + 1];
    for (int oh = 0; oh <= outH; oh++) startH[oh] = (int) Math.floor((double) oh * H / outH);
    for (int ow = 0; ow <= outW; ow++) startW[ow] = (int) Math.floor((double) ow * W / outW);

    // Try HPC accelerated path first
    boolean hpcOk = com.yishape.lab.math.compute.hpc.HpcPool.tryAdaptiveAvgPool2dForward(
            xd, N, C, H, W, outH, outW, y);

    if (!hpcOk) {
        // SISD fallback
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int oh = 0; oh < outH; oh++) {
                    int hStart = startH[oh], hEnd = startH[oh + 1];
                    for (int ow = 0; ow < outW; ow++) {
                        int wStart = startW[ow], wEnd = startW[ow + 1];
                        double sum = 0;
                        int cnt = 0;
                        for (int h = hStart; h < hEnd; h++) {
                            for (int w = wStart; w < wEnd; w++) {
                                sum += xd[((n * C + c) * H + h) * W + w];
                                cnt++;
                            }
                        }
                        y[((n * C + c) * outH + oh) * outW + ow] = (cnt > 0) ? sum / cnt : 0;
                    }
                }
            }
        }
    }

    int[] outShape = (s.length == 4) ? new int[]{N, C, outH, outW}
        : (s.length == 3) ? new int[]{C, outH, outW} : buildOutShape(s, N, C, outH, outW);
    int[] savedStartH = startH, savedStartW = startW;
    int fH = H, fW = W, fOutH = outH, fOutW = outW, fN = N, fC = C;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inp = self.inputs.get(0);
        double[] g = self.grad;
        double[] dx = new double[(int) inp.value.totalSize()];
        if (!com.yishape.lab.math.compute.hpc.HpcPool.tryAdaptiveAvgPool2dBackward(
                g, fN, fC, fH, fW, fOutH, fOutW, dx)) {
            // SISD fallback
            for (int n2 = 0; n2 < fN; n2++) {
                for (int c2 = 0; c2 < fC; c2++) {
                    for (int oh2 = 0; oh2 < fOutH; oh2++) {
                        int hStart2 = savedStartH[oh2], hEnd2 = savedStartH[oh2 + 1];
                        for (int ow2 = 0; ow2 < fOutW; ow2++) {
                            int wStart2 = savedStartW[ow2], wEnd2 = savedStartW[ow2 + 1];
                            double gradVal = g[((n2 * fC + c2) * fOutH + oh2) * fOutW + ow2];
                            double pixelCnt = (hEnd2 - hStart2) * (wEnd2 - wStart2);
                            if (pixelCnt == 0) continue;
                            double perPixel = gradVal / pixelCnt;
                            for (int h2 = hStart2; h2 < hEnd2; h2++) {
                                for (int w2 = wStart2; w2 < wEnd2; w2++) {
                                    dx[((n2 * fC + c2) * fH + h2) * fW + w2] += perPixel;
                                }
                            }
                        }
                    }
                }
            }
        }
        inp.accGrad(dx);
    };
    RereDiffTensor result = new RereDiffTensor(y, outShape, List.of(tensor), bw, "adaptiveAvgPool2d");
    result.scalarParam = H;
    result.scalarParam2 = W;
    return result;
}

public static IDiffTensor oneHot(RereDiffTensor tensor, int numClasses) {
    // Decompose into scatter: create zero tensor, scatter 1 at class indices
    int[] s = tensor.shape();
    long n = tensor.value.totalSize();
    int[] outShape = new int[s.length + 1];
    System.arraycopy(s, 0, outShape, 0, s.length);
    outShape[s.length] = numClasses;
    // Create zero tensor of output shape
    double[] y = new double[(int) (n * numClasses)];
    double[] xd = tensor.value.toDoubleArray();
    int classStride = 1;
    for (int i = 0; i < n; i++) {
        int cls = (int) Math.round(xd[i]);
        if (cls >= 0 && cls < numClasses) {
            y[i * numClasses + cls] = 1.0;
        }
    }
    if (!tensor.requiresGrad) return tensor.toNonDiff(new RereDoubleTensor(y, outShape));
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inp = self.inputs.get(0);
        double[] g = self.grad;
        double[] dx = new double[(int) inp.value.totalSize()];
        for (int i = 0; i < n; i++) {
            int cls = (int) Math.round(xd[i]);
            if (cls >= 0 && cls < numClasses) {
                dx[i] = g[i * numClasses + cls];
            }
        }
        inp.accGrad(dx);
    };
    return new RereDiffTensor(y, outShape, List.of(tensor), bw, "oneHot");
}

/** Build output shape for adaptiveAvgPool2d with non-standard input rank. */
private static int[] buildOutShape(int[] inShape, int N, int C, int outH, int outW) {
    if (inShape.length <= 3) return new int[]{C, outH, outW};
    int prefixRank = inShape.length - 3;
    int prefix = 1;
    for (int i = 0; i < prefixRank; i++) prefix *= inShape[i];
    if (prefix != N) throw new IllegalStateException("Shape mismatch");
    int[] out = new int[inShape.length];
    for (int i = 0; i < prefixRank; i++) out[i] = inShape[i];
    out[prefixRank] = C;
    out[prefixRank + 1] = outH;
    out[prefixRank + 2] = outW;
    return out;
}

}
