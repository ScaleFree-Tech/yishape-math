package com.yishape.lab.math.autodiff.impl.delegate;

import com.yishape.lab.math.autodiff.impl.DiffTensorUtil;
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
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;

/**
 * Extracted from {@link RereDiffTensor}.
 * All methods are static, taking the tensor as first parameter.
 */
public final class DiffTensorExtOps {
    private DiffTensorExtOps() { /* utility class */ }

// ==================== Phase 5: instanceNorm ====================

public static IDiffTensor instanceNorm(RereDiffTensor tensor, IDiffTensor gamma, IDiffTensor beta, double eps) {
    RereDiffTensor gr = (RereDiffTensor) gamma;
    int[] s = tensor.shape();
    int rank = tensor.rank();
    if (rank < 2) throw new IllegalArgumentException("instanceNorm requires rank >= 2, got " + rank);
    // instanceNorm currently supports rank-4 [N, C, H, W].
    // rank-3 [N, C, L] and rank-5+ [N, C, D, H, W] need generalized spatial computation.
    if (rank != 4) {
        throw new IllegalArgumentException(
            "instanceNorm requires rank=4 [N,C,H,W], got rank=" + rank
            + ", shape=" + java.util.Arrays.toString(s));
    }
    int N = s[0], C = s[1], H = s[2], W = s[3];
    int spatial = H * W;

    double[] xd = tensor.value.toDoubleArray();
    double[] gd = gr.value.toDoubleArray();
    double[] bd = (beta != null) ? beta.toDoubleArray() : null;
    double[] y = new double[xd.length];
    double[] means = new double[N * C];
    double[] sigmas = new double[N * C];

    // Try HPC accelerated path first
    boolean hpcOk = com.yishape.lab.math.compute.hpc.HpcNorm.tryInstanceNormForward(
            xd, gd, bd, N, C, spatial, eps, y);

    if (!hpcOk) {
        // SISD fallback
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                int instIdx = n * C + c;
                double mean = 0;
                for (int sp = 0; sp < spatial; sp++) {
                    mean += xd[(n * C + c) * spatial + sp];
                }
                mean /= spatial;
                means[instIdx] = mean;
                double var = 0;
                for (int sp = 0; sp < spatial; sp++) {
                    double d = xd[(n * C + c) * spatial + sp] - mean;
                    var += d * d;
                }
                var /= spatial;
                double sigma = Math.sqrt(var + eps);
                sigmas[instIdx] = sigma;
                double inv = 1.0 / sigma;
                for (int sp = 0; sp < spatial; sp++) {
                    int idx = (n * C + c) * spatial + sp;
                    y[idx] = (xd[idx] - mean) * inv * gd[c] + (bd != null ? bd[c] : 0);
                }
            }
        }
    }

    final int fN = N, fC = C, fSpatial = spatial;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inpX = self.inputs.get(0);
        RereDiffTensor inpG = self.inputs.get(1);
        RereDiffTensor inpB = (beta != null) ? self.inputs.get(2) : null;
        double[] g = self.grad;
        double[] dx = new double[(int) inpX.value.totalSize()];
        double[] dGamma = new double[fC];
        double[] dBeta = (bd != null) ? new double[fC] : null;
        // Clone captured arrays inside backward closure to prevent stale reads
        // on backward re-entry (HPC fail → SISD retry or multi-thread graph exec)
        double[] lMeans = means.clone();
        double[] lSigmas = sigmas.clone();
        double[] lXd = xd.clone();
        double[] lGd = gd.clone();

        if (!com.yishape.lab.math.compute.hpc.HpcNorm.tryInstanceNormBackward(
                lXd, lGd, g, fN, fC, fSpatial, eps, dx, dGamma, dBeta != null ? dBeta : new double[fC])) {
            // SISD fallback
            for (int n2 = 0; n2 < fN; n2++) {
                for (int c2 = 0; c2 < fC; c2++) {
                    int instIdx = n2 * fC + c2;
                    double sigma = lSigmas[instIdx];
                    double invSigma = 1.0 / sigma;
                    double sumG = 0, sumGXH = 0;
                    for (int sp2 = 0; sp2 < fSpatial; sp2++) {
                        int idx2 = (n2 * fC + c2) * fSpatial + sp2;
                        double gScaled = g[idx2] * lGd[c2];
                        sumG += gScaled;
                        sumGXH += gScaled * (lXd[idx2] - lMeans[instIdx]) / sigma;
                    }
                    for (int sp2 = 0; sp2 < fSpatial; sp2++) {
                        int idx2 = (n2 * fC + c2) * fSpatial + sp2;
                        dx[idx2] = (g[idx2] * lGd[c2] - sumG / fSpatial - (lXd[idx2] - lMeans[instIdx]) / sigma * sumGXH / fSpatial) * invSigma;
                        dGamma[c2] += g[idx2] * (lXd[idx2] - lMeans[instIdx]) / sigma;
                        if (dBeta != null) dBeta[c2] += g[idx2];
                    }
                }
            }
        }
        inpX.accGrad(dx);
        inpG.accGrad(dGamma);
        if (inpB != null && dBeta != null) inpB.accGrad(dBeta);
    };

    List<RereDiffTensor> inputs = (beta != null)
        ? List.of(tensor, gr, (RereDiffTensor) beta)
        : List.of(tensor, gr);
    RereDiffTensor result = new RereDiffTensor(y, s, inputs, bw, "instanceNorm");
    result.scalarParam = eps;
    return result;
}

// ==================== Phase 5: diagEmbed ====================

	public static IDiffTensor diagEmbed(RereDiffTensor tensor, int offset, int dim1, int dim2) {
	    // Input: [..., D] where D = s[rank-1] are the diagonal elements.
	    // Output: batch prefix with D inserted at dim1 and dim2 positions.
	    int[] s = tensor.shape();
	    int rank = s.length;
	    int D = s[rank - 1];
	    int batchRank = rank - 1;

	    int outRank = rank + 1;
	    int fDim1 = (dim1 < 0) ? dim1 + outRank : dim1;
	    int fDim2 = (dim2 < 0) ? dim2 + outRank : dim2;
	    if (fDim1 < 0 || fDim1 >= outRank || fDim2 < 0 || fDim2 >= outRank || fDim1 == fDim2) {
	        throw new IllegalArgumentException(
	            "diagEmbed: dim1=" + dim1 + " dim2=" + dim2 + " invalid for input rank " + rank);
	    }

	    int[] outShape = new int[outRank];
	    int[] prefixShape = new int[batchRank];
	    System.arraycopy(s, 0, prefixShape, 0, batchRank);
	    for (int i = 0, pi = 0; i < outRank; i++) {
	        outShape[i] = (i == fDim1 || i == fDim2) ? D : prefixShape[pi++];
	    }

	    long outer = tensor.value.totalSize() / D;
	    double[] xd = tensor.value.toDoubleArray();
	    double[] y = new double[(int)(outer * D * D)];

	    for (int o = 0; o < outer; o++) {
	        int[] batchCoord = DiffTensorUtil.decomposeFlatIndex(o, prefixShape);
	        for (int k = 0; k < D; k++) {
	            int row = (offset >= 0) ? k : k - offset;
	            int col = (offset >= 0) ? k + offset : k;
	            if (row < 0 || row >= D || col < 0 || col >= D) continue;
	            y[DiffTensorUtil.buildDiagEmbedOutputIndex(batchCoord, row, col, fDim1, fDim2, outShape)] = xd[o * D + k];
	        }
	    }

	    if (!tensor.requiresGrad) return tensor.toNonDiff(new RereDoubleTensor(y, outShape));

	    final int fD = D, fOffset = offset;
	    final int ffDim1 = fDim1, ffDim2 = fDim2;
	    final long fOuter = outer;
	    final int[] fPrefixShape = prefixShape.clone();
	    final int[] fOutShape = outShape.clone();

	    Consumer<RereDiffTensor> bw = self -> {
	        RereDiffTensor inp = self.inputs.get(0);
	        double[] g = self.gradData();
	        double[] dx = new double[(int)(fOuter * fD)];
	        for (int o = 0; o < fOuter; o++) {
	            int[] batchCoord = DiffTensorUtil.decomposeFlatIndex(o, fPrefixShape);
	            for (int k = 0; k < fD; k++) {
	                int row = (fOffset >= 0) ? k : k - fOffset;
	                int col = (fOffset >= 0) ? k + fOffset : k;
	                if (row < 0 || row >= fD || col < 0 || col >= fD) continue;
	                dx[o * fD + k] += g[DiffTensorUtil.buildDiagEmbedOutputIndex(batchCoord, row, col, ffDim1, ffDim2, fOutShape)];
	            }
	        }
	        inp.accGrad(dx);
	    };
	    return new RereDiffTensor(y, outShape, List.of(tensor), bw, "diagEmbed");
	}



// ==================== Phase 5: dropout2d ====================

public static IDiffTensor dropout2d(RereDiffTensor tensor, double p) {
    if (p <= 0) return tensor;
    if (!tensor.requiresGrad) return tensor;
    int[] s = tensor.shape();
    if (s.length < 3) throw new IllegalArgumentException("dropout2d requires rank >= 3, got " + s.length);
    int rankLocal = s.length;
    int nn = 1;
    for (int i = 0; i < rankLocal - 2; i++) nn *= s[i];
    final int N = nn;
    final int C = s[rankLocal - 2];

    double[] xd = tensor.value.toDoubleArray();
    final int total = xd.length;
    double[] y = new double[total];
    // Create channel-wise mask: same for all spatial positions
    double scale = 1.0 / (1.0 - p);
    double[] chMask = new double[N * C];
    java.util.Random rng = new java.util.Random();
    for (int i = 0; i < N * C; i++) {
        chMask[i] = (rng.nextDouble() >= p) ? scale : 0;
    }

    final int spatial = total / (N * C);
    for (int nc = 0; nc < N * C; nc++) {
        double m = chMask[nc];
        for (int sp = 0; sp < spatial; sp++) {
            y[nc * spatial + sp] = xd[nc * spatial + sp] * m;
        }
    }

    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inp = self.inputs.get(0);
        double[] g = self.grad;
        double[] dx = new double[total];
        for (int nc2 = 0; nc2 < N * C; nc2++) {
            double m = chMask[nc2];
            for (int sp2 = 0; sp2 < spatial; sp2++) {
                dx[nc2 * spatial + sp2] = g[nc2 * spatial + sp2] * m;
            }
        }
        inp.accGrad(dx);
    };
    return new RereDiffTensor(y, s, List.of(tensor), bw, "dropout2d");
}

// ==================== Phase 5: depthwiseConv1d ====================

public static IDiffTensor depthwiseConv1d(RereDiffTensor tensor, IDiffTensor weight, int stride, int padding) {
    RereDiffTensor w = (RereDiffTensor) weight;
    int[] s = tensor.shape();
    if (s.length != 3) throw new IllegalArgumentException("depthwiseConv1d requires 3D input [N,C,L], got rank " + s.length);
    int N = s[0], C = s[1], L = s[2];
    int kSize = w.dim(w.rank() - 1);
    int effStride = (stride <= 0) ? 1 : stride;
    int outL = (L + 2 * padding - kSize) / effStride + 1;

    double[] xd = tensor.value.toDoubleArray();
    double[] wd = w.value.toDoubleArray();
    double[] y = new double[N * C * outL];

    // C14: try HPC-accelerated forward first, fall back to SISD
    if (!com.yishape.lab.math.compute.hpc.HpcDepthwiseConv1d.tryForward(
            xd, wd, null, L, C, kSize, y)) {
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int ol = 0; ol < outL; ol++) {
                    double sum = 0;
                    for (int k = 0; k < kSize; k++) {
                        int inIdx = ol * effStride + k - padding;
                        if (inIdx >= 0 && inIdx < L) {
                            sum += xd[(n * C + c) * L + inIdx] * wd[c * kSize + k];
                        }
                    }
                    y[(n * C + c) * outL + ol] = sum;
                }
            }
        }
    }

    int[] outShape = new int[]{N, C, outL};
    final int fN = N, fC = C, fL = L, fOutL = outL, fK = kSize, fStride = effStride, fPad = padding;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inpX = self.inputs.get(0);
        RereDiffTensor inpW = self.inputs.get(1);
        double[] g = self.grad;
        double[] dx = new double[(int) inpX.value.totalSize()];
        double[] dw = new double[(int) inpW.value.totalSize()];

        // Try HPC backward first (the HPC forward is causal-only, so forward stays SISD)
        int perSampleIn = fC * fL;
        int perSampleOut = fC * fOutL;
        for (int n2 = 0; n2 < fN; n2++) {
            double[] xSlice = new double[perSampleIn];
            double[] gSlice = new double[perSampleOut];
            double[] dxSlice = new double[perSampleIn];
            double[] dwSlice = new double[(int) fC * fK];
            System.arraycopy(xd, n2 * perSampleIn, xSlice, 0, perSampleIn);
            System.arraycopy(g, n2 * perSampleOut, gSlice, 0, perSampleOut);

            if (!com.yishape.lab.math.compute.hpc.HpcDepthwiseConv1d.tryBackward(
                    xSlice, wd, gSlice, fL, fC, fK, fStride, fPad, dxSlice, dwSlice)) {
                // SISD fallback for this sample
                for (int c2 = 0; c2 < fC; c2++) {
                    for (int ol2 = 0; ol2 < fOutL; ol2++) {
                        double gradVal = gSlice[(c2) * fOutL + ol2];
                        for (int k2 = 0; k2 < fK; k2++) {
                            int inIdx2 = ol2 * fStride + k2 - fPad;
                            if (inIdx2 >= 0 && inIdx2 < fL) {
                                int xIdx = c2 * fL + inIdx2;
                                dxSlice[xIdx] += gradVal * wd[c2 * fK + k2];
                                dwSlice[c2 * fK + k2] += gradVal * xSlice[xIdx];
                            }
                        }
                    }
                }
            }
            // Accumulate back
            for (int i = 0; i < perSampleIn; i++) dx[n2 * perSampleIn + i] += dxSlice[i];
            for (int i = 0; i < fC * fK; i++) dw[i] += dwSlice[i];
        }
        inpX.accGrad(dx);
        inpW.accGrad(dw);
    };
    RereDiffTensor result = new RereDiffTensor(y, outShape, List.of(tensor, w), bw, "depthwiseConv1d");
    result.scalarParam = Double.longBitsToDouble(((long) L << 16) | (long) kSize);
    result.scalarParam2 = C;
    return result;
}

// ==================== Phase 5: interpolate ====================

	public static IDiffTensor interpolate(RereDiffTensor tensor, double scaleFactor, String mode) {
	    int[] s = tensor.shape();
	    if (s.length < 3) throw new IllegalArgumentException("interpolate requires rank >= 3, got " + s.length);
	    int N, C, H, W;
	    if (s.length == 3) { N = 1; C = s[0]; H = s[1]; W = s[2]; }
	    else if (s.length == 4) { N = s[0]; C = s[1]; H = s[2]; W = s[3]; }
	    else { N = 1; for (int i = 0; i < s.length - 3; i++) N *= s[i]; C = s[s.length - 3]; H = s[s.length - 2]; W = s[s.length - 1]; }

	    int outH = (int) Math.floor(H * scaleFactor);
	    int outW = (int) Math.floor(W * scaleFactor);
	    boolean bilinear = "bilinear".equals(mode);
	    int totalOut = N * C * outH * outW;

	    double[] xd = tensor.value.toDoubleArray();
	    double[] y = new double[totalOut];

	    // Flat arrays for backward SISD fallback — allocated only when HPC fails
	    final double[] savedWeightsFlat;
	    final int[] savedIndicesFlat;
	    final int savedStride; // 4 for bilinear, 1 for nearest

	    // Try HPC forward first
	    boolean hpcFwdOk = false;
	    if (bilinear) {
	        hpcFwdOk = com.yishape.lab.math.compute.hpc.HpcInterpolate.tryBilinearForward(
	                xd, N, C, H, W, outH, outW, y);
	    } else {
	        hpcFwdOk = com.yishape.lab.math.compute.hpc.HpcInterpolate.tryNearestForward(
	                xd, N, C, H, W, outH, outW, y);
	    }

	    if (!hpcFwdOk) {
	        // SISD fallback — allocate flat arrays (single allocation, no per-pixel objects)
	        savedStride = bilinear ? 4 : 1;
	        savedWeightsFlat = bilinear ? new double[totalOut * 4] : null;
	        savedIndicesFlat = new int[totalOut * savedStride];

	        // Structural loops: per-pixel coordinate transform + index computation — not element-wise arithmetic
	        for (int n = 0; n < N; n++) {
	            for (int c = 0; c < C; c++) {
	                for (int oh = 0; oh < outH; oh++) {
	                    double srcH = (oh + 0.5) / scaleFactor - 0.5;
	                    int h0 = (int) Math.floor(srcH);
	                    int h1 = Math.min(h0 + 1, H - 1);
	                    h0 = Math.max(h0, 0);
	                    double dh = srcH - h0;
	                    for (int ow = 0; ow < outW; ow++) {
	                        double srcW = (ow + 0.5) / scaleFactor - 0.5;
	                        int w0 = (int) Math.floor(srcW);
	                        int w1 = Math.min(w0 + 1, W - 1);
	                        w0 = Math.max(w0, 0);
	                        double dw = srcW - w0;
	                        int outIdx = ((n * C + c) * outH + oh) * outW + ow;

	                        if (bilinear) {
	                            double v00 = xd[((n * C + c) * H + h0) * W + w0];
	                            double v01 = xd[((n * C + c) * H + h0) * W + w1];
	                            double v10 = xd[((n * C + c) * H + h1) * W + w0];
	                            double v11 = xd[((n * C + c) * H + h1) * W + w1];
	                            y[outIdx] = (1 - dh) * (1 - dw) * v00 + (1 - dh) * dw * v01
	                                       + dh * (1 - dw) * v10 + dh * dw * v11;
	                            int base = outIdx * 4;
	                            savedWeightsFlat[base]     = (1 - dh) * (1 - dw);
	                            savedWeightsFlat[base + 1] = (1 - dh) * dw;
	                            savedWeightsFlat[base + 2] = dh * (1 - dw);
	                            savedWeightsFlat[base + 3] = dh * dw;
	                            savedIndicesFlat[base]     = h0 * W + w0;
	                            savedIndicesFlat[base + 1] = h0 * W + w1;
	                            savedIndicesFlat[base + 2] = h1 * W + w0;
	                            savedIndicesFlat[base + 3] = h1 * W + w1;
	                        } else {
	                            int hNear = (int) Math.round(srcH);
	                            int wNear = (int) Math.round(srcW);
	                            hNear = Math.max(0, Math.min(H - 1, hNear));
	                            wNear = Math.max(0, Math.min(W - 1, wNear));
	                            y[outIdx] = xd[((n * C + c) * H + hNear) * W + wNear];
	                            savedIndicesFlat[outIdx] = hNear * W + wNear;
	                        }
	                    }
	                }
	            }
	        }
	    } else {
	        savedWeightsFlat = null;
	        savedIndicesFlat = null;
	        savedStride = 0;
	    }

	    int[] outShape = (s.length == 4) ? new int[]{N, C, outH, outW}
	        : (s.length == 3) ? new int[]{C, outH, outW} : RereDiffTensor.buildOutShape(s, N, C, outH, outW);
	    final int fH = H, fW = W, fOutH = outH, fOutW = outW, fN = N, fC = C;
	    final boolean fBilinear = bilinear;
	    Consumer<RereDiffTensor> bw = self -> {
	        RereDiffTensor inp = self.inputs.get(0);
	        double[] g = self.grad;
	        double[] dx = new double[(int) inp.value.totalSize()];

	        // Try HPC backward first
	        boolean hpcBwdOk = false;
	        if (fBilinear) {
	            hpcBwdOk = com.yishape.lab.math.compute.hpc.HpcInterpolate.tryBilinearBackward(
	                    g, fN, fC, fH, fW, fOutH, fOutW, dx);
	        } else {
	            hpcBwdOk = com.yishape.lab.math.compute.hpc.HpcInterpolate.tryNearestBackward(
	                    g, fN, fC, fH, fW, fOutH, fOutW, dx);
	        }

	        if (!hpcBwdOk) {
	            // SISD fallback — structural loops reading from flat arrays
	            final double[] swf = savedWeightsFlat;
	            final int[] sif = savedIndicesFlat;
	            for (int n2 = 0; n2 < fN; n2++) {
	                for (int c2 = 0; c2 < fC; c2++) {
	                    for (int oh2 = 0; oh2 < fOutH; oh2++) {
	                        for (int ow2 = 0; ow2 < fOutW; ow2++) {
	                            int outIdx2 = ((n2 * fC + c2) * fOutH + oh2) * fOutW + ow2;
	                            double gradVal = g[outIdx2];
	                            int baseIn = ((n2 * fC + c2) * fH);
	                            if (fBilinear && swf != null) {
	                                int base = outIdx2 * 4;
	                                dx[baseIn * fW + sif[base]]     += gradVal * swf[base];
	                                dx[baseIn * fW + sif[base + 1]] += gradVal * swf[base + 1];
	                                dx[baseIn * fW + sif[base + 2]] += gradVal * swf[base + 2];
	                                dx[baseIn * fW + sif[base + 3]] += gradVal * swf[base + 3];
	                            } else {
	                                dx[baseIn * fW + sif[outIdx2]] += gradVal;
	                            }
	                        }
	                    }
	                }
	            }
	        }
	        inp.accGrad(dx);
	    };
	    RereDiffTensor result = new RereDiffTensor(y, outShape, List.of(tensor), bw, "interpolate");
	    result.scalarParam = Double.longBitsToDouble(((long) H << 32) | ((long) W & 0xFFFF_FFFFL));
	    result.scalarParam2 = bilinear ? 0.0 : 1.0;
	    return result;
	}

}
