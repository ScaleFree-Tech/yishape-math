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
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.ReduceOperation;

/**
 * Extracted from {@link RereDiffTensor}.
 * All methods are static, taking the tensor as first parameter.
 */
public final class DiffTensorNormNN {
    private DiffTensorNormNN() { /* utility class */ }

// ==================== Layer/Batch Normalization ====================

public static IDiffTensor layerNorm(RereDiffTensor tensor, IDiffTensor gamma, IDiffTensor beta, double eps) {
    RereDiffTensor gr = (RereDiffTensor) gamma;
    RereDiffTensor br = (RereDiffTensor) beta;
    long totalSize = tensor.value.totalSize();
    int features = tensor.value.dim(tensor.rank() - 1);
    if (totalSize % features != 0) {
        throw new IllegalArgumentException(
            "Input size (" + totalSize + ") not divisible by features (" + features + ")");
    }
    int batch = (int) (totalSize / features);

    double[] xd = tensor.value.toDoubleArray();
    double[] gd = gr.value.toDoubleArray();
    double[] bd = br.value.toDoubleArray();

    double[] y = new double[(int) totalSize];
    double[] xHat = new double[(int) totalSize];
    double[] means = new double[batch];
    double[] sigmas = new double[batch];

    for (int p = 0; p < batch; p++) {
        int off = p * features;
        double mean = 0;
        for (int j = 0; j < features; j++) mean += xd[off + j];
        mean /= features;
        means[p] = mean;
        double var = 0;
        for (int j = 0; j < features; j++) { double d = xd[off + j] - mean; var += d * d; }
        var /= features;
        double sigma = Math.sqrt(var + eps);
        sigmas[p] = sigma;
        for (int j = 0; j < features; j++) {
            xHat[off + j] = (xd[off + j] - mean) / sigma;
            y[off + j] = gd[j] * xHat[off + j] + bd[j];
        }
    }

    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inpX = self.inputs.get(0);
        RereDiffTensor inpG = self.inputs.get(1);
        RereDiffTensor inpB = self.inputs.get(2);
        double[] g = self.grad;
        double[] cg = gr.value.toDoubleArray();
        int m = (int) inpX.value.totalSize();
        double[] dx = AutodiffBufferPool.acquire(m);
        double[] dGamma = AutodiffBufferPool.acquire(features);
        double[] dBeta = AutodiffBufferPool.acquire(features);
        for (int p = 0; p < batch; p++) {
            int off = p * features;
            double sigma = sigmas[p];
            double sumGT = 0, sumGTXH = 0;
            for (int j = 0; j < features; j++) {
                double gt = g[off + j] * cg[j];
                sumGT += gt;
                sumGTXH += gt * xHat[off + j];
            }
            double invFS = 1.0 / (features * sigma);
            for (int j = 0; j < features; j++) {
                double gt = g[off + j] * cg[j];
                dx[off + j] = (features * gt - sumGT - xHat[off + j] * sumGTXH) * invFS;
            }
            for (int j = 0; j < features; j++) {
                dGamma[j] += g[off + j] * xHat[off + j];
                dBeta[j] += g[off + j];
            }
        }
        inpX.accGradFromPooled(dx, m);
        inpG.accGradFromPooled(dGamma, features);
        inpB.accGradFromPooled(dBeta, features);
    };
    RereDiffTensor result = new RereDiffTensor(y, tensor.shape(), List.of(tensor, gr, br), bw, "layerNorm");
    result.scalarParam = eps;
    return result;
}

public static IDiffTensor batchNorm(RereDiffTensor tensor, IDiffTensor gamma, IDiffTensor beta, double eps) {
    RereDiffTensor gr = (RereDiffTensor) gamma;
    RereDiffTensor br = (RereDiffTensor) beta;
    long totalSize = tensor.value.totalSize();
    int features = tensor.value.dim(tensor.rank() - 1);
    if (totalSize % features != 0) {
        throw new IllegalArgumentException(
            "Input size (" + totalSize + ") not divisible by features (" + features + ")");
    }
    int batch = (int) (totalSize / features);

    double[] xd = tensor.value.toDoubleArray();
    double[] gd = gr.value.toDoubleArray();
    double[] bd = br.value.toDoubleArray();

    double[] y = new double[(int) totalSize];
    double[] means = new double[features];
    double[] invSigmas = new double[features];

    for (int j = 0; j < features; j++) {
        double mean = 0;
        for (int i = 0; i < batch; i++) mean += xd[i * features + j];
        mean /= batch;
        means[j] = mean;
        double var = 0;
        for (int i = 0; i < batch; i++) { double d = xd[i * features + j] - mean; var += d * d; }
        var /= batch;
        double invSigma = 1.0 / Math.sqrt(var + eps);
        invSigmas[j] = invSigma;
        for (int i = 0; i < batch; i++) {
            int idx = i * features + j;
            double xHat = (xd[idx] - mean) * invSigma;
            y[idx] = gd[j] * xHat + bd[j];
        }
    }

    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inpX = self.inputs.get(0);
        RereDiffTensor inpG = self.inputs.get(1);
        RereDiffTensor inpB = self.inputs.get(2);
        double[] g = self.grad;
        double[] cg = gr.value.toDoubleArray();
        int m = (int) inpX.value.totalSize();
        double[] dx = AutodiffBufferPool.acquire(m);
        double[] dGamma = AutodiffBufferPool.acquire(features);
        double[] dBeta = AutodiffBufferPool.acquire(features);
        for (int j = 0; j < features; j++) {
            double invSig = invSigmas[j];
            double mean = means[j];
            double dg = 0, db = 0, sumG = 0, sumGXHat = 0;
            double[] xHatCache = new double[batch];
            for (int i = 0; i < batch; i++) {
                int idx = i * features + j;
                double xHat = (xd[idx] - mean) * invSig;
                xHatCache[i] = xHat;
                dg += g[idx] * xHat;
                db += g[idx];
                sumG += g[idx];
                sumGXHat += g[idx] * xHat;
            }
            dGamma[j] = dg;
            dBeta[j] = db;
            double scale = cg[j] * invSig / batch;
            for (int i = 0; i < batch; i++) {
                int idx = i * features + j;
                dx[idx] = scale * (batch * g[idx] - sumG - xHatCache[i] * sumGXHat);
            }
        }
        inpX.accGradFromPooled(dx, m);
        inpG.accGradFromPooled(dGamma, features);
        inpB.accGradFromPooled(dBeta, features);
    };
    RereDiffTensor result = new RereDiffTensor(y, tensor.shape(), List.of(tensor, gr, br), bw, "batchNorm");
    result.scalarParam = eps;
    return result;
}

public static IDiffTensor rmsNorm(RereDiffTensor tensor, IDiffTensor gamma, double eps) {
    RereDiffTensor gr = (RereDiffTensor) gamma;
    long totalSize = tensor.value.totalSize();
    int features = tensor.value.dim(tensor.rank() - 1);
    if (totalSize % features != 0) {
        throw new IllegalArgumentException(
            "Input size (" + totalSize + ") not divisible by features (" + features + ")");
    }
    int batch = (int) (totalSize / features);

    double[] xd = tensor.value.toDoubleArray();
    double[] gd = gr.value.toDoubleArray();
    double[] y = new double[(int) totalSize];
    double[] rmsVals = new double[batch];

    // Try HPC accelerated path first
    boolean hpcOk = com.yishape.lab.math.compute.hpc.HpcNorm.tryRMSNormForward(
            xd, gd, batch, features, eps, y, rmsVals);

    if (!hpcOk) {
        // SISD fallback: y = x / sqrt(mean(x^2) + eps) * gamma
        for (int p = 0; p < batch; p++) {
            int off = p * features;
            double sumSq = 0;
            for (int j = 0; j < features; j++) sumSq += xd[off + j] * xd[off + j];
            double rms = Math.sqrt(sumSq / features + eps);
            rmsVals[p] = rms;
            double invRms = 1.0 / rms;
            for (int j = 0; j < features; j++) {
                y[off + j] = xd[off + j] * invRms * gd[j];
            }
        }
    }

    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inpX = self.inputs.get(0);
        RereDiffTensor inpG = self.inputs.get(1);
        double[] g = self.grad;
        int m = (int) inpX.value.totalSize();
        double[] dx = AutodiffBufferPool.acquire(m);
        double[] dGamma = new double[features];

        if (!com.yishape.lab.math.compute.hpc.HpcNorm.tryRMSNormBackward(
                xd, gd, g, rmsVals, batch, features, eps, dx, dGamma)) {
            // SISD fallback
            for (int p = 0; p < batch; p++) {
                int off = p * features;
                double rms = rmsVals[p];
                double invRms = 1.0 / rms;
                double invRms3 = invRms * invRms * invRms; // 1/rms^3
                double sumGX = 0;
                for (int j = 0; j < features; j++) {
                    sumGX += g[off + j] * xd[off + j];
                }
                double scale = sumGX * invRms3 / features;
                for (int j = 0; j < features; j++) {
                    int idx = off + j;
                    double xi = xd[idx];
                    dx[idx] = g[idx] * invRms * gd[j] - gd[j] * xi * scale;
                }
                for (int j = 0; j < features; j++) {
                    int idx = off + j;
                    dGamma[j] += g[idx] * xd[idx] * invRms;
                }
            }
        }
        inpX.accGradFromPooled(dx, m);
        inpG.accGradFromPooled(dGamma, features);
    };
    RereDiffTensor result = new RereDiffTensor(y, tensor.shape(), List.of(tensor, gr), bw, "rmsNorm");
    result.scalarParam = eps;
    return result;
}

public static IDiffTensor embedding(RereDiffTensor tensor, IDiffTensor indices) {
    // Differentiable embedding lookup: gather rows from this embedding table.
    // tensor.shape = [vocabSize, embeddingDim], indices can be any integer tensor.
    // Returns [*indices.shape, embeddingDim]
    RereDiffTensor idx = (RereDiffTensor) indices;
    int[] idxShape = idx.shape();
    int embeddingDim = tensor.dim(tensor.rank() - 1);
    // Flatten indices to 1D, gather, then reshape.
    // NOTE: opTag is deliberately NOT overridden to "embedding" on the reshape node.
    // The previous setOpTag caused GPU graph execution to crash at yishape_math_gpu
    // graph.rs:906 (index out of bounds: len 1, idx 1) because the Rust GPU handler
    // for "embedding" expected 2 direct inputs (weight + indices), but the tag was on
    // the reshape node which has only 1 input (the gather result).
    // Individual indexSelect(gather) + reshape are both fully supported by GPU/HPC.
    IDiffTensor flatIdx = idx.reshape((int) idx.totalSize());
    return tensor.indexSelect(0, flatIdx).reshape(outShape(idxShape, embeddingDim));
}

/** Helper: compute output shape [*idxShape, embeddingDim]. */
private static int[] outShape(int[] idxShape, int embeddingDim) {
    int[] outShape = new int[idxShape.length + 1];
    System.arraycopy(idxShape, 0, outShape, 0, idxShape.length);
    outShape[idxShape.length] = embeddingDim;
    return outShape;
}

public static IDiffTensor rope(RereDiffTensor tensor, int dim, int maxLen, double base) {
    // Rotary Position Embedding: apply rotation to pairs in the last dimension.
    // dim is typically headDim/2 (half the actual dimension).
    // Uses structural loops over positions (OK per CLAUDE.md exception for structural loops).
    int[] sh = tensor.shape();
    int lastDim = sh[tensor.rank() - 1];
    long totalSize = tensor.value.totalSize();
    int headDim = lastDim;
    if (headDim != dim * 2) {
        // If dim doesn't match half of last dim, use dim as-is
        headDim = dim * 2;
    }

    int fHeadDim = headDim; // final copy for lambda
    double[] xd = tensor.value.toDoubleArray();
    double[] y = new double[(int) totalSize];
    int seqLen = (int) (totalSize / fHeadDim);
    int fSeqLen = seqLen; // final copy for lambda

    // Pre-compute sin/cos tables for all positions and all pairs
    // Each position pos and pair i uses angle = pos / base^(2i/dim)
    // where i ranges 0..dim-1 (half-dim)
    int halfDim = dim;
    int fHalfDim = halfDim; // final copy for lambda
    double[] cosTable = new double[seqLen * halfDim];
    double[] sinTable = new double[seqLen * halfDim];
    for (int pos = 0; pos < seqLen; pos++) {
        int baseOff = pos * halfDim;
        for (int i = 0; i < halfDim; i++) {
            double theta = pos / Math.pow(base, 2.0 * i / dim);
            cosTable[baseOff + i] = Math.cos(theta);
            sinTable[baseOff + i] = Math.sin(theta);
        }
    }

    // Apply rotation: for each position pair, [x1, x2] → [x1*c - x2*s, x1*s + x2*c]
    for (int pos = 0; pos < seqLen; pos++) {
        int baseOff = pos * fHeadDim;
        int tblOff = pos * fHalfDim;
        for (int i = 0; i < fHalfDim; i++) {
            int idx2i = baseOff + 2 * i;
            int idx2i1 = idx2i + 1;
            double x1 = xd[idx2i];
            double x2 = xd[idx2i1];
            double c = cosTable[tblOff + i];
            double s = sinTable[tblOff + i];
            y[idx2i] = x1 * c - x2 * s;
            y[idx2i1] = x1 * s + x2 * c;
        }
    }

    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inpX = self.inputs.get(0);
        double[] g = self.grad;
        double[] dx = new double[g.length];
        for (int pos = 0; pos < fSeqLen; pos++) {
            int baseOff = pos * fHeadDim;
            int tblOff = pos * fHalfDim;
            for (int i = 0; i < fHalfDim; i++) {
                int idx2i = baseOff + 2 * i;
                int idx2i1 = idx2i + 1;
                double dY1 = g[idx2i];
                double dY2 = g[idx2i1];
                double c = cosTable[tblOff + i];
                double s = sinTable[tblOff + i];
                // Forward: [y1, y2] = [c, -s; s, c] @ [x1, x2]
                // dL/d[x1,x2] = R^T @ [dY1, dY2] = [c, s; -s, c] @ [dY1, dY2]
                dx[idx2i] = dY1 * c + dY2 * s;
                dx[idx2i1] = -dY1 * s + dY2 * c;
            }
        }
        inpX.accGrad(dx);
    };
    RereDiffTensor result = new RereDiffTensor(y, tensor.shape(), List.of(tensor), bw, "rope");
    result.scalarParam = dim;
    result.scalarParam2 = base;
    return result;
}

public static IDiffTensor[] lstmCell(RereDiffTensor tensor, IDiffTensor x, IDiffTensor hPrev, IDiffTensor cPrev,
                               IDiffTensor wInput, IDiffTensor wHidden, IDiffTensor bias) {
    // gates = x @ W_i^T + hPrev @ W_h^T + bias
    // x: [batch, inputSize], wInput: [4H, inputSize] → wInput^T: [inputSize, 4H]
    // hPrev: [batch, hiddenSize], wHidden: [4H, hiddenSize] → wHidden^T: [hiddenSize, 4H]
    IDiffTensor gates = x.mmul(wInput.transpose()).add(hPrev.mmul(wHidden.transpose()));
    if (bias != null) gates = gates.add(bias);
    IDiffTensor[] splitGates = gates.chunk(4, gates.rank() - 1);
    IDiffTensor i = splitGates[0].sigmoid();
    IDiffTensor f = splitGates[1].sigmoid();
    IDiffTensor o = splitGates[2].sigmoid();
    IDiffTensor g = splitGates[3].tanh();
    IDiffTensor c = f.mul(cPrev).add(i.mul(g));
    IDiffTensor h = o.mul(c.tanh());
    return new IDiffTensor[]{h, c};
}

public static IDiffTensor gruCell(RereDiffTensor tensor, IDiffTensor x, IDiffTensor hPrev,
                           IDiffTensor wInput, IDiffTensor wHidden, IDiffTensor bias) {
    // gates = x @ W_i^T + hPrev @ W_h^T + bias
    // x: [batch, inputSize], wInput: [3H, inputSize]
    // hPrev: [batch, hiddenSize], wHidden: [3H, hiddenSize]
    // GRU: z/r/n with r gating hidden part of n
    IDiffTensor xGates = x.mmul(wInput.transpose());
    IDiffTensor hGates = hPrev.mmul(wHidden.transpose());
    IDiffTensor[] xParts = xGates.chunk(3, xGates.rank() - 1);
    IDiffTensor[] hParts = hGates.chunk(3, hGates.rank() - 1);
    IDiffTensor z = xParts[0].add(hParts[0]).sigmoid();
    IDiffTensor r = xParts[1].add(hParts[1]).sigmoid();
    IDiffTensor nPre = xParts[2];
    if (bias != null) {
        IDiffTensor[] biasParts = bias.chunk(3, bias.rank() - 1);
        nPre = nPre.add(biasParts[2]);
    }
    IDiffTensor n = nPre.add(r.mul(hParts[2])).tanh();
    IDiffTensor h = z.neg().add(1.0).mul(n).add(z.mul(hPrev));
    return h;
}

public static IDiffTensor conv2d(RereDiffTensor tensor, IDiffTensor weight, IDiffTensor bias,
                           int stride, int padding, int dilation) {
    RereDiffTensor w = (RereDiffTensor) weight;
    RereDiffTensor b = (RereDiffTensor) bias;
    int[] inShape = tensor.shape();
    if (inShape.length != 4) {
        throw new IllegalArgumentException("conv2d: input must be 4-D [N,C,H,W], got rank=" + inShape.length);
    }
    int N = inShape[0], C = inShape[1], H = inShape[2], W_in = inShape[3];
    int[] wShape = w.shape();
    int outC = wShape[0], inC = wShape[1], kH = wShape[2], kW = wShape[3];
    if (inC != C) {
        throw new IllegalArgumentException("conv2d: weight inCh=" + inC + " != input C=" + C);
    }
    int outH = (H + 2 * padding - dilation * (kH - 1) - 1) / stride + 1;
    int outW = (W_in + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;
    if (outH <= 0 || outW <= 0) {
        throw new IllegalArgumentException("conv2d: output size invalid outH=" + outH + " outW=" + outW);
    }
    int outHW = outH * outW;
    long M = (long) N * outHW;
    if (M > Integer.MAX_VALUE / 8) {
        throw new IllegalArgumentException("conv2d: output too large");
    }
    int mM = (int) M;
    int Kcol = C * kH * kW;

    double[] xd = tensor.value.toDoubleArray();
    double[] wd = w.value.toDoubleArray();
    double[] bd = b != null ? b.value.toDoubleArray() : null;

    // im2col: [N*outH*outW, C*kH*kW], HPC→SISD fallback
    double[] col = new double[mM * Kcol];
    int kH_kW = kH * kW;
    if (HpcIm2col.tryBatchIm2col(xd, N, C, H, W_in, kH, kW, stride, padding, dilation, col)) {
        // HPC succeeded
    } else {
        // SISD fallback
        int H_W = H * W_in;
        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    int colRow = n * outHW + oh * outW + ow;
                    int colBase = colRow * Kcol;
                    for (int c = 0; c < C; c++) {
                        int cOff = c * kH_kW;
                        for (int kh = 0; kh < kH; kh++) {
                            int ih = oh * stride + kh * dilation - padding;
                            int khOff = kh * kW;
                            for (int kw = 0; kw < kW; kw++) {
                                int iw = ow * stride + kw * dilation - padding;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W_in) {
                                    col[colBase + cOff + khOff + kw] = xd[n * C * H_W + c * H_W + ih * W_in + iw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // gemm: col @ w^T → [N*outH*outW, outC]
    // w is [outC, C*kH*kW], we need w^T = [C*kH*kW, outC]
    double[] wT = DoubleFlatGemm.flatTranspose(wd, outC, Kcol);
    double[] outCol = DoubleFlatGemm.flatMmul(col, mM, Kcol, wT, outC);

    // Add bias via acceleration chain (tile bias vector across rows)
    if (bd != null) {
        double[] biasTiled = new double[mM * outC];
        for (int i = 0; i < mM; i++) {
            System.arraycopy(bd, 0, biasTiled, i * outC, outC);
        }
        outCol = new DoubleVectorComputer().binaryOperate(outCol, biasTiled, BinaryOperation.ADD);
    }

    int[] outShape = {N, outC, outH, outW};
    // Reshape output: [N*outH*outW, outC] → [N, outC, outH, outW]
    double[] y = new double[(int) ((long) N * outC * outHW)];
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                int colRow = n * outHW + oh * outW + ow;
                int outBase = n * outC * outHW + oh * outW + ow;
                for (int oc = 0; oc < outC; oc++) {
                    y[outBase + oc * outHW] = outCol[colRow * outC + oc];
                }
            }
        }
    }

    // Build input list
    List<RereDiffTensor> inputs = new ArrayList<>();
    inputs.add(tensor);
    inputs.add(w);
    if (b != null && b.requiresGrad()) inputs.add(b);

    // Capture for backward
    double[] savedCol = col;
    double[] savedWd = wd;
    int fN = N, fC = C, fH = H, fW = W_in, fOutC = outC, fOutH = outH, fOutW = outW;
    int fKH = kH, fKW = kW, fStride = stride, fPad = padding, fDil = dilation;

    Consumer<RereDiffTensor> bw = self -> {
        int inpIdx = 0;
        RereDiffTensor inpX = self.inputs.get(inpIdx++);
        RereDiffTensor inpW = self.inputs.get(inpIdx++);
        RereDiffTensor inpB = (b != null && b.requiresGrad()) ? self.inputs.get(inpIdx) : null;

        // Reshape grad from [N, outC, outH, outW] → [N*outH*outW, outC]
        int fOutHW = fOutH * fOutW;
        int fM = fN * fOutHW;
        double[] dOutCol = new double[fM * fOutC];
        for (int n2 = 0; n2 < fN; n2++) {
            for (int oh = 0; oh < fOutH; oh++) {
                for (int ow = 0; ow < fOutW; ow++) {
                    int colRow = n2 * fOutHW + oh * fOutW + ow;
                    int gradBase = n2 * fOutC * fOutHW + oh * fOutW + ow;
                    for (int oc = 0; oc < fOutC; oc++) {
                        dOutCol[colRow * fOutC + oc] = self.grad[gradBase + oc * fOutHW];
                    }
                }
            }
        }

        // d_weight = dOutCol^T @ col → [outC, C*kH*kW]
        int fKcol = fC * fKH * fKW;
        double[] dOutT = DoubleFlatGemm.flatTranspose(dOutCol, fM, fOutC);
        double[] dW = DoubleFlatGemm.flatMmul(dOutT, fOutC, fM, savedCol, fKcol);
        inpW.accGrad(dW);

        // d_bias = sum over batch+spatial (column-wise sum of [fM, fOutC])
        // Reuses dOutT ([fOutC, fM]) computed above for d_weight; reduce over last dim via GPU→SISD chain
        if (inpB != null) {
            double[] dB = GpuReduce.tryReduce(GpuReduce.SUM, dOutT, fOutC, fM);
            if (dB == null) {
                dB = new double[fOutC];
                DoubleVectorComputer bwVc2 = new DoubleVectorComputer();
                for (int oc = 0; oc < fOutC; oc++) {
                    dB[oc] = bwVc2.reduceOperate(
                        java.util.Arrays.copyOfRange(dOutT, oc * fM, (oc + 1) * fM),
                        ReduceOperation.SUM);
                }
            }
            inpB.accGrad(dB);
        }

        // d_input: dOutCol @ w → [N*outH*outW, C*kH*kW] → col2im, HPC→SISD fallback
        double[] dCol = DoubleFlatGemm.flatMmul(dOutCol, fM, fOutC, savedWd, fKcol);
        int dXsize = fN * fC * fH * fW;
        double[] dX = new double[dXsize];
        if (HpcIm2col.tryBatchCol2im(dCol, fN, fC, fH, fW, fKH, fKW, fStride, fPad, fDil, dX)) {
            // HPC succeeded
        } else {
            // SISD fallback
            int fHW = fH * fW;
            for (int n2 = 0; n2 < fN; n2++) {
                for (int oh = 0; oh < fOutH; oh++) {
                    for (int ow = 0; ow < fOutW; ow++) {
                        int colRow = n2 * fOutHW + oh * fOutW + ow;
                        int colBase = colRow * fKcol;
                        for (int c = 0; c < fC; c++) {
                            int cOff = c * fKH * fKW;
                            for (int kh = 0; kh < fKH; kh++) {
                                int ih = oh * fStride + kh * fDil - fPad;
                                int khOff = kh * fKW;
                                for (int kw = 0; kw < fKW; kw++) {
                                    int iw = ow * fStride + kw * fDil - fPad;
                                    if (ih >= 0 && ih < fH && iw >= 0 && iw < fW) {
                                        int idx = n2 * fC * fHW + c * fHW + ih * fW + iw;
                                        dX[idx] += dCol[colBase + cOff + khOff + kw];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        inpX.accGrad(dX);
    };

    RereDiffTensor result = new RereDiffTensor(y, outShape, inputs, bw, "conv2d");
    // B13: 6D exportShape provides actual H/W to HPC/GPU backends, avoiding stride-divisibility bugs
    result.exportShape = new int[]{N, C, H, W_in, outH, outW};
    result.scalarParam = Double.longBitsToDouble(
        ((long) stride << 32) | ((long) padding & 0xFFFFFFFFL));
    result.scalarParam2 = Double.longBitsToDouble(
        ((long) dilation << 32));
    return result;
}

/**
 * Stable softmax over the last dimension of a flat 2-D matrix.
 * Uses GPU→SIMD→SISD acceleration chain for row-reduce (max/sum),
 * element-wise exp, and per-row scalar arithmetic.
 *
 * @param scores  flat row-major array [rows, cols], modified in-place
 * @param rows    number of rows
 * @param cols    number of columns (softmax dim)
 * @return the same {@code scores} array, now containing row-wise softmax probabilities
 */
private static double[] softmaxRowsStable(double[] scores, int rows, int cols) {
    int total = rows * cols;
    DoubleVectorComputer vc = new DoubleVectorComputer();

    // Step 1: Row-wise max via GPU reduce → SISD fallback
    double[] rowMax = GpuReduce.tryReduce(GpuReduce.MAX, scores, rows, cols);
    if (rowMax == null) {
        rowMax = new double[rows];
        for (int r = 0; r < rows; r++) {
            double[] rowSlice = java.util.Arrays.copyOfRange(scores, r * cols, r * cols + cols);
            rowMax[r] = vc.reduceOperate(rowSlice, ReduceOperation.MAX);
        }
    }

    // Step 2: Subtract row max (per-row scalar add) → collect into shifted for batch exp
    double[] shifted = new double[total];
    for (int r = 0; r < rows; r++) {
        int rowOff = r * cols;
        double[] rowSlice = java.util.Arrays.copyOfRange(scores, rowOff, rowOff + cols);
        double[] shiftRow = vc.binaryOperate(rowSlice, -rowMax[r], BinaryOperation.ADD);
        System.arraycopy(shiftRow, 0, shifted, rowOff, cols);
    }

    // Step 3: Batch exp via GPU → SIMD/SISD fallback
    double[] exped = GpuActivation.tryExp(shifted);
    if (exped == null) {
        exped = vc.universalOperate(shifted, UniversalOperation.EXP, 0);
    }

    // Step 4: Row-wise sum via GPU reduce → SISD fallback
    double[] rowSum = GpuReduce.tryReduce(GpuReduce.SUM, exped, rows, cols);
    if (rowSum == null) {
        rowSum = new double[rows];
        for (int r = 0; r < rows; r++) {
            double[] rowSlice = java.util.Arrays.copyOfRange(exped, r * cols, r * cols + cols);
            rowSum[r] = vc.reduceOperate(rowSlice, ReduceOperation.SUM);
        }
    }

    // Step 5: Normalize (per-row scalar multiply) into scores
    for (int r = 0; r < rows; r++) {
        int rowOff = r * cols;
        double[] rowSlice = java.util.Arrays.copyOfRange(exped, rowOff, rowOff + cols);
        double[] normRow = vc.binaryOperate(rowSlice, 1.0 / rowSum[r], BinaryOperation.MULTIPLY);
        System.arraycopy(normRow, 0, scores, rowOff, cols);
    }
    return scores;
}

public static IDiffTensor scaledDotProductAttention(RereDiffTensor tensor, IDiffTensor key, IDiffTensor vTensor,
                                              IDiffTensor mask, double dropout) {
    RereDiffTensor k = (RereDiffTensor) key;
    RereDiffTensor v = (RereDiffTensor) vTensor;
    RereDiffTensor m = (RereDiffTensor) mask;
    int[] qShape = tensor.shape();
    int[] kShape = k.shape();
    int[] vShape = v.shape();
    if (qShape.length != 3 || kShape.length != 3 || vShape.length != 3) {
        throw new IllegalArgumentException(
            "scaledDotProductAttention: inputs must be 3-D [batch,seq,dim]");
    }
    int batch = qShape[0], seqQ = qShape[1], dk = qShape[2];
    int seqK = kShape[1], dk2 = kShape[2];
    int seqV = vShape[1], dv = vShape[2];
    if (batch != kShape[0] || batch != vShape[0]) {
        throw new IllegalArgumentException("scaledDotProductAttention: batch mismatch");
    }
    if (dk != dk2) {
        throw new IllegalArgumentException("scaledDotProductAttention: Q.d_k=" + dk + " != K.d_k=" + dk2);
    }
    if (seqK != seqV) {
        throw new IllegalArgumentException("scaledDotProductAttention: K.seq=" + seqK + " != V.seq=" + seqV);
    }

    double[] qd = tensor.value.toDoubleArray();
    double[] kd = k.value.toDoubleArray();
    double[] vd = v.value.toDoubleArray();
    double[] md = m != null ? m.value.toDoubleArray() : null;

    double scale = 1.0 / Math.sqrt(dk);

    int qStride = seqQ * dk;
    int kStride = seqK * dk;
    int vStride = seqK * dv;
    int scoresStride = seqQ * seqK;
    double[] attnWeights = new double[batch * scoresStride];

    DoubleVectorComputer vc = new DoubleVectorComputer();
    for (int b = 0; b < batch; b++) {
        // scores = Q @ K^T / sqrt(dk)
        double[] qSlice = java.util.Arrays.copyOfRange(qd, b * qStride, b * qStride + qStride);
        double[] kSlice = java.util.Arrays.copyOfRange(kd, b * kStride, b * kStride + kStride);
        double[] kT = DoubleFlatGemm.flatTranspose(kSlice, seqK, dk);
        // GEMM + scale via acceleration chain
        double[] rawScores = DoubleFlatGemm.flatMmul(qSlice, seqQ, dk, kT, seqK);
        double[] scaledScores = vc.binaryOperate(rawScores, scale, BinaryOperation.MULTIPLY);

        // Add mask per-batch via acceleration chain (handles broadcast internally)
        if (md != null) {
            if (md.length == scoresStride) {
                scaledScores = vc.binaryOperate(scaledScores, md, BinaryOperation.ADD);
            } else {
                // Tile mask to scores shape using data-movement primitives, then accelerate the add
                double[] maskTiled = new double[scoresStride];
                if (md.length == 1) {
                    Arrays.fill(maskTiled, md[0]);
                } else {
                    int fullReps = scoresStride / md.length;
                    for (int r = 0; r < fullReps; r++) {
                        System.arraycopy(md, 0, maskTiled, r * md.length, md.length);
                    }
                    int rem = scoresStride % md.length;
                    if (rem > 0) {
                        System.arraycopy(md, 0, maskTiled, fullReps * md.length, rem);
                    }
                }
                scaledScores = vc.binaryOperate(scaledScores, maskTiled, BinaryOperation.ADD);
            }
        }

        // Stable softmax via acceleration-eligible helper
        double[] attnB = softmaxRowsStable(scaledScores, seqQ, seqK);
        System.arraycopy(attnB, 0, attnWeights, b * scoresStride, scoresStride);
    }

    // Dropout with per-element mask (stored for backward)
    double[] attnOut = attnWeights;
    double dropoutScale = 1.0;
    double[] dropoutMask = null;
    if (dropout > 0 && dropout < 1) {
        dropoutScale = 1.0 / (1.0 - dropout);
        int totalScores = batch * scoresStride;
        dropoutMask = new double[totalScores];
        // SISD: dropout mask generation — Random.nextDouble() has no accelerated alternative (§7a exception)
        java.util.Random rng = new java.util.Random(RereDiffVector.DROPOUT_SEED_COUNTER.incrementAndGet());
        for (int i = 0; i < totalScores; i++) {
            dropoutMask[i] = (rng.nextDouble() > dropout) ? dropoutScale : 0.0;
        }
        attnOut = vc.binaryOperate(attnWeights, dropoutMask, BinaryOperation.MULTIPLY);
    }

    // output = attn @ V → [batch, seqQ, dv]
    int outStride = seqQ * dv;
    double[] y = new double[batch * outStride];
    for (int b = 0; b < batch; b++) {
        double[] outB = DoubleFlatGemm.flatMmul(
            attnOut, b * scoresStride, seqQ, seqK, vd, b * vStride, dv);
        System.arraycopy(outB, 0, y, b * outStride, outStride);
    }

    // Build inputs — fused backward handles all three
    List<RereDiffTensor> inputs = new ArrayList<>();
    inputs.add(tensor);
    inputs.add(k);
    inputs.add(v);

    // Capture for backward
    DoubleVectorComputer fvc = vc;
    double[] savedAttn = attnWeights;
    double[] savedAttnOut = attnOut;
    double[] savedQd = qd;
    double[] savedKd = kd;
    double[] savedVd = vd;
    double fScale = scale;
    double[] fDropoutMask = dropoutMask;
    int fBatch = batch, fSeqQ = seqQ, fSeqK = seqK, fDk = dk, fDv = dv;

    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inpQ = self.inputs.get(0);
        RereDiffTensor inpK = self.inputs.get(1);
        RereDiffTensor inpV = self.inputs.get(2);

        int fSoStride = fSeqQ * fSeqK;
        int fQStride = fSeqQ * fDk;
        int fKStride = fSeqK * fDk;
        int fVStride = fSeqK * fDv;
        int fOutStride = fSeqQ * fDv;

        int dQsize = fBatch * fQStride;
        int dKsize = fBatch * fKStride;
        int dVsize = fBatch * fVStride;
        double[] dQ = new double[dQsize];
        double[] dK = new double[dKsize];
        double[] dV = new double[dVsize];

        for (int b = 0; b < fBatch; b++) {
            int gOff = b * fOutStride;
            int attnOff = b * fSoStride;

            // Extract slices
            double[] attnSlice = java.util.Arrays.copyOfRange(savedAttn, attnOff, attnOff + fSoStride);
            double[] attnOutSlice = fDropoutMask != null
                ? java.util.Arrays.copyOfRange(savedAttnOut, attnOff, attnOff + fSoStride)
                : attnSlice;
            double[] vSlice = java.util.Arrays.copyOfRange(savedVd, b * fVStride, b * fVStride + fVStride);
            double[] kSlice = java.util.Arrays.copyOfRange(savedKd, b * fKStride, b * fKStride + fKStride);
            double[] qSlice = java.util.Arrays.copyOfRange(savedQd, b * fQStride, b * fQStride + fQStride);
            double[] gSlice = java.util.Arrays.copyOfRange(self.grad, gOff, gOff + fOutStride);

            // dV = attn_dropped^T @ d_output → [seqK, dv] (uses post-dropout attention when dropout>0)
            double[] attnOutT = DoubleFlatGemm.flatTranspose(attnOutSlice, fSeqQ, fSeqK);
            double[] dVB = DoubleFlatGemm.flatMmul(attnOutT, fSeqK, fSeqQ, gSlice, fDv);
            System.arraycopy(dVB, 0, dV, b * fVStride, fVStride);

            // d_attn = d_output @ V^T → [seqQ, seqK]
            double[] vT = DoubleFlatGemm.flatTranspose(vSlice, fSeqK, fDv);
            double[] dAttnB = DoubleFlatGemm.flatMmul(gSlice, fSeqQ, fDv, vT, fSeqK);

            // Dropout backward: apply per-element mask (not uniform scale)
            if (fDropoutMask != null) {
                double[] maskSlice = java.util.Arrays.copyOfRange(fDropoutMask, attnOff, attnOff + fSoStride);
                dAttnB = fvc.binaryOperate(dAttnB, maskSlice, BinaryOperation.MULTIPLY);
            }

            // Softmax backward: ds_i = p_i * (dp_i - sum_j(p_j * dp_j))
            double[] dScoresB = new double[fSeqQ * fSeqK];
            for (int q = 0; q < fSeqQ; q++) {
                int rowOff = q * fSeqK;
                double dot = 0;
                for (int j = 0; j < fSeqK; j++) {
                    dot += attnSlice[rowOff + j] * dAttnB[rowOff + j];
                }
                for (int j = 0; j < fSeqK; j++) {
                    double p = attnSlice[rowOff + j];
                    dScoresB[rowOff + j] = p * (dAttnB[rowOff + j] - dot) * fScale;
                }
            }

            // dQ = dScores @ K → [seqQ, dk]
            double[] dQB = DoubleFlatGemm.flatMmul(dScoresB, seqQ, fSeqK, kSlice, fDk);
            System.arraycopy(dQB, 0, dQ, b * fQStride, fQStride);

            // dK = dScores^T @ Q → [seqK, dk]
            double[] dScoresT = DoubleFlatGemm.flatTranspose(dScoresB, fSeqQ, fSeqK);
            double[] dKB = DoubleFlatGemm.flatMmul(dScoresT, fSeqK, fSeqQ, qSlice, fDk);
            System.arraycopy(dKB, 0, dK, b * fKStride, fKStride);
        }

        inpQ.accGrad(dQ);
        inpK.accGrad(dK);
        inpV.accGrad(dV);
    };

    RereDiffTensor result = new RereDiffTensor(y, new int[]{batch, seqQ, dv}, inputs, bw,
        "scaledDotProductAttention");
    result.scalarParam = dropout;
    return result;
}

}
