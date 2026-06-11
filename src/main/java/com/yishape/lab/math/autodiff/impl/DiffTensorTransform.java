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
public final class DiffTensorTransform {
    private DiffTensorTransform() { /* utility class */ }

// ==================== Phase 2: flip, roll, repeatInterleave ====================

public static IDiffTensor flip(RereDiffTensor tensor, int... dims) {
    if (!tensor.requiresGrad) {
        // Inline flip for constant tensors (RereDoubleTensor has no flip method)
        int[] s = tensor.shape();
        double[] xd = tensor.value.toDoubleArray();
        double[] y = new double[xd.length];
        int[] fDims = new int[dims.length];
        for (int i = 0; i < dims.length; i++) fDims[i] = (dims[i] < 0 ? dims[i] + tensor.rank() : dims[i]);
        long n = tensor.value.totalSize();
        for (long flatIdx = 0; flatIdx < n; flatIdx++) {
            long dstIdx = 0;
            long srcIdx2 = flatIdx;
            long stride = 1;
            for (int d = tensor.rank() - 1; d >= 0; d--) {
                long coord = srcIdx2 % s[d];
                srcIdx2 /= s[d];
                boolean match = false;
                for (int fd : fDims) { if (fd == d) { match = true; break; } }
                if (match) coord = s[d] - 1 - coord;
                dstIdx += coord * stride;
                stride *= s[d];
            }
            y[(int) dstIdx] = xd[(int) flatIdx];
        }
        return tensor.toNonDiff(new RereDoubleTensor(y, s));
    }
    int[] s = tensor.shape();
    double[] xd = tensor.value.toDoubleArray();
    double[] y = new double[xd.length];
    int[] flippedDims = new int[dims.length];
    for (int i = 0; i < dims.length; i++) flippedDims[i] = (dims[i] < 0 ? dims[i] + tensor.rank() : dims[i]);
    // Forward: copy data flipping along each dim
    long n = tensor.value.totalSize();
    for (long flatIdx = 0; flatIdx < n; flatIdx++) {
        long srcIdx = flatIdx;
        long dstIdx = 0;
        long stride = 1;
        for (int d = tensor.rank() - 1; d >= 0; d--) {
            long coord = srcIdx % s[d];
            srcIdx /= s[d];
            boolean flipDim = false;
            for (int fd : flippedDims) { if (fd == d) { flipDim = true; break; } }
            if (flipDim) coord = s[d] - 1 - coord;
            dstIdx += coord * stride;
            stride *= s[d];
        }
        y[(int) dstIdx] = xd[(int) flatIdx];
    }
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inp = self.inputs.get(0);
        double[] g = self.grad;
        double[] dx = new double[g.length];
        // flip is self-inverse: apply same coordinate mapping to gradient
        for (long flatIdx = 0; flatIdx < n; flatIdx++) {
            long dstIdx2 = 0;
            long srcIdx2 = flatIdx;
            long stride2 = 1;
            for (int d = tensor.rank() - 1; d >= 0; d--) {
                long coord = srcIdx2 % s[d];
                srcIdx2 /= s[d];
                boolean flipDim2 = false;
                for (int fd : flippedDims) { if (fd == d) { flipDim2 = true; break; } }
                if (flipDim2) coord = s[d] - 1 - coord;
                dstIdx2 += coord * stride2;
                stride2 *= s[d];
            }
            dx[(int) flatIdx] = g[(int) dstIdx2];
        }
        inp.accGrad(dx);
    };
    return new RereDiffTensor(y, s, List.of(tensor), bw, "flip");
}

public static IDiffTensor roll(RereDiffTensor tensor, int[] shifts, int[] dims) {
    if (!tensor.requiresGrad) {
        // Use split+cat for constant tensors
        IDiffTensor result = tensor;
        for (int i = 0; i < shifts.length; i++) {
            int d = (dims[i] < 0 ? dims[i] + tensor.rank() : dims[i]);
            int dimSize = tensor.dim(d);
            int shift = ((shifts[i] % dimSize) + dimSize) % dimSize;
            if (shift == 0) continue;
            IDiffTensor[] parts = { result.narrow(d, dimSize - shift, shift),
                                    result.narrow(d, 0, dimSize - shift) };
            result = parts[0].cat(d, parts[1]);
        }
        return result;
    }
    // For differentiable tensors, also use composition (each sub-op is differentiable)
    IDiffTensor result = tensor;
    for (int i = 0; i < shifts.length; i++) {
        int d = (dims[i] < 0 ? dims[i] + tensor.rank() : dims[i]);
        int dimSize = tensor.dim(d);
        int shift = ((shifts[i] % dimSize) + dimSize) % dimSize;
        if (shift == 0) continue;
        IDiffTensor[] parts = { result.narrow(d, dimSize - shift, shift),
                                result.narrow(d, 0, dimSize - shift) };
        result = parts[0].cat(d, parts[1]);
    }
    return result;
}

public static IDiffTensor repeatInterleave(RereDiffTensor tensor, int repeats, int dim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    int dimSize = tensor.dim(d);
    // Build index array: [0,0,...,0, 1,1,...,1, ...] each repeated `repeats` times
    double[] idxData = new double[dimSize * repeats];
    for (int i = 0; i < dimSize; i++) {
        for (int r = 0; r < repeats; r++) {
            idxData[i * repeats + r] = i;
        }
    }
    IDiffTensor indices = new RereDiffTensor(idxData, new int[]{dimSize * repeats});
    indices.setRequiresGrad(false);
    IDiffTensor result = tensor.indexSelect(d, indices);
    if (result instanceof RereDiffTensor rt && tensor.requiresGrad) {
        rt.setOpTag("repeatInterleave");
    }
    return result;
}

// ==================== Phase 2: groupNorm ====================

public static IDiffTensor groupNorm(RereDiffTensor tensor, int numGroups, IDiffTensor gamma, IDiffTensor beta, double eps) {
    RereDiffTensor gr = (RereDiffTensor) gamma;
    long totalSize = tensor.value.totalSize();
    int[] s = tensor.shape();
    int rank = tensor.rank();
    if (rank < 2) throw new IllegalArgumentException("groupNorm requires rank >= 2, got " + rank);
    int C = s[rank - 2]; // channels — second to last dim for [N,C,H,W] or [N,C,L]
    if (C % numGroups != 0) throw new IllegalArgumentException("Channels (" + C + ") must be divisible by numGroups (" + numGroups + ")");
    int groupCh = C / numGroups; // channels per group
    // Compute outer dims product (batch or batch*spatial prefix) and spatial dims product
    int outer = 1;
    for (int i = 0; i < rank - 2; i++) outer *= s[i];
    int spatialPerSample = 1;
    for (int i = rank - 1; i < rank; i++) spatialPerSample *= s[i];
    int N = outer;

    double[] xd = tensor.value.toDoubleArray();
    double[] gd = gr.value.toDoubleArray();
    double[] bd = (beta != null) ? beta.toDoubleArray() : null;
    double[] y = new double[(int) totalSize];

    // Saved for backward
    double[] means = new double[N * numGroups];
    double[] sigmas = new double[N * numGroups];
    double[] xHat = new double[(int) totalSize];

    // Forward: normalize within each group
    int groupSize = groupCh * spatialPerSample;
    // Try HPC forward first
    boolean hpcFwdOk = com.yishape.lab.math.compute.hpc.HpcGroupNorm.tryForward(
            xd, gd, bd, C, numGroups, spatialPerSample, 1, eps, y);

    if (!hpcFwdOk) {
        // SISD fallback
        for (int n = 0; n < N; n++) {
            for (int g = 0; g < numGroups; g++) {
                int groupIdx = n * numGroups + g;
                double mean = 0;
                int count = 0;
                for (int c = g * groupCh; c < (g + 1) * groupCh; c++) {
                    for (int sp = 0; sp < spatialPerSample; sp++) {
                        int idx = n * C * spatialPerSample + c * spatialPerSample + sp;
                        mean += xd[idx];
                        count++;
                    }
                }
                mean /= count;
                means[groupIdx] = mean;
                double var = 0;
                for (int c = g * groupCh; c < (g + 1) * groupCh; c++) {
                    for (int sp = 0; sp < spatialPerSample; sp++) {
                        int idx = n * C * spatialPerSample + c * spatialPerSample + sp;
                        double d = xd[idx] - mean;
                        var += d * d;
                    }
                }
                var /= count;
                double sigma = Math.sqrt(var + eps);
                sigmas[groupIdx] = sigma;
                double invSigma = 1.0 / sigma;
                for (int c = g * groupCh; c < (g + 1) * groupCh; c++) {
                    for (int sp = 0; sp < spatialPerSample; sp++) {
                        int idx = n * C * spatialPerSample + c * spatialPerSample + sp;
                        double xh = (xd[idx] - mean) * invSigma;
                        xHat[idx] = xh;
                        y[idx] = xh * gd[c] + (bd != null ? bd[c] : 0);
                    }
                }
            }
        }
    }

    // Try GPU backward first
    final int fGroupCh = groupCh;
    final int fHW = spatialPerSample;
    final int fN = N;
    final int fC = C;
    final int fNumGroups = numGroups;
    final double fEps = eps;

    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inpX = self.inputs.get(0);
        RereDiffTensor inpG = self.inputs.get(1);
        RereDiffTensor inpB = (beta != null) ? self.inputs.get(2) : null;
        double[] g = self.grad;
        long m = inpX.value.totalSize();

        // Try GPU accelerated backward
        double[][] gpuResult = GpuGroupNorm.tryGroupNormBackward(
            xd, gd, g, fNumGroups, fGroupCh, fHW, fEps);
        if (gpuResult != null) {
            inpX.accGrad(gpuResult[0]);
            inpG.accGrad(gpuResult[1]);
            if (inpB != null) inpB.accGrad(gpuResult[2]);
            return;
        }

        // Try HPC accelerated backward
        double[] dxHpc = new double[(int) m];
        double[] dGammaHpc = new double[fC];
        double[] dBetaHpc = (bd != null) ? new double[fC] : new double[fC];
        if (com.yishape.lab.math.compute.hpc.HpcGroupNorm.tryBackward(
                xd, gd, g, fC, fNumGroups, fHW, 1, fEps, dxHpc, dGammaHpc, dBetaHpc)) {
            inpX.accGrad(dxHpc);
            inpG.accGrad(dGammaHpc);
            if (inpB != null) inpB.accGrad(bd != null ? dBetaHpc : new double[fC]);
            return;
        }

        // CPU fallback
        double[] dx = AutodiffBufferPool.acquire((int) m);
        double[] dGamma = new double[fC];
        double[] dBeta = (bd != null) ? new double[fC] : null;
        int grpSize = fGroupCh * fHW;

        for (int n = 0; n < fN; n++) {
            for (int gIdx = 0; gIdx < fNumGroups; gIdx++) {
                int groupIdx = n * fNumGroups + gIdx;
                double sigma = sigmas[groupIdx];
                double invSigma = 1.0 / sigma;
                double invSigma2 = invSigma * invSigma;

                // Compute sumG and sumGXH for this group
                double sumG = 0, sumGXH = 0;
                for (int c = gIdx * fGroupCh; c < (gIdx + 1) * fGroupCh; c++) {
                    for (int sp = 0; sp < fHW; sp++) {
                        int idx = n * fC * fHW + c * fHW + sp;
                        double gScaled = g[idx] * gd[c];
                        sumG += gScaled;
                        sumGXH += gScaled * xHat[idx];
                    }
                }
                double invGS = 1.0 / groupSize;

                for (int c = gIdx * fGroupCh; c < (gIdx + 1) * fGroupCh; c++) {
                    for (int sp = 0; sp < fHW; sp++) {
                        int idx = n * fC * fHW + c * fHW + sp;
                        double gScaled = g[idx] * gd[c];
                        // Standard GroupNorm backward
                        dx[idx] = (gScaled - sumG * invGS - xHat[idx] * sumGXH * invGS) * invSigma;
                        dGamma[c] += g[idx] * xHat[idx];
                        if (dBeta != null) dBeta[c] += g[idx];
                    }
                }
            }
        }
        inpX.accGradFromPooled(dx, (int) m);
        inpG.accGrad(dGamma);
        if (inpB != null && dBeta != null) inpB.accGrad(dBeta);
    };

    List<RereDiffTensor> inputs = (beta != null)
        ? List.of(tensor, gr, (RereDiffTensor) beta)
        : List.of(tensor, gr);
    RereDiffTensor result = new RereDiffTensor(y, s, inputs, bw, "groupNorm");
    result.scalarParam = eps;
    return result;
}

}
