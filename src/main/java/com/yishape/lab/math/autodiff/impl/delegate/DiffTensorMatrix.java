package com.yishape.lab.math.autodiff.impl.delegate;

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
public final class DiffTensorMatrix {
    private DiffTensorMatrix() { /* utility class */ }

// ==================== Matrix ops ====================

public static IDiffTensor mmul(RereDiffTensor tensor, IDoubleTensor other) {
    boolean otherNeedsGrad = other instanceof IDiffTensor od && od.requiresGrad();
    if (!tensor.requiresGrad && !otherNeedsGrad) {
        IDoubleTensor detOther = (other instanceof IDiffTensor dt) ? dt.detach() : other;
        return tensor.toNonDiff(tensor.value.mmul(detOther));
    }
    if (tensor.rank() != 2 || other.rank() != 2) {
        IDoubleTensor detOther = (other instanceof IDiffTensor dt) ? dt.detach() : other;
        return tensor.toNonDiff(tensor.value.mmul(detOther));
    }
    int M = tensor.dim(0), K = tensor.dim(1), N = other.dim(1);
    if (K != other.dim(0)) {
        throw new IllegalArgumentException("mmul: shape mismatch " + M + "x" + K + " @ " + other.dim(0) + "x" + N);
    }
    double[] aData = tensor.value.toDoubleArray();
    IDoubleTensor bVal = (other instanceof RereDiffTensor rt) ? rt.value
        : (other instanceof IDiffTensor dt) ? dt.detach() : other;
    double[] bData = bVal.toDoubleArray();
    double[] resultData = DoubleFlatGemm.flatMmul(aData, M, K, bData, N);
    int[] resultShape = {M, N};

    // Defer both transposes to the backward closure (PERF-2): precomputing bT and aT
    // in the forward pass holds up to two full transposes (each ~8MB for 1024×1024)
    // simultaneously through to backward. Computing them lazily inside backward — one
    // at a time, released after each operand's gradient — halves peak memory, matching
    // the bmm strategy. Correctness is unchanged (the closure already had a lazy fallback).
    boolean aNeedsGrad = tensor.requiresGrad;
    boolean bNeedsGrad = other instanceof IDiffTensor od && od.requiresGrad();
    double[] bT = null;
    double[] aT = null;

    // Dynamic inputs list: only include tensors that require gradients
    List<RereDiffTensor> inputs = new ArrayList<>();
    if (aNeedsGrad) inputs.add(tensor);
    RereDiffTensor otherNode = bNeedsGrad ? (RereDiffTensor) other : null;
    if (otherNode != null) inputs.add(otherNode);

    int fM = M, fK = K, fN = N;
    Consumer<RereDiffTensor> bw = self -> {
        int idx = 0;
        if (aNeedsGrad) {
            RereDiffTensor inpA = self.inputs.get(idx++);
            // Compute bT lazily on first backward if not pre-computed
            double[] bt = bT != null ? bT : DoubleFlatGemm.flatTranspose(bData, fK, fN);
            double[] dA = DoubleFlatGemm.flatMmul(self.grad, fM, fN, bt, fK);
            inpA.accGrad(dA);
        }
        if (bNeedsGrad) {
            RereDiffTensor inpB = self.inputs.get(idx);
            // Compute aT lazily on first backward if not pre-computed
            double[] at = aT != null ? aT : DoubleFlatGemm.flatTranspose(aData, fM, fK);
            double[] dB = DoubleFlatGemm.flatMmul(at, fK, fM, self.grad, fN);
            inpB.accGrad(dB);
        }
    };
    RereDiffTensor resultMmul = new RereDiffTensor(resultData, resultShape, inputs, bw, "mmul");
    // Symbolic backward: dA = g @ B^T, dB = A^T @ g (tape-of-tape)
    // Always capture the other operand even if it doesn't need grad,
    // so the tape-of-tape graph can reference it.
    final IDiffTensor aRefSym = tensor;
    final IDiffTensor bRefSym = (other instanceof IDiffTensor dt) ? dt
        : IDiffTensor.constantTensor(bData, new int[]{fK, fN});
    resultMmul.symbolicBackwardFn = g -> {
        java.util.List<IDiffTensor> grads = new java.util.ArrayList<>(2);
        if (aNeedsGrad) grads.add(g.mmul(bRefSym.transpose()));
        if (bNeedsGrad) grads.add(aRefSym.transpose().mmul(g));
        return grads.toArray(new IDiffTensor[0]);
    };
    return resultMmul;
}

public static IDiffTensor bmm(RereDiffTensor tensor, IDoubleTensor other) {
    boolean otherNeedsGrad = other instanceof IDiffTensor od && od.requiresGrad();
    if (!tensor.requiresGrad && !otherNeedsGrad) {
        IDoubleTensor detOther = (other instanceof IDiffTensor dt) ? dt.detach() : other;
        return tensor.toNonDiff(tensor.value.bmm(detOther));
    }
    if (tensor.rank() != 3 || other.rank() != 3) {
        IDoubleTensor detOther = (other instanceof IDiffTensor dt) ? dt.detach() : other;
        return tensor.toNonDiff(tensor.value.bmm(detOther));
    }
    int B = tensor.dim(0), M = tensor.dim(1), K = tensor.dim(2);
    int B2 = other.dim(0), K2 = other.dim(1), N = other.dim(2);
    if (B != B2 || K != K2) {
        throw new IllegalArgumentException("bmm: shape mismatch");
    }
    int[] resultShape = {B, M, N};
    double[] aData = tensor.value.toDoubleArray();
    IDoubleTensor bTensor = (other instanceof RereDiffTensor rt) ? rt.value
        : (other instanceof IDiffTensor dt) ? dt.detach() : other;
    double[] bData = bTensor.toDoubleArray();
    double[] resultData = DoubleFlatGemm.flatMmulBatched(aData, bData, B, M, K, N);

    // Lazy transpose: only pre-allocate slices for operands that need gradients;
    // compute each slice's transpose on-the-fly in the backward closure
    // to avoid storing all B transposes simultaneously in memory (PERF-3).
    boolean aNeedsGrad = tensor.requiresGrad;
    boolean bNeedsGrad = other instanceof IDiffTensor od && od.requiresGrad();

    // Dynamic inputs list
    List<RereDiffTensor> inputs = new ArrayList<>();
    if (aNeedsGrad) inputs.add(tensor);
    RereDiffTensor otherNode = (bNeedsGrad && other instanceof RereDiffTensor) ? (RereDiffTensor) other : null;
    if (otherNode != null) inputs.add(otherNode);

    int fB = B, fM = M, fK = K, fN = N;
    Consumer<RereDiffTensor> bw = self -> {
        int idx = 0;
        int aStride = fM * fK, bStride = fK * fN, gStride = fM * fN;
        if (aNeedsGrad) {
            RereDiffTensor inpA = self.inputs.get(idx++);
            double[] dA = new double[fB * aStride];
            double[] bSliceBuf = new double[bStride]; // C13: reuse buffer, avoid copyOfRange per batch
            for (int bi = 0; bi < fB; bi++) {
                int aOff = bi * aStride, bOff = bi * bStride, gOff = bi * gStride;
                System.arraycopy(bData, bOff, bSliceBuf, 0, bStride);
                double[] bT = DoubleFlatGemm.flatTranspose(bSliceBuf, fK, fN);
                double[] dASlice = DoubleFlatGemm.flatMmul(self.grad, gOff, fM, fN, bT, 0, fK);
                System.arraycopy(dASlice, 0, dA, aOff, aStride);
            }
            inpA.accGrad(dA);
        }
        if (bNeedsGrad) {
            RereDiffTensor inpB = self.inputs.get(idx);
            double[] dB = new double[fB * bStride];
            double[] aSliceBuf = new double[aStride];
            for (int bi = 0; bi < fB; bi++) {
                int aOff = bi * aStride, bOff = bi * bStride, gOff = bi * gStride;
                System.arraycopy(aData, aOff, aSliceBuf, 0, aStride);
                double[] aT = DoubleFlatGemm.flatTranspose(aSliceBuf, fM, fK);
                double[] dBSlice = DoubleFlatGemm.flatMmul(aT, 0, fK, fM, self.grad, gOff, fN);
                System.arraycopy(dBSlice, 0, dB, bOff, bStride);
            }
            inpB.accGrad(dB);
        }
    };
    RereDiffTensor resultBmm = new RereDiffTensor(resultData, resultShape, inputs, bw, "bmm");
    // Symbolic backward: dA[b]=g[b]@B[b]^T, dB[b]=A[b]^T@g[b] (tape-of-tape)
    final IDiffTensor aRefSymB = tensor;
    final IDiffTensor bRefSym2 = (other instanceof IDiffTensor dt) ? dt
        : IDiffTensor.constantTensor(bData, new int[]{fB, fK, fN});
    resultBmm.symbolicBackwardFn = g -> {
        java.util.List<IDiffTensor> grads = new java.util.ArrayList<>(2);
        if (aNeedsGrad) grads.add(g.bmm(bRefSym2.transpose(1, 2)));
        if (bNeedsGrad) grads.add(aRefSymB.transpose(1, 2).bmm(g));
        return grads.toArray(new IDiffTensor[0]);
    };
    return resultBmm;
}

public static IDiffTensor einsum(RereDiffTensor tensor, String subscript, IDoubleTensor... others) {
    if (others.length == 0) {
        return einsumSingle(tensor, subscript);
    }
    // Non-differentiable path: delegate to linalg einsum
    boolean otherNeedsGrad = others[0] instanceof IDiffTensor od && od.requiresGrad();
    if (!tensor.requiresGrad && !otherNeedsGrad) {
        IDoubleTensor[] detOthers = new IDoubleTensor[others.length];
        for (int i = 0; i < others.length; i++) {
            detOthers[i] = (others[i] instanceof IDiffTensor dt) ? dt.detach() : others[i];
        }
        return tensor.toNonDiff(tensor.value.einsum(subscript, detOthers));
    }
    IDiffTensor other;
    if (others[0] instanceof IDiffTensor od) {
        other = od;
    } else {
        // Plain IDoubleTensor — wrap as constant (no gradient flows to it)
        other = tensor.toNonDiff(others[0]);
    }
    if (others.length > 1) {
        throw new UnsupportedOperationException(
            "einsum with >2 inputs not yet supported: " + subscript);
    }
    return einsumPair(tensor, subscript, other);
}

static IDiffTensor einsumPair(RereDiffTensor tensor, String subscript, IDiffTensor other) {
    EinsumParser.EinsumSpec spec = EinsumParser.parse(subscript, tensor.shape(), other.shape());

    if (spec.contractAxes.isEmpty()) {
        // No contraction: treat as element-wise multiply with broadcasting
        return tensor.mul(other);
    }

    // Per-slice batch loop using DoubleFlatGemm for forward, custom backward for autodiff.
    // bmm([B*keptA,1,J]@[B,J,K]) doesn't work because bmm M=1 (not keptA) and batch dims may not broadcast.
    int B = 1, keptA = 1, K = 1;
    for (char c : spec.inputLabels[0].toCharArray()) {
        if (spec.batchAxes.contains(c)) B *= spec.axisSizes.get(c);
        else if (spec.contractAxes.contains(c)) { /* skip */ }
        else keptA *= spec.axisSizes.get(c);
    }
    for (char c : spec.inputLabels[1].toCharArray()) {
        if (spec.batchAxes.contains(c)) { /* already counted */ }
        else if (spec.contractAxes.contains(c)) { /* skip */ }
        else K *= spec.axisSizes.get(c);
    }
    int contractSize = 1;
    for (char c : spec.contractAxes) {
        contractSize *= spec.axisSizes.get(c);
    }

    double[] aData = tensor.value.toDoubleArray();
    IDoubleTensor bVal = (other instanceof RereDiffTensor rt) ? rt.value
        : (other instanceof IDiffTensor dt) ? dt.detach() : other;
    double[] bData = bVal.toDoubleArray();

    int outTotal = B * keptA * K;
    double[] outData = new double[outTotal];

    for (int bi = 0; bi < B * keptA; bi++) {
        int bj = B > 1 ? bi / keptA : 0;
        double[] aSlice = java.util.Arrays.copyOfRange(aData, bi * contractSize, (bi + 1) * contractSize);
        double[] bSlice = java.util.Arrays.copyOfRange(bData, bj * contractSize * K, (bj + 1) * contractSize * K);
        double[] cSlice = com.yishape.lab.math.compute.DoubleFlatGemm.flatMmul(
            aSlice, 0, 1, contractSize, bSlice, 0, K);
        System.arraycopy(cSlice, 0, outData, bi * K, K);
    }

    int[] outShape = spec.outputShape(tensor.shape(), other.shape());

    // Build custom backward — capture final copies for lambda
    final int fB = B, fKeptA = keptA, fK = K, fContractSize = contractSize;
    java.util.List<RereDiffTensor> inputs = new java.util.ArrayList<>();
    inputs.add(tensor);
    final RereDiffTensor fOtherNode = other instanceof IDiffTensor od && od.requiresGrad()
        ? (RereDiffTensor) other : null;
    if (fOtherNode != null) inputs.add(fOtherNode);

    java.util.function.Consumer<RereDiffTensor> bw = self -> {
        double[] dC = self.grad;
        double[] dA = new double[aData.length];
        double[] dB = new double[bData.length];

        for (int bi = 0; bi < fB * fKeptA; bi++) {
            int bj = fB > 1 ? bi / fKeptA : 0;
            double[] dCSlice = java.util.Arrays.copyOfRange(dC, bi * fK, (bi + 1) * fK);
            double[] bSlice = java.util.Arrays.copyOfRange(bData, bj * fContractSize * fK, (bj + 1) * fContractSize * fK);

            double[] bT = com.yishape.lab.math.compute.DoubleFlatGemm.flatTranspose(bSlice, fContractSize, fK);
            double[] dASlice = com.yishape.lab.math.compute.DoubleFlatGemm.flatMmul(dCSlice, 0, fK, bT, fContractSize);
            System.arraycopy(dASlice, 0, dA, bi * fContractSize, fContractSize);

            double[] aSlice = java.util.Arrays.copyOfRange(aData, bi * fContractSize, (bi + 1) * fContractSize);
            double[] aT = com.yishape.lab.math.compute.DoubleFlatGemm.flatTranspose(aSlice, 1, fContractSize);
            double[] dBSlice = com.yishape.lab.math.compute.DoubleFlatGemm.flatMmul(aT, fContractSize, 1, dCSlice, fK);
            int dBOff = bj * fContractSize * fK;
            for (int i = 0; i < fContractSize * fK; i++) {
                dB[dBOff + i] += dBSlice[i];
            }
        }

        if (tensor.requiresGrad) tensor.accGrad(dA);
        if (fOtherNode != null && fOtherNode.requiresGrad) fOtherNode.accGrad(dB);
    };

    return new RereDiffTensor(outData, outShape, inputs, bw, "einsum");
}

static IDiffTensor einsumSingle(RereDiffTensor tensor, String subscript) {
    EinsumParser.EinsumSpec spec = EinsumParser.parse(subscript, tensor.shape());

    if (!spec.contractAxes.isEmpty()) {
        // Summation or trace: sum over contract dims
        IDiffTensor result = tensor;
        // Sum over contract axes (highest dim first to avoid index shifts)
        int[] sortedDims = spec.contractAxes.stream()
            .mapToInt(c -> spec.inputLabels[0].indexOf(c))
            .sorted().toArray();
        for (int i = sortedDims.length - 1; i >= 0; i--) {
            result = result.sum(sortedDims[i], spec.outputLabels.indexOf(spec.inputLabels[0].charAt(sortedDims[i])) >= 0);
        }
        // If output is scalar (empty), squeeze remaining dims
        if (spec.outputLabels.isEmpty()) {
            while (result.rank() > 1) result = result.sum(0, false);
        }
        return result;
    }

    // Pure permutation
    int[] perm = new int[spec.inputLabels[0].length()];
    for (int i = 0; i < spec.outputLabels.length(); i++) {
        char c = spec.outputLabels.charAt(i);
        perm[i] = spec.inputLabels[0].indexOf(c);
    }
    return tensor.permute(perm);
}

}
