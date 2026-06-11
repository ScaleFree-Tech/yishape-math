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
        return tensor.toNonDiff(tensor.value.mmul(((IDiffTensor) other).detach()));
    }
    int M = tensor.dim(0), K = tensor.dim(1), N = other.dim(1);
    if (K != other.dim(0)) {
        throw new IllegalArgumentException("mmul: shape mismatch " + M + "x" + K + " @ " + other.dim(0) + "x" + N);
    }
    double[] aData = tensor.value.toDoubleArray();
    double[] bData = ((RereDiffTensor) other).value.toDoubleArray();
    double[] resultData = DoubleFlatGemm.flatMmul(aData, M, K, bData, N);
    int[] resultShape = {M, N};

    // Lazy transpose: only allocate when the corresponding operand needs gradients
    boolean aNeedsGrad = tensor.requiresGrad;
    boolean bNeedsGrad = other instanceof IDiffTensor od && od.requiresGrad();
    double[] bT = aNeedsGrad ? DoubleFlatGemm.flatTranspose(bData, K, N) : null;
    double[] aT = bNeedsGrad ? DoubleFlatGemm.flatTranspose(aData, M, K) : null;

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
    return new RereDiffTensor(resultData, resultShape, inputs, bw, "mmul");
}

public static IDiffTensor bmm(RereDiffTensor tensor, IDoubleTensor other) {
    boolean otherNeedsGrad = other instanceof IDiffTensor od && od.requiresGrad();
    if (!tensor.requiresGrad && !otherNeedsGrad) {
        IDoubleTensor detOther = (other instanceof IDiffTensor dt) ? dt.detach() : other;
        return tensor.toNonDiff(tensor.value.bmm(detOther));
    }
    if (tensor.rank() != 3 || other.rank() != 3) return tensor.toNonDiff(tensor.value.bmm(((IDiffTensor) other).detach()));
    int B = tensor.dim(0), M = tensor.dim(1), K = tensor.dim(2);
    int B2 = other.dim(0), K2 = other.dim(1), N = other.dim(2);
    if (B != B2 || K != K2) {
        throw new IllegalArgumentException("bmm: shape mismatch");
    }
    int[] resultShape = {B, M, N};
    double[] aData = tensor.value.toDoubleArray();
    double[] bData = ((RereDiffTensor) other).value.toDoubleArray();
    double[] resultData = DoubleFlatGemm.flatMmulBatched(aData, bData, B, M, K, N);

    // Lazy transpose: only pre-allocate slices for operands that need gradients;
    // compute each slice's transpose on-the-fly in the backward closure
    // to avoid storing all B transposes simultaneously in memory (PERF-3).
    boolean aNeedsGrad = tensor.requiresGrad;
    boolean bNeedsGrad = other instanceof IDiffTensor od && od.requiresGrad();

    // Dynamic inputs list
    List<RereDiffTensor> inputs = new ArrayList<>();
    if (aNeedsGrad) inputs.add(tensor);
    RereDiffTensor otherNode = bNeedsGrad ? (RereDiffTensor) other : null;
    if (otherNode != null) inputs.add(otherNode);

    int fB = B, fM = M, fK = K, fN = N;
    Consumer<RereDiffTensor> bw = self -> {
        int idx = 0;
        int aStride = fM * fK, bStride = fK * fN, gStride = fM * fN;
        if (aNeedsGrad) {
            RereDiffTensor inpA = self.inputs.get(idx++);
            double[] dA = new double[fB * aStride];
            for (int bi = 0; bi < fB; bi++) {
                int aOff = bi * aStride, bOff = bi * bStride, gOff = bi * gStride;
                // Compute bT on-the-fly for this batch element
                double[] bSlice = Arrays.copyOfRange(bData, bOff, bOff + bStride);
                double[] bT = DoubleFlatGemm.flatTranspose(bSlice, fK, fN);
                double[] dASlice = DoubleFlatGemm.flatMmul(self.grad, gOff, fM, fN, bT, 0, fK);
                System.arraycopy(dASlice, 0, dA, aOff, aStride);
            }
            inpA.accGrad(dA);
        }
        if (bNeedsGrad) {
            RereDiffTensor inpB = self.inputs.get(idx);
            double[] dB = new double[fB * bStride];
            for (int bi = 0; bi < fB; bi++) {
                int aOff = bi * aStride, bOff = bi * bStride, gOff = bi * gStride;
                // Compute aT on-the-fly for this batch element
                double[] aSlice = Arrays.copyOfRange(aData, aOff, aOff + aStride);
                double[] aT = DoubleFlatGemm.flatTranspose(aSlice, fM, fK);
                double[] dBSlice = DoubleFlatGemm.flatMmul(aT, 0, fK, fM, self.grad, gOff, fN);
                System.arraycopy(dBSlice, 0, dB, bOff, bStride);
            }
            inpB.accGrad(dB);
        }
    };
    return new RereDiffTensor(resultData, resultShape, inputs, bw, "bmm");
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
    IDiffTensor other = (IDiffTensor) others[0];
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

    // Compositional approach: permute → reshape → bmm → reshape
    IDiffTensor aP = tensor.permute(spec.permuteA());
    IDiffTensor bP = other.permute(spec.permuteB());

    IDiffTensor aR = aP.reshape(spec.reshapeTo3D(0, aP.shape()));
    IDiffTensor bR = bP.reshape(spec.reshapeTo3D(1, bP.shape()));

    IDiffTensor result = aR.bmm(bR);

    int[] outShape = spec.outputShape(tensor.shape(), other.shape());
    return result.reshape(outShape);
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
