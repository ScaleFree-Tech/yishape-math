package com.yishape.lab.math.autodiff.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.DiffTensorUtil.BinaryBackward;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import com.yishape.lab.math.linalg.tensor.TensorShape;

/**
 * Element-wise binary operations extracted from {@link RereDiffTensor}.
 * All methods are static, taking the tensor as first parameter.
 */
public final class DiffTensorBinary {
    private DiffTensorBinary() { /* utility class */ }

// ==================== Element-wise binary ops — scalar ====================

public static IDiffTensor add(RereDiffTensor tensor, double scalar) { return scalarOp(tensor, scalar, (a, b) -> a + b, (g, v) -> g, "addScalar"); }
public static IDiffTensor sub(RereDiffTensor tensor, double scalar) { return scalarOp(tensor, scalar, (a, b) -> a - b, (g, v) -> g, "subScalar"); }
public static IDiffTensor mul(RereDiffTensor tensor, double scalar) { return scalarOp(tensor, scalar, (a, b) -> a * b, (g, v) -> g * scalar, "mulScalar"); }
public static IDiffTensor div(RereDiffTensor tensor, double scalar) { return scalarOp(tensor, scalar, (a, b) -> a / b, (g, v) -> g / scalar, "divScalar"); }

public static IDiffTensor rsub(RereDiffTensor tensor, double scalar) {
    if (!tensor.requiresGrad) {
        double[] data = tensor.value.toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] = scalar - data[i];
        return tensor.toNonDiff(new RereDoubleTensor(data, tensor.shape()));
    }
    int n = (int) tensor.value.totalSize();
    double[] out = new double[n];
    for (int i = 0; i < n; i++) out[i] = scalar - tensor.value.linearGet(i);
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        double[] inGrad = AutodiffBufferPool.acquire(n);
        for (int i = 0; i < n; i++) inGrad[i] = -self.grad[i];
        input.accGradFromPooled(inGrad, n);
    };
    RereDiffTensor r = new RereDiffTensor(out, tensor.shape(), List.of(tensor), bw, "rsub", scalar);
    // tape-of-tape: d²(rsub)/dx² = 0. Connect to input via mul(0).
    RereDiffTensor xRefRsub = tensor;
    int[] rsubShape = tensor.shape().clone();
    double[] rsubFactor = new double[n];
    Arrays.fill(rsubFactor, -1.0);
    r.symbolicBackwardFn = g -> new IDiffTensor[]{
        g.mul(xRefRsub.mul(0.0).add(IDiffTensor.constantTensor(rsubFactor, rsubShape)))
    };
    return r;
}

public static IDiffTensor rdiv(RereDiffTensor tensor, double scalar) {
    if (!tensor.requiresGrad) {
        double[] data = tensor.value.toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] = scalar / data[i];
        return tensor.toNonDiff(new RereDoubleTensor(data, tensor.shape()));
    }
    int n = (int) tensor.value.totalSize();
    double[] out = new double[n];
    double[] xd = tensor.value.toDoubleArray();
    for (int i = 0; i < n; i++) out[i] = scalar / xd[i];
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        int m = (int) input.value.totalSize();
        double[] inGrad = AutodiffBufferPool.acquire(m);
        for (int i = 0; i < m; i++) inGrad[i] = -self.grad[i] * scalar / (xd[i] * xd[i]);
        input.accGradFromPooled(inGrad, m);
    };
    RereDiffTensor r = new RereDiffTensor(out, tensor.shape(), List.of(tensor), bw, "rdiv", scalar);
    // tape-of-tape: d²(scalar/x)/dx² = 2*scalar/x³ ≠ 0.
    // Use tensor ops on xRef so MixedMode.hvp() gradients flow back.
    RereDiffTensor xRefRdiv = tensor;
    double rdivScalar = scalar;
    r.symbolicBackwardFn = g -> new IDiffTensor[]{
        g.mul(xRefRdiv.pow(2).reciprocal()).mul(-rdivScalar)
    };
    return r;
}

public static IDiffTensor reciprocal(RereDiffTensor tensor) {
    if (!tensor.requiresGrad) {
        double[] data = tensor.value.toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] = 1.0 / data[i];
        return tensor.toNonDiff(new RereDoubleTensor(data, tensor.shape()));
    }
    int n = (int) tensor.value.totalSize();
    double[] out = new double[n];
    double[] xd = tensor.value.toDoubleArray();
    for (int i = 0; i < n; i++) out[i] = 1.0 / xd[i];
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        int m = (int) input.value.totalSize();
        double[] inGrad = AutodiffBufferPool.acquire(m);
        for (int i = 0; i < m; i++) inGrad[i] = -self.grad[i] / (xd[i] * xd[i]);
        input.accGradFromPooled(inGrad, m);
    };
    RereDiffTensor r = new RereDiffTensor(out, tensor.shape(), List.of(tensor), bw, "reciprocal");
    // tape-of-tape: d²(1/x)/dx² = 2/x³ ≠ 0.
    // Use tensor ops on xRef so MixedMode.hvp() gradients flow back.
    RereDiffTensor xRefRecip = tensor;
    r.symbolicBackwardFn = g -> new IDiffTensor[]{
        g.mul(xRefRecip.pow(2).reciprocal()).neg()
    };
    return r;
}

static IDiffTensor scalarOp(RereDiffTensor tensor, double scalar, DoubleBinaryOperator forward,
                              DoubleBinaryOperator backward, String tag) {
    if (!tensor.requiresGrad) {
        double[] data = tensor.value.toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] = forward.applyAsDouble(data[i], scalar);
        return tensor.toNonDiff(new RereDoubleTensor(data, tensor.shape()));
    }
    int n = (int) tensor.value.totalSize();
    double[] out = new double[n];
    for (int i = 0; i < n; i++) out[i] = forward.applyAsDouble(tensor.value.linearGet(i), scalar);
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        double[] inGrad = AutodiffBufferPool.acquire(n);
        for (int i = 0; i < n; i++) {
            inGrad[i] = backward.applyAsDouble(self.grad[i], input.value.linearGet(i));
        }
        input.accGradFromPooled(inGrad, n);
    };
    RereDiffTensor result = new RereDiffTensor(out, tensor.shape(), List.of(tensor), bw, tag, scalar);
    // tape-of-tape: d²(scalar_op)/dx² = 0 (scalar ops don't change curvature).
    // Connect factor to input via mul(0) so gradient can flow (as zero).
    RereDiffTensor xRefScalar = tensor;
    int[] scShape = tensor.shape().clone();
    result.symbolicBackwardFn = g -> {
        double[] f = new double[n];
        for (int i = 0; i < n; i++) f[i] = backward.applyAsDouble(1.0, 0.0); // same for all x
        return new IDiffTensor[]{
            g.mul(xRefScalar.mul(0.0).add(IDiffTensor.constantTensor(f, scShape)))
        };
    };
    return result;
}

// ==================== Element-wise binary ops — tensor ====================

public static IDiffTensor add(RereDiffTensor tensor, IDoubleTensor other) { return binaryTensorOp(tensor, other, (a,b)->a+b, (g,a,b)->g, (g,a,b)->g, "add"); }
public static IDiffTensor sub(RereDiffTensor tensor, IDoubleTensor other) { return binaryTensorOp(tensor, other, (a,b)->a-b, (g,a,b)->g, (g,a,b)->-g, "sub"); }
public static IDiffTensor mul(RereDiffTensor tensor, IDoubleTensor other) { return binaryTensorOp(tensor, other, (a,b)->a*b, (g,a,b)->g*b, (g,a,b)->g*a, "mul"); }
public static IDiffTensor div(RereDiffTensor tensor, IDoubleTensor other) { return binaryTensorOp(tensor, other, (a,b)->a/b, (g,a,b)->g/b, (g,a,b)->-g*a/(b*b), "div"); }

static IDiffTensor binaryTensorOp(RereDiffTensor tensor, IDoubleTensor other,
                                    DoubleBinaryOperator forward,
                                    BinaryBackward gradA,
                                    BinaryBackward gradB,
                                    String tag) {
    // Detach if needed
    IDoubleTensor detOther = (other instanceof IDiffTensor dt) ? dt.detach() : other;
    boolean otherDiff = other instanceof IDiffTensor && ((IDiffTensor) other).requiresGrad();

    if (Arrays.equals(tensor.shape(), other.shape())) {
        // Same shape
        if (!tensor.requiresGrad && !otherDiff) {
            double[] aData = tensor.value.toDoubleArray();
            double[] bData = detOther.toDoubleArray();
            double[] out = new double[aData.length];
            for (int i = 0; i < out.length; i++) out[i] = forward.applyAsDouble(aData[i], bData[i]);
            return tensor.toNonDiff(new RereDoubleTensor(out, tensor.shape()));
        }
        int n = (int) tensor.value.totalSize();
        double[] out = new double[n];
        double[] aData = tensor.value.toDoubleArray();
        double[] bData = detOther.toDoubleArray();
        double[] savedA = tensor.requiresGrad ? aData.clone() : null;
        double[] savedB = (tensor.requiresGrad || otherDiff) ? bData.clone() : null;
        for (int i = 0; i < n; i++) out[i] = forward.applyAsDouble(aData[i], bData[i]);

        List<RereDiffTensor> inputs = new ArrayList<>();
        if (tensor.requiresGrad) inputs.add(tensor);
        RereDiffTensor otherNode = (other instanceof RereDiffTensor rt && rt.requiresGrad) ? rt : null;
        // Always include the other tensor in inputs for GPU/HPC graph serialization,
        // even if it doesn't require gradients. The backward function below only
        // propagates to inputs that require grad (controlled by otherNode flag).
        if (other instanceof RereDiffTensor) inputs.add((RereDiffTensor) other);

        Consumer<RereDiffTensor> bw = self -> {
            int idx = 0;
            if (tensor.requiresGrad) {
                RereDiffTensor inpA = self.inputs.get(idx++);
                double[] dA = AutodiffBufferPool.acquire(n);
                for (int i = 0; i < n; i++) dA[i] = gradA.apply(self.grad[i], savedA[i], savedB[i]);
                inpA.accGradFromPooled(dA, n);
            }
            if (otherNode != null) {
                RereDiffTensor inpB = self.inputs.get(idx);
                double[] dB = AutodiffBufferPool.acquire(n);
                for (int i = 0; i < n; i++) dB[i] = gradB.apply(self.grad[i], savedA[i], savedB[i]);
                inpB.accGradFromPooled(dB, n);
            }
        };
        RereDiffTensor result = new RereDiffTensor(out, tensor.shape(), inputs, bw, tag);
        boolean hasA = tensor.requiresGrad;
        boolean hasB = otherNode != null;
        // Per-op symbolic backward: for mul/div, use original tensor references
        // so that tape-of-tape AD (MixedMode.hvp) can flow gradients back
        // to the primal variables. For add/sub, constant factors are fine.
        if ("mul".equals(tag)) {
            RereDiffTensor aRef = tensor;
            RereDiffTensor bRef = otherNode;
            result.symbolicBackwardFn = g -> {
                if (hasA && hasB) {
                    return new IDiffTensor[]{g.mul(bRef), g.mul(aRef)};
                }
                IDiffTensor gradAT = (bRef != null) ? g.mul(bRef)
                    : g.mul(IDiffTensor.constantTensor(bData, tensor.shape()));
                IDiffTensor gradBT = g.mul(aRef);
                return new IDiffTensor[]{hasA ? gradAT : gradBT};
            };
        } else if ("div".equals(tag)) {
            RereDiffTensor aRef = tensor;
            RereDiffTensor bRef = otherNode;
            result.symbolicBackwardFn = g -> {
                if (hasA && hasB) {
                    return new IDiffTensor[]{g.div(bRef),
                        g.neg().mul(aRef).div(bRef.mul(bRef))};
                } else if (hasA) {
                    IDiffTensor bDiv = (bRef != null) ? bRef
                        : IDiffTensor.constantTensor(bData, tensor.shape());
                    return new IDiffTensor[]{g.div(bDiv)};
                } else {
                    IDiffTensor aConst = IDiffTensor.constantTensor(aData, tensor.shape());
                    return new IDiffTensor[]{g.neg().mul(aConst).div(bRef.mul(bRef))};
                }
            };
        } else {
            result.symbolicBackwardFn = DiffTensorUtil.binarySameSymbolicFn(n, gradA, gradB, aData, bData, tensor.shape(),
                hasA, hasB, tensor, otherNode);
        }
        return result;
    }

    // Broadcast case
    int[] sA = tensor.shape();
    int[] sB = other.shape();
    int[] resultShape = TensorShape.broadcastShape(sA, sB);
    long total = 1;
    for (int d : resultShape) total *= d;
    int n = (int) total;

    double[] aData = tensor.value.toDoubleArray();
    double[] bData = detOther.toDoubleArray();
    double[] out = new double[n];
    double[] bcA = new double[n];
    double[] bcB = new double[n];

    for (int i = 0; i < n; i++) {
        int[] idx = DiffTensorUtil.unlinearizeInt(i, resultShape);
        int flatA = DiffTensorUtil.flatIndexFromBroadcast(idx, sA, resultShape);
        int flatB = DiffTensorUtil.flatIndexFromBroadcast(idx, sB, resultShape);
        bcA[i] = aData[flatA];
        bcB[i] = bData[flatB];
        out[i] = forward.applyAsDouble(bcA[i], bcB[i]);
    }

    if (!tensor.requiresGrad && !otherDiff) return tensor.toNonDiff(new RereDoubleTensor(out, resultShape));

    List<RereDiffTensor> inputs = new ArrayList<>();
    if (tensor.requiresGrad) inputs.add(tensor);
    RereDiffTensor otherNode = (other instanceof RereDiffTensor rt && rt.requiresGrad) ? rt : null;
    // Always include the other tensor in inputs for GPU/HPC graph serialization,
    // even if it doesn't require gradients. The backward function below only
    // propagates to inputs that require grad (controlled by otherNode flag).
    if (other instanceof RereDiffTensor) inputs.add((RereDiffTensor) other);

    Consumer<RereDiffTensor> bw = self -> {
        int idx = 0;
        if (tensor.requiresGrad) {
            RereDiffTensor inpA = self.inputs.get(idx++);
            int aTotal = (int) DiffTensorUtil.computeSize(sA);
            double[] dA = AutodiffBufferPool.acquire(aTotal);
            for (int i = 0; i < n; i++) {
                int flatA = DiffTensorUtil.flatIndexFromBroadcast(DiffTensorUtil.unlinearizeInt(i, resultShape), sA, resultShape);
                dA[flatA] += gradA.apply(self.grad[i], bcA[i], bcB[i]);
            }
            inpA.accGradFromPooled(dA, aTotal);
        }
        if (otherNode != null) {
            RereDiffTensor inpB = self.inputs.get(idx);
            int bTotal = (int) DiffTensorUtil.computeSize(sB);
            double[] dB = AutodiffBufferPool.acquire(bTotal);
            for (int i = 0; i < n; i++) {
                int flatB = DiffTensorUtil.flatIndexFromBroadcast(DiffTensorUtil.unlinearizeInt(i, resultShape), sB, resultShape);
                dB[flatB] += gradB.apply(self.grad[i], bcA[i], bcB[i]);
            }
            inpB.accGradFromPooled(dB, bTotal);
        }
    };
    RereDiffTensor result = new RereDiffTensor(out, resultShape, inputs, bw, tag);
    boolean bHasA = tensor.requiresGrad;
    boolean bHasB = otherNode != null;
    // Per-op symbolic backward for broadcast: use original tensor references
    // for mul/div to preserve tape-of-tape connections.
    if ("mul".equals(tag)) {
        RereDiffTensor bRefBc = otherNode;
        result.symbolicBackwardFn = g -> {
            if (bHasA && bHasB) {
                return new IDiffTensor[]{g.mul(bRefBc), g.mul(tensor)};
            }
            IDiffTensor gradAT = (bRefBc != null) ? g.mul(bRefBc)
                : g.mul(IDiffTensor.constantTensor(bData, sB));
            return new IDiffTensor[]{bHasA ? gradAT : g.mul(tensor)};
        };
    } else if ("div".equals(tag)) {
        RereDiffTensor bRefBc = otherNode;
        result.symbolicBackwardFn = g -> {
            if (bHasA && bHasB) {
                return new IDiffTensor[]{g.div(bRefBc),
                    g.neg().mul(tensor).div(bRefBc.mul(bRefBc))};
            } else if (bHasA) {
                IDiffTensor bDiv = (bRefBc != null) ? bRefBc
                    : IDiffTensor.constantTensor(bData, sB);
                return new IDiffTensor[]{g.div(bDiv)};
            } else {
                IDiffTensor aConst = IDiffTensor.constantTensor(aData, sA);
                return new IDiffTensor[]{g.neg().mul(aConst).div(bRefBc.mul(bRefBc))};
            }
        };
    } else {
        result.symbolicBackwardFn = DiffTensorUtil.binaryBroadcastSymbolicFn(
            n, gradA, gradB, bcA, bcB, sA, sB, resultShape, bHasA, bHasB,
            tensor, otherNode);
    }
    return result;
}

}
