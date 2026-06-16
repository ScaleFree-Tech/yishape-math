package com.yishape.lab.math.autodiff.impl.delegate;

import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.AutodiffBufferPool;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;

/**
 * Reduction operations extracted from {@link RereDiffTensor}.
 * All methods are static, taking the tensor as first parameter.
 */
public final class DiffTensorReduce {
    private DiffTensorReduce() { /* utility class */ }

// ==================== Reduction ops ====================

public static IDiffTensor sum(RereDiffTensor tensor, int dim, boolean keepdim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.sum(d, keepdim));

    // Pattern fusion: detect unaryOp + sum(dim) patterns
    IDiffTensor fused = tensor.tryFuseSumDim(d, keepdim);
    if (fused != null) return fused;

    return sumDimImpl(tensor, d, keepdim);
}

/** Regular sum(dim) implementation (no fusion). */
static IDiffTensor sumDimImpl(RereDiffTensor tensor, int d, boolean keepdim) {
    int[] s = tensor.shape();
    int outer = 1;
    for (int i = 0; i < d; i++) outer *= s[i];
    int reduce = s[d];
    int inner = 1;
    for (int i = d + 1; i < tensor.rank(); i++) inner *= s[i];

    double[] vals = tensor.value.toDoubleArray();
    int[] resultShape = tensor.reducedShape(d, keepdim);
    int resultLen = outer * inner;
    double[] result = new double[resultLen];
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            double sum = 0;
            for (int r = 0; r < reduce; r++) {
                sum += vals[(o * reduce + r) * inner + i];
            }
            result[o * inner + i] = sum;
        }
    }
    int fOuter = outer, fReduce = reduce, fInner = inner;
    // SISD backward with buffer pooling: broadcasts grad to all reduced positions.
    // Data is strided by inner: elements for a given (o,i) are at offset
    // (o*reduce+r)*inner+i, which is non-contiguous when inner > 1.
    // DoubleVectorComputer.binaryOperate() requires contiguous arrays; gathering
    // strided elements + processing + scattering back costs more than the raw loop.
    // When inner==1 (last-dim reduction), Arrays.fill would suffice; the general
    // inner>1 case is also cheap (pure copy, no arithmetic). Pooling amortizes
    // buffer allocation across backward calls.
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        int total = fOuter * fReduce * fInner;
        double[] inGrad = AutodiffBufferPool.acquire(total);
        for (int o = 0; o < fOuter; o++) {
            for (int r = 0; r < fReduce; r++) {
                for (int i = 0; i < fInner; i++) {
                    inGrad[(o * fReduce + r) * fInner + i] = self.grad[o * fInner + i];
                }
            }
        }
        input.accGradFromPooled(inGrad, total);
    };
    RereDiffTensor rd = new RereDiffTensor(result, resultShape, List.of(tensor), bw, "sum");
    // ╔══════════════════════════════════════════════════════════════════════════╗
    // ║  ☠️ STRIDE ENCODING PITFALL — ROOT CAUSE                              ║
    // ║                                                                         ║
    // ║  THE BUG (2026-06-08): GPU sum(dim) always reduced ALL elements to 1    ║
    // ║  value because the Rust graph executor's sum/mean dispatch used         ║
    // ║  outer=1 (flat reduce) for ALL cases. It didn't know the stride.        ║
    // ║                                                                         ║
    // ║  ROOT CAUSE: This method added dimension-specific sum with proper       ║
    // ║  stride calculation (inner = product of dims after reduced dim), but     ║
    // ║  the Rust GPU graph executor (`graph.rs`) was NOT updated to read       ║
    // ║  the stride from scalarParam. It kept using outer=1 (flat sum) for      ║
    // ║  every "sum" op node, ignoring the stride entirely.                     ║
    // ║                                                                         ║
    // ║  DEFENSE: This `inner` value (= stride = product of dims after the      ║
    // ║  reduced dimension) is encoded as scalarParam on the sum tensor node.   ║
    // ║  The Rust graph executor reads scalarParam as the stride parameter:     ║
    // ║    n==1 (flat): outer=1, inner=in_size, stride=1                        ║
    // ║    n>1  (dim):  outer=n, inner=in_size/n, stride=scalarParam            ║
    // ║                                                                         ║
    // ║  If scalarParam is NaN (sum() with no dim argument → flat), the Rust    ║
    // ║  side defaults to stride=1 (contiguous flat reduction).                 ║
    // ║                                                                         ║
    // ║  RULE: Any modification to this encoding MUST also update:              ║
    // ║  1. graph.rs → reduce::dispatch call (outer/inner/stride logic)         ║
    // ║  2. reduce.wgsl → WGSL access formula                                   ║
    // ║  3. reduce.rs → dispatch signature                                      ║
    // ║  4. broadcast.wgsl → strided backward broadcast                         ║
    // ║  5. broadcast.rs → dispatch_strided function                            ║
    // ║  6. Test: SumDimGpuDiagnostic.java                                      ║
    // ╚══════════════════════════════════════════════════════════════════════════╝
    rd.setScalarParam((double) inner);
    return rd;
}

public static IDiffTensor mean(RereDiffTensor tensor, int dim, boolean keepdim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.mean(d, keepdim));
    // Pattern fusion: detect unaryOp + mean(dim) → single fused node (e.g. "reluMean")
    IDiffTensor fused = tensor.tryFuseMeanDim(d, keepdim);
    if (fused != null) return fused;
    IDiffTensor s = sum(tensor, d, keepdim);
    double scale = 1.0 / tensor.dim(d);
    return s.mul(scale);
}

public static IDiffTensor logSumExp(RereDiffTensor tensor, int dim, boolean keepdim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    if (!tensor.requiresGrad) {
        double[] vals = tensor.value.toDoubleArray();
        int[] s = tensor.shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < tensor.rank(); i++) inner *= s[i];
        double[] result = new double[outer * inner];
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                // Find max for numerical stability
                double max = vals[o * reduce * inner + i];
                for (int r = 1; r < reduce; r++) {
                    double v = vals[(o * reduce + r) * inner + i];
                    if (v > max) max = v;
                }
                double sumExp = 0;
                for (int r = 0; r < reduce; r++) {
                    sumExp += Math.exp(vals[(o * reduce + r) * inner + i] - max);
                }
                result[o * inner + i] = Math.log(sumExp) + max;
            }
        }
        return tensor.toNonDiff(new RereDoubleTensor(result, tensor.reducedShape(d, keepdim)));
    }

    int[] s = tensor.shape();
    int outer = 1;
    for (int i = 0; i < d; i++) outer *= s[i];
    int reduce = s[d];
    int inner = 1;
    for (int i = d + 1; i < tensor.rank(); i++) inner *= s[i];

    double[] vals = tensor.value.toDoubleArray();
    int[] resultShape = tensor.reducedShape(d, keepdim);
    int resultLen = outer * inner;

    // Forward: max → exp(x-max) → sum → log + max
    double[] maxVals = new double[resultLen];
    double[] sumExpVals = new double[resultLen];
    double[] result = new double[resultLen];

    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            double max = vals[o * reduce * inner + i];
            for (int r = 1; r < reduce; r++) {
                double v = vals[(o * reduce + r) * inner + i];
                if (v > max) max = v;
            }
            double sumExp = 0;
            for (int r = 0; r < reduce; r++) {
                sumExp += Math.exp(vals[(o * reduce + r) * inner + i] - max);
            }
            maxVals[o * inner + i] = max;
            sumExpVals[o * inner + i] = sumExp;
            result[o * inner + i] = Math.log(sumExp) + max;
        }
    }

    int fOuter = outer, fReduce = reduce, fInner = inner;
    double[] fMaxVals = maxVals;
    double[] fSumExpVals = sumExpVals;
    // SISD backward with buffer pooling: softmax-weighted gradient distribution.
    // The exp/sumExp computation is per-element and strided (same as sumDimImpl).
    // When inner>1, elements are non-contiguous; gathering for vc processing costs
    // more than the raw loop. Pooling amortizes buffer allocation.
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        int total = fOuter * fReduce * fInner;
        double[] inGrad = AutodiffBufferPool.acquire(total);
        for (int o = 0; o < fOuter; o++) {
            for (int r = 0; r < fReduce; r++) {
                for (int i = 0; i < fInner; i++) {
                    int flatIdx = (o * fReduce + r) * fInner + i;
                    int gradIdx = o * fInner + i;
                    // d(lse)/dx = exp(x - max) / sumExp
                    double weight = Math.exp(vals[flatIdx] - fMaxVals[gradIdx]) / fSumExpVals[gradIdx];
                    inGrad[flatIdx] = self.grad[gradIdx] * weight;
                }
            }
        }
        input.accGradFromPooled(inGrad, total);
    };
    RereDiffTensor rd = new RereDiffTensor(result, resultShape, List.of(tensor), bw, "logSumExp");
    rd.setScalarParam((double) fInner);
    return rd;
}

public static IDiffTensor max(RereDiffTensor tensor, int dim, boolean keepdim) { return minMax(tensor, dim, keepdim, true); }
public static IDiffTensor min(RereDiffTensor tensor, int dim, boolean keepdim) { return minMax(tensor, dim, keepdim, false); }

static IDiffTensor minMax(RereDiffTensor tensor, int dim, boolean keepdim, boolean isMax) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    if (!tensor.requiresGrad) {
        return tensor.toNonDiff(isMax ? tensor.value.max(d, keepdim) : tensor.value.min(d, keepdim));
    }
    int[] s = tensor.shape();
    int outer = 1;
    for (int i = 0; i < d; i++) outer *= s[i];
    int reduce = s[d];
    int inner = 1;
    for (int i = d + 1; i < tensor.rank(); i++) inner *= s[i];

    double[] vals = tensor.value.toDoubleArray();
    double[] result = new double[outer * inner];
    int[] argIdx = new int[outer * inner];
    double init = isMax ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
    Arrays.fill(result, init);
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            for (int r = 0; r < reduce; r++) {
                double v = vals[(o * reduce + r) * inner + i];
                if ((isMax && v > result[o * inner + i]) || (!isMax && v < result[o * inner + i])) {
                    result[o * inner + i] = v;
                    argIdx[o * inner + i] = r;
                }
            }
        }
    }
    int fOuter = outer, fReduce = reduce, fInner = inner;
    int[] fArg = argIdx;
    // SISD backward with buffer pooling: routes grad only to the arg position
    // (sparse pattern, typically 1 element per reduce group). Cannot be
    // vectorized — the argIdx indirection is inherently scalar. Pooling
    // amortizes buffer allocation; most elements remain at 0 (pooled default).
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        int total = fOuter * fReduce * fInner;
        double[] inGrad = AutodiffBufferPool.acquire(total);
        for (int o = 0; o < fOuter; o++) {
            for (int i = 0; i < fInner; i++) {
                int ri = fArg[o * fInner + i];
                inGrad[(o * fReduce + ri) * fInner + i] = self.grad[o * fInner + i];
            }
        }
        input.accGradFromPooled(inGrad, total);
    };
    return new RereDiffTensor(result, tensor.reducedShape(d, keepdim), List.of(tensor), bw, isMax ? "max" : "min");
}

public static IDiffTensor prod(RereDiffTensor tensor, int dim, boolean keepdim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.prod(d, keepdim));
    int[] s = tensor.shape();
    int outer = 1;
    for (int i = 0; i < d; i++) outer *= s[i];
    int reduce = s[d];
    int inner = 1;
    for (int i = d + 1; i < tensor.rank(); i++) inner *= s[i];

    double[] vals = tensor.value.toDoubleArray();
    double[] result = new double[outer * inner];
    Arrays.fill(result, 1.0);
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            for (int r = 0; r < reduce; r++) {
                result[o * inner + i] *= vals[(o * reduce + r) * inner + i];
            }
        }
    }
    int fOuter = outer, fReduce = reduce, fInner = inner;
    double[] savedVals = vals;
    double[] savedResult = result;
    // SISD backward with buffer pooling: grad_i = grad * prod / x_i for each i.
    // Requires per-element division by savedVals[idx] which varies per position;
    // the strided access pattern (same as sumDimImpl) prevents contiguous-slice
    // vectorization. Pooling amortizes buffer allocation.
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        int total = fOuter * fReduce * fInner;
        double[] inGrad = AutodiffBufferPool.acquire(total);
        for (int o = 0; o < fOuter; o++) {
            for (int i = 0; i < fInner; i++) {
                double prodVal = savedResult[o * fInner + i];
                double gi = self.grad[o * fInner + i];
                for (int r = 0; r < fReduce; r++) {
                    int idx = (o * fReduce + r) * fInner + i;
                    double xi = savedVals[idx];
                    inGrad[idx] = (xi != 0.0) ? gi * prodVal / xi : 0.0;
                }
            }
        }
        input.accGradFromPooled(inGrad, total);
    };
    return new RereDiffTensor(result, tensor.reducedShape(d, keepdim), List.of(tensor), bw, "prod");
}

public static IDiffTensor std(RereDiffTensor tensor, int dim, boolean keepdim) {
    return var(tensor, dim, keepdim).sqrt();
}

public static IDiffTensor var(RereDiffTensor tensor, int dim, boolean keepdim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.var(d, keepdim));
    int[] s = tensor.shape();
    int outer = 1;
    for (int i = 0; i < d; i++) outer *= s[i];
    int reduce = s[d];
    int inner = 1;
    for (int i = d + 1; i < tensor.rank(); i++) inner *= s[i];

    double[] vals = tensor.value.toDoubleArray();
    // Compute means
    double[] means = new double[outer * inner];
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            double sum = 0;
            for (int r = 0; r < reduce; r++) sum += vals[(o * reduce + r) * inner + i];
            means[o * inner + i] = sum / reduce;
        }
    }
    double[] varData = new double[outer * inner];
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            double sumSq = 0;
            for (int r = 0; r < reduce; r++) {
                double diff = vals[(o * reduce + r) * inner + i] - means[o * inner + i];
                sumSq += diff * diff;
            }
            varData[o * inner + i] = sumSq / reduce;
        }
    }
    int fOuter = outer, fReduce = reduce, fInner = inner;
    double[] fMeans = means;
    // SISD backward with buffer pooling: grad_i = 2*grad/n * (x_i - mean).
    // Linear per-element computation with strided access (same as sumDimImpl).
    // When inner==1 (last-dim reduction), the slice is contiguous and could use
    // vc.binaryOperate() with copy-in/copy-out, but the allocation overhead of
    // fresh arrays per (o,i) pair exceeds the per-element loop cost for typical
    // reduce sizes. Pooling amortizes buffer allocation.
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        int total = fOuter * fReduce * fInner;
        double[] inGrad = AutodiffBufferPool.acquire(total);
        for (int o = 0; o < fOuter; o++) {
            for (int i = 0; i < fInner; i++) {
                double m = fMeans[o * fInner + i];
                double scale = 2.0 * self.grad[o * fInner + i] / fReduce;
                for (int r = 0; r < fReduce; r++) {
                    inGrad[(o * fReduce + r) * fInner + i] = scale * (vals[(o * fReduce + r) * fInner + i] - m);
                }
            }
        }
        input.accGradFromPooled(inGrad, total);
    };
    return new RereDiffTensor(varData, tensor.reducedShape(d, keepdim), List.of(tensor), bw, "var");
}

}
