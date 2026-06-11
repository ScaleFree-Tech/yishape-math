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
public final class DiffTensorSoftmax {
    private DiffTensorSoftmax() { /* utility class */ }

// ==================== Softmax / logSoftmax ====================

public static IDiffTensor softmax(RereDiffTensor tensor, int dim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.softmax(d));
    int[] s = tensor.shape();
    int outer = 1;
    for (int i = 0; i < d; i++) outer *= s[i];
    int reduce = s[d];
    int inner = 1;
    for (int i = d + 1; i < tensor.rank(); i++) inner *= s[i];

    double[] inData = tensor.value.toDoubleArray();
    double[] out = new double[inData.length];
    // Compute softmax: out = exp(x - max) / sum(exp(x - max))
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            double maxVal = Double.NEGATIVE_INFINITY;
            for (int r = 0; r < reduce; r++) {
                double v = inData[(o * reduce + r) * inner + i];
                if (v > maxVal) maxVal = v;
            }
            double sumExp = 0;
            for (int r = 0; r < reduce; r++) {
                int idx = (o * reduce + r) * inner + i;
                out[idx] = Math.exp(inData[idx] - maxVal);
                sumExp += out[idx];
            }
            for (int r = 0; r < reduce; r++) {
                int idx = (o * reduce + r) * inner + i;
                out[idx] /= sumExp;
            }
        }
    }
    int fOuter = outer, fReduce = reduce, fInner = inner;
    double[] sm = out.clone();
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        int total = fOuter * fReduce * fInner;
        double[] inGrad = AutodiffBufferPool.acquire(total);
        for (int o = 0; o < fOuter; o++) {
            for (int i = 0; i < fInner; i++) {
                double dot = 0;
                for (int r = 0; r < fReduce; r++) {
                    int idx = (o * fReduce + r) * fInner + i;
                    dot += self.grad[idx] * sm[idx];
                }
                for (int r = 0; r < fReduce; r++) {
                    int idx = (o * fReduce + r) * fInner + i;
                    inGrad[idx] = sm[idx] * (self.grad[idx] - dot);
                }
            }
        }
        input.accGradFromPooled(inGrad, total);
    };
    return new RereDiffTensor(out, s, List.of(tensor), bw, "softmax");
}

public static IDiffTensor logSoftmax(RereDiffTensor tensor, int dim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.logSoftmax(d));
    IDiffTensor sm = DiffTensorSoftmax.softmax(tensor, d);
    if (!(sm instanceof RereDiffTensor rsm)) return tensor.toNonDiff(tensor.value.logSoftmax(d));

    // Compute log(sm) forward, with proper backward
    double[] smData = rsm.value.toDoubleArray();
    double[] logData = new double[smData.length];
    for (int i = 0; i < smData.length; i++) logData[i] = Math.log(smData[i]);

    int[] s = tensor.shape();
    int outer = 1;
    for (int i = 0; i < d; i++) outer *= s[i];
    int reduce = s[d];
    int inner = 1;
    for (int i = d + 1; i < tensor.rank(); i++) inner *= s[i];

    int fOuter = outer, fReduce = reduce, fInner = inner;
    double[] fSmData = smData;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        int total = fOuter * fReduce * fInner;
        double[] inGrad = AutodiffBufferPool.acquire(total);
        for (int o = 0; o < fOuter; o++) {
            for (int i = 0; i < fInner; i++) {
                double gSum = 0;
                for (int r = 0; r < fReduce; r++) {
                    int idx = (o * fReduce + r) * fInner + i;
                    gSum += self.grad[idx];
                }
                for (int r = 0; r < fReduce; r++) {
                    int idx = (o * fReduce + r) * fInner + i;
                    inGrad[idx] = self.grad[idx] - fSmData[idx] * gSum;
                }
            }
        }
        input.accGradFromPooled(inGrad, total);
    };
    return new RereDiffTensor(logData, s, List.of(tensor), bw, "logSoftmax");
}

public static IDiffTensor softmaxCrossEntropy(RereDiffTensor tensor, IDoubleTensor labels, int dim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    int[] s = tensor.shape();
    int r = tensor.rank();
    int outerSize = 1;
    for (int i = 0; i < d; i++) outerSize *= s[i];
    int classSize = s[d];
    int innerSize = 1;
    for (int i = d + 1; i < r; i++) innerSize *= s[i];
    int totalSamples = outerSize * innerSize;

    double[] logits = tensor.value.toDoubleArray();
    double[] labelData = labels instanceof RereDiffTensor rl ? rl.value.toDoubleArray() : labels.toDoubleArray();

    double[] softmax = new double[logits.length];
    double totalLoss = 0;
    for (int o = 0; o < outerSize; o++) {
        for (int in = 0; in < innerSize; in++) {
            int base = (o * classSize) * innerSize + in;
            double mx = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < classSize; c++) {
                double v = logits[base + c * innerSize];
                if (v > mx) mx = v;
            }
            double sumExp = 0;
            for (int c = 0; c < classSize; c++) {
                double ex = Math.exp(logits[base + c * innerSize] - mx);
                softmax[base + c * innerSize] = ex;
                sumExp += ex;
            }
            double invSum = 1.0 / sumExp;
            for (int c = 0; c < classSize; c++) {
                softmax[base + c * innerSize] *= invSum;
                double p = softmax[base + c * innerSize];
                double y = labelData[base + c * innerSize];
                totalLoss += -y * Math.log(Math.max(p, 1e-30));
            }
        }
    }
    double meanLoss = totalLoss / totalSamples;
    double[] resultData = new double[] { meanLoss };
    int[] resultShape = new int[] { 1 };

    if (!tensor.requiresGrad) return new ConstantDiffTensor(new RereDoubleTensor(resultData, resultShape));

    int fOuter = outerSize, fClassSize = classSize, fInner = innerSize, fTotal = totalSamples;
    double[] fSoftmax = softmax;
    double[] fLabelData = labelData;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        double gradScale = self.grad[0] / fTotal;
        int m = fSoftmax.length;
        double[] inGrad = AutodiffBufferPool.acquire(m);
        for (int i = 0; i < m; i++) {
            inGrad[i] = gradScale * (fSoftmax[i] - fLabelData[i]);
        }
        input.accGradFromPooled(inGrad, m);
    };
    return new RereDiffTensor(resultData, resultShape, List.of(tensor), bw, "softmaxCrossEntropy");
}

/**
 * Fused softmax + cross-entropy with sparse integer labels (class indices).
 * Memory: O(B*C) for softmax probabilities (required for backward), but avoids
 * allocating a [B, C] one-hot tensor. Saves 5 intermediate graph nodes vs
 * the manual logSumExp→sub→gather→sum→div→neg chain.
 *
 * @param labels integer class indices, length = outerSize * innerSize, each in [0, classSize)
 * @param dim    the class dimension
 * @return scalar loss = mean(-log(softmax[target]))
 */
public static IDiffTensor softmaxCrossEntropySparse(RereDiffTensor tensor, int[] labels, int dim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    int[] s = tensor.shape();
    int r = tensor.rank();
    int outerSize = 1;
    for (int i = 0; i < d; i++) outerSize *= s[i];
    int classSize = s[d];
    int innerSize = 1;
    for (int i = d + 1; i < r; i++) innerSize *= s[i];
    int totalSamples = outerSize * innerSize;
    if (labels.length != totalSamples) {
        throw new IllegalArgumentException(
            "labels length " + labels.length + " != totalSamples " + totalSamples);
    }

    double[] logits = tensor.value.toDoubleArray();

    // Forward: softmax → -log(softmax[target]) → mean
    double[] softmax = new double[logits.length];
    double totalLoss = 0;
    for (int o = 0; o < outerSize; o++) {
        for (int in = 0; in < innerSize; in++) {
            int base = (o * classSize) * innerSize + in;
            double mx = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < classSize; c++) {
                double v = logits[base + c * innerSize];
                if (v > mx) mx = v;
            }
            double sumExp = 0;
            for (int c = 0; c < classSize; c++) {
                double ex = Math.exp(logits[base + c * innerSize] - mx);
                softmax[base + c * innerSize] = ex;
                sumExp += ex;
            }
            double invSum = 1.0 / sumExp;
            int t = labels[o * innerSize + in];
            if (t < 0 || t >= classSize) {
                throw new IllegalArgumentException(
                    "Label " + t + " out of range [0, " + classSize + ") at sample " + (o * innerSize + in));
            }
            for (int c = 0; c < classSize; c++) {
                softmax[base + c * innerSize] *= invSum;
            }
            double pt = softmax[base + t * innerSize];
            totalLoss += -Math.log(Math.max(pt, 1e-30));
        }
    }
    double meanLoss = totalLoss / totalSamples;
    double[] resultData = new double[] { meanLoss };
    int[] resultShape = new int[] { 1 };

    if (!tensor.requiresGrad) return new ConstantDiffTensor(new RereDoubleTensor(resultData, resultShape));

    int fOuter = outerSize, fClassSize = classSize, fInner = innerSize, fTotal = totalSamples;
    double[] fSoftmax = softmax;
    int[] fLabels = labels.clone();
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        double gradScale = self.grad[0] / fTotal;
        int m = fSoftmax.length;
        double[] inGrad = AutodiffBufferPool.acquire(m);
        System.arraycopy(fSoftmax, 0, inGrad, 0, m);
        // Subtract 1 at target positions: grad = (softmax - oneHot) / totalSamples
        for (int o = 0; o < fOuter; o++) {
            for (int in = 0; in < fInner; in++) {
                int t = fLabels[o * fInner + in];
                int idx = (o * fClassSize + t) * fInner + in;
                inGrad[idx] -= 1.0;
            }
        }
        for (int i = 0; i < m; i++) inGrad[i] *= gradScale;
        input.accGradFromPooled(inGrad, m);
    };
    return new RereDiffTensor(resultData, resultShape, List.of(tensor), bw, "softmaxCrossEntropySparse");
}

// ==================== cumsum / cumprod ====================

public static IDiffTensor cumsum(RereDiffTensor tensor, int dim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.cumsum(d));
    int[] s = tensor.shape();
    int outer = 1;
    for (int i = 0; i < d; i++) outer *= s[i];
    int reduce = s[d];
    int inner = 1;
    for (int i = d + 1; i < tensor.rank(); i++) inner *= s[i];

    double[] vals = tensor.value.toDoubleArray();
    double[] result = new double[vals.length];
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            double sum = 0;
            for (int r = 0; r < reduce; r++) {
                int idx = (o * reduce + r) * inner + i;
                sum += vals[idx];
                result[idx] = sum;
            }
        }
    }
    int fOuter = outer, fReduce = reduce, fInner = inner;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        int total = fOuter * fReduce * fInner;
        double[] inGrad = AutodiffBufferPool.acquire(total);
        for (int o = 0; o < fOuter; o++) {
            for (int i = 0; i < fInner; i++) {
                double cum = 0;
                for (int r = fReduce - 1; r >= 0; r--) {
                    int idx = (o * fReduce + r) * fInner + i;
                    cum += self.grad[idx];
                    inGrad[idx] = cum;
                }
            }
        }
        input.accGradFromPooled(inGrad, total);
    };
    return new RereDiffTensor(result, s, List.of(tensor), bw, "cumsum");
}

public static IDiffTensor cumprod(RereDiffTensor tensor, int dim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.cumprod(d));
    int[] s = tensor.shape();
    int outer = 1;
    for (int i = 0; i < d; i++) outer *= s[i];
    int reduce = s[d];
    int inner = 1;
    for (int i = d + 1; i < tensor.rank(); i++) inner *= s[i];

    double[] vals = tensor.value.toDoubleArray();
    double[] result = new double[vals.length];
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            double prod = 1;
            for (int r = 0; r < reduce; r++) {
                int idx = (o * reduce + r) * inner + i;
                prod *= vals[idx];
                result[idx] = prod;
            }
        }
    }
    int fOuter = outer, fReduce = reduce, fInner = inner;
    double[] savedVals = vals;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        int total = fOuter * fReduce * fInner;
        double[] inGrad = AutodiffBufferPool.acquire(total);
        for (int o = 0; o < fOuter; o++) {
            for (int i = 0; i < fInner; i++) {
                double[] cp = new double[fReduce];
                double p = 1;
                for (int r = 0; r < fReduce; r++) {
                    int idx = (o * fReduce + r) * fInner + i;
                    p *= savedVals[idx];
                    cp[r] = p;
                }
                double[] q = new double[fReduce];
                for (int r = 0; r < fReduce; r++) {
                    int idx = (o * fReduce + r) * fInner + i;
                    double xi = savedVals[idx];
                    q[r] = (xi != 0.0) ? self.grad[idx] * cp[r] / xi : 0.0;
                }
                double cum = 0;
                for (int r = fReduce - 1; r >= 0; r--) {
                    cum += q[r];
                    inGrad[(o * fReduce + r) * fInner + i] = cum;
                }
            }
        }
        input.accGradFromPooled(inGrad, total);
    };
    return new RereDiffTensor(result, s, List.of(tensor), bw, "cumprod");
}

// ==================== argmax / argmin ====================

public static IDiffTensor argmax(RereDiffTensor tensor, int dim) {
    IDoubleTensor r = tensor.value.argmax(dim);
    return tensor.toNonDiff(r);
}

public static IDiffTensor argmin(RereDiffTensor tensor, int dim) {
    IDoubleTensor r = tensor.value.argmin(dim);
    return tensor.toNonDiff(r);
}

}
