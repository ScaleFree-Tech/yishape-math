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
public final class DiffTensorAdvanced {
    private DiffTensorAdvanced() { /* utility class */ }

// ==================== Advanced ops ====================

public static IDiffTensor gather(RereDiffTensor tensor, int dim, IDoubleTensor index) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.gather(d, index instanceof IDiffTensor ? ((IDiffTensor) index).detach() : index));
    int[] s = tensor.shape();
    int r = tensor.rank();
    int idxRank = index.rank();
    int[] idxShape = index.shape();
    int trailingRank = r - d - 1;
    int[] resultShape = new int[idxRank + trailingRank];
    System.arraycopy(idxShape, 0, resultShape, 0, idxRank);
    for (int i = 0; i < trailingRank; i++) resultShape[idxRank + i] = s[d + 1 + i];
    int resultTotal = (int) DiffTensorUtil.computeSize(resultShape);

    double[] inData = tensor.value.toDoubleArray();
    double[] resultData = new double[resultTotal];
    int[] gatherIndices = new int[resultTotal];
    int[] flatSourcePos = new int[resultTotal];
    for (int i = 0; i < resultTotal; i++) {
        int[] outIdx = DiffTensorUtil.unlinearizeInt(i, resultShape);
        int[] idxIdx = new int[idxRank];
        System.arraycopy(outIdx, 0, idxIdx, 0, idxRank);
        int gatherIdx = (int) index.get(idxIdx);
        gatherIndices[i] = gatherIdx;
        int[] srcIdx = new int[r];
        for (int j = 0; j < d; j++) srcIdx[j] = outIdx[j];
        srcIdx[d] = gatherIdx;
        for (int j = 0; j < trailingRank; j++) srcIdx[d + 1 + j] = outIdx[idxRank + j];
        flatSourcePos[i] = DiffTensorUtil.flatIndex(srcIdx, s);
        resultData[i] = inData[flatSourcePos[i]];
    }
    int fD = d, fIdxRank = idxRank, fTrailingRank = trailingRank;
    int[] fSrcShape = s, fResultShape = resultShape;
    int[] fGatherIndices = gatherIndices;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        double[] inGrad = new double[(int) DiffTensorUtil.computeSize(fSrcShape)];
        for (int i = 0; i < self.grad.length; i++) {
            int[] outIdx = DiffTensorUtil.unlinearizeInt(i, fResultShape);
            int[] srcIdx = new int[fSrcShape.length];
            for (int j = 0; j < fD; j++) srcIdx[j] = outIdx[j];
            srcIdx[fD] = fGatherIndices[i];
            for (int j = 0; j < fTrailingRank; j++) srcIdx[fD + 1 + j] = outIdx[fIdxRank + j];
            inGrad[DiffTensorUtil.flatIndex(srcIdx, fSrcShape)] += self.grad[i];
        }
        input.accGrad(inGrad);
    };
    // Build inputs for GPU/HPC graph execution.
    // The GPU gather kernel treats the source as [vocab_size, embedding_dim]
    // where embedding_dim = product of source dims after the gather dim.
    // It expects "row indices" (= flat_source_position / embedding_dim) rather
    // than dimension-relative index values (e.g. raw class labels 0-9).
    // We pre-compute row indices so the GPU can do:
    //   output[row*dim+j] = weight[rowIdx[row] * dim + j]
    // The CPU backward uses closure-captured fGatherIndices and is unaffected.
    //
    // Always compute rowIdxTensor: flatSourcePos is already computed for forward,
    // and index may be a ConstantDiffTensor (from tensor.toNonDiff() reshape on
    // tensor.requiresGrad=false indices). The instanceof RereDiffTensor gate was
    // incorrect — it skipped rowIdx creation for non-RereDiffTensor indices,
    // causing "missing inputs" GPU worker errors.
    int trailingProduct = 1;
    for (int i = d + 1; i < r; i++) trailingProduct *= s[i];
    int numOutputRows = resultTotal / trailingProduct;
    double[] rowIdxData = new double[numOutputRows];
    for (int row = 0; row < numOutputRows; row++) {
        rowIdxData[row] = (double) (flatSourcePos[row * trailingProduct] / trailingProduct);
    }
    RereDiffTensor rowIdxTensor = new RereDiffTensor(rowIdxData, numOutputRows);
    rowIdxTensor.setRequiresGrad(false);
    List<RereDiffTensor> gatherInputs = List.of(tensor, rowIdxTensor);
    return new RereDiffTensor(resultData, resultShape, gatherInputs, bw, "gather");
}

public static IDiffTensor indexSelect(RereDiffTensor tensor, int dim, IDoubleTensor index) {
    return gather(tensor, dim, index);
}

public static IDiffTensor argsort(RereDiffTensor tensor, int dim, boolean descending) {
    // Non-differentiable
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    int[] s = tensor.shape();
    int r = tensor.rank();
    int dimSize = s[d];
    int outerTotal = 1;
    for (int i = 0; i < d; i++) outerTotal *= s[i];
    int innerTotal = 1;
    for (int i = d + 1; i < r; i++) innerTotal *= s[i];

    double[] inData = tensor.value.toDoubleArray();
    double[] outData = new double[inData.length];
    for (int outer = 0; outer < outerTotal; outer++) {
        for (int inner = 0; inner < innerTotal; inner++) {
            double[] sliceVals = new double[dimSize];
            Integer[] indices = new Integer[dimSize];
            for (int i = 0; i < dimSize; i++) {
                sliceVals[i] = inData[(outer * dimSize + i) * innerTotal + inner];
                indices[i] = i;
            }
            java.util.Arrays.sort(indices, (a, b) -> {
                int cmp = Double.compare(sliceVals[a], sliceVals[b]);
                return descending ? -cmp : cmp;
            });
            for (int i = 0; i < dimSize; i++) {
                outData[(outer * dimSize + i) * innerTotal + inner] = indices[i];
            }
        }
    }
    return tensor.toNonDiff(new RereDoubleTensor(outData, s));
}

public static IDiffTensor scatter(RereDiffTensor tensor, int dim, IDoubleTensor index, IDoubleTensor source) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    if (!tensor.requiresGrad) {
        IDoubleTensor detSrc = (source instanceof IDiffTensor ds) ? ds.detach() : source;
        IDoubleTensor detIdx = (index instanceof IDiffTensor di) ? di.detach() : index;
        return tensor.toNonDiff(tensor.value.scatter(d, detIdx, detSrc));
    }
    int[] resultShape = tensor.shape();
    int r = tensor.rank();
    double[] resultData = tensor.value.toDoubleArray();
    double[] srcData = source instanceof RereDiffTensor rdt ? rdt.value.toDoubleArray() : source.toDoubleArray();
    int[] srcShape = source.shape();
    int idxRank = index.rank();
    int[] idxShape = index.shape();
    int idxTotal = (int) DiffTensorUtil.computeSize(idxShape);

    int[] scatterSrcFlat = new int[idxTotal];
    int[] scatterTgtFlat = new int[idxTotal];
    for (int i = 0; i < idxTotal; i++) {
        int[] idx = DiffTensorUtil.unlinearizeInt(i, idxShape);
        int scatterIdx = (int) index.get(idx);
        int[] tgtIdx = new int[r];
        for (int j = 0; j < r; j++) tgtIdx[j] = j == d ? scatterIdx : idx[j];
        int tgtFlat = DiffTensorUtil.flatIndex(tgtIdx, resultShape);
        int srcFlat = DiffTensorUtil.flatIndex(idx, srcShape);
        resultData[tgtFlat] = srcData[srcFlat];
        scatterSrcFlat[i] = srcFlat;
        scatterTgtFlat[i] = tgtFlat;
    }

    List<RereDiffTensor> inputs = new ArrayList<>();
    inputs.add(tensor);
    RereDiffTensor srcNode = (source instanceof RereDiffTensor rdt2 && rdt2.requiresGrad) ? rdt2 : null;
    if (srcNode != null) inputs.add(srcNode);

    int fIdxTotal = idxTotal;
    int[] fScatterSrcFlat = scatterSrcFlat;
    int[] fScatterTgtFlat = scatterTgtFlat;
    int[] fSrcShape = srcShape;

    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inpSelf = self.inputs.get(0);
        // Self gradient: pass through except at overwritten positions
        double[] dxSelf = self.grad.clone();
        for (int i = 0; i < fIdxTotal; i++) dxSelf[fScatterTgtFlat[i]] = 0.0;
        inpSelf.accGrad(dxSelf);
        // Source gradient
        if (srcNode != null) {
            RereDiffTensor inpSrc = self.inputs.get(1);
            double[] dxSrc = new double[(int) DiffTensorUtil.computeSize(fSrcShape)];
            for (int i = 0; i < fIdxTotal; i++) dxSrc[fScatterSrcFlat[i]] += self.grad[fScatterTgtFlat[i]];
            inpSrc.accGrad(dxSrc);
        }
    };
    return new RereDiffTensor(resultData, resultShape, inputs, bw, "scatter");
}

public static IDiffTensor scatterAdd(RereDiffTensor tensor, int dim, IDoubleTensor index, IDoubleTensor source) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    if (!tensor.requiresGrad) {
        IDoubleTensor detSrc = (source instanceof IDiffTensor ds) ? ds.detach() : source;
        IDoubleTensor detIdx = (index instanceof IDiffTensor di) ? di.detach() : index;
        return tensor.toNonDiff(tensor.value.scatterAdd(d, detIdx, detSrc));
    }
    int[] resultShape = tensor.shape();
    int r = tensor.rank();
    double[] resultData = tensor.value.toDoubleArray();
    double[] srcData = source instanceof RereDiffTensor rdt ? rdt.value.toDoubleArray() : source.toDoubleArray();
    int[] srcShape = source.shape();
    int idxRank = index.rank();
    int[] idxShape = index.shape();
    int idxTotal = (int) DiffTensorUtil.computeSize(idxShape);

    int[] scatterSrcFlat = new int[idxTotal];
    int[] scatterTgtFlat = new int[idxTotal];
    for (int i = 0; i < idxTotal; i++) {
        int[] idx = DiffTensorUtil.unlinearizeInt(i, idxShape);
        int scatterIdx = (int) index.get(idx);
        int[] tgtIdx = new int[r];
        for (int j = 0; j < r; j++) tgtIdx[j] = j == d ? scatterIdx : idx[j];
        int tgtFlat = DiffTensorUtil.flatIndex(tgtIdx, resultShape);
        int srcFlat = DiffTensorUtil.flatIndex(idx, srcShape);
        resultData[tgtFlat] += srcData[srcFlat];
        scatterSrcFlat[i] = srcFlat;
        scatterTgtFlat[i] = tgtFlat;
    }

    List<RereDiffTensor> inputs = new ArrayList<>();
    inputs.add(tensor);
    RereDiffTensor srcNode = (source instanceof RereDiffTensor rdt2 && rdt2.requiresGrad) ? rdt2 : null;
    if (srcNode != null) inputs.add(srcNode);

    int fIdxTotal = idxTotal;
    int[] fScatterSrcFlat = scatterSrcFlat;
    int[] fScatterTgtFlat = scatterTgtFlat;
    int[] fSrcShape = srcShape;

    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inpSelf = self.inputs.get(0);
        inpSelf.accGrad(self.grad.clone());
        if (srcNode != null) {
            RereDiffTensor inpSrc = self.inputs.get(1);
            double[] dxSrc = new double[(int) DiffTensorUtil.computeSize(fSrcShape)];
            for (int i = 0; i < fIdxTotal; i++) dxSrc[fScatterSrcFlat[i]] += self.grad[fScatterTgtFlat[i]];
            inpSrc.accGrad(dxSrc);
        }
    };
    return new RereDiffTensor(resultData, resultShape, inputs, bw, "scatterAdd");
}

public static IDiffTensor where(RereDiffTensor tensor, IDoubleTensor condition, IDoubleTensor other) {
    if (!tensor.requiresGrad) {
        IDoubleTensor detCond = (condition instanceof IDiffTensor dc) ? dc.detach() : condition;
        IDoubleTensor detOther = (other instanceof IDiffTensor dt) ? dt.detach() : other;
        return tensor.toNonDiff(tensor.value.where(detCond, detOther));
    }
    IDoubleTensor detCond = (condition instanceof IDiffTensor dc) ? dc.detach() : condition;
    IDoubleTensor detOther = (other instanceof IDiffTensor dt) ? dt.detach() : other;
    int[] sSelf = tensor.shape(), sOther = detOther.shape(), sCond = detCond.shape();
    int[] resultShape = TensorShape.broadcastShape(sSelf, TensorShape.broadcastShape(sOther, sCond));
    long resultTotal = 1;
    for (int d : resultShape) resultTotal *= d;
    int n = (int) resultTotal;

    double[] aData = tensor.value.toDoubleArray();
    double[] bData = detOther.toDoubleArray();
    double[] condData = detCond.toDoubleArray();
    double[] resultData = new double[n];
    boolean[] condMask = new boolean[n];
    for (int i = 0; i < n; i++) {
        int[] idx = DiffTensorUtil.unlinearizeInt(i, resultShape);
        double condVal = DiffTensorUtil.broadcastGetFlat(idx, condData, sCond, resultShape);
        condMask[i] = condVal > 0.5;
        resultData[i] = condMask[i] ? DiffTensorUtil.broadcastGetFlat(idx, aData, sSelf, resultShape)
                                    : DiffTensorUtil.broadcastGetFlat(idx, bData, sOther, resultShape);
    }

    List<RereDiffTensor> inputs = new ArrayList<>();
    inputs.add(tensor);
    RereDiffTensor otherNode = (other instanceof RereDiffTensor rt && rt.requiresGrad) ? rt : null;
    if (otherNode != null) inputs.add(otherNode);

    int[] fResultShape = resultShape;
    int[] fSSelf = sSelf, fSOther = sOther;
    boolean[] fCondMask = condMask;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inpSelf = self.inputs.get(0);
        double[] dxSelf = new double[(int) DiffTensorUtil.computeSize(fSSelf)];
        double[] dxOther = null;
        RereDiffTensor inpOther = null;
        if (otherNode != null) {
            inpOther = self.inputs.get(1);
            dxOther = new double[(int) DiffTensorUtil.computeSize(fSOther)];
        }
        for (int i = 0; i < n; i++) {
            int[] idx = DiffTensorUtil.unlinearizeInt(i, fResultShape);
            if (fCondMask[i]) {
                dxSelf[DiffTensorUtil.flatIndexFromBroadcast(idx, fSSelf, fResultShape)] += self.grad[i];
            } else if (dxOther != null) {
                dxOther[DiffTensorUtil.flatIndexFromBroadcast(idx, fSOther, fResultShape)] += self.grad[i];
            }
        }
        inpSelf.accGrad(dxSelf);
        if (inpOther != null) inpOther.accGrad(dxOther);
    };
    return new RereDiffTensor(resultData, resultShape, inputs, bw, "where");
}

public static IDiffTensor topk(RereDiffTensor tensor, int k, int dim, boolean largest) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.topk(k, d, largest));
    int[] s = tensor.shape();
    int n = s[d];
    int outer = 1;
    for (int i = 0; i < d; i++) outer *= s[i];
    int inner = 1;
    for (int i = d + 1; i < tensor.rank(); i++) inner *= s[i];
    int[] resultShape = s.clone();
    resultShape[d] = k;
    int resultSize = outer * k * inner;
    double[] resultData = new double[resultSize];
    int[] argIdx = new int[resultSize];

    double[] inData = tensor.value.toDoubleArray();
    for (int o = 0; o < outer; o++) {
        for (int ii = 0; ii < inner; ii++) {
            double[] vals = new double[n];
            for (int r = 0; r < n; r++) vals[r] = inData[(o * n + r) * inner + ii];
            Integer[] idxs = new Integer[n];
            for (int r = 0; r < n; r++) idxs[r] = r;
            final boolean l = largest;
            Arrays.sort(idxs, (a, b) -> l ? Double.compare(vals[b], vals[a]) : Double.compare(vals[a], vals[b]));
            for (int r = 0; r < k; r++) {
                int flatIdx = (o * k + r) * inner + ii;
                resultData[flatIdx] = vals[idxs[r]];
                argIdx[flatIdx] = idxs[r];
            }
        }
    }
    int fOuter = outer, fReduce = n, fInner = inner, fK = k;
    int[] fArgIdx = argIdx;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        double[] inGrad = new double[fOuter * fReduce * fInner];
        for (int o = 0; o < fOuter; o++) {
            for (int r = 0; r < fK; r++) {
                for (int ii = 0; ii < fInner; ii++) {
                    int outIdx = (o * fK + r) * fInner + ii;
                    int origR = fArgIdx[outIdx];
                    inGrad[(o * fReduce + origR) * fInner + ii] += self.grad[outIdx];
                }
            }
        }
        input.accGrad(inGrad);
    };
    return new RereDiffTensor(resultData, resultShape, List.of(tensor), bw, "topk");
}

public static IDiffTensor pad(RereDiffTensor tensor, int[][] padding, String mode, double padValue) {
    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.pad(padding, mode, padValue));
    int[] s = tensor.shape();
    int r = tensor.rank();
    int[] resultShape = new int[r];
    for (int i = 0; i < r; i++) resultShape[i] = s[i] + padding[i][0] + padding[i][1];
    long total = 1;
    for (int d : resultShape) total *= d;
    int n = (int) total;
    double[] resultData = new double[n];
    Arrays.fill(resultData, padValue);
    double[] inData = tensor.value.toDoubleArray();
    for (int i = 0; i < inData.length; i++) {
        int[] srcIdx = DiffTensorUtil.unlinearizeInt(i, s);
        int[] tgtIdx = new int[r];
        for (int j = 0; j < r; j++) tgtIdx[j] = srcIdx[j] + padding[j][0];
        resultData[DiffTensorUtil.flatIndex(tgtIdx, resultShape)] = inData[i];
    }
    int[] fResultShape = resultShape;
    int[] fOrigShape = s;
    int[][] fPadding = padding;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        int origTotal = (int) DiffTensorUtil.computeSize(fOrigShape);
        double[] inGrad = new double[origTotal];
        for (int i = 0; i < origTotal; i++) {
            int[] srcIdx = DiffTensorUtil.unlinearizeInt(i, fOrigShape);
            int[] paddedIdx = new int[r];
            for (int j = 0; j < r; j++) paddedIdx[j] = srcIdx[j] + fPadding[j][0];
            inGrad[i] = self.grad[DiffTensorUtil.flatIndex(paddedIdx, fResultShape)];
        }
        input.accGrad(inGrad);
    };
    return new RereDiffTensor(resultData, resultShape, List.of(tensor), bw, "pad");
}

public static IDiffTensor tril(RereDiffTensor tensor, int diagonal) {
    int r = tensor.rank();
    if (r < 2) return tensor; // scalar/vector: no-op
    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.tril(diagonal));

    int[] s = tensor.shape();
    int M = s[r - 2];
    int N = s[r - 1];
    double[] resultData = tensor.value.toDoubleArray().clone();
    int batchStride = M * N;
    int batchCount = resultData.length / batchStride;

    for (int b = 0; b < batchCount; b++) {
        int base = b * batchStride;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                if (j > i + diagonal) {
                    resultData[base + i * N + j] = 0.0;
                }
            }
        }
    }

    int fDiagonal = diagonal;
    int[] fShape = s;
    int fTotalElements = (int) DiffTensorUtil.computeSize(s);
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        int bStride = fShape[fShape.length - 2] * fShape[fShape.length - 1];
        int bCount = fTotalElements / bStride;
        // C6: clone grad to avoid mutating self.grad in-place (violates immutability contract)
        double[] maskedGrad = self.grad.clone();
        for (int b = 0; b < bCount; b++) {
            int base = b * bStride;
            for (int i = 0; i < fShape[fShape.length - 2]; i++) {
                for (int j = 0; j < fShape[fShape.length - 1]; j++) {
                    if (j > i + fDiagonal) {
                        maskedGrad[base + i * fShape[fShape.length - 1] + j] = 0.0;
                    }
                }
            }
        }
        input.accGrad(maskedGrad);
    };
    return new RereDiffTensor(resultData, s, List.of(tensor), bw, "tril");
}

public static IDiffTensor triu(RereDiffTensor tensor, int diagonal) {
    int r = tensor.rank();
    if (r < 2) return tensor; // scalar/vector: no-op
    if (!tensor.requiresGrad) {
        // Manual triu: clone and zero lower triangle
        double[] d = tensor.value.toDoubleArray().clone();
        int M = tensor.value.dim(r - 2);
        int N = tensor.value.dim(r - 1);
        int batchStride = M * N;
        int batchCount = d.length / batchStride;
        for (int b = 0; b < batchCount; b++) {
            int base = b * batchStride;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    if (j < i + diagonal) {
                        d[base + i * N + j] = 0.0;
                    }
                }
            }
        }
        return tensor.toNonDiff(new RereDoubleTensor(d, tensor.shape()));
    }

    int[] s = tensor.shape();
    int M = s[r - 2];
    int N = s[r - 1];
    double[] resultData = tensor.value.toDoubleArray().clone();
    int batchStride = M * N;
    int batchCount = resultData.length / batchStride;

    for (int b = 0; b < batchCount; b++) {
        int base = b * batchStride;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                if (j < i + diagonal) {
                    resultData[base + i * N + j] = 0.0;
                }
            }
        }
    }

    int fDiagonal = diagonal;
    int[] fShape = s;
    int fTotalElements = (int) DiffTensorUtil.computeSize(s);
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        int bStride = fShape[fShape.length - 2] * fShape[fShape.length - 1];
        int bCount = fTotalElements / bStride;
        // C6: clone to avoid mutating self.grad in-place
        // SISD: triangular masking is a structural loop over batch/row/col with a positional
        // condition (j < i + diagonal), not element-wise arithmetic (§7a structural-loop exception)
        double[] gradCopy = self.grad.clone();
        for (int b = 0; b < bCount; b++) {
            int base = b * bStride;
            for (int i = 0; i < fShape[fShape.length - 2]; i++) {
                for (int j = 0; j < fShape[fShape.length - 1]; j++) {
                    if (j < i + fDiagonal) {
                        gradCopy[base + i * fShape[fShape.length - 1] + j] = 0.0;
                    }
                }
            }
        }
        input.accGrad(gradCopy);
    };
    return new RereDiffTensor(resultData, s, List.of(tensor), bw, "triu");
}

public static IDiffTensor diag(RereDiffTensor tensor) {
    int r = tensor.rank();
    if (r < 2) return tensor;
    int[] s = tensor.shape();
    int M = s[r - 2];
    int N = s[r - 1];
    if (M != N) {
        throw new IllegalArgumentException("diag() requires a square matrix (last two dims equal), got " + M + "x" + N);
    }
    // Batch support: leading dims treated as batch
    int batchDim = 1;
    for (int i = 0; i < r - 2; i++) batchDim *= s[i];
    int n = (int) Math.min(M, N);

    double[] vals = tensor.value.toDoubleArray();
    int totalDiagSize = batchDim * n;
    double[] resultData = new double[totalDiagSize];

    for (int b = 0; b < batchDim; b++) {
        int base = b * M * N;
        for (int i = 0; i < n; i++) {
            resultData[b * n + i] = vals[base + i * N + i];
        }
    }

    int[] resultShape;
    if (r == 2) {
        resultShape = new int[]{n};
    } else {
        resultShape = new int[r - 1];
        for (int i = 0; i < r - 2; i++) resultShape[i] = s[i];
        resultShape[r - 2] = n;
    }

    if (!tensor.requiresGrad) return tensor.toNonDiff(new RereDoubleTensor(resultData, resultShape));

    int fM = M, fN = N, fBatchDim = batchDim;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        int total = fBatchDim * fM * fN;
        double[] inGrad = AutodiffBufferPool.acquire(total);
        for (int b = 0; b < fBatchDim; b++) {
            int base = b * fM * fN;
            for (int i = 0; i < n; i++) {
                inGrad[base + i * fN + i] = self.grad[b * n + i];
            }
        }
        input.accGradFromPooled(inGrad, total);
    };
    return new RereDiffTensor(resultData, resultShape, List.of(tensor), bw, "diag");
}

public static IDiffTensor diagonal(RereDiffTensor tensor, int offset, int dim1, int dim2) {
    int r = tensor.rank();
    int fDim1 = (dim1 < 0 ? dim1 + r : dim1);
    int fDim2 = (dim2 < 0 ? dim2 + r : dim2);
    int[] s = tensor.shape();
    int size = (int) Math.min(s[fDim1], s[fDim2]);
    int effectiveSize = offset >= 0
        ? Math.max(0, Math.min(s[fDim1] - offset, s[fDim2]))
        : Math.max(0, Math.min(s[fDim1], s[fDim2] + offset));
    if (effectiveSize == 0) {
        throw new IllegalArgumentException("diagonal: offset " + offset + " yields empty diagonal for shape " + java.util.Arrays.toString(s));
    }
    int minDim = Math.min(fDim1, fDim2);
    int maxDim = Math.max(fDim1, fDim2);

    // Flatten leading and trailing dims
    int outer = 1;
    for (int i = 0; i < minDim; i++) outer *= s[i];
    int inner = 1;
    for (int i = maxDim + 1; i < r; i++) inner *= s[i];

    double[] vals = tensor.value.toDoubleArray();
    double[] resultData = new double[outer * effectiveSize * inner];

    // Compute diagonal extraction
    for (int o = 0; o < outer; o++) {
        for (int k = 0; k < effectiveSize; k++) {
            for (int i = 0; i < inner; i++) {
                int[] idx = new int[r];
                int remaining = o;
                for (int j = minDim - 1; j >= 0; j--) {
                    idx[j] = remaining % s[j];
                    remaining /= s[j];
                }
                // Positive offset shifts toward dim2 (away from dim1).
                // PyTorch/Numpy convention: diagonal(x, offset=1) gives super-diagonal.
                idx[fDim1] = offset >= 0 ? k : k - offset;
                idx[fDim2] = offset >= 0 ? k + offset : k;
                remaining = i;
                for (int j = r - 1; j > maxDim; j--) {
                    idx[j] = remaining % s[j];
                    remaining /= s[j];
                }
                resultData[(o * effectiveSize + k) * inner + i] = vals[DiffTensorUtil.flatIndex(idx, s)];
            }
        }
    }

    int[] resultShape;
    if (r == 2) {
        resultShape = new int[]{effectiveSize};
    } else {
        resultShape = new int[r - 1];
        int pos = 0;
        for (int j = 0; j < fDim1; j++) resultShape[pos++] = s[j];
        for (int j = fDim1 + 1; j < fDim2; j++) resultShape[pos++] = s[j];
        for (int j = fDim2 + 1; j < r; j++) resultShape[pos++] = s[j];
        resultShape[minDim] = effectiveSize;
    }

    if (!tensor.requiresGrad) return tensor.toNonDiff(new RereDoubleTensor(resultData, resultShape));

    int fOffset = offset, fEffSize = effectiveSize, fOuter = outer, fInner = inner;
    int[] fShape = s;
    int ffDim1 = fDim1, ffDim2 = fDim2;
    int fMinDim = minDim, fMaxDim = maxDim;
    int fR = r;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        int total = (int) DiffTensorUtil.computeSize(fShape);
        double[] inGrad = AutodiffBufferPool.acquire(total);
        for (int o = 0; o < fOuter; o++) {
            for (int k = 0; k < fEffSize; k++) {
                for (int i = 0; i < fInner; i++) {
                    int[] idx = new int[fR];
                    int remaining = o;
                    for (int j = fMinDim - 1; j >= 0; j--) {
                        idx[j] = remaining % fShape[j];
                        remaining /= fShape[j];
                    }
                    idx[ffDim1] = fOffset >= 0 ? k : k - fOffset;
                    idx[ffDim2] = fOffset >= 0 ? k + fOffset : k;
                    remaining = i;
                    for (int j = fR - 1; j > fMaxDim; j--) {
                        idx[j] = remaining % fShape[j];
                        remaining /= fShape[j];
                    }
                    inGrad[DiffTensorUtil.flatIndex(idx, fShape)] = self.grad[(o * fEffSize + k) * fInner + i];
                }
            }
        }
        input.accGradFromPooled(inGrad, total);
    };
    return new RereDiffTensor(resultData, resultShape, List.of(tensor), bw, "diagonal");
}

public static IDiffTensor trace(RereDiffTensor tensor) {
    return diag(tensor).sum();
}

public static IDiffTensor unfold(RereDiffTensor tensor, int dim, int size, int stride, int dilation) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.unfold(d, size, stride, dilation));
    int[] s = tensor.shape();
    int dimSize = s[d];
    int numPatches = (dimSize - dilation * (size - 1) - 1) / stride + 1;
    int outerElems = 1;
    for (int i = 0; i < d; i++) outerElems *= s[i];
    int innerElems = 1;
    for (int i = d + 1; i < tensor.rank(); i++) innerElems *= s[i];

    int[] resultShape = new int[tensor.rank() + 1];
    for (int i = 0; i < d; i++) resultShape[i] = s[i];
    resultShape[d] = numPatches;
    for (int i = d + 1; i < tensor.rank(); i++) resultShape[i] = s[i];
    resultShape[tensor.rank()] = size;

    double[] vals = tensor.value.toDoubleArray();
    double[] result = new double[outerElems * numPatches * innerElems * size];
    for (int o = 0; o < outerElems; o++) {
        for (int p = 0; p < numPatches; p++) {
            for (int ii = 0; ii < innerElems; ii++) {
                for (int k = 0; k < size; k++) {
                    int srcIdx = (o * dimSize + (p * stride + k * dilation)) * innerElems + ii;
                    int dstIdx = ((o * numPatches + p) * innerElems + ii) * size + k;
                    result[dstIdx] = vals[srcIdx];
                }
            }
        }
    }
    int fNumPatches = numPatches, fSize = size, fStride = stride, fDilation = dilation;
    int fOuterElems = outerElems, fDimSize = dimSize, fInnerElems = innerElems;
    int total = (int) tensor.value.totalSize();
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        double[] inGrad = new double[total];
        for (int o = 0; o < fOuterElems; o++) {
            for (int p = 0; p < fNumPatches; p++) {
                for (int ii = 0; ii < fInnerElems; ii++) {
                    for (int k = 0; k < fSize; k++) {
                        int dstIdx = ((o * fNumPatches + p) * fInnerElems + ii) * fSize + k;
                        int srcIdx = (o * fDimSize + (p * fStride + k * fDilation)) * fInnerElems + ii;
                        inGrad[srcIdx] += self.grad[dstIdx];
                    }
                }
            }
        }
        input.accGrad(inGrad);
    };
    return new RereDiffTensor(result, resultShape, List.of(tensor), bw, "unfold");
}

public static IDiffTensor nonzero(RereDiffTensor tensor) {
    IDoubleTensor r = tensor.value.nonzero();
    return tensor.toNonDiff(r);
}

public static IDiffTensor maskedSelect(RereDiffTensor tensor, IDoubleTensor mask) {
    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.maskedSelect(mask));
    int[] s = tensor.shape();
    int total = (int) tensor.value.totalSize();
    double[] inData = tensor.value.toDoubleArray();
    ArrayList<Double> selected = new ArrayList<>();
    ArrayList<Integer> selectedIndices = new ArrayList<>();
    for (int i = 0; i < total; i++) {
        int[] idx = DiffTensorUtil.unlinearizeInt(i, s);
        if (mask.get(idx) > 0.5) {
            selected.add(inData[i]);
            selectedIndices.add(i);
        }
    }
    int outLen = selected.size();
    double[] resultData = new double[outLen];
    for (int i = 0; i < outLen; i++) resultData[i] = selected.get(i);
    int[] fSelectedIdx = new int[outLen];
    for (int i = 0; i < outLen; i++) fSelectedIdx[i] = selectedIndices.get(i);
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        double[] inGrad = new double[total];
        for (int i = 0; i < self.grad.length; i++) inGrad[fSelectedIdx[i]] += self.grad[i];
        input.accGrad(inGrad);
    };
    return new RereDiffTensor(resultData, new int[]{outLen}, List.of(tensor), bw, "maskedSelect");
}

public static IDiffTensor maskedFill(RereDiffTensor tensor, IDoubleTensor mask, double fillValue) {
    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.maskedFill(mask, fillValue));
    int[] s = tensor.shape();
    int total = (int) tensor.value.totalSize();
    double[] inData = tensor.value.toDoubleArray();
    boolean[] maskArr = new boolean[total];
    for (int i = 0; i < total; i++) {
        int[] idx = DiffTensorUtil.unlinearizeInt(i, s);
        maskArr[i] = mask.get(idx) > 0.5;
    }
    double[] resultData = new double[total];
    for (int i = 0; i < total; i++) resultData[i] = maskArr[i] ? fillValue : inData[i];
    boolean[] fMask = maskArr;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        double[] inGrad = new double[self.grad.length];
        for (int i = 0; i < self.grad.length; i++) inGrad[i] = fMask[i] ? 0.0 : self.grad[i];
        input.accGrad(inGrad);
    };
    return new RereDiffTensor(resultData, s, List.of(tensor), bw, "maskedFill");
}

public static IDiffTensor cat(RereDiffTensor tensor, int dim, IDoubleTensor... others) {
    if (dim < 0) dim += tensor.rank();
    int d = dim;
    if (!tensor.requiresGrad) {
        IDoubleTensor[] detached = new IDoubleTensor[others.length];
        for (int i = 0; i < others.length; i++) detached[i] = others[i] instanceof IDiffTensor dt ? dt.detach() : others[i];
        return tensor.toNonDiff(tensor.value.cat(d, detached));
    }
    int[] resultShape = tensor.shape().clone();
    int[] sizes = new int[1 + others.length];
    sizes[0] = tensor.dim(d);
    for (int i = 0; i < others.length; i++) { sizes[i + 1] = others[i].dim(d); resultShape[d] += sizes[i + 1]; }

    long total = 1;
    for (int rs : resultShape) total *= rs;
    int totalSize = (int) total;
    double[] resultData = new double[totalSize];
    int[] shapeA = tensor.shape();
    double[] aData = tensor.value.toDoubleArray();
    int r = tensor.rank();
    int innerSize = 1;
    for (int j = d + 1; j < r; j++) innerSize *= shapeA[j];
    int aBlockSize = shapeA[d] * innerSize;
    int outerGroups = aData.length / aBlockSize;
    int resultBlockSize = resultShape[d] * innerSize;
    // Copy self data in bulk contiguous blocks
    for (int g = 0; g < outerGroups; g++) {
        System.arraycopy(aData, g * aBlockSize, resultData, g * resultBlockSize, aBlockSize);
    }
    // Copy other tensors in bulk
    // ✓ offsetInBlock = dimOffset * innerSize (flat units, used as flat array offset in arraycopy)
    // This IS correct: dimension offset × innerSize = flat element offset.
    // The backward code MUST do the same (see ⚠️ below).
    int offsetInBlock = shapeA[d] * innerSize;
    for (int ti = 0; ti < others.length; ti++) {
        IDoubleTensor t = others[ti];
        double[] tData = t instanceof RereDiffTensor rdt3 ? rdt3.value.toDoubleArray() : t.toDoubleArray();
        int tBlockSize = t.dim(d) * innerSize;
        for (int g = 0; g < outerGroups; g++) {
            System.arraycopy(tData, g * tBlockSize, resultData, g * resultBlockSize + offsetInBlock, tBlockSize);
        }
        offsetInBlock += tBlockSize;
    }

    // Build gradient inputs
    List<RereDiffTensor> allInputs = new java.util.ArrayList<>();
    allInputs.add(tensor);
    List<int[]> allShapes = new java.util.ArrayList<>();
    allShapes.add(tensor.shape());
    for (IDoubleTensor other : others) {
        if (other instanceof RereDiffTensor rdt4 && rdt4.requiresGrad) {
            allInputs.add(rdt4);
            allShapes.add(other.shape());
        }
    }

    int[] fResultShape = resultShape;
    // ⚠️ cumOffset is in dimension units (count along concat dim), NOT flat array offset.
    // When used in System.arraycopy below, MUST multiply by fInnerSize.
    // For cat([1,H], dim=0) + [1,H]: cumOffset[1]=1 means "1 row offset" = 1*H flat elements.
    // Without *fInnerSize, non-first inputs read shifted gradients (dc = dh[1:] instead of dc).
    // See git history for the bug this comment prevents.
    int[] cumOffset = new int[allInputs.size()];
    cumOffset[0] = 0;
    int off = sizes[0];
    int inpIdx = 1;
    for (int i = 0; i < others.length; i++) {
        if (others[i] instanceof RereDiffTensor rdt4 && rdt4.requiresGrad) {
            cumOffset[inpIdx] = off;
            inpIdx++;
        }
        off += sizes[i + 1];
    }

    int fDim = d;
    int[][] fAllShapes = allShapes.toArray(new int[0][]);
    int fInnerSize = innerSize;
    int fResultBlockSize = resultBlockSize;
    int fOuterGroups = outerGroups;
    Consumer<RereDiffTensor> bw = self -> {
        for (int ai = 0; ai < allInputs.size(); ai++) {
            RereDiffTensor inp = allInputs.get(ai);
            int[] tShape = fAllShapes[ai];
            int tFlat = (int) DiffTensorUtil.computeSize(tShape);
            // ⚠️ startOffset is in dimension units (e.g., row count). cumOffset[1]=1 means "row 1".
            // flatArrayPos = startOffset * fInnerSize — do NOT use startOffset alone.
            int startOffset = cumOffset[ai];
            double[] subGrad = new double[tFlat];
            int tBlockSize = tShape[fDim] * fInnerSize;
            for (int g = 0; g < fOuterGroups; g++) {
                // ⚠️ CRITICAL: startOffset is in dim-units, fResultBlockSize/tBlockSize are flat-units.
                // Both must use the same unit: multiply startOffset by fInnerSize to convert to flat units.
                // Bug history: missing *fInnerSize caused dc to read dh[1:] (offset=1 vs offset=1*H).
                System.arraycopy(self.grad, g * fResultBlockSize + startOffset * fInnerSize,
                        subGrad, g * tBlockSize, tBlockSize);
            }
            inp.accGrad(subGrad);
        }
    };
    RereDiffTensor result = new RereDiffTensor(resultData, resultShape, allInputs, bw, "cat");
    // Encode concatenation dimension for GPU/HPC backends.
    // GPU cat forward uses scalar to perform block-interleaved concatenation;
    // dim=0 is equivalent to flat concatenation (the legacy GPU path).
    result.setScalarParam((double) d);
    return result;
}

public static IDiffTensor stack(RereDiffTensor tensor, int dim, IDoubleTensor... others) {
    if (!tensor.requiresGrad) {
        IDoubleTensor[] detached = new IDoubleTensor[others.length];
        for (int i = 0; i < others.length; i++) detached[i] = others[i] instanceof IDiffTensor dt ? dt.detach() : others[i];
        return tensor.toNonDiff(tensor.value.stack(dim, detached));
    }
    int d = (dim < 0 ? tensor.rank() + 1 + dim : dim);
    IDiffTensor[] all = new IDiffTensor[1 + others.length];
    all[0] = tensor.unsqueeze(d);
    for (int i = 0; i < others.length; i++) {
        if (others[i] instanceof IDiffTensor dt && !dt.requiresGrad()) {
            all[i + 1] = IDiffTensor.constantTensor(others[i].toDoubleArray(), others[i].shape()).unsqueeze(d);
        } else if (others[i] instanceof IDiffTensor dt) {
            all[i + 1] = dt.unsqueeze(d);
        } else {
            all[i + 1] = IDiffTensor.constantTensor(others[i].toDoubleArray(), others[i].shape()).unsqueeze(d);
        }
    }
    return all[0].cat(d, java.util.Arrays.copyOfRange(all, 1, all.length));
}

public static IDiffTensor[] split(RereDiffTensor tensor, int splitSize, int dim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    int dimSize = tensor.dim(d);
    if (splitSize <= 0) throw new IllegalArgumentException("splitSize must be positive, got " + splitSize);
    java.util.ArrayList<Integer> sizes = new java.util.ArrayList<>();
    int remaining = dimSize;
    while (remaining > 0) {
        sizes.add(Math.min(splitSize, remaining));
        remaining -= splitSize;
    }
    int[] sizeArray = new int[sizes.size()];
    for (int i = 0; i < sizes.size(); i++) sizeArray[i] = sizes.get(i);
    return split(tensor, sizeArray, d);
}

public static IDiffTensor[] split(RereDiffTensor tensor, int[] splitSizes, int dim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    int dimSize = tensor.dim(d);
    int sumSizes = 0;
    for (int sz : splitSizes) sumSizes += sz;
    if (sumSizes != dimSize) {
        throw new IllegalArgumentException("split sizes sum to " + sumSizes + " but dim " + dim + " has size " + dimSize);
    }
    int n = splitSizes.length;
    IDiffTensor[] result = new IDiffTensor[n];
    int offset = 0;
    for (int i = 0; i < n; i++) {
        result[i] = DiffTensorView.narrow(tensor, d, offset, splitSizes[i]);
        offset += splitSizes[i];
    }
    return result;
}

public static IDiffTensor[] chunk(RereDiffTensor tensor, int chunks, int dim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    int dimSize = tensor.dim(d);
    int chunkSize = (dimSize + chunks - 1) / chunks; // ceil division
    // ensure at most chunks pieces
    java.util.ArrayList<Integer> sizes = new java.util.ArrayList<>();
    int remaining = dimSize;
    for (int i = 0; i < chunks && remaining > 0; i++) {
        sizes.add(Math.min(chunkSize, remaining));
        remaining -= chunkSize;
    }
    int[] sizeArray = new int[sizes.size()];
    for (int i = 0; i < sizes.size(); i++) sizeArray[i] = sizes.get(i);
    return split(tensor, sizeArray, d);
}

public static IDiffTensor[] unbind(RereDiffTensor tensor, int dim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    int dimSize = tensor.dim(d);
    IDiffTensor[] result = new IDiffTensor[dimSize];
    for (int i = 0; i < dimSize; i++) {
        // slice(d, i, i+1) removes the dim, so squeeze back
        IDiffTensor sliced = DiffTensorView.narrow(tensor, d, i, 1);
        result[i] = sliced.squeeze(d);
    }
    return result;
}

public static List<IDoubleTensor> unstack(RereDiffTensor tensor, int dim) {
    return tensor.value.unstack(dim);
}

public static IDiffTensor normalize(RereDiffTensor tensor, double p, int dim) {
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.normalize(p, d));
    int[] s = tensor.shape();
    int outer = 1;
    for (int i = 0; i < d; i++) outer *= s[i];
    int reduce = s[d];
    int inner = 1;
    for (int i = d + 1; i < tensor.rank(); i++) inner *= s[i];

    double[] inData = tensor.value.toDoubleArray();
    double[] resultData = new double[inData.length];
    double[] normVals = new double[outer * inner];
    for (int o = 0; o < outer; o++) {
        for (int ii = 0; ii < inner; ii++) {
            double normP = 0;
            for (int r = 0; r < reduce; r++) normP += Math.pow(Math.abs(inData[(o * reduce + r) * inner + ii]), p);
            double norm = Math.pow(normP, 1.0 / p);
            normVals[o * inner + ii] = norm;
            if (norm > 0) {
                for (int r = 0; r < reduce; r++) {
                    int idx = (o * reduce + r) * inner + ii;
                    resultData[idx] = inData[idx] / norm;
                }
            }
        }
    }
    int fOuter = outer, fReduce = reduce, fInner = inner;
    double fP = p;
    double[] savedNorms = normVals;
    double[] savedIn = inData;
    double[] savedResult = resultData;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        double[] inGrad = new double[self.grad.length];
        for (int o = 0; o < fOuter; o++) {
            for (int ii = 0; ii < fInner; ii++) {
                double norm = savedNorms[o * fInner + ii];
                if (norm == 0) continue;
                double dot = 0;
                for (int r = 0; r < fReduce; r++) {
                    int idx = (o * fReduce + r) * fInner + ii;
                    dot += self.grad[idx] * savedResult[idx];
                }
                for (int r = 0; r < fReduce; r++) {
                    int idx = (o * fReduce + r) * fInner + ii;
                    double xi = savedIn[idx];
                    if (fP == 2.0) {
                        inGrad[idx] = (self.grad[idx] - xi * dot / norm) / norm;
                    } else {
                        double signX = xi >= 0 ? 1.0 : -1.0;
                        inGrad[idx] = (self.grad[idx] - signX * Math.pow(Math.abs(xi), fP - 1) * dot / Math.pow(norm, fP - 1)) / norm;
                    }
                }
            }
        }
        input.accGrad(inGrad);
    };
    return new RereDiffTensor(resultData, s, List.of(tensor), bw, "normalize");
}

}
