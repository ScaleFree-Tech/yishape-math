package com.yishape.lab.math.autodiff.impl;

import java.util.List;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.ITensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import com.yishape.lab.math.linalg.tensor.TensorShape;

/**
 * Zero-overhead non-differentiable tensor wrapping a {@link RereDoubleTensor}.
 * All ops delegate to the underlying value; {@link #backward()} is a no-op.
 * Used by {@link RereDiffTensor#toNonDiff} to avoid allocating full graph nodes
 * during eval/inference.
 */

public final class ConstantDiffTensor implements IDiffTensor {
    final RereDoubleTensor value;

    ConstantDiffTensor(RereDoubleTensor value) { this.value = value; }

    // Shape delegation
    @Override public int rank() { return value.rank(); }
    @Override public int[] shape() { return value.shape(); }
    @Override public int dim(int axis) { return value.dim(axis); }
    @Override public long totalSize() { return value.totalSize(); }
    @Override public int[] strides() { return value.strides(); }
    @Override public int stride(int axis) { return value.stride(axis); }
    @Override public int offset() { return value.offset(); }
    @Override public boolean isContiguous() { return value.isContiguous(); }
    @Override public double item() { return value.item(); }
    @Override public double get(int... indices) { return value.get(indices); }
    @Override public ITensor set(double val, int... indices) { value.set(val, indices); return this; }
    @Override public ITensor fill(double val) { value.fill(val); return this; }
    @Override public IDoubleTensor copy() { return new RereDoubleTensor(value.toDoubleArray(), shape()); }
    @Override public double[] toDoubleArray() { return value.toDoubleArray(); }
    @Override public IMatrix toMatrix() { return value.toMatrix(); }
    @Override public IDoubleVector toVector() { return value.toVector(); }
    @Override public IDoubleVector toVectorCopy() { return value.toVectorCopy(); }

    // Non-diff ops delegate to value, wrap result
    @Override public IDiffTensor add(IDoubleTensor other) { return wrap(value.add(detachOther(other))); }
    @Override public IDiffTensor sub(IDoubleTensor other) { return wrap(value.sub(detachOther(other))); }
    @Override public IDiffTensor mul(IDoubleTensor other) { return wrap(value.mul(detachOther(other))); }
    @Override public IDiffTensor div(IDoubleTensor other) { return wrap(value.div(detachOther(other))); }
    @Override public IDiffTensor add(double scalar) { return wrap(value.add(scalar)); }
    @Override public IDiffTensor sub(double scalar) { return wrap(value.sub(scalar)); }
    @Override public IDiffTensor mul(double scalar) { return wrap(value.mul(scalar)); }
    @Override public IDiffTensor div(double scalar) { return wrap(value.div(scalar)); }
    @Override public IDiffTensor neg() { return wrap(value.neg()); }
    @Override public IDiffTensor abs() { return wrap(value.abs()); }
    @Override public IDiffTensor sqrt() { return wrap(value.sqrt()); }
    @Override public IDiffTensor exp() { return wrap(value.exp()); }
    @Override public IDiffTensor log() { return wrap(value.log()); }
    @Override public IDiffTensor sigmoid() { return wrap(value.sigmoid()); }
    @Override public IDiffTensor relu() { return wrap(value.relu()); }
    @Override public IDiffTensor square() { return wrap(value.square()); }
    @Override public IDiffTensor pow(double n) { return wrap(value.pow(n)); }
    @Override public IDiffTensor clamp(double min, double max) { return wrap(value.clamp(min, max)); }
    @Override public IDiffTensor sin() { return wrap(value.sin()); }
    @Override public IDiffTensor cos() { return wrap(value.cos()); }
    @Override public IDiffTensor tan() { return wrap(value.tan()); }
    @Override public IDiffTensor tanh() { return wrap(value.tanh()); }
    @Override public IDiffTensor silu() { return wrap(value.silu()); }
    @Override public IDiffTensor gelu() { return wrap(value.gelu()); }
    @Override public IDiffTensor softplus(double beta) { return wrap(value.softplus(beta)); }
    @Override public IDiffTensor mish() { return wrap(value.mish()); }
    @Override public IDiffTensor elu(double alpha) { return wrap(value.elu(alpha)); }
    @Override public IDiffTensor leakyRelu(double alpha) { return wrap(value.leakyRelu(alpha)); }
    @Override public IDiffTensor selu() { return wrap(value.selu()); }
    @Override public IDiffTensor hardtanh(double minVal, double maxVal) { return wrap(value.hardtanh(minVal, maxVal)); }
    @Override public IDiffTensor rsub(double scalar) { return wrap(value.neg().add(scalar)); }
    @Override public IDiffTensor rdiv(double scalar) { return wrap(value.pow(-1).mul(scalar)); }
    @Override public IDiffTensor reciprocal() { return wrap(value.pow(-1)); }
    @Override public IDiffTensor dropout(double p) { return this; }
    @Override public IDiffTensor sum(int dim, boolean keepdim) { return wrap(value.sum(dim, keepdim)); }
    @Override public IDiffTensor mean(int dim, boolean keepdim) { return wrap(value.mean(dim, keepdim)); }
    @Override public IDiffTensor max(int dim, boolean keepdim) { return wrap(value.max(dim, keepdim)); }
    @Override public IDiffTensor min(int dim, boolean keepdim) { return wrap(value.min(dim, keepdim)); }
    @Override public IDiffTensor prod(int dim, boolean keepdim) { return wrap(value.prod(dim, keepdim)); }
    @Override public IDiffTensor cumsum(int dim) { return wrap(value.cumsum(dim)); }
    @Override public IDiffTensor cumprod(int dim) { return wrap(value.cumprod(dim)); }
    @Override public IDiffTensor argmax(int dim) { return wrap(value.argmax(dim)); }
    @Override public IDiffTensor argmin(int dim) { return wrap(value.argmin(dim)); }
    @Override public IDiffTensor std(int dim, boolean keepdim) { return wrap(value.std(dim, keepdim)); }
    @Override public IDiffTensor var(int dim, boolean keepdim) { return wrap(value.var(dim, keepdim)); }
    @Override public IDiffTensor logSumExp(int dim, boolean keepdim) {
        int d = (dim < 0 ? dim + value.rank() : dim);
        double[] vals = value.toDoubleArray();
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];
        double[] result = new double[outer * inner];
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
                result[o * inner + i] = Math.log(sumExp + 1e-15) + max;
            }
        }
        int[] reducedShape = new int[s.length];
        System.arraycopy(s, 0, reducedShape, 0, s.length);
        reducedShape[d] = keepdim ? 1 : 0;
        if (!keepdim) {
            int[] trimmed = new int[s.length - 1];
            int pos = 0;
            for (int j = 0; j < s.length; j++) if (j != d) trimmed[pos++] = reducedShape[j];
            reducedShape = trimmed;
        } else {
            reducedShape[d] = 1;
        }
        return wrap(new RereDoubleTensor(result, reducedShape));
    }
    @Override public IDiffTensor softmax(int dim) { return wrap(value.softmax(dim)); }
    @Override public IDiffTensor logSoftmax(int dim) { return wrap(value.logSoftmax(dim)); }
    @Override public IDiffTensor softmaxCrossEntropy(IDoubleTensor labels, int dim) {
        int d = (dim < 0 ? dim + value.rank() : dim);
        IDoubleTensor sm = value.softmax(d);
        double[] smData = sm.toDoubleArray();
        double[] lbData = labels.toDoubleArray();
        double totalLoss = 0;
        for (int i = 0; i < smData.length; i++) totalLoss += -lbData[i] * Math.log(smData[i]);
        int totalSamples = smData.length / value.dim(d);
        return wrap(new RereDoubleTensor(new double[]{totalLoss / totalSamples}, new int[]{1}));
    }
    @Override public IDiffTensor softmaxCrossEntropySparse(int[] labels, int dim) {
        int d = (dim < 0 ? dim + value.rank() : dim);
        IDoubleTensor sm = value.softmax(d);
        double[] smData = sm.toDoubleArray();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= value.dim(i);
        int C = value.dim(d);
        int inner = 1;
        for (int i = d + 1; i < value.rank(); i++) inner *= value.dim(i);
        double totalLoss = 0;
        for (int o = 0; o < outer; o++) {
            for (int in = 0; in < inner; in++) {
                int t = labels[o * inner + in];
                int idx = (o * C + t) * inner + in;
                totalLoss += -Math.log(Math.max(smData[idx], 1e-30));
            }
        }
        int totalSamples = outer * inner;
        return wrap(new RereDoubleTensor(new double[]{totalLoss / totalSamples}, new int[]{1}));
    }
    @Override public IDiffTensor conv2d(IDiffTensor weight, IDiffTensor bias,
            int stride, int padding, int dilation) {
        throw new UnsupportedOperationException("conv2d is not available on constant tensors");
    }
    @Override public IDiffTensor[] lstmCell(IDiffTensor x, IDiffTensor hPrev, IDiffTensor cPrev,
            IDiffTensor wInput, IDiffTensor wHidden, IDiffTensor bias) {
        throw new UnsupportedOperationException("lstmCell not available on constant tensors");
    }
    @Override public IDiffTensor gruCell(IDiffTensor x, IDiffTensor hPrev,
            IDiffTensor wInput, IDiffTensor wHidden, IDiffTensor bias) {
        throw new UnsupportedOperationException("gruCell not available on constant tensors");
    }
    @Override public IDiffTensor groupNorm(int numGroups, IDiffTensor gamma, IDiffTensor beta, double eps) {
        return computeGroupNorm(numGroups, gamma, beta, eps);
    }
    @Override public IDiffTensor flip(int... dims) {
        int[] s = shape();
        double[] xd = value.toDoubleArray();
        double[] y = new double[xd.length];
        int[] fd = new int[dims.length];
        for (int i = 0; i < dims.length; i++) fd[i] = (dims[i] < 0 ? dims[i] + value.rank() : dims[i]);
        long n = value.totalSize();
        for (long flatIdx = 0; flatIdx < n; flatIdx++) {
            long dstIdx = 0, srcIdx = flatIdx, stride = 1;
            for (int d = value.rank() - 1; d >= 0; d--) {
                long coord = srcIdx % s[d]; srcIdx /= s[d];
                boolean match = false;
                for (int f : fd) { if (f == d) { match = true; break; } }
                if (match) coord = s[d] - 1 - coord;
                dstIdx += coord * stride;
                stride *= s[d];
            }
            y[(int) dstIdx] = xd[(int) flatIdx];
        }
        return wrap(new RereDoubleTensor(y, s));
    }
    @Override public IDiffTensor roll(int[] shifts, int[] dims) {
        IDoubleTensor result = value;
        for (int i = 0; i < shifts.length; i++) {
            int d = (dims[i] < 0 ? dims[i] + value.rank() : dims[i]);
            int dimSize = value.dim(d);
            int shift = ((shifts[i] % dimSize) + dimSize) % dimSize;
            if (shift == 0) continue;
            IDoubleTensor[] parts = { result.narrow(d, dimSize - shift, shift),
                                      result.narrow(d, 0, dimSize - shift) };
            result = parts[0].cat(d, parts[1]);
        }
        return wrap(result);
    }
    @Override public IDiffTensor repeatInterleave(int repeats, int dim) {
        int d = (dim < 0 ? dim + value.rank() : dim);
        int dimSize = value.dim(d);
        double[] idxData = new double[dimSize * repeats];
        for (int i = 0; i < dimSize; i++) {
            for (int r = 0; r < repeats; r++) idxData[i * repeats + r] = i;
        }
        return wrap(value.indexSelect(d, new RereDoubleTensor(idxData, new int[]{dimSize * repeats})));
    }
    @Override public IDiffTensor smoothL1Loss(IDiffTensor target, double beta) {
        double[] xd = value.toDoubleArray();
        double[] td = target.toDoubleArray();
        double total = 0;
        double halfBeta = 0.5 * beta;
        for (int i = 0; i < xd.length; i++) {
            double d = Math.abs(xd[i] - td[i]);
            total += (d <= beta) ? 0.5 * d * d / beta : d - halfBeta;
        }
        return wrap(new RereDoubleTensor(new double[]{total / xd.length}, new int[]{1}));
    }
	        @Override public IDiffTensor bceLoss(IDiffTensor target) {
	            double[] xd = value.toDoubleArray();
	            double[] td = target.toDoubleArray();
	            double total = 0;
	            final double eps = 1e-7;
	            for (int i = 0; i < xd.length; i++) {
	                double p = Math.max(eps, Math.min(1.0 - eps, xd[i]));
	                double y = td[i];
	                total += -y * Math.log(p) - (1.0 - y) * Math.log(1.0 - p);
	            }
	            return wrap(new RereDoubleTensor(new double[]{total / xd.length}, new int[]{1}));
	        }
	        @Override public IDiffTensor focalLoss(IDiffTensor target, double alpha, double gamma) {
	            double[] xd = value.toDoubleArray();
	            double[] td = target.toDoubleArray();
	            double total = 0;
	            final double eps = 1e-7;
	            double oma = 1.0 - alpha;
	            for (int i = 0; i < xd.length; i++) {
	                double p = Math.max(eps, Math.min(1.0 - eps, xd[i]));
	                double y = td[i];
	                double pT = (y > 0.5) ? p : 1.0 - p;
	                double aT = (y > 0.5) ? alpha : oma;
	                total += aT * Math.pow(1.0 - pT, gamma) * (-Math.log(pT));
	            }
	            return wrap(new RereDoubleTensor(new double[]{total / xd.length}, new int[]{1}));
	        }
	        @Override public IDiffTensor diceLoss(IDiffTensor target, double smooth) {
	            double[] xd = value.toDoubleArray();
	            double[] td = target.toDoubleArray();
	            double I = 0, Sp = 0, St = 0;
	            for (int i = 0; i < xd.length; i++) {
	                I += xd[i] * td[i];
	                Sp += xd[i];
	                St += td[i];
	            }
	            double denom = Sp + St + smooth;
	            double dice = (2.0 * I + smooth) / denom;
	            return wrap(new RereDoubleTensor(new double[]{1.0 - dice}, new int[]{1}));
	        }
    @Override public IDiffTensor nllLoss(IDiffTensor target, int classDim) {
        int d = (classDim < 0 ? classDim + value.rank() : classDim);
        IDoubleTensor gathered = value.gather(d, target);
        double total = 0;
        double[] gd = gathered.toDoubleArray();
        for (double v : gd) total += v;
        return wrap(new RereDoubleTensor(new double[]{-total / gd.length}, new int[]{1}));
    }
    @Override public IDiffTensor maxPool2d(int kH, int kW, int stride, int padding) {
        throw new UnsupportedOperationException("maxPool2d is not available on constant tensors");
    }
    @Override public IDiffTensor avgPool2d(int kH, int kW, int stride, int padding) {
        throw new UnsupportedOperationException("avgPool2d is not available on constant tensors");
    }
    @Override public IDiffTensor adaptiveAvgPool2d(int outH, int outW) {
        int[] s = shape();
        int rank = value.rank();
        int N, C, H, W;
        if (rank == 3) { N = 1; C = s[0]; H = s[1]; W = s[2]; }
        else if (rank == 4) { N = s[0]; C = s[1]; H = s[2]; W = s[3]; }
        else { N = 1; for (int i = 0; i < rank - 3; i++) N *= s[i]; C = s[rank - 3]; H = s[rank - 2]; W = s[rank - 1]; }
        double[] xd = value.toDoubleArray();
        double[] y = new double[N * C * outH * outW];
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int oh = 0; oh < outH; oh++) {
                    int hStart = (int) Math.floor((double) oh * H / outH);
                    int hEnd = (int) Math.floor((double) (oh + 1) * H / outH);
                    for (int ow = 0; ow < outW; ow++) {
                        int wStart = (int) Math.floor((double) ow * W / outW);
                        int wEnd = (int) Math.floor((double) (ow + 1) * W / outW);
                        double sum = 0;
                        for (int h = hStart; h < hEnd; h++)
                            for (int w = wStart; w < wEnd; w++)
                                sum += xd[((n * C + c) * H + h) * W + w];
                        y[((n * C + c) * outH + oh) * outW + ow] = sum / ((hEnd - hStart) * (wEnd - wStart));
                    }
                }
            }
        }
        // Build output shape preserving all leading batch dimensions
        int[] outShape;
        if (rank == 4) {
            outShape = new int[]{N, C, outH, outW};
        } else if (rank == 3) {
            outShape = new int[]{C, outH, outW};
        } else {
            // rank > 4: preserve leading dims [B1,...,Bn, C, outH, outW]
            int leadingDims = rank - 3;
            outShape = new int[rank];
            System.arraycopy(s, 0, outShape, 0, leadingDims);
            outShape[leadingDims] = C;
            outShape[leadingDims + 1] = outH;
            outShape[leadingDims + 2] = outW;
        }
        return wrap(new RereDoubleTensor(y, outShape));
    }
    @Override public IDiffTensor oneHot(int numClasses) {
        int[] s = shape();
        int[] outShape = new int[s.length + 1];
        System.arraycopy(s, 0, outShape, 0, s.length);
        outShape[s.length] = numClasses;
        long n = value.totalSize();
        double[] y = new double[(int) (n * numClasses)];
        double[] xd = value.toDoubleArray();
        for (int i = 0; i < n; i++) {
            int cls = (int) Math.round(xd[i]);
            if (cls >= 0 && cls < numClasses) y[i * numClasses + cls] = 1.0;
        }
        return wrap(new RereDoubleTensor(y, outShape));
    }
    @Override public IDiffTensor instanceNorm(IDiffTensor gamma, IDiffTensor beta, double eps) {
        return computeInstanceNorm(gamma, beta, eps);
    }
    @Override public IDiffTensor diagEmbed(int offset, int dim1, int dim2) {
        long n = value.totalSize();
        int M = (int) n + Math.abs(offset);
        double[] xd = value.toDoubleArray();
        double[] y = new double[M * M];
        for (int i = 0; i < n; i++) {
            int row = (offset >= 0) ? i : i - offset;
            int col = (offset >= 0) ? i + offset : i;
            if (row >= 0 && row < M && col >= 0 && col < M) y[row * M + col] = xd[i];
        }
        return wrap(new RereDoubleTensor(y, new int[]{M, M}));
    }
    @Override public IDiffTensor dropout2d(double p) {
        if (p <= 0) return this;
        int[] s = shape();
        int rank = value.rank();
        int N, C;
        if (rank == 3) { N = 1; C = s[0]; }
        else if (rank == 4) { N = s[0]; C = s[1]; }
        else { N = 1; for (int i = 0; i < rank - 2; i++) N *= s[i]; C = s[rank - 2]; }
        double[] xd = value.toDoubleArray();
        double[] y = new double[xd.length];
        double scale = 1.0 / (1.0 - p);
        int spatial = xd.length / (N * C);
        java.util.Random rng = new java.util.Random();
        for (int nc = 0; nc < N * C; nc++) {
            double m = (rng.nextDouble() >= p) ? scale : 0;
            for (int sp = 0; sp < spatial; sp++) y[nc * spatial + sp] = xd[nc * spatial + sp] * m;
        }
        return wrap(new RereDoubleTensor(y, s));
    }
    @Override public IDiffTensor depthwiseConv1d(IDiffTensor weight, int stride, int padding) {
        throw new UnsupportedOperationException("depthwiseConv1d is not available on constant tensors");
    }
    @Override public IDiffTensor interpolate(double scaleFactor, String mode) {
        throw new UnsupportedOperationException("interpolate is not available on constant tensors");
    }
    @Override public IDiffTensor logDet() {
        int[] s = shape();
        if (s.length != 2 || s[0] != s[1])
            throw new UnsupportedOperationException("logDet on constant tensor requires square 2D");
        double[] xd = value.toDoubleArray();
        int n = s[0];
        double logDet;
        try {
            IMatrix<Double> A = Linalg.fromArray(xd, n, n);
            var luDecomp = com.yishape.lab.math.linalg.decomposition.Decomps.createLU();
            luDecomp.decompose(A);
            logDet = Math.log(Math.abs(luDecomp.getDeterminant()));
        } catch (Exception e) {
            logDet = Double.NEGATIVE_INFINITY;
        }
        return wrap(new RereDoubleTensor(new double[]{logDet}, 1));
    }
    @Override public IDiffTensor[] slogDet() {
        int[] s = shape();
        if (s.length != 2 || s[0] != s[1])
            throw new UnsupportedOperationException("slogDet on constant tensor requires square 2D");
        double[] xd = value.toDoubleArray();
        int n = s[0];
        IMatrix<Double> A = Linalg.fromArray(xd, n, n);
        var luDecomp = com.yishape.lab.math.linalg.decomposition.Decomps.createLU();
        luDecomp.decompose(A);
        double det = luDecomp.getDeterminant();
        IDiffTensor signT = wrap(new RereDoubleTensor(new double[]{Math.signum(det)}, 1));
        IDiffTensor ldT = wrap(new RereDoubleTensor(new double[]{Math.log(Math.abs(det))}, 1));
        return new IDiffTensor[]{signT, ldT};
    }
    @Override public IDiffTensor nuclearNorm() {
        // Constant tensor path: SVD is expensive, just compute Frobenius-like bound
        double[] xd = value.toDoubleArray();
        long n = value.totalSize();
        if (value.rank() == 2) {
            IMatrix<Double> A = Linalg.fromArray(xd, value.dim(0), value.dim(1));
            var svdResult = A.svd();
            var sVec = svdResult.getSecond();
            double[] sv = sVec.toDoubleArray();
            double sum = 0;
            for (double v : sv) sum += v;
            return wrap(new RereDoubleTensor(new double[]{sum}, 1));
        }
        throw new UnsupportedOperationException("nuclearNorm on constant tensor requires 2D matrix");
    }
    @Override public IDiffTensor ctcLoss(IDiffTensor targets, IDiffTensor inputLengths, IDiffTensor targetLengths) {
        throw new UnsupportedOperationException("ctcLoss is not available on constant tensors");
    }
    @Override public IDiffTensor cross(IDiffTensor other) {
        // SISD: constant-tensor cross product
        int[] sA = shape();
        int[] sB = other.shape();
        if (sA[sA.length - 1] != 3 || sB[sB.length - 1] != 3) {
            throw new IllegalArgumentException("cross requires last dim = 3");
        }
        int[] bcShape = TensorShape.broadcastShape(sA, sB);
        long outSize = 1;
        for (int d : bcShape) outSize *= d;
        double[] aData = value.toDoubleArray();
        double[] bData = other.toDoubleArray();
        double[] y = new double[(int) outSize];
        long numTriplets = outSize / 3;
        for (long t = 0; t < numTriplets; t++) {
            long flatIdx = t * 3;
            int[] bcIdx = DiffTensorUtil.unlinearizeInt((int) flatIdx, bcShape);
            int ai = DiffTensorUtil.flatIndexFromBroadcast(bcIdx, sA, bcShape);
            int bi = DiffTensorUtil.flatIndexFromBroadcast(bcIdx, sB, bcShape);
            int aBase = (ai / 3) * 3, bBase = (bi / 3) * 3;
            double a0 = aData[aBase], a1 = aData[aBase + 1], a2 = aData[aBase + 2];
            double b0 = bData[bBase], b1 = bData[bBase + 1], b2 = bData[bBase + 2];
            y[(int) flatIdx]     = a1 * b2 - a2 * b1;
            y[(int) flatIdx + 1] = a2 * b0 - a0 * b2;
            y[(int) flatIdx + 2] = a0 * b1 - a1 * b0;
        }
        return wrap(new RereDoubleTensor(y, bcShape));
    }
    @Override public IDiffTensor gridSample(IDiffTensor grid, String mode, String paddingMode) {
        // SISD fallback for constant grid_sample
        int[] s = shape();
        int N = s[0], C = s[1], H = s[2], W = s[3];
        int[] gs = grid.shape();
        int outH = gs[1], outW = gs[2];
        double[] xd = value.toDoubleArray();
        double[] gd = grid.toDoubleArray();
        double[] y = new double[N * C * outH * outW];
        boolean bilinear = "bilinear".equals(mode);
        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    double gx = gd[((n * outH + oh) * outW + ow) * 2];
                    double gy = gd[((n * outH + oh) * outW + ow) * 2 + 1];
                    double px = (gx + 1.0) * 0.5 * (W - 1);
                    double py = (gy + 1.0) * 0.5 * (H - 1);
                    if (bilinear) {
                        int ix0 = (int) Math.floor(px), iy0 = (int) Math.floor(py);
                        int ix1 = ix0 + 1, iy1 = iy0 + 1;
                        double wx1 = px - ix0, wy1 = py - iy0;
                        double wx0 = 1.0 - wx1, wy0 = 1.0 - wy1;
                        for (int c = 0; c < C; c++) {
                            double val = 0;
                            int inBase = ((n * C + c) * H);
                            val += sampleBorder(xd, inBase, H, W, iy0, ix0, paddingMode) * wx0 * wy0;
                            val += sampleBorder(xd, inBase, H, W, iy0, ix1, paddingMode) * wx1 * wy0;
                            val += sampleBorder(xd, inBase, H, W, iy1, ix0, paddingMode) * wx0 * wy1;
                            val += sampleBorder(xd, inBase, H, W, iy1, ix1, paddingMode) * wx1 * wy1;
                            y[((n * C + c) * outH + oh) * outW + ow] = val;
                        }
                    } else {
                        int ix = (int) Math.round(px), iy = (int) Math.round(py);
                        for (int c = 0; c < C; c++) {
                            int inBase = ((n * C + c) * H);
                            y[((n * C + c) * outH + oh) * outW + ow] =
                                sampleBorder(xd, inBase, H, W, iy, ix, paddingMode);
                        }
                    }
                }
            }
        }
        return wrap(new RereDoubleTensor(y, new int[]{N, C, outH, outW}));
    }
    private static double sampleBorder(double[] data, int inBase, int H, int W, int y, int x, String pad) {
        int cy = switch (pad) {
            case "border" -> Math.clamp(y, 0, H - 1);
            case "reflection" -> { int r = Math.abs(y) % (2 * H); yield r >= H ? 2 * H - 1 - r : r; }
            default -> y;
        };
        int cx = switch (pad) {
            case "border" -> Math.clamp(x, 0, W - 1);
            case "reflection" -> { int r = Math.abs(x) % (2 * W); yield r >= W ? 2 * W - 1 - r : r; }
            default -> x;
        };
        if (cy < 0 || cy >= H || cx < 0 || cx >= W) return 0;
        return data[inBase + cy * W + cx];
    }
    @Override public IDiffTensor trapezoidalScan(IDiffTensor delta, IDiffTensor A, IDiffTensor B,
                                                  IDiffTensor C, IDiffTensor D) {
        throw new UnsupportedOperationException("trapezoidalScan is not available on constant tensors");
    }
    @Override public IDiffTensor embedding(IDiffTensor indices) {
        int[] idxShape = indices.shape();
        int embeddingDim = value.dim(value.rank() - 1);
        IDoubleTensor flatIdx = indices.reshape((int) indices.totalSize());
        IDoubleTensor gathered = value.indexSelect(0, flatIdx);
        int[] outShape = new int[idxShape.length + 1];
        System.arraycopy(idxShape, 0, outShape, 0, idxShape.length);
        outShape[idxShape.length] = embeddingDim;
        return wrap(gathered.reshape(outShape));
    }
    @Override public IDiffTensor rope(int dim, int maxLen, double base) {
        double[] xd = value.toDoubleArray();
        int headDim = value.dim(value.rank() - 1);
        long totalSize = value.totalSize();
        int seqLen = (int) (totalSize / headDim);
        int numPairs = headDim / 2;
        double[] y = new double[xd.length];
        for (int pos = 0; pos < seqLen; pos++) {
            int baseOff = pos * headDim;
            for (int i = 0; i < numPairs; i++) {
                double theta = pos / Math.pow(base, 2.0 * i / dim);
                double c = Math.cos(theta), s = Math.sin(theta);
                int idx2i = baseOff + 2 * i, idx2i1 = idx2i + 1;
                double x1 = xd[idx2i], x2 = xd[idx2i1];
                y[idx2i] = x1 * c - x2 * s;
                y[idx2i1] = x1 * s + x2 * c;
            }
        }
        return wrap(new RereDoubleTensor(y, shape()));
    }
    @Override public IDiffTensor scaledDotProductAttention(IDiffTensor key, IDiffTensor vTensor,
            IDiffTensor mask, double dropout) {
        throw new UnsupportedOperationException("scaledDotProductAttention is not available on constant tensors");
    }
    @Override public IDiffTensor layerNorm(IDiffTensor gamma, IDiffTensor beta, double eps) { return computeNorm(gamma, beta, eps, true); }
    @Override public IDiffTensor batchNorm(IDiffTensor gamma, IDiffTensor beta, double eps) { return computeNorm(gamma, beta, eps, false); }
    @Override public IDiffTensor rmsNorm(IDiffTensor gamma, double eps) {
        int features = value.dim(value.rank() - 1);
        long totalSize = value.totalSize();
        int batch = (int) (totalSize / features);
        double[] xd = value.toDoubleArray();
        double[] gd = gamma.toDoubleArray();
        double[] y = new double[(int) totalSize];
        for (int p = 0; p < batch; p++) {
            int off = p * features;
            double sumSq = 0;
            for (int j = 0; j < features; j++) sumSq += xd[off + j] * xd[off + j];
            double invRms = 1.0 / Math.sqrt(sumSq / features + eps);
            for (int j = 0; j < features; j++) {
                y[off + j] = xd[off + j] * invRms * gd[j];
            }
        }
        return wrap(new RereDoubleTensor(y, shape()));
    }
    private IDiffTensor computeNorm(IDiffTensor gamma, IDiffTensor beta, double eps, boolean overLastDim) {
        int features = overLastDim ? value.dim(value.rank() - 1) : value.dim(value.rank() - 1);
        long totalSize = value.totalSize();
        int batch = (int) (totalSize / features);
        double[] xd = value.toDoubleArray();
        double[] gd = gamma.toDoubleArray();
        double[] bd = beta.toDoubleArray();
        double[] y = new double[(int) totalSize];
        for (int p = 0; p < batch; p++) {
            int off = p * features;
            double mean = 0;
            for (int j = 0; j < features; j++) mean += xd[off + j];
            mean /= features;
            double var = 0;
            for (int j = 0; j < features; j++) { double d = xd[off + j] - mean; var += d * d; }
            var /= features;
            double inv = 1.0 / Math.sqrt(var + eps);
            for (int j = 0; j < features; j++) {
                y[off + j] = gd[j] * (xd[off + j] - mean) * inv + bd[j];
            }
        }
        return wrap(new RereDoubleTensor(y, shape()));
    }
    private IDiffTensor computeGroupNorm(int numGroups, IDiffTensor gamma, IDiffTensor beta, double eps) {
        int[] s = shape();
        int rank = value.rank();
        int C = s[rank - 2];
        int groupCh = C / numGroups;
        int outer = 1;
        for (int i = 0; i < rank - 2; i++) outer *= s[i];
        int spatial = (int) value.totalSize() / (outer * C);
        double[] xd = value.toDoubleArray();
        double[] gd = gamma.toDoubleArray();
        double[] bd = (beta != null) ? beta.toDoubleArray() : null;
        double[] y = new double[xd.length];
        for (int n = 0; n < outer; n++) {
            for (int g = 0; g < numGroups; g++) {
                int count = 0;
                double mean = 0;
                for (int c = g * groupCh; c < (g + 1) * groupCh; c++) {
                    for (int sp = 0; sp < spatial; sp++) {
                        mean += xd[n * C * spatial + c * spatial + sp];
                        count++;
                    }
                }
                mean /= count;
                double var = 0;
                for (int c = g * groupCh; c < (g + 1) * groupCh; c++) {
                    for (int sp = 0; sp < spatial; sp++) {
                        double d = xd[n * C * spatial + c * spatial + sp] - mean;
                        var += d * d;
                    }
                }
                var /= count;
                double inv = 1.0 / Math.sqrt(var + eps);
                for (int c = g * groupCh; c < (g + 1) * groupCh; c++) {
                    for (int sp = 0; sp < spatial; sp++) {
                        int idx = n * C * spatial + c * spatial + sp;
                        y[idx] = (xd[idx] - mean) * inv * gd[c] + (bd != null ? bd[c] : 0);
                    }
                }
            }
        }
        return wrap(new RereDoubleTensor(y, s));
    }
    private IDiffTensor computeInstanceNorm(IDiffTensor gamma, IDiffTensor beta, double eps) {
        int[] s = shape();
        int rank = value.rank();
        int N = 1;
        for (int i = 0; i < rank - 2; i++) N *= s[i];
        int C = s[rank - 2];
        int spatial = 1;
        for (int i = rank - 1; i < rank; i++) spatial *= s[i];
        double[] xd = value.toDoubleArray();
        double[] gd = gamma.toDoubleArray();
        double[] bd = (beta != null) ? beta.toDoubleArray() : null;
        double[] y = new double[xd.length];
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                double mean = 0;
                for (int sp = 0; sp < spatial; sp++) mean += xd[(n * C + c) * spatial + sp];
                mean /= spatial;
                double var = 0;
                for (int sp = 0; sp < spatial; sp++) { double d = xd[(n * C + c) * spatial + sp] - mean; var += d * d; }
                var /= spatial;
                double inv = 1.0 / Math.sqrt(var + eps);
                for (int sp = 0; sp < spatial; sp++) {
                    int idx = (n * C + c) * spatial + sp;
                    y[idx] = (xd[idx] - mean) * inv * gd[c] + (bd != null ? bd[c] : 0);
                }
            }
        }
        return wrap(new RereDoubleTensor(y, s));
    }
    @Override public IDiffTensor mmul(IDoubleTensor other) { return wrap(value.mmul(detachOther(other))); }
    @Override public IDiffTensor bmm(IDoubleTensor other) { return wrap(value.bmm(detachOther(other))); }
    @Override public IDiffTensor einsum(String subscript, IDoubleTensor... others) {
        IDoubleTensor[] detOthers = new IDoubleTensor[others.length];
        for (int i = 0; i < others.length; i++) {
            detOthers[i] = others[i] instanceof IDiffTensor dt ? dt.detach() : others[i];
        }
        return wrap(value.einsum(subscript, detOthers));
    }
    @Override public IDiffTensor reshape(int... newShape) { return wrap(value.reshape(newShape)); }
    @Override public IDiffTensor permute(int... dims) { return wrap(value.permute(dims)); }
    @Override public IDiffTensor transpose(int dim0, int dim1) { return wrap(value.transpose(dim0, dim1)); }
    @Override public IDiffTensor transpose() { return wrap(value.transpose()); }
    @Override public IDiffTensor squeeze(int... dims) { return wrap(value.squeeze(dims)); }
    @Override public IDiffTensor unsqueeze(int dim) { return wrap(value.unsqueeze(dim)); }
    @Override public IDiffTensor select(int dim, long index) { return wrap(value.select(dim, index)); }
    @Override public IDiffTensor flatten(int startDim, int endDim) { return wrap(value.flatten(startDim, endDim)); }
    @Override public IDiffTensor slice(int dim, long start, long end) { return wrap(value.slice(dim, start, end)); }
    @Override public IDiffTensor narrow(int dim, long start, long length) { return wrap(value.narrow(dim, start, length)); }
    @Override public IDiffTensor expand(int... shape) { return wrap(value.expand(shape)); }
    @Override public IDiffTensor contiguous() { return wrap(value.contiguous()); }
    @Override public IDiffTensor tile(int... repeats) { return wrap(value.tile(repeats)); }
    @Override public IDiffTensor broadcastTo(int... shape) { return wrap(value.broadcastTo(shape)); }
    @Override public IDiffTensor clone() { return new ConstantDiffTensor(new RereDoubleTensor(value.toDoubleArray(), shape())); }
    @Override public IDiffTensor gather(int dim, IDoubleTensor index) { return wrap(value.gather(dim, detachOther(index))); }
    @Override public IDiffTensor indexSelect(int dim, IDoubleTensor index) { return wrap(value.indexSelect(dim, detachOther(index))); }
    @Override public IDiffTensor argsort(int dim, boolean descending) { return wrap(value.argsort(dim, descending)); }
    @Override public IDiffTensor scatter(int dim, IDoubleTensor index, IDoubleTensor source) { return wrap(value.scatter(dim, detachOther(index), detachOther(source))); }
    @Override public IDiffTensor scatterAdd(int dim, IDoubleTensor index, IDoubleTensor source) { return wrap(value.scatterAdd(dim, detachOther(index), detachOther(source))); }
    @Override public IDiffTensor where(IDoubleTensor condition, IDoubleTensor other) { return wrap(value.where(detachOther(condition), detachOther(other))); }
    @Override public IDiffTensor topk(int k, int dim, boolean largest) { return wrap(value.topk(k, dim, largest)); }
    @Override public IDiffTensor pad(int[][] padding, String mode, double padValue) { return wrap(value.pad(padding, mode, padValue)); }
    @Override public IDiffTensor tril(int diagonal) { return wrap(value.tril(diagonal)); }
    @Override public IDiffTensor triu(int diagonal) {
        // Manual triu: clone and zero lower triangle
        double[] d = value.toDoubleArray().clone();
        int r = rank();
        int M = value.dim(r - 2);
        int N = value.dim(r - 1);
        int batchStride = M * N;
        int batchCount = d.length / batchStride;
        for (int b = 0; b < batchCount; b++) {
            int base = b * batchStride;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    if (j < i + diagonal) d[base + i * N + j] = 0.0;
                }
            }
        }
        return wrap(new RereDoubleTensor(d, shape()));
    }
    @Override public IDiffTensor diag() {
        int r = rank();
        int[] s = shape();
        int M = (r >= 2) ? value.dim(r - 2) : 1;
        int N = (r >= 2) ? value.dim(r - 1) : 1;
        if (r < 2) return this;
        if (M != N) throw new IllegalArgumentException("diag() requires square matrix, got " + M + "x" + N);
        int batchDim = 1;
        for (int i = 0; i < r - 2; i++) batchDim *= s[i];
        double[] vals = value.toDoubleArray();
        double[] resultData = new double[batchDim * M];
        for (int b = 0; b < batchDim; b++) {
            int base = b * M * N;
            for (int i = 0; i < M; i++) resultData[b * M + i] = vals[base + i * N + i];
        }
        int[] resultShape = (r == 2) ? new int[]{M} : java.util.Arrays.copyOf(s, r - 1);
        if (r > 2) resultShape[r - 2] = M;
        return wrap(new RereDoubleTensor(resultData, resultShape));
    }
    @Override public IDiffTensor diagonal(int offset, int dim1, int dim2) {
        // For constant tensors, just compute via diag path
        int r = rank();
        int[] s = shape();
        if (dim1 < 0) dim1 += r;
        if (dim2 < 0) dim2 += r;
        int size = (int) Math.min(s[dim1], s[dim2]);
        int effSize = offset >= 0
            ? Math.max(0, Math.min(s[dim1] - offset, s[dim2]))
            : Math.max(0, Math.min(s[dim1], s[dim2] + offset));
        double[] vals = value.toDoubleArray();
        int outer = 1;
        for (int i = 0; i < Math.min(dim1, dim2); i++) outer *= s[i];
        int inner = 1;
        for (int i = Math.max(dim1, dim2) + 1; i < r; i++) inner *= s[i];
        double[] resultData = new double[outer * effSize * inner];
        for (int o = 0; o < outer; o++) {
            for (int k = 0; k < effSize; k++) {
                for (int i = 0; i < inner; i++) {
                    int[] idx = new int[r];
                    int remaining = o;
                    for (int j = Math.min(dim1, dim2) - 1; j >= 0; j--) {
                        idx[j] = remaining % s[j]; remaining /= s[j];
                    }
                    idx[dim1] = offset >= 0 ? k : k - offset;
                    idx[dim2] = offset >= 0 ? k + offset : k;
                    remaining = i;
                    for (int j = r - 1; j > Math.max(dim1, dim2); j--) {
                        idx[j] = remaining % s[j]; remaining /= s[j];
                    }
                    resultData[(o * effSize + k) * inner + i] = vals[DiffTensorUtil.flatIndex(idx, s)];
                }
            }
        }
        int[] resultShape;
        if (r == 2) { resultShape = new int[]{effSize}; }
        else {
            resultShape = new int[r - 1];
            int pos = 0;
            for (int j = 0; j < dim1; j++) resultShape[pos++] = s[j];
            for (int j = dim1 + 1; j < dim2; j++) resultShape[pos++] = s[j];
            for (int j = dim2 + 1; j < r; j++) resultShape[pos++] = s[j];
            resultShape[Math.min(dim1, dim2)] = effSize;
        }
        return wrap(new RereDoubleTensor(resultData, resultShape));
    }
    @Override public IDiffTensor trace() { return diag().sum(); }
    @Override public IDiffTensor unfold(int dim, int size, int stride, int dilation) { return wrap(value.unfold(dim, size, stride, dilation)); }
    @Override public IDiffTensor nonzero() { return wrap(value.nonzero()); }
    @Override public IDiffTensor maskedSelect(IDoubleTensor mask) { return wrap(value.maskedSelect(detachOther(mask))); }
    @Override public IDiffTensor maskedFill(IDoubleTensor mask, double fillValue) { return wrap(value.maskedFill(detachOther(mask), fillValue)); }
    @Override public IDiffTensor cat(int dim, IDoubleTensor... others) {
        IDoubleTensor[] d = new IDoubleTensor[others.length];
        for (int i = 0; i < others.length; i++) d[i] = detachOther(others[i]);
        return wrap(value.cat(dim, d));
    }
    @Override public IDiffTensor stack(int dim, IDoubleTensor... others) {
        IDoubleTensor[] d = new IDoubleTensor[others.length];
        for (int i = 0; i < others.length; i++) d[i] = detachOther(others[i]);
        return wrap(value.stack(dim, d));
    }
    @Override public List<IDoubleTensor> unstack(int dim) { return value.unstack(dim); }
    @Override public IDiffTensor normalize(double p, int dim) { return wrap(value.normalize(p, dim)); }

    @Override public IDiffTensor[] split(int splitSize, int dim) {
        int d = (dim < 0 ? dim + rank() : dim);
        int dimSize = dim(d);
        java.util.ArrayList<Integer> sizes = new java.util.ArrayList<>();
        int remaining = dimSize;
        while (remaining > 0) { sizes.add(Math.min(splitSize, remaining)); remaining -= splitSize; }
        int[] sizeArray = new int[sizes.size()];
        for (int i = 0; i < sizes.size(); i++) sizeArray[i] = sizes.get(i);
        return split(sizeArray, d);
    }
    @Override public IDiffTensor[] split(int[] splitSizes, int dim) {
        int d = (dim < 0 ? dim + rank() : dim);
        int n = splitSizes.length;
        IDiffTensor[] result = new IDiffTensor[n];
        int offset = 0;
        for (int i = 0; i < n; i++) {
            result[i] = narrow(d, offset, splitSizes[i]);
            offset += splitSizes[i];
        }
        return result;
    }
    @Override public IDiffTensor[] chunk(int chunks, int dim) {
        int d = (dim < 0 ? dim + rank() : dim);
        int dimSize = dim(d);
        int chunkSize = (dimSize + chunks - 1) / chunks;
        java.util.ArrayList<Integer> sizes = new java.util.ArrayList<>();
        int remaining = dimSize;
        for (int i = 0; i < chunks && remaining > 0; i++) {
            sizes.add(Math.min(chunkSize, remaining));
            remaining -= chunkSize;
        }
        int[] sizeArray = new int[sizes.size()];
        for (int i = 0; i < sizes.size(); i++) sizeArray[i] = sizes.get(i);
        return split(sizeArray, d);
    }
    @Override public IDiffTensor[] unbind(int dim) {
        int d = (dim < 0 ? dim + rank() : dim);
        int dimSize = dim(d);
        IDiffTensor[] result = new IDiffTensor[dimSize];
        for (int i = 0; i < dimSize; i++) {
            result[i] = narrow(d, i, 1).squeeze(d);
        }
        return result;
    }

    // In-place ops
    @Override public IDiffTensor add_(IDoubleTensor other) { value.add_(detachOther(other)); return this; }
    @Override public IDiffTensor sub_(IDoubleTensor other) { value.sub_(detachOther(other)); return this; }
    @Override public IDiffTensor mul_(IDoubleTensor other) { value.mul_(detachOther(other)); return this; }
    @Override public IDiffTensor div_(IDoubleTensor other) { value.div_(detachOther(other)); return this; }
    @Override public IDiffTensor fill_(double val) { value.fill(val); return this; }
    @Override public IDiffTensor copy_(IDoubleTensor src) { value.copy_(detachOther(src)); return this; }

    // Non-diff — full reductions
    @Override public double sumAll() { return value.sumAll(); }
    @Override public double meanAll() { return value.meanAll(); }
    @Override public double maxAll() { return value.maxAll(); }
    @Override public double minAll() { return value.minAll(); }
    @Override public double prodAll() { return value.prodAll(); }

    // Gradient ops: all no-ops
    @Override public void backward() {}
    @Override public void backward(IDoubleTensor gradient) {}
    @Override public void zeroGradient() {}
    @Override public void clipGradNorm(double maxNorm) {}
    @Override public void clipGradValue(double maxValue) {}
    @Override public IDiffVector flattenGrad() { return null; }
    @Override public IDiffVector flattenValue() {
        double[] arr = value.toDoubleArray();
        return new RereDiffVector(arr);
    }
    @Override public IDoubleTensor detach() { return new RereDoubleTensor(value.toDoubleArray(), shape()); }
    @Override public boolean requiresGrad() { return false; }
    @Override public IDiffTensor setRequiresGrad(boolean rg) { return this; }
    @Override public IDoubleTensor grad() { return null; }

    @Override
    public IDiffTensor sum() {
        double total = value.sumAll();
        return new ConstantDiffTensor(new RereDoubleTensor(new double[]{total}, new int[]{1}));
    }

    private static IDiffTensor wrap(IDoubleTensor t) {
        if (t instanceof RereDoubleTensor rdt) return new ConstantDiffTensor(rdt);
        return new ConstantDiffTensor(new RereDoubleTensor(t.toDoubleArray(), t.shape()));
    }
    private static IDoubleTensor detachOther(IDoubleTensor t) {
        return (t instanceof IDiffTensor dt) ? dt.detach() : t;
    }
}
