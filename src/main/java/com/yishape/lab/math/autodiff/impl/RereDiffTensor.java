package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.ITensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;

import com.yishape.lab.math.linalg.tensor.TensorShape;
import com.yishape.lab.math.compute.FlatGemm;

/**
 * 可微张量实现：通过组合 RereDoubleTensor（数据）+ IDiffVector（梯度追踪）实现.
 */
public class RereDiffTensor implements IDiffTensor {

    private IDiffVector vec;
    private RereDoubleTensor tensor;
    private boolean requiresGrad = true;

    // ==================== 构造 ====================

    public RereDiffTensor(IDiffVector vec, int... shape) {
        this.vec = vec;
        this.tensor = new RereDoubleTensor(vec.getValue().toDoubleArray(), shape);
    }

    /** 视图构造 */
    private RereDiffTensor(IDiffVector vec, RereDoubleTensor view) {
        this.vec = vec;
        this.tensor = view;
    }

    // ==================== Shape 查询（委托给 tensor） ====================

    @Override public int rank() { return tensor.rank(); }
    @Override public int[] shape() { return tensor.shape(); }
    @Override public int dim(int axis) { return tensor.dim(axis); }
    @Override public long totalSize() { return tensor.totalSize(); }
    @Override public int[] strides() { return tensor.strides(); }
    @Override public int stride(int axis) { return tensor.stride(axis); }
    @Override public int offset() { return tensor.offset(); }
    @Override public boolean isContiguous() { return tensor.isContiguous(); }
    @Override public double item() { return tensor.item(); }
    @Override public double get(int... indices) { return tensor.get(indices); }
    @Override public ITensor set(double value, int... indices) { return tensor.set(value, indices); }
    @Override public ITensor fill(double value) { return tensor.fill(value); }
    @Override public IDoubleTensor copy() { return tensor.copy(); }
    @Override public double[] toDoubleArray() { return tensor.toDoubleArray(); }

    // ==================== 梯度方法 ====================

    @Override
    public void backward() {
        if (!requiresGrad) return;
        vec.backward();
    }

    @Override
    public void backward(IDoubleTensor gradient) {
        if (!requiresGrad) return;
        IDoubleVector gv = com.yishape.lab.math.linalg.IDoubleVector.of(gradient.toDoubleArray());
        vec.backward(gv);
    }

    @Override
    public void zeroGradient() {
        vec.zeroGradient();
    }

    @Override
    public IDiffVector flattenGrad() {
        IDoubleVector g = vec.getGradient();
        if (g == null) return null;
        return AD.vector(g.toDoubleArray());
    }

    @Override
    public IDiffVector flattenValue() {
        // For contiguous tensors, return the original vec to preserve the autodiff graph.
        // For non-contiguous views (e.g. after permute/slice), contiguous() materializes
        // the data in logical order while preserving the AD graph, then flattenValue()
        // on the contiguous result returns its vec directly.
        if (tensor.isContiguous() && tensor.totalSize() == vec.getValue().size()) {
            return vec;
        }
        return this.contiguous().flattenValue();
    }

    @Override
    public IDoubleTensor detach() {
        return tensor.clone();
    }

    @Override
    public boolean requiresGrad() { return requiresGrad; }

    @Override
    public IDiffTensor setRequiresGrad(boolean requiresGrad) {
        this.requiresGrad = requiresGrad;
        if (!requiresGrad && vec.isLeaf()) {
            this.vec.zeroGradient();
        }
        return this;
    }

    @Override
    public IDoubleTensor grad() {
        IDoubleVector gradVal = vec.getGradient();
        if (gradVal == null) return null;
        return new RereDoubleTensor(gradVal.toDoubleArray(), shape());
    }

    // ==================== 视图操作 ====================

    @Override
    public IDiffTensor permute(int... dims) {
        return new RereDiffTensor(vec, (RereDoubleTensor) tensor.permute(dims));
    }

    @Override
    public IDiffTensor transpose(int dim0, int dim1) {
        return new RereDiffTensor(vec, (RereDoubleTensor) tensor.transpose(dim0, dim1));
    }

    @Override
    public IDiffTensor transpose() {
        return new RereDiffTensor(vec, (RereDoubleTensor) tensor.transpose());
    }

    @Override
    public IDiffTensor squeeze(int... dims) {
        return new RereDiffTensor(vec, (RereDoubleTensor) tensor.squeeze(dims));
    }

    @Override
    public IDiffTensor unsqueeze(int dim) {
        return new RereDiffTensor(vec, (RereDoubleTensor) tensor.unsqueeze(dim));
    }

    @Override
    public IDiffTensor slice(int dim, long start, long end) {
        return new RereDiffTensor(vec, (RereDoubleTensor) tensor.slice(dim, start, end));
    }

    @Override
    public IDiffTensor narrow(int dim, long start, long length) {
        return new RereDiffTensor(vec, (RereDoubleTensor) tensor.narrow(dim, start, length));
    }

    @Override
    public IDiffTensor select(int dim, long index) {
        return new RereDiffTensor(vec, (RereDoubleTensor) tensor.select(dim, index));
    }

    @Override
    public IDiffTensor expand(int... shape) {
        return new RereDiffTensor(vec, (RereDoubleTensor) tensor.expand(shape));
    }

    @Override
    public IDiffTensor flatten(int startDim, int endDim) {
        return new RereDiffTensor(vec, (RereDoubleTensor) tensor.flatten(startDim, endDim));
    }

    @Override
    public IDiffTensor contiguous() {
        if (tensor.isContiguous()) return this;
        if (!requiresGrad) return new RereDiffTensor(AD.vector(toDoubleArray()), shape());

        int[] s = shape();
        int[] st = strides();
        int off = offset();
        int total = (int) totalSize();
        double[] contigData = toDoubleArray();

        // Compute source storage index for each contiguous position
        int[] srcIdx = new int[total];
        for (int flat = 0; flat < total; flat++) {
            int remaining = flat;
            int storagePos = off;
            for (int j = s.length - 1; j >= 0; j--) {
                storagePos += (remaining % s[j]) * st[j];
                remaining /= s[j];
            }
            srcIdx[flat] = storagePos;
        }

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int[] fSrcIdx = srcIdx;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dx = new double[fSrcIdx.length];
            for (int i = 0; i < fSrcIdx.length; i++) {
                dx[fSrcIdx[i]] = g[i];
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(contigData), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, s);
    }

    @Override
    public IDiffTensor reshape(int... newShape) {
        if (isContiguous() && offset() == 0) {
            return new RereDiffTensor(vec, (RereDoubleTensor) tensor.reshape(newShape));
        }
        // Non-contiguous (e.g. after permute): contiguous() preserves the AD graph,
        // then reshape on the contiguous result is a zero-copy view.
        return this.contiguous().reshape(newShape);
    }

    @Override
    public IDiffTensor tile(int... repeats) {
        int rank = rank();
        int[] rep = new int[rank];
        for (int i = 0; i < rank; i++) {
            rep[i] = (i < repeats.length) ? repeats[i] : 1;
        }

        int[] inShape = shape();
        int[] outShape = new int[rank];
        for (int i = 0; i < rank; i++) outShape[i] = inShape[i] * rep[i];

        if (!requiresGrad) {
            return toNonDiff(tensor.tile(repeats));
        }

        double[] inData = tensor.toDoubleArray();
        int outTotal = 1;
        for (int d : outShape) outTotal *= d;
        double[] outData = new double[outTotal];

        // Forward: tile by copying input data with modular indexing
        for (int flat = 0; flat < outTotal; flat++) {
            int remaining = flat;
            int inFlat = 0;
            int inStride = 1;
            for (int j = rank - 1; j >= 0; j--) {
                int coord = remaining % outShape[j];
                remaining /= outShape[j];
                inFlat += (coord % inShape[j]) * inStride;
                inStride *= inShape[j];
            }
            outData[flat] = inData[inFlat];
        }

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int fRank = rank;
        int[] fInShape = inShape;
        int[] fOutShape = outShape;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            int inTotal = 1;
            for (int d : fInShape) inTotal *= d;
            double[] dx = new double[inTotal];
            // Sum-reduce: for each output position, accumulate gradient into the
            // corresponding input position (via modular indexing)
            for (int flat = 0; flat < g.length; flat++) {
                int remaining = flat;
                int inFlat = 0;
                int inStride = 1;
                for (int j = fRank - 1; j >= 0; j--) {
                    int coord = remaining % fOutShape[j];
                    remaining /= fOutShape[j];
                    inFlat += (coord % fInShape[j]) * inStride;
                    inStride *= fInShape[j];
                }
                dx[inFlat] += g[flat];
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(outData), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, outShape);
    }

    @Override
    public IDiffTensor broadcastTo(int... targetShape) {
        return new RereDiffTensor(vec,
            (RereDoubleTensor) tensor.broadcastTo(targetShape));
    }

    // ==================== 逐元素运算 ====================

    @Override
    public IDiffTensor add(IDoubleTensor other) {
        return binaryDiffOp(other, IDiffVector::add, (a, b) -> a + b);
    }

    @Override
    public IDiffTensor sub(IDoubleTensor other) {
        return binaryDiffOp(other, IDiffVector::sub, (a, b) -> a - b);
    }

    @Override
    public IDiffTensor mul(IDoubleTensor other) {
        return binaryDiffOp(other, IDiffVector::mul, (a, b) -> a * b);
    }

    @Override
    public IDiffTensor div(IDoubleTensor other) {
        return binaryDiffOp(other, IDiffVector::div, (a, b) -> a / b);
    }

    @Override
    public IDiffTensor add(double scalar) {
        if (!requiresGrad) return toNonDiff(tensor.add(scalar));
        return new RereDiffTensor(vec.add(scalar), shape());
    }

    @Override
    public IDiffTensor sub(double scalar) {
        if (!requiresGrad) return toNonDiff(tensor.sub(scalar));
        return new RereDiffTensor(vec.sub(scalar), shape());
    }

    @Override
    public IDiffTensor mul(double scalar) {
        if (!requiresGrad) return toNonDiff(tensor.mul(scalar));
        return new RereDiffTensor(vec.mul(scalar), shape());
    }

    @Override
    public IDiffTensor div(double scalar) {
        if (!requiresGrad) return toNonDiff(tensor.div(scalar));
        return new RereDiffTensor(vec.div(scalar), shape());
    }

    @Override
    public IDiffTensor neg() {
        if (!requiresGrad) return toNonDiff(tensor.neg());
        return new RereDiffTensor(vec.neg(), shape());
    }

    @Override
    public IDiffTensor abs() {
        if (!requiresGrad) return toNonDiff(tensor.abs());
        return new RereDiffTensor(vec.abs(), shape());
    }

    @Override
    public IDiffTensor sqrt() {
        if (!requiresGrad) return toNonDiff(tensor.sqrt());
        return new RereDiffTensor(vec.sqrt(), shape());
    }

    @Override
    public IDiffTensor exp() {
        if (!requiresGrad) return toNonDiff(tensor.exp());
        return new RereDiffTensor(vec.exp(), shape());
    }

    @Override
    public IDiffTensor log() {
        if (!requiresGrad) return toNonDiff(tensor.log());
        return new RereDiffTensor(vec.log(), shape());
    }

    @Override
    public IDiffTensor sin() {
        if (!requiresGrad) return toNonDiff(tensor.sin());
        return new RereDiffTensor(vec.sin(), shape());
    }

    @Override
    public IDiffTensor cos() {
        if (!requiresGrad) return toNonDiff(tensor.cos());
        return new RereDiffTensor(vec.cos(), shape());
    }

    @Override
    public IDiffTensor tan() {
        if (!requiresGrad) return toNonDiff(tensor.tan());
        return new RereDiffTensor(vec.tan(), shape());
    }

    @Override
    public IDiffTensor tanh() {
        if (!requiresGrad) return toNonDiff(computeElementwise(Math::tanh));
        return new RereDiffTensor(vec.tanh(), shape());
    }

    @Override
    public IDiffTensor silu() {
        if (!requiresGrad) return toNonDiff(computeElementwise(x -> x / (1.0 + Math.exp(-x))));
        return new RereDiffTensor(vec.silu(), shape());
    }

    @Override
    public IDiffTensor gelu() {
        if (!requiresGrad) return toNonDiff(computeElementwise(x -> {
            double cdf = 0.5 * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
            return x * cdf;
        }));
        return new RereDiffTensor(vec.gelu(), shape());
    }

    @Override
    public IDiffTensor softplus(double beta) {
        if (!requiresGrad) return toNonDiff(computeElementwise(x -> {
            double bx = beta * x;
            return bx > 20 ? x : Math.log(1.0 + Math.exp(bx)) / beta;
        }));
        return new RereDiffTensor(vec.softplus(beta), shape());
    }

    @Override
    public IDiffTensor mish() {
        if (!requiresGrad) return toNonDiff(computeElementwise(
            x -> x * Math.tanh(Math.log(1.0 + Math.exp(x)))));
        return new RereDiffTensor(vec.mish(), shape());
    }

    @Override
    public IDiffTensor elu(double alpha) {
        if (!requiresGrad) return toNonDiff(computeElementwise(
            x -> x >= 0 ? x : alpha * (Math.exp(x) - 1)));
        return new RereDiffTensor(vec.elu(alpha), shape());
    }

    @Override
    public IDiffTensor leakyRelu(double alpha) {
        if (!requiresGrad) return toNonDiff(computeElementwise(
            x -> x >= 0 ? x : alpha * x));
        return new RereDiffTensor(vec.leakyRelu(alpha), shape());
    }

    @Override
    public IDiffTensor selu() {
        double alpha = 1.6732632423543772, scale = 1.0507009873554804;
        if (!requiresGrad) return toNonDiff(computeElementwise(
            x -> scale * (x >= 0 ? x : alpha * (Math.exp(x) - 1))));
        return new RereDiffTensor(vec.selu(), shape());
    }

    @Override
    public IDiffTensor hardtanh(double minVal, double maxVal) {
        if (!requiresGrad) return toNonDiff(computeElementwise(
            x -> Math.min(Math.max(x, minVal), maxVal)));
        return new RereDiffTensor(vec.hardtanh(minVal, maxVal), shape());
    }

    @Override
    public IDiffTensor dropout(double p) {
        if (!requiresGrad) return toNonDiff(tensor.clone());
        return new RereDiffTensor(vec.dropout(p), shape());
    }

    private IDoubleTensor computeElementwise(java.util.function.DoubleUnaryOperator fn) {
        double[] data = tensor.toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] = fn.applyAsDouble(data[i]);
        return new RereDoubleTensor(data, shape());
    }

    @Override
    public IDiffTensor sigmoid() {
        if (!requiresGrad) return toNonDiff(tensor.sigmoid());
        return new RereDiffTensor(vec.sigmoid(), shape());
    }

    @Override
    public IDiffTensor relu() {
        if (!requiresGrad) return toNonDiff(tensor.relu());
        return new RereDiffTensor(vec.relu(), shape());
    }

    @Override
    public IDiffTensor square() {
        if (!requiresGrad) return toNonDiff(tensor.square());
        return new RereDiffTensor(vec.square(), shape());
    }

    @Override
    public IDiffTensor pow(double n) {
        if (!requiresGrad) return toNonDiff(tensor.pow(n));
        return new RereDiffTensor(vec.pow(n), shape());
    }

    @Override
    public IDiffTensor clamp(double min, double max) {
        if (!requiresGrad) return toNonDiff(tensor.clamp(min, max));
        return new RereDiffTensor(vec.clamp(min, max), shape());
    }

    // ==================== 归约 ====================

    @Override
    public IDiffTensor sum(int dim, boolean keepdim) {
        int d = dim < 0 ? dim + rank() : dim;
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        if (outer == 1 && requiresGrad) {
            // dim=0: gradient-preserving flatSum using vec slices
            int reduce = s[d];
            int inner = 1;
            for (int i = d + 1; i < rank(); i++) inner *= s[i];
            IDiffVector result = vec.slice(0, inner);
            for (int r = 1; r < reduce; r++) {
                result = result.add(vec.slice(r * inner, (r + 1) * inner));
            }
            return new RereDiffTensor(result, reducedShape(d, keepdim));
        }
        if (!requiresGrad) return toNonDiff(tensor.sum(d, keepdim));

        // dim>0: gradient-expansion backward
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];
        double[] resultData = tensor.sum(d, keepdim).toDoubleArray();
        int[] resultShape = reducedShape(d, keepdim);

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int fOuter = outer, fReduce = reduce, fInner = inner;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dx = new double[fOuter * fReduce * fInner];
            // Each reduced-dim element receives the same gradient
            for (int o = 0; o < fOuter; o++) {
                for (int r = 0; r < fReduce; r++) {
                    for (int i = 0; i < fInner; i++) {
                        dx[(o * fReduce + r) * fInner + i] = g[o * fInner + i];
                    }
                }
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(resultData), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, resultShape);
    }

    @Override
    public IDiffTensor mean(int dim, boolean keepdim) {
        int d = dim < 0 ? dim + rank() : dim;
        if (!requiresGrad) return toNonDiff(tensor.mean(d, keepdim));
        IDiffTensor s = sum(d, keepdim);
        return new RereDiffTensor(
            ((RereDiffTensor) s).vec.div(dim(d)),
            ((RereDiffTensor) s).tensor.shape());
    }

    @Override
    public IDiffTensor max(int dim, boolean keepdim) {
        int d = dim < 0 ? dim + rank() : dim;
        double[] vals = tensor.toDoubleArray();
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        double[] result = new double[outer * inner];
        int[] argIdx = new int[outer * inner]; // argmax indices along reduce dim
        Arrays.fill(result, Double.NEGATIVE_INFINITY);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                for (int r = 0; r < reduce; r++) {
                    double v = vals[(o * reduce + r) * inner + i];
                    if (v > result[o * inner + i]) {
                        result[o * inner + i] = v;
                        argIdx[o * inner + i] = r;
                    }
                }
            }
        }

        if (!requiresGrad) return toNonDiff(new RereDoubleTensor(result, reducedShape(d, keepdim)));

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int fOuter = outer, fReduce = reduce, fInner = inner;
        int[] fArgIdx = argIdx;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dx = new double[fOuter * fReduce * fInner];
            for (int o = 0; o < fOuter; o++) {
                for (int i = 0; i < fInner; i++) {
                    int ri = fArgIdx[o * fInner + i];
                    dx[(o * fReduce + ri) * fInner + i] = g[o * fInner + i];
                }
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(result), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, reducedShape(d, keepdim));
    }

    @Override
    public IDiffTensor min(int dim, boolean keepdim) {
        double[] vals = tensor.toDoubleArray();
        int[] s = shape();
        int d = dim < 0 ? dim + rank() : dim;
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        double[] result = new double[outer * inner];
        int[] argIdx = new int[outer * inner]; // argmin indices along reduce dim
        Arrays.fill(result, Double.POSITIVE_INFINITY);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                for (int r = 0; r < reduce; r++) {
                    double v = vals[(o * reduce + r) * inner + i];
                    if (v < result[o * inner + i]) {
                        result[o * inner + i] = v;
                        argIdx[o * inner + i] = r;
                    }
                }
            }
        }

        if (!requiresGrad) return toNonDiff(new RereDoubleTensor(result, reducedShape(d, keepdim)));

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int fOuter = outer, fReduce = reduce, fInner = inner;
        int[] fArgIdx = argIdx;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dx = new double[fOuter * fReduce * fInner];
            for (int o = 0; o < fOuter; o++) {
                for (int i = 0; i < fInner; i++) {
                    int ri = fArgIdx[o * fInner + i];
                    dx[(o * fReduce + ri) * fInner + i] = g[o * fInner + i];
                }
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(result), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, reducedShape(d, keepdim));
    }

    @Override
    public IDiffTensor prod(int dim, boolean keepdim) {
        int d = dim < 0 ? dim + rank() : dim;
        double[] vals = tensor.toDoubleArray();
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        double[] result = new double[outer * inner];
        Arrays.fill(result, 1.0);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                for (int r = 0; r < reduce; r++) {
                    result[o * inner + i] *= vals[(o * reduce + r) * inner + i];
                }
            }
        }

        if (!requiresGrad) return toNonDiff(new RereDoubleTensor(result, reducedShape(d, keepdim)));

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int fOuter = outer, fReduce = reduce, fInner = inner;
        double[] savedVals = vals;
        double[] savedResult = result;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dx = new double[fOuter * fReduce * fInner];
            // d(prod)/dx_i = prod / x_i = (prod of all other elements in the reduce group)
            for (int o = 0; o < fOuter; o++) {
                for (int i = 0; i < fInner; i++) {
                    double prodVal = savedResult[o * fInner + i];
                    double gi = g[o * fInner + i];
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + i;
                        double xi = savedVals[idx];
                        // prod / x_i (handles x_i == 0 safely when grad is 0)
                        dx[idx] = (xi != 0.0) ? gi * prodVal / xi : 0.0;
                    }
                }
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(result), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, reducedShape(d, keepdim));
    }

    private int[] reducedShape(int dim, boolean keepdim) {
        if (keepdim) {
            int[] r = shape().clone();
            r[dim] = 1;
            return r;
        }
        if (rank() == 1) {
            return new int[]{1}; // scalar -> 1-element tensor
        }
        int[] r = new int[rank() - 1];
        int idx = 0;
        for (int i = 0; i < rank(); i++) {
            if (i != dim) r[idx++] = dim(i);
        }
        return r;
    }

    // ==================== 全量归约 ====================

    @Override
    public double sumAll() { return tensor.sumAll(); }
    @Override
    public double meanAll() { return tensor.meanAll(); }
    @Override
    public double maxAll() { return tensor.maxAll(); }
    @Override
    public double minAll() { return tensor.minAll(); }
    @Override
    public double prodAll() { return tensor.prodAll(); }

    @Override
    public IDiffTensor cumsum(int dim) {
        int d = dim < 0 ? dim + rank() : dim;
        if (!requiresGrad) return toNonDiff(tensor.cumsum(d));

        double[] vals = tensor.toDoubleArray();
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

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

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int fOuter = outer, fReduce = reduce, fInner = inner;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dx = new double[fOuter * fReduce * fInner];
            // backward of cumsum is reverse cumsum: dx[r] = sum_{j>=r} g[j]
            for (int o = 0; o < fOuter; o++) {
                for (int i = 0; i < fInner; i++) {
                    double cum = 0;
                    for (int r = fReduce - 1; r >= 0; r--) {
                        int idx = (o * fReduce + r) * fInner + i;
                        cum += g[idx];
                        dx[idx] = cum;
                    }
                }
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(result), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, shape());
    }

    @Override
    public IDiffTensor cumprod(int dim) {
        int d = dim < 0 ? dim + rank() : dim;
        if (!requiresGrad) return toNonDiff(tensor.cumprod(d));

        double[] vals = tensor.toDoubleArray();
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

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

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int fOuter = outer, fReduce = reduce, fInner = inner;
        double[] savedVals = vals;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dx = new double[fOuter * fReduce * fInner];
            // cumprod backward: reverse cumulative sum of (g * result / x) for each slice
            for (int o = 0; o < fOuter; o++) {
                for (int i = 0; i < fInner; i++) {
                    // Compute forward cumprod for this slice
                    double[] cp = new double[fReduce];
                    double p = 1;
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + i;
                        p *= savedVals[idx];
                        cp[r] = p;
                    }
                    // Compute g * result / x = g[r] * cp[r] / x[r]
                    double[] q = new double[fReduce];
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + i;
                        double xi = savedVals[idx];
                        q[r] = (xi != 0.0) ? g[idx] * cp[r] / xi : 0.0;
                    }
                    // Reverse cumulative sum
                    double cum = 0;
                    for (int r = fReduce - 1; r >= 0; r--) {
                        cum += q[r];
                        dx[(o * fReduce + r) * fInner + i] = cum;
                    }
                }
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(result), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, shape());
    }

    @Override
    public IDiffTensor argmax(int dim) { return toNonDiff(tensor.argmax(dim)); }

    @Override
    public IDiffTensor argmin(int dim) { return toNonDiff(tensor.argmin(dim)); }

    @Override
    public IDiffTensor std(int dim, boolean keepdim) {
        int d = dim < 0 ? dim + rank() : dim;
        if (!requiresGrad) return toNonDiff(tensor.std(d, keepdim));
        double[] vals = tensor.toDoubleArray();
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

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
        double[] stdData = new double[outer * inner];
        for (int i = 0; i < stdData.length; i++) stdData[i] = Math.sqrt(varData[i]);

        if (!requiresGrad) return toNonDiff(new RereDoubleTensor(stdData, reducedShape(d, keepdim)));

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        double[] fMeans = means;
        int fOuter = outer, fReduce = reduce, fInner = inner;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dx = new double[fOuter * fReduce * fInner];
            for (int o = 0; o < fOuter; o++) {
                for (int i = 0; i < fInner; i++) {
                    int oi = o * fInner + i;
                    double m = fMeans[oi];
                    double st = stdData[oi];
                    if (st > 1e-15) {
                        double scale = g[oi] / (fReduce * st);
                        for (int r = 0; r < fReduce; r++) {
                            dx[(o * fReduce + r) * fInner + i] = scale * (vals[(o * fReduce + r) * fInner + i] - m);
                        }
                    }
                }
            }
            selfVec.accGradDirect(dx);
        };
        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(stdData), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, reducedShape(d, keepdim));
    }

    @Override
    public IDiffTensor var(int dim, boolean keepdim) {
        int d = dim < 0 ? dim + rank() : dim;
        if (!requiresGrad) return toNonDiff(tensor.var(d, keepdim));
        double[] vals = tensor.toDoubleArray();
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

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

        if (!requiresGrad) return toNonDiff(new RereDoubleTensor(varData, reducedShape(d, keepdim)));

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        double[] fMeans = means;
        int fOuter = outer, fReduce = reduce, fInner = inner;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dx = new double[fOuter * fReduce * fInner];
            for (int o = 0; o < fOuter; o++) {
                for (int i = 0; i < fInner; i++) {
                    double m = fMeans[o * fInner + i];
                    double scale = 2.0 * g[o * fInner + i] / fReduce;
                    for (int r = 0; r < fReduce; r++) {
                        dx[(o * fReduce + r) * fInner + i] = scale * (vals[(o * fReduce + r) * fInner + i] - m);
                    }
                }
            }
            selfVec.accGradDirect(dx);
        };
        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(varData), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, reducedShape(d, keepdim));
    }

    // ==================== Softmax ====================

    @Override
    public IDiffTensor softmax(int dim) {
        if (!requiresGrad) return toNonDiff(tensor.softmax(dim));
        int d = dim < 0 ? dim + rank() : dim;
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        // Numerically stable softmax: exp(x - max) / sum(exp(x - max))
        double[] vals = tensor.toDoubleArray();
        double[] maxVals = new double[outer * inner];
        Arrays.fill(maxVals, Double.NEGATIVE_INFINITY);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                for (int r = 0; r < reduce; r++) {
                    double v = vals[(o * reduce + r) * inner + i];
                    if (v > maxVals[o * inner + i]) maxVals[o * inner + i] = v;
                }
            }
        }
        double[] softmaxData = new double[vals.length];
        double[] sumExps = new double[outer * inner];
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double sum = 0;
                for (int r = 0; r < reduce; r++) {
                    int idx = (o * reduce + r) * inner + i;
                    double ex = Math.exp(vals[idx] - maxVals[o * inner + i]);
                    softmaxData[idx] = ex;
                    sum += ex;
                }
                sumExps[o * inner + i] = sum;
            }
        }
        for (int idx = 0; idx < softmaxData.length; idx++) {
            int oi = (idx / (reduce * inner)) * inner + (idx % inner);
            softmaxData[idx] /= sumExps[oi];
        }

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int fOuter = outer, fReduce = reduce, fInner = inner;
        double[] sm = softmaxData.clone();

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dx = new double[fOuter * fReduce * fInner];
            // dx = s * (g - dot(g, s)) for each outer-inner slice
            // dot[o,i] = sum_r(g[o,r,i] * sm[o,r,i])
            for (int o = 0; o < fOuter; o++) {
                for (int i = 0; i < fInner; i++) {
                    double dot = 0;
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + i;
                        dot += g[idx] * sm[idx];
                    }
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + i;
                        dx[idx] = sm[idx] * (g[idx] - dot);
                    }
                }
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(softmaxData), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, shape());
    }

    @Override
    public IDiffTensor logSoftmax(int dim) {
        if (!requiresGrad) return toNonDiff(tensor.logSoftmax(dim));
        int d = dim < 0 ? dim + rank() : dim;
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        double[] vals = tensor.toDoubleArray();
        double[] maxVals = new double[outer * inner];
        Arrays.fill(maxVals, Double.NEGATIVE_INFINITY);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                for (int r = 0; r < reduce; r++) {
                    double v = vals[(o * reduce + r) * inner + i];
                    if (v > maxVals[o * inner + i]) maxVals[o * inner + i] = v;
                }
            }
        }
        double[] logSoftmaxData = new double[vals.length];
        double[] softmaxData = new double[vals.length];
        double[] sumExps = new double[outer * inner];
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double sum = 0;
                for (int r = 0; r < reduce; r++) {
                    int idx = (o * reduce + r) * inner + i;
                    double ex = Math.exp(vals[idx] - maxVals[o * inner + i]);
                    softmaxData[idx] = ex;
                    sum += ex;
                }
                sumExps[o * inner + i] = sum;
            }
        }
        for (int idx = 0; idx < softmaxData.length; idx++) {
            int oi = (idx / (reduce * inner)) * inner + (idx % inner);
            softmaxData[idx] /= sumExps[oi];
            logSoftmaxData[idx] = Math.log(softmaxData[idx]);
        }

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int fOuter = outer, fReduce = reduce, fInner = inner;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dx = new double[fOuter * fReduce * fInner];
            // d(log_softmax) = g - softmax * sum(g, dim) -- same as softmax but without g*sm term
            for (int o = 0; o < fOuter; o++) {
                for (int i = 0; i < fInner; i++) {
                    double gSum = 0;
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + i;
                        gSum += g[idx];
                    }
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + i;
                        dx[idx] = g[idx] - softmaxData[idx] * gSum;
                    }
                }
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(logSoftmaxData), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, shape());
    }

    // ==================== 线性代数 ====================

    @Override
    public IDiffTensor mmul(IDoubleTensor other) {
        if (!(other instanceof IDiffTensor otherDiff)) {
            IDoubleTensor detOther = other instanceof IDiffTensor ?
                ((IDiffTensor) other).detach() : other;
            return toNonDiff(tensor.mmul(detOther));
        }

        // 2D matmul with full gradient: [M,K] @ [K,N] = [M,N]
        if (rank() == 2 && other.rank() == 2 && requiresGrad && otherDiff.requiresGrad()) {
            int M = dim(0), K = dim(1), N = other.dim(1);
            if (K != other.dim(0)) {
                throw new IllegalArgumentException(
                    "mmul: " + M + "x" + K + " @ " + other.dim(0) + "x" + N + " incompatible");
            }
            // Forward
            IDoubleTensor resultRaw = tensor.mmul(otherDiff.detach());
            double[] resultData = resultRaw.toDoubleArray();
            int[] resultShape = resultRaw.shape();
            // Save for backward
            double[] aData = this.tensor.toDoubleArray();
            double[] bData = ((RereDiffTensor) otherDiff).tensor.toDoubleArray();
            RereDiffVector selfVec = (RereDiffVector) this.vec;
            RereDiffVector otherVec = (RereDiffVector) ((RereDiffTensor) otherDiff).vec;
            int fM = M, fK = K, fN = N;

            boolean selfContig = isContiguous();
            boolean otherContig = otherDiff.isContiguous();
            int[] selfStrides = selfContig ? null : strides();
            int[] otherStrides = otherContig ? null : ((RereDiffTensor) otherDiff).strides();

            // Pre-compute transposes for backward (avoid manual loops)
            double[] bT = FlatGemm.flatTranspose(bData, fK, fN);
            double[] aT = FlatGemm.flatTranspose(aData, fM, fK);

            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double[] g = gradOut.getData();
                // dA[M,K] = dC[M,N] @ B^T[N,K]  →  (M,N) @ (N,K) = (M,K)
                double[] dA = FlatGemm.flatMmul(g, fM, fN, bT, fK);
                if (selfContig) {
                    selfVec.accGradDirect(dA);
                } else {
                    double[] dARemap = new double[fM * fK];
                    for (int i = 0; i < fM; i++) {
                        for (int k = 0; k < fK; k++) {
                            int origPos = i * selfStrides[0] + k * selfStrides[1];
                            dARemap[origPos] = dA[i * fK + k];
                        }
                    }
                    selfVec.accGradDirect(dARemap);
                }
                // dB[K,N] = A^T[K,M] @ dC[M,N]  →  (K,M) @ (M,N) = (K,N)
                double[] dB = FlatGemm.flatMmul(aT, fK, fM, g, fN);
                if (otherContig) {
                    otherVec.accGradDirect(dB);
                } else {
                    double[] dBRemap = new double[fK * fN];
                    for (int k = 0; k < fK; k++) {
                        for (int n = 0; n < fN; n++) {
                            int origPos = k * otherStrides[0] + n * otherStrides[1];
                            dBRemap[origPos] = dB[k * fN + n];
                        }
                    }
                    otherVec.accGradDirect(dBRemap);
                }
            };

            IDiffVector resultVec = new RereDiffVector(
                IDoubleVector.of(resultData), List.of(selfVec, otherVec), backwardFn);
            return new RereDiffTensor(resultVec, resultShape);
        }

        // Fallback: no gradient tracking
        if (!requiresGrad) {
            IDoubleTensor detOther = other instanceof IDiffTensor ?
                ((IDiffTensor) other).detach() : other;
            return toNonDiff(tensor.mmul(detOther));
        }
        return toNonDiff(tensor.mmul(otherDiff.detach()));
    }

    @Override
    public IDiffTensor bmm(IDoubleTensor other) {
        if (!(other instanceof IDiffTensor otherDiff) || !requiresGrad || !otherDiff.requiresGrad()) {
            IDoubleTensor detOther = other instanceof IDiffTensor ?
                ((IDiffTensor) other).detach() : other;
            return toNonDiff(tensor.bmm(detOther));
        }
        if (rank() != 3 || other.rank() != 3) {
            return toNonDiff(tensor.bmm(otherDiff.detach()));
        }
        int B = dim(0), M = dim(1), K = dim(2);
        int B2 = other.dim(0), K2 = other.dim(1), N = other.dim(2);
        if (B != B2 || K != K2) {
            throw new IllegalArgumentException(
                "bmm: [" + B + "," + M + "," + K + "] @ [" + B2 + "," + K2 + "," + N + "] incompatible");
        }

        IDoubleTensor resultRaw = tensor.bmm(otherDiff.detach());
        int[] resultShape = resultRaw.shape();
        double[] aData = this.tensor.toDoubleArray();
        double[] bData = ((RereDiffTensor) otherDiff).tensor.toDoubleArray();
        RereDiffVector selfVec = (RereDiffVector) this.vec;
        RereDiffVector otherVec = (RereDiffVector) ((RereDiffTensor) otherDiff).vec;
        int fB = B, fM = M, fK = K, fN = N;

        // Pre-compute per-batch transposes for backward
        double[][] bT_slices = new double[fB][];
        double[][] aT_slices = new double[fB][];
        for (int bi = 0; bi < fB; bi++) {
            int aOff = bi * fM * fK, bOff = bi * fK * fN;
            double[] bSlice = Arrays.copyOfRange(bData, bOff, bOff + fK * fN);
            double[] aSlice = Arrays.copyOfRange(aData, aOff, aOff + fM * fK);
            bT_slices[bi] = FlatGemm.flatTranspose(bSlice, fK, fN);
            aT_slices[bi] = FlatGemm.flatTranspose(aSlice, fM, fK);
        }

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            int aStride = fM * fK, bStride = fK * fN, gStride = fM * fN;
            double[] dA = new double[fB * aStride];
            double[] dB = new double[fB * bStride];
            for (int bi = 0; bi < fB; bi++) {
                int aOff = bi * aStride, bOff = bi * bStride, gOff = bi * gStride;
                // dA[M,K] = g[M,N] @ B^T[N,K]
                double[] dASlice = FlatGemm.flatMmul(g, gOff, fM, fN, bT_slices[bi], 0, fK);
                System.arraycopy(dASlice, 0, dA, aOff, aStride);
                // dB[K,N] = A^T[K,M] @ g[M,N]
                double[] dBSlice = FlatGemm.flatMmul(aT_slices[bi], 0, fK, fM, g, gOff, fN);
                System.arraycopy(dBSlice, 0, dB, bOff, bStride);
            }
            selfVec.accGradDirect(dA);
            otherVec.accGradDirect(dB);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(resultRaw.toDoubleArray()), List.of(selfVec, otherVec), backwardFn);
        return new RereDiffTensor(resultVec, resultShape);
    }

    @Override
    public IDiffTensor einsum(String subscript, IDoubleTensor... others) {
        throw new UnsupportedOperationException(
            "einsum() does not support automatic differentiation. "
            + "Use Einstein summation only in non-training contexts.");
    }

    // ==================== 高级操作（非可微） ====================

    @Override
    public IDiffTensor gather(int dim, IDoubleTensor index) {
        if (!requiresGrad) return toNonDiff(tensor.gather(dim, index));
        int d = dim < 0 ? dim + rank() : dim;
        int[] s = shape();
        int r = rank();
        int idxRank = index.rank();
        int[] idxShape = index.shape();
        int trailingRank = r - d - 1;
        int[] resultShape = new int[idxRank + trailingRank];
        System.arraycopy(idxShape, 0, resultShape, 0, idxRank);
        for (int i = 0; i < trailingRank; i++) resultShape[idxRank + i] = s[d + 1 + i];

        int resultTotal = (int) computeSize(resultShape);
        double[] resultData = new double[resultTotal];
        double[] inData = tensor.toDoubleArray();
        // Save gather indices for backward
        int[] gatherIndices = new int[resultTotal];

        for (int i = 0; i < resultTotal; i++) {
            int[] outIdx = unlinearizeInt(i, resultShape);
            int[] idxIdx = new int[idxRank];
            System.arraycopy(outIdx, 0, idxIdx, 0, idxRank);
            int gatherIdx = (int) index.get(idxIdx);
            gatherIndices[i] = gatherIdx;
            // Build source index
            int[] srcIdx = new int[r];
            for (int j = 0; j < d; j++) srcIdx[j] = outIdx[j];
            srcIdx[d] = gatherIdx;
            for (int j = 0; j < trailingRank; j++) srcIdx[d + 1 + j] = outIdx[idxRank + j];
            resultData[i] = inData[flatIndex(srcIdx, s)];
        }

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int fD = d;
        int fIdxRank = idxRank;
        int fTrailingRank = trailingRank;
        int[] fSrcShape = s;
        int[] fResultShape = resultShape;
        int[] fGatherIndices = gatherIndices;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            int srcTotal = 1;
            for (int dd : fSrcShape) srcTotal *= dd;
            double[] dx = new double[srcTotal];
            for (int i = 0; i < g.length; i++) {
                int[] outIdx = unlinearizeInt(i, fResultShape);
                int[] srcIdx = new int[fSrcShape.length];
                for (int j = 0; j < fD; j++) srcIdx[j] = outIdx[j];
                srcIdx[fD] = fGatherIndices[i];
                for (int j = 0; j < fTrailingRank; j++) srcIdx[fD + 1 + j] = outIdx[fIdxRank + j];
                dx[flatIndex(srcIdx, fSrcShape)] += g[i];
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(resultData), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, resultShape);
    }

    @Override
    public IDiffTensor indexSelect(int dim, IDoubleTensor index) {
        return gather(dim, index);
    }

    @Override
    public IDiffTensor argsort(int dim, boolean descending) {
        int d = dim < 0 ? dim + rank() : dim;
        int[] s = shape();
        int r = rank();
        int dimSize = s[d];
        int outerTotal = 1;
        for (int i = 0; i < d; i++) outerTotal *= s[i];
        int innerTotal = 1;
        for (int i = d + 1; i < r; i++) innerTotal *= s[i];

        double[] inData = tensor.toDoubleArray();
        double[] outData = new double[inData.length];

        for (int outer = 0; outer < outerTotal; outer++) {
            for (int inner = 0; inner < innerTotal; inner++) {
                // Collect (value, originalIndex) pairs for this slice
                int sliceBase = (outer * dimSize + 0) * innerTotal + inner;
                double[] sliceVals = new double[dimSize];
                int[] origIdx = new int[dimSize];
                for (int i = 0; i < dimSize; i++) {
                    sliceVals[i] = inData[(outer * dimSize + i) * innerTotal + inner];
                    origIdx[i] = i;
                }
                // Simple insertion sort on indices by value
                for (int i = 1; i < dimSize; i++) {
                    int keyIdx = origIdx[i];
                    double keyVal = sliceVals[i];
                    int j = i - 1;
                    while (j >= 0 && (descending ? sliceVals[j] < keyVal : sliceVals[j] > keyVal)) {
                        origIdx[j + 1] = origIdx[j];
                        sliceVals[j + 1] = sliceVals[j];
                        j--;
                    }
                    origIdx[j + 1] = keyIdx;
                    sliceVals[j + 1] = keyVal;
                }
                // Write sorted indices to output
                for (int i = 0; i < dimSize; i++) {
                    outData[(outer * dimSize + i) * innerTotal + inner] = origIdx[i];
                }
            }
        }

        RereDiffVector resultVec = new RereDiffVector(IDoubleVector.of(outData));
        return new RereDiffTensor(resultVec, s);
    }

    @Override
    public IDiffTensor scatter(int dim, IDoubleTensor index, IDoubleTensor source) {
        if (!requiresGrad) return toNonDiff(tensor.scatter(dim, index,
            source instanceof IDiffTensor ? ((IDiffTensor) source).detach() : source));
        int d = dim < 0 ? dim + rank() : dim;
        int[] resultShape = shape();
        int r = rank();

        // Forward: start from this tensor, overwrite with source at scatter positions
        double[] resultData = tensor.toDoubleArray().clone();
        double[] srcData = source instanceof IDiffTensor ?
            ((RereDiffTensor) source).tensor.toDoubleArray() : source.toDoubleArray();
        int[] srcShape = source.shape();
        int idxRank = index.rank();
        int[] idxShape = index.shape();
        int idxTotal = (int) computeSize(idxShape);

        // Save scatter index mapping for backward
        int[] scatterSrcFlat = new int[idxTotal]; // flat index in source
        int[] scatterTgtFlat = new int[idxTotal]; // flat index in result

        for (int i = 0; i < idxTotal; i++) {
            int[] idx = unlinearizeInt(i, idxShape);
            int scatterIdx = (int) index.get(idx);
            int[] tgtIdx = new int[r];
            for (int j = 0; j < r; j++) tgtIdx[j] = j == d ? scatterIdx : idx[j];
            resultData[flatIndex(tgtIdx, resultShape)] = srcData[flatIndex(idx, srcShape)];
            scatterSrcFlat[i] = flatIndex(idx, srcShape);
            scatterTgtFlat[i] = flatIndex(tgtIdx, resultShape);
        }

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        ArrayList<RereDiffVector> allVecs = new ArrayList<>();
        allVecs.add(selfVec);

        RereDiffVector srcVec = null;
        if (source instanceof IDiffTensor srcDiff && srcDiff.requiresGrad()) {
            srcVec = (RereDiffVector) ((RereDiffTensor) source).vec;
            allVecs.add(srcVec);
        }
        RereDiffVector fSrcVec = srcVec;
        int[] fScatterSrcFlat = scatterSrcFlat;
        int[] fScatterTgtFlat = scatterTgtFlat;
        int fIdxTotal = idxTotal;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            // Gradient for this: grad where not overwritten, 0 where overwritten
            // We know which positions were overwritten (scatterTgtFlat)
            double[] dxSelf = g.clone();
            // Zero out positions that were overwritten by scatter (they got source values, not this values)
            for (int i = 0; i < fIdxTotal; i++) {
                dxSelf[fScatterTgtFlat[i]] = 0.0;
            }
            selfVec.accGradDirect(dxSelf);

            // Gradient for source: gather gradient from the scattered positions
            if (fSrcVec != null) {
                int srcTotal = 1;
                for (int dd : srcShape) srcTotal *= dd;
                double[] dxSrc = new double[srcTotal];
                for (int i = 0; i < fIdxTotal; i++) {
                    dxSrc[fScatterSrcFlat[i]] += g[fScatterTgtFlat[i]];
                }
                fSrcVec.accGradDirect(dxSrc);
            }
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(resultData), allVecs, backwardFn);
        return new RereDiffTensor(resultVec, resultShape);
    }

    @Override
    public IDiffTensor scatterAdd(int dim, IDoubleTensor index, IDoubleTensor source) {
        if (!requiresGrad) return toNonDiff(tensor.scatterAdd(dim, index,
            source instanceof IDiffTensor ? ((IDiffTensor) source).detach() : source));
        int d = dim < 0 ? dim + rank() : dim;
        int[] resultShape = shape();
        int r = rank();

        // Forward: start from this tensor, add source at scatter positions
        double[] resultData = tensor.toDoubleArray().clone();
        double[] srcData = source instanceof IDiffTensor ?
            ((RereDiffTensor) source).tensor.toDoubleArray() : source.toDoubleArray();
        int[] srcShape = source.shape();
        int idxRank = index.rank();
        int[] idxShape = index.shape();
        int idxTotal = (int) computeSize(idxShape);

        int[] scatterSrcFlat = new int[idxTotal];
        int[] scatterTgtFlat = new int[idxTotal];

        for (int i = 0; i < idxTotal; i++) {
            int[] idx = unlinearizeInt(i, idxShape);
            int scatterIdx = (int) index.get(idx);
            int[] tgtIdx = new int[r];
            for (int j = 0; j < r; j++) tgtIdx[j] = j == d ? scatterIdx : idx[j];
            int tgtFlat = flatIndex(tgtIdx, resultShape);
            resultData[tgtFlat] += srcData[flatIndex(idx, srcShape)];
            scatterSrcFlat[i] = flatIndex(idx, srcShape);
            scatterTgtFlat[i] = tgtFlat;
        }

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        ArrayList<RereDiffVector> allVecs = new ArrayList<>();
        allVecs.add(selfVec);

        RereDiffVector srcVec = null;
        if (source instanceof IDiffTensor srcDiff && srcDiff.requiresGrad()) {
            srcVec = (RereDiffVector) ((RereDiffTensor) source).vec;
            allVecs.add(srcVec);
        }
        RereDiffVector fSrcVec = srcVec;
        int[] fScatterSrcFlat = scatterSrcFlat;
        int[] fScatterTgtFlat = scatterTgtFlat;
        int fIdxTotal = idxTotal;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            // Gradient for this: pass through (scatterAdd adds to existing values)
            selfVec.accGradDirect(g.clone());

            // Gradient for source: gather gradient from the target positions
            if (fSrcVec != null) {
                int srcTotal = 1;
                for (int dd : srcShape) srcTotal *= dd;
                double[] dxSrc = new double[srcTotal];
                for (int i = 0; i < fIdxTotal; i++) {
                    dxSrc[fScatterSrcFlat[i]] += g[fScatterTgtFlat[i]];
                }
                fSrcVec.accGradDirect(dxSrc);
            }
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(resultData), allVecs, backwardFn);
        return new RereDiffTensor(resultVec, resultShape);
    }

    @Override
    public IDiffTensor where(IDoubleTensor condition, IDoubleTensor other) {
        if (!requiresGrad) return toNonDiff(tensor.where(condition,
            other instanceof IDiffTensor ? ((IDiffTensor) other).detach() : other));

        IDoubleTensor condContig = condition instanceof IDiffTensor ?
            ((IDiffTensor) condition).detach() : condition;
        IDoubleTensor detOther = other instanceof IDiffTensor ?
            ((IDiffTensor) other).detach() : other;

        int[] sSelf = shape();
        int[] sOther = detOther.shape();
        int[] sCond = condContig.shape();
        int[] resultShape = TensorShape.broadcastShape(sSelf, TensorShape.broadcastShape(sOther, sCond));

        long resultTotal = 1;
        for (int d : resultShape) resultTotal *= d;
        double[] resultData = new double[(int) resultTotal];
        boolean[] condMask = new boolean[(int) resultTotal];

        double[] aData = tensor.toDoubleArray();
        double[] bData = detOther.toDoubleArray();
        double[] condData = condContig.toDoubleArray();

        for (int i = 0; i < (int) resultTotal; i++) {
            int[] idx = unlinearizeInt(i, resultShape);
            double condVal = broadcastGetFlat(idx, condData, sCond, resultShape);
            condMask[i] = condVal > 0.5;
            resultData[i] = condMask[i] ?
                broadcastGetFlat(idx, aData, sSelf, resultShape) :
                broadcastGetFlat(idx, bData, sOther, resultShape);
        }

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int[] fResultShape = resultShape;
        int[] fSSelf = sSelf;
        boolean[] fCondMask = condMask;

        ArrayList<RereDiffVector> allVecs = new ArrayList<>();
        allVecs.add(selfVec);

        RereDiffVector otherVec = null;
        if (other instanceof IDiffTensor otherDiff && otherDiff.requiresGrad()) {
            otherVec = (RereDiffVector) ((RereDiffTensor) other).vec;
            allVecs.add(otherVec);
        }
        RereDiffVector fOtherVec = otherVec;
        int[] fSOther = sOther;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dxSelf = new double[(int) computeSize(fSSelf)];
            double[] dxOther = fOtherVec != null ? new double[(int) computeSize(fSOther)] : null;

            for (int i = 0; i < g.length; i++) {
                int[] idx = unlinearizeInt(i, fResultShape);
                if (fCondMask[i]) {
                    int flat = flatIndexFromBroadcast(idx, fSSelf, fResultShape);
                    dxSelf[flat] += g[i];
                } else if (dxOther != null) {
                    int flat = flatIndexFromBroadcast(idx, fSOther, fResultShape);
                    dxOther[flat] += g[i];
                }
            }
            selfVec.accGradDirect(dxSelf);
            if (fOtherVec != null) fOtherVec.accGradDirect(dxOther);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(resultData), allVecs, backwardFn);
        return new RereDiffTensor(resultVec, resultShape);
    }

    @Override
    public IDiffTensor topk(int k, int dim, boolean largest) {
        if (!requiresGrad) return toNonDiff(tensor.topk(k, dim, largest));
        int d = dim < 0 ? dim + rank() : dim;
        int[] s = shape();
        int n = s[d];
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        int[] resultShape = s.clone();
        resultShape[d] = k;
        int resultSize = outer * k * inner;
        double[] resultData = new double[resultSize];
        int[] argIdx = new int[resultSize]; // index along dim of each top-k element

        double[] inData = tensor.toDoubleArray();
        for (int o = 0; o < outer; o++) {
            for (int ii = 0; ii < inner; ii++) {
                // Collect values along dim
                double[] vals = new double[n];
                for (int r = 0; r < n; r++) vals[r] = inData[(o * n + r) * inner + ii];
                // Argsort
                Integer[] idxs = new Integer[n];
                for (int r = 0; r < n; r++) idxs[r] = r;
                final boolean l = largest;
                Arrays.sort(idxs, (a, b) -> l ? Double.compare(vals[b], vals[a]) : Double.compare(vals[a], vals[b]));
                // Write top k
                for (int r = 0; r < k; r++) {
                    int flatIdx = (o * k + r) * inner + ii;
                    resultData[flatIdx] = vals[idxs[r]];
                    argIdx[flatIdx] = idxs[r];
                }
            }
        }

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int fOuter = outer, fReduce = n, fInner = inner, fK = k;
        int[] fArgIdx = argIdx;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dx = new double[fOuter * fReduce * fInner];
            for (int o = 0; o < fOuter; o++) {
                for (int r = 0; r < fK; r++) {
                    for (int ii = 0; ii < fInner; ii++) {
                        int outIdx = (o * fK + r) * fInner + ii;
                        int origR = fArgIdx[outIdx];
                        dx[(o * fReduce + origR) * fInner + ii] += g[outIdx];
                    }
                }
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(resultData), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, resultShape);
    }

    @Override
    public IDiffTensor pad(int[][] padding, String mode, double value) {
        if (!requiresGrad) return toNonDiff(tensor.pad(padding, mode, value));

        int[] s = shape();
        int r = rank();
        int[] resultShape = new int[r];
        for (int i = 0; i < r; i++) resultShape[i] = s[i] + padding[i][0] + padding[i][1];

        long total = 1;
        for (int d : resultShape) total *= d;
        double[] resultData = new double[(int) total];
        Arrays.fill(resultData, value);

        // Copy original data into padded result
        double[] inData = tensor.toDoubleArray();
        for (int i = 0; i < inData.length; i++) {
            int[] srcIdx = unlinearizeInt(i, s);
            int[] tgtIdx = new int[r];
            for (int j = 0; j < r; j++) tgtIdx[j] = srcIdx[j] + padding[j][0];
            resultData[flatIndex(tgtIdx, resultShape)] = inData[i];
        }

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int[] fResultShape = resultShape;
        int[] fOrigShape = s;
        int[][] fPadding = padding;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            int origTotal = 1;
            for (int d : fOrigShape) origTotal *= d;
            double[] dx = new double[origTotal];
            // Slice out the gradient corresponding to the original (unpadded) region
            for (int i = 0; i < origTotal; i++) {
                int[] srcIdx = unlinearizeInt(i, fOrigShape);
                int[] paddedIdx = new int[r];
                for (int j = 0; j < r; j++) paddedIdx[j] = srcIdx[j] + fPadding[j][0];
                dx[i] = g[flatIndex(paddedIdx, fResultShape)];
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(resultData), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, resultShape);
    }

    @Override
    public IDiffTensor unfold(int dim, int size, int stride, int dilation) {
        if (!requiresGrad) return toNonDiff(tensor.unfold(dim, size, stride, dilation));
        int d = dim < 0 ? dim + rank() : dim;
        int[] s = shape();
        int total = (int) totalSize();

        // Compute stride array from shape
        int[] strides = new int[rank()];
        strides[rank() - 1] = 1;
        for (int i = rank() - 2; i >= 0; i--) strides[i] = strides[i + 1] * s[i + 1];

        int dimSize = s[d];
        int numPatches = (dimSize - dilation * (size - 1) - 1) / stride + 1;

        // Build unfolded output manually
        int outerElems = 1;
        for (int i = 0; i < d; i++) outerElems *= s[i];
        int innerElems = 1;
        for (int i = d + 1; i < rank(); i++) innerElems *= s[i];

        double[] vals = tensor.toDoubleArray();
        double[] result = new double[outerElems * numPatches * innerElems * size];

        for (int o = 0; o < outerElems; o++) {
            for (int p = 0; p < numPatches; p++) {
                for (int i = 0; i < innerElems; i++) {
                    for (int k = 0; k < size; k++) {
                        int srcIdx = (o * dimSize + (p * stride + k * dilation)) * innerElems + i;
                        int dstIdx = ((o * numPatches + p) * innerElems + i) * size + k;
                        result[dstIdx] = vals[srcIdx];
                    }
                }
            }
        }

        int[] resultShape = new int[rank() + 1];
        for (int i = 0; i < d; i++) resultShape[i] = s[i];
        resultShape[d] = numPatches;
        for (int i = d + 1; i < rank(); i++) resultShape[i] = s[i];
        resultShape[rank()] = size;

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int fNumPatches = numPatches, fSize = size, fStride = stride, fDilation = dilation;
        int fOuterElems = outerElems, fDimSize = dimSize, fInnerElems = innerElems;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dx = new double[total];
            for (int o = 0; o < fOuterElems; o++) {
                for (int p = 0; p < fNumPatches; p++) {
                    for (int i = 0; i < fInnerElems; i++) {
                        for (int k = 0; k < fSize; k++) {
                            int dstIdx = ((o * fNumPatches + p) * fInnerElems + i) * fSize + k;
                            int srcIdx = (o * fDimSize + (p * fStride + k * fDilation)) * fInnerElems + i;
                            dx[srcIdx] += g[dstIdx];
                        }
                    }
                }
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(result), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, resultShape);
    }

    @Override
    public IDiffTensor nonzero() { return toNonDiff(tensor.nonzero()); }

    @Override
    public IDiffTensor maskedSelect(IDoubleTensor mask) {
        if (!requiresGrad) return toNonDiff(tensor.maskedSelect(mask));

        int[] s = shape();
        int total = (int) totalSize();
        double[] inData = tensor.toDoubleArray();

        // Forward: collect elements where mask > 0.5
        ArrayList<Double> selected = new ArrayList<>();
        ArrayList<Integer> selectedIndices = new ArrayList<>();
        for (int i = 0; i < total; i++) {
            int[] idx = unlinearizeInt(i, s);
            if (mask.get(idx) > 0.5) {
                selected.add(inData[i]);
                selectedIndices.add(i);
            }
        }

        int outLen = selected.size();
        double[] resultData = new double[outLen];
        for (int i = 0; i < outLen; i++) resultData[i] = selected.get(i);

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int[] fSelectedIdx = new int[outLen];
        for (int i = 0; i < outLen; i++) fSelectedIdx[i] = selectedIndices.get(i);

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dx = new double[total];
            for (int i = 0; i < g.length; i++) {
                dx[fSelectedIdx[i]] += g[i];
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(resultData), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, outLen);
    }

    @Override
    public IDiffTensor maskedFill(IDoubleTensor mask, double value) {
        if (!requiresGrad) return toNonDiff(tensor.maskedFill(mask, value));

        int[] s = shape();
        int total = (int) totalSize();
        double[] inData = tensor.toDoubleArray();
        boolean[] maskArr = new boolean[total];
        for (int i = 0; i < total; i++) {
            int[] idx = unlinearizeInt(i, s);
            maskArr[i] = mask.get(idx) > 0.5;
        }

        double[] resultData = new double[total];
        for (int i = 0; i < total; i++) {
            resultData[i] = maskArr[i] ? value : inData[i];
        }

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        boolean[] fMask = maskArr;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dx = new double[g.length];
            for (int i = 0; i < g.length; i++) {
                dx[i] = fMask[i] ? 0.0 : g[i];
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(resultData), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, s);
    }

    @Override
    public IDiffTensor cat(int dim, IDoubleTensor... others) {
        if (!requiresGrad) {
            IDoubleTensor[] detached = new IDoubleTensor[others.length];
            for (int i = 0; i < others.length; i++) {
                detached[i] = others[i] instanceof IDiffTensor ?
                    ((IDiffTensor) others[i]).detach() : others[i];
            }
            return toNonDiff(tensor.cat(dim, detached));
        }
        if (dim < 0) dim += rank();
        int d = dim;
        int[] resultShape = shape().clone();
        // Add all others' dim(d) contributions
        int[] sizes = new int[1 + others.length];
        sizes[0] = dim(d);
        for (int i = 0; i < others.length; i++) {
            sizes[i + 1] = others[i].dim(d);
            resultShape[d] += sizes[i + 1];
        }

        // Forward: compute concatenated data
        long total = 1;
        for (int rs : resultShape) total *= rs;
        int totalSize = (int) total;
        double[] resultData = new double[totalSize];

        // Copy this tensor's data
        int[] shapeA = shape();
        double[] aData = tensor.toDoubleArray();
        int[] idx = new int[rank()];
        for (int flat = 0; flat < aData.length; flat++) {
            int remaining = flat;
            for (int j = rank() - 1; j >= 0; j--) {
                idx[j] = remaining % shapeA[j];
                remaining /= shapeA[j];
            }
            idx[d] = idx[d]; // same coord along dim
            int resFlat = 0;
            int stride = 1;
            for (int j = resultShape.length - 1; j >= 0; j--) {
                resFlat += idx[j] * stride;
                stride *= resultShape[j];
            }
            resultData[resFlat] = aData[flat];
        }

        // Copy each other tensor's data
        int offset = sizes[0];
        for (int ti = 0; ti < others.length; ti++) {
            IDoubleTensor t = others[ti];
            int[] tShape = t.shape();
            double[] tData = t instanceof IDiffTensor ?
                ((RereDiffTensor) t).tensor.toDoubleArray() : t.toDoubleArray();
            for (int flat = 0; flat < tData.length; flat++) {
                int remaining = flat;
                for (int j = tShape.length - 1; j >= 0; j--) {
                    idx[j] = remaining % tShape[j];
                    remaining /= tShape[j];
                }
                idx[d] = idx[d] + offset; // adjust by offset
                int resFlat = 0;
                int stride = 1;
                for (int j = resultShape.length - 1; j >= 0; j--) {
                    resFlat += idx[j] * stride;
                    stride *= resultShape[j];
                }
                resultData[resFlat] = tData[flat];
            }
            offset += sizes[ti + 1];
        }

        // Collect all gradient-tracking vecs and their sizes
        List<RereDiffVector> allVecs = new java.util.ArrayList<>();
        List<int[]> allShapes = new java.util.ArrayList<>();
        List<Integer> allSizes = new java.util.ArrayList<>();
        int[][] allOffsets = new int[1 + others.length][]; // [tensorIdx][dim_coord ranges]

        // This tensor
        allVecs.add((RereDiffVector) this.vec);
        allShapes.add(shape());
        int totalFlat = 1;
        for (int s : shape()) totalFlat *= s;
        allSizes.add(totalFlat);
        int[] cumOffset = new int[sizes.length];
        cumOffset[0] = 0;
        for (int i = 1; i < sizes.length; i++) cumOffset[i] = cumOffset[i-1] + sizes[i-1];

        for (int ti = 0; ti < others.length; ti++) {
            if (others[ti] instanceof IDiffTensor otherDiff && otherDiff.requiresGrad()) {
                allVecs.add((RereDiffVector) ((RereDiffTensor) others[ti]).vec);
                allShapes.add(others[ti].shape());
                int tf = 1;
                for (int s : others[ti].shape()) tf *= s;
                allSizes.add(tf);
            } else {
                allVecs.add(null); // no gradient tracking
                allSizes.add(0);
                allShapes.add(null);
            }
        }

        int fDim = d;
        int[] fSizes = sizes;
        int[] fResultShape = resultShape;
        int[] fCumOffset = cumOffset;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            int totalIn = 1 + others.length;
            for (int ti = 0; ti < totalIn; ti++) {
                if (allVecs.get(ti) == null) continue;
                int[] tShape = allShapes.get(ti);
                int tFlat = allSizes.get(ti);
                int startOffset = fCumOffset[ti];
                int dimSize = fSizes[ti];

                // Extract sub-gradient for input ti using the offset in dim fDim
                double[] subGrad = new double[tFlat];
                for (int flat = 0; flat < tFlat; flat++) {
                    int remaining = flat;
                    for (int j = tShape.length - 1; j >= 0; j--) {
                        idx[j] = remaining % tShape[j];
                        remaining /= tShape[j];
                    }
                    // Map to result index
                    idx[fDim] = idx[fDim] + startOffset;
                    // Compute flat index in result
                    int resFlat = 0;
                    int stride = 1;
                    for (int j = fResultShape.length - 1; j >= 0; j--) {
                        resFlat += idx[j] * stride;
                        stride *= fResultShape[j];
                    }
                    subGrad[flat] = g[resFlat];
                }
                allVecs.get(ti).accGradDirect(subGrad);
            }
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(resultData), allVecs, backwardFn);
        return new RereDiffTensor(resultVec, resultShape);
    }

    @Override
    public IDiffTensor stack(int dim, IDoubleTensor... others) {
        if (!requiresGrad) {
            IDoubleTensor[] detached = new IDoubleTensor[others.length];
            for (int i = 0; i < others.length; i++) {
                detached[i] = (others[i] instanceof IDiffTensor dt) ? dt.detach() : others[i];
            }
            return toNonDiff(tensor.stack(dim, detached));
        }
        // stack = unsqueeze each along dim, then cat along dim
        int d = dim < 0 ? rank() + 1 + dim : dim;
        IDoubleTensor[] all = new IDoubleTensor[1 + others.length];
        all[0] = this.unsqueeze(d);
        for (int i = 0; i < others.length; i++) {
            IDoubleTensor t = others[i];
            if (t instanceof IDiffTensor dt && !dt.requiresGrad()) {
                t = IDiffTensor.constantTensor(dt.detach().toDoubleArray(), dt.shape());
            }
            all[i + 1] = t instanceof IDiffTensor dt2 ? dt2.unsqueeze(d) : IDiffTensor.fromTensor(t, false).unsqueeze(d);
        }
        return ((IDiffTensor) all[0]).cat(d, java.util.Arrays.copyOfRange(all, 1, all.length));
    }

    @Override
    public List<IDoubleTensor> unstack(int dim) {
        return tensor.unstack(dim);
    }

    @Override
    public IDiffTensor normalize(double p, int dim) {
        if (!requiresGrad) return toNonDiff(tensor.normalize(p, dim));
        int d = dim < 0 ? dim + rank() : dim;
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        double[] inData = tensor.toDoubleArray();
        double[] resultData = new double[inData.length];
        double[] normVals = new double[outer * inner]; // per-slice norm

        // Forward: compute norm and normalized result
        for (int o = 0; o < outer; o++) {
            for (int ii = 0; ii < inner; ii++) {
                double normP = 0;
                for (int r = 0; r < reduce; r++) {
                    double v = Math.abs(inData[(o * reduce + r) * inner + ii]);
                    normP += Math.pow(v, p);
                }
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

        RereDiffVector selfVec = (RereDiffVector) this.vec;
        int fOuter = outer, fReduce = reduce, fInner = inner;
        double fP = p;
        double[] savedNorms = normVals;
        double[] savedIn = inData;
        double[] savedResult = resultData;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            double[] dx = new double[g.length];
            // d(x/norm)/dx = 1/norm * I - x*norm^(p-2) * |x|^(p-2) * sign(x) / norm^(p+1)
            // For L2 (p=2): d(x/norm)/dx = (I - x*x^T/norm^2) / norm
            for (int o = 0; o < fOuter; o++) {
                for (int ii = 0; ii < fInner; ii++) {
                    double norm = savedNorms[o * fInner + ii];
                    if (norm == 0) continue;
                    // dot product of grad and normalized output along reduce dim
                    double dot = 0;
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + ii;
                        dot += g[idx] * savedResult[idx];
                    }
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + ii;
                        double xi = savedIn[idx];
                        double absXi = Math.abs(xi);
                        if (fP == 2.0) {
                            // L2 special case: dx = (g - x * dot/norm) / norm
                            dx[idx] = (g[idx] - xi * dot / norm) / norm;
                        } else {
                            // General p-norm: dx = (g - sign(x) * |x|^(p-1) * dot / norm^p) / norm
                            double signX = xi >= 0 ? 1.0 : -1.0;
                            dx[idx] = (g[idx] - signX * Math.pow(absXi, fP - 1) * dot / Math.pow(norm, fP)) / norm;
                        }
                    }
                }
            }
            selfVec.accGradDirect(dx);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(resultData), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, s);
    }

    // ==================== 就地操作 ====================

    @Override
    public IDiffTensor add_(IDoubleTensor other) {
        if (other instanceof IDiffTensor && requiresGrad) {
            vec = (RereDiffVector) vec.add(((IDiffTensor) other).flattenValue());
        } else {
            // Non-differentiable: mutate tensor data directly, sync vec
            tensor.add_(other instanceof IDiffTensor ? ((IDiffTensor) other).detach() : other);
            if (requiresGrad) {
                ((RereDiffVector) vec).updateData(tensor.toDoubleArray());
            }
        }
        syncTensor();
        return this;
    }

    @Override
    public IDiffTensor sub_(IDoubleTensor other) {
        if (other instanceof IDiffTensor && requiresGrad) {
            vec = (RereDiffVector) vec.sub(((IDiffTensor) other).flattenValue());
        } else {
            tensor.sub_(other instanceof IDiffTensor ? ((IDiffTensor) other).detach() : other);
            if (requiresGrad) {
                ((RereDiffVector) vec).updateData(tensor.toDoubleArray());
            }
        }
        syncTensor();
        return this;
    }

    @Override
    public IDiffTensor mul_(IDoubleTensor other) {
        if (other instanceof IDiffTensor && requiresGrad) {
            vec = (RereDiffVector) vec.mul(((IDiffTensor) other).flattenValue());
        } else {
            tensor.mul_(other instanceof IDiffTensor ? ((IDiffTensor) other).detach() : other);
            if (requiresGrad) {
                ((RereDiffVector) vec).updateData(tensor.toDoubleArray());
            }
        }
        syncTensor();
        return this;
    }

    @Override
    public IDiffTensor div_(IDoubleTensor other) {
        if (other instanceof IDiffTensor && requiresGrad) {
            vec = (RereDiffVector) vec.div(((IDiffTensor) other).flattenValue());
        } else {
            tensor.div_(other instanceof IDiffTensor ? ((IDiffTensor) other).detach() : other);
            if (requiresGrad) {
                ((RereDiffVector) vec).updateData(tensor.toDoubleArray());
            }
        }
        syncTensor();
        return this;
    }

    @Override
    public IDiffTensor fill_(double value) {
        tensor.fill_(value);
        if (requiresGrad) {
            ((RereDiffVector) vec).updateData(tensor.toDoubleArray());
        }
        return this;
    }

    @Override
    public IDiffTensor copy_(IDoubleTensor src) {
        if (src instanceof IDiffTensor && requiresGrad) {
            tensor.copy_(((IDiffTensor) src).detach());
            ((RereDiffVector) vec).updateData(tensor.toDoubleArray());
        } else {
            tensor.copy_(src instanceof IDiffTensor ? ((IDiffTensor) src).detach() : src);
            if (requiresGrad) {
                ((RereDiffVector) vec).updateData(tensor.toDoubleArray());
            }
        }
        syncTensor();
        return this;
    }

    // ==================== 转换 ====================

    @Override
    public IDoubleVector toVector() { return tensor.toVector(); }
    @Override
    public IDoubleVector toVectorCopy() { return tensor.toVectorCopy(); }
    @Override
    public IMatrix toMatrix() { return tensor.toMatrix(); }
    @Override
    public IDiffTensor clone() {
        if (!requiresGrad) return toNonDiff(tensor.clone());
        RereDiffVector selfVec = (RereDiffVector) this.vec;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            selfVec.accGradDirect(gradOut.getData());
        };
        IDiffVector resultVec = new RereDiffVector(
            this.vec.getValue().copy(), List.of(selfVec), backwardFn);
        return new RereDiffTensor(resultVec, shape());
    }

    @Override
    public String toString() { return tensor.toString(); }

    // ==================== 工具 ====================

    private void syncTensor() {
        this.tensor = new RereDoubleTensor(vec.getValue().toDoubleArray(), shape());
    }

    private IDiffTensor toNonDiff(IDoubleTensor t) {
        return new RereDiffTensor(AD.vector(t.toDoubleArray()), t.shape());
    }

    private IDiffTensor binaryDiffOp(IDoubleTensor other,
                                      java.util.function.BinaryOperator<IDiffVector> op,
                                      java.util.function.DoubleBinaryOperator scalarOp) {
        if (!requiresGrad) {
            IDoubleTensor detOther = other instanceof IDiffTensor ?
                ((IDiffTensor) other).detach() : other;
            IDiffVector selfVec = AD.vector(tensor.toDoubleArray());
            IDiffVector otherVec = AD.vector(detOther.toDoubleArray());
            IDoubleVector result = op.apply(selfVec, otherVec).getValue();
            return toNonDiff(new RereDoubleTensor(result.toDoubleArray(), shape()));
        }
        if (other instanceof IDiffTensor && Arrays.equals(shape(), other.shape())) {
            IDiffVector result = op.apply(vec, ((IDiffTensor) other).flattenValue());
            return new RereDiffTensor(result, shape());
        }
        // Same-shape other is not an IDiffTensor: gradient only for this
        if (Arrays.equals(shape(), other.shape())) {
            IDiffVector result = op.apply(vec, AD.vector(other.toDoubleArray()));
            return new RereDiffTensor(result, shape());
        }
        // Broadcast case: compute broadcast shape and do element-wise op
        IDoubleTensor detOther = other instanceof IDiffTensor ?
            ((IDiffTensor) other).detach() : other;
        double[] resultData;
        int[] resultShape;
        if (other instanceof IDiffTensor otherIDiff) {
            // Gradient tracking only when both are IDiffTensor
            int[] sA = shape();
            int[] sB = other.shape();
            resultShape = TensorShape.broadcastShape(sA, sB);
            long total = 1;
            for (int d : resultShape) total *= d;
            resultData = new double[(int) total];
            double[] aData = tensor.toDoubleArray();
            double[] bData = ((RereDiffTensor) other).tensor.toDoubleArray();
            for (long i = 0; i < total; i++) {
                int[] idxA = broadcastIndex(i, resultShape, sA);
                int[] idxB = broadcastIndex(i, resultShape, sB);
                int flatA = flatIndex(idxA, sA);
                int flatB = flatIndex(idxB, sB);
                resultData[(int) i] = scalarOp.applyAsDouble(aData[flatA], bData[flatB]);
            }
        } else {
            // Broadcast without gradient tracking for other: compute forward using scalar op
            int[] sA = shape();
            int[] sB = detOther.shape();
            int[] bShape = TensorShape.broadcastShape(sA, sB);
            long total = 1;
            for (int d : bShape) total *= d;
            double[] data = new double[(int) total];
            double[] aData = tensor.toDoubleArray();
            double[] bData = detOther.toDoubleArray();
            for (long i = 0; i < total; i++) {
                int[] idxA = broadcastIndex(i, bShape, sA);
                int[] idxB = broadcastIndex(i, bShape, sB);
                int flatA = flatIndex(idxA, sA);
                int flatB = flatIndex(idxB, sB);
                data[(int) i] = scalarOp.applyAsDouble(aData[flatA], bData[flatB]);
            }
            return toNonDiff(new RereDoubleTensor(data, bShape));
        }

        // Backward: sum-reduce gradient back to each input's original shape
        RereDiffVector selfVec = (RereDiffVector) this.vec;
        RereDiffVector otherVec = (RereDiffVector) ((RereDiffTensor) other).vec;
        int[] fResultShape = resultShape;
        int[] fShapeA = shape();
        int[] fShapeB = other.shape();

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] g = gradOut.getData();
            // dA: sum-reduce g along dimensions where sA had size 1
            double[] dA = reduceToShape(g, fResultShape, fShapeA);
            selfVec.accGradDirect(dA);
            // dB: sum-reduce g along dimensions where sB had size 1
            double[] dB = reduceToShape(g, fResultShape, fShapeB);
            otherVec.accGradDirect(dB);
        };

        IDiffVector resultVec = new RereDiffVector(
            IDoubleVector.of(resultData), List.of(selfVec, otherVec), backwardFn);
        return new RereDiffTensor(resultVec, resultShape);
    }

    /** Reduce gradient from resultShape down to targetShape by summing over broadcast dims. */
    private static double[] reduceToShape(double[] grad, int[] resultShape, int[] targetShape) {
        int[] tShape = new int[resultShape.length];
        int diff = resultShape.length - targetShape.length;
        for (int i = 0; i < diff; i++) tShape[i] = 1;
        for (int i = 0; i < targetShape.length; i++) tShape[diff + i] = targetShape[i];

        long totalTarget = 1;
        for (int d : tShape) totalTarget *= d;
        double[] result = new double[(int) totalTarget];
        Arrays.fill(result, 0.0);
        // Iterate over result and accumulate into target
        int resSize = 1;
        for (int d : resultShape) resSize *= d;
        for (int i = 0; i < resSize; i++) {
            int[] idx = unlinearizeInt(i, resultShape);
            int[] tIdx = new int[resultShape.length];
            for (int j = 0; j < resultShape.length; j++) {
                tIdx[j] = tShape[j] == 1 ? 0 : idx[j];
            }
            int targetFlat = flatIndex(tIdx, tShape);
            result[targetFlat] += grad[i];
        }
        return result;
    }

    /** Convert linear index to multi-dimensional index. */
    private static int[] unlinearizeInt(int flat, int[] shape) {
        int[] idx = new int[shape.length];
        int remaining = flat;
        for (int j = shape.length - 1; j >= 0; j--) {
            idx[j] = remaining % shape[j];
            remaining /= shape[j];
        }
        return idx;
    }

    /** Compute flat index from multi-dimensional index. */
    private static int flatIndex(int[] indices, int[] shape) {
        int idx = 0;
        int stride = 1;
        for (int j = shape.length - 1; j >= 0; j--) {
            idx += indices[j] * stride;
            stride *= shape[j];
        }
        return idx;
    }

    /** Map a flat index in the broadcast result to indices in the original shape. */
    private static int[] broadcastIndex(long flat, int[] resultShape, int[] originalShape) {
        int diff = resultShape.length - originalShape.length;
        int[] idx = new int[resultShape.length];
        long remaining = flat;
        for (int j = resultShape.length - 1; j >= 0; j--) {
            int coord = (int) (remaining % resultShape[j]);
            remaining /= resultShape[j];
            idx[j] = (j >= diff && originalShape[j - diff] > 1) ? coord : 0;
        }
        return idx;
    }

    /** Get value from a flat data array using broadcast indexing. */
    private static double broadcastGetFlat(int[] resultIdx, double[] srcData, int[] srcShape, int[] resultShape) {
        int diff = resultShape.length - srcShape.length;
        int srcFlat = 0;
        int srcStride = 1;
        for (int j = srcShape.length - 1; j >= 0; j--) {
            int coord = (diff + j < resultShape.length) ? resultIdx[diff + j] : 0;
            int srcCoord = (srcShape[j] == 1) ? 0 : coord;
            srcFlat += srcCoord * srcStride;
            srcStride *= srcShape[j];
        }
        return srcData[srcFlat];
    }

    /** Compute flat index in src from broadcast result index, with broadcast dimension mapping. */
    private static int flatIndexFromBroadcast(int[] resultIdx, int[] srcShape, int[] resultShape) {
        int diff = resultShape.length - srcShape.length;
        int srcFlat = 0;
        int srcStride = 1;
        for (int j = srcShape.length - 1; j >= 0; j--) {
            int coord = (diff + j < resultShape.length) ? resultIdx[diff + j] : 0;
            int srcCoord = (srcShape[j] == 1) ? 0 : coord;
            srcFlat += srcCoord * srcStride;
            srcStride *= srcShape[j];
        }
        return srcFlat;
    }

    /** Compute size of a shape array. */
    private static long computeSize(int[] shape) {
        long size = 1;
        for (int d : shape) size *= d;
        return size;
    }
}
