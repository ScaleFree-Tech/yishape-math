package com.yishape.lab.math.autodiff.impl;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import com.yishape.lab.util.YishapeLogger;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.IDiffMatrix;

/**
 * Default reverse-mode AD implementation for {@link IDiffMatrix}.
 * Now a thin proxy over {@link RereDiffTensor} (shape=[rows, cols]).
 *
 * <p>All operations delegate to the underlying tensor graph. The tensor graph
 * is the single source of truth for gradient propagation. GPU/HPC execution
 * is automatically available through the tensor graph.</p>
 */
public class RereDiffMatrix implements IDiffMatrix, Serializable {

    private static final long serialVersionUID = 5L;
    private static final YishapeLogger log = YishapeLogger.getLogger(RereDiffMatrix.class);

    // ==================== Core field ====================

    /** Underlying rank-2 tensor (shape=[rows, cols]). The single source of truth for the computation graph. */
    public final RereDiffTensor tensor;

    // ==================== Constructors ====================

    /** Wrap an existing tensor as an IDiffMatrix. Package-private: for autodiff package use. */
    RereDiffMatrix(RereDiffTensor tensor) {
        this.tensor = tensor;
    }

    /** Create a leaf matrix from an IDoubleMatrix. The underlying data is defensively copied. */
    public RereDiffMatrix(IDoubleMatrix value) {
        double[][] data = value.getData();
        int rows = data.length;
        int cols = rows > 0 ? data[0].length : 0;
        double[] flat = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, flat, i * cols, cols);
        }
        this.tensor = new RereDiffTensor(flat, new int[]{rows, cols});
    }

    // ==================== Package-private accessors ====================

    /** Unwrap to the underlying tensor. Package-private: only for autodiff package use. */
    RereDiffTensor unwrap() { return tensor; }

    /** Wrap a tensor as an IDiffMatrix. */
    static RereDiffMatrix wrap(RereDiffTensor t) {
        return new RereDiffMatrix(t);
    }

    // ==================== Value / gradient access ====================

    @Override
    public IDoubleMatrix getValue() {
        double[] flat = tensor.value().toDoubleArray();
        int[] shape = tensor.shape();
        int rows, cols;
        if (shape.length == 2) {
            rows = shape[0];
            cols = shape[1];
        } else if (shape.length == 1) {
            // 1D tensor → treat as column vector (n×1) or row vector (1×n) — use rows×1
            rows = shape[0];
            cols = 1;
        } else if (shape.length == 0) {
            // Scalar → 1×1 matrix
            rows = 1;
            cols = 1;
        } else {
            throw new IllegalStateException("Expected 1D or 2D tensor, got shape: " + java.util.Arrays.toString(shape));
        }
        return IDoubleMatrix.fromArray(flat, rows, cols);
    }

    @Override
    public IDoubleMatrix getGradient() {
        double[] g = tensor.gradData();
        if (g == null) return null;
        int[] shape = tensor.shape();
        int rows, cols;
        if (shape.length == 2) {
            rows = shape[0];
            cols = shape[1];
        } else if (shape.length == 1) {
            rows = shape[0];
            cols = 1;
        } else if (shape.length == 0) {
            rows = 1;
            cols = 1;
        } else {
            throw new IllegalStateException("Expected 1D or 2D tensor, got shape: " + java.util.Arrays.toString(shape));
        }
        double[] copy = g.clone();
        return IDoubleMatrix.fromArray(copy, rows, cols);
    }

    @Override
    public boolean isLeaf() {
        return tensor.isLeaf();
    }

    // ==================== Gradient operations ====================

    @Override
    public void backward() {
        tensor.backward();
    }

    @Override
    public void backward(boolean retainGraph) {
        tensor.backward(retainGraph);
    }

    @Override
    public void backward(IDoubleMatrix initialGradient) {
        backward(initialGradient, false);
    }

    @Override
    public void backward(IDoubleMatrix initialGradient, boolean retainGraph) {
        if (!tensor.requiresGrad()) return;
        double[][] gd = initialGradient.getData();
        int rows = gd.length;
        int cols = rows > 0 ? gd[0].length : 0;
        double[] flat = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(gd[i], 0, flat, i * cols, cols);
        }
        tensor.setGradData(flat);
        tensor.backwardImpl(retainGraph);
    }

    @Override
    public void zeroGradient() {
        tensor.zeroGradient();
    }

    @Override
    public IDiffMatrix grad() {
        if (tensor.gradData() == null) {
            throw new IllegalStateException("Gradient is null — call backward() first");
        }
        RereDiffTensor gradTensor = new RereDiffTensor(tensor.gradData().clone(), tensor.shape());
        gradTensor.setRequiresGrad(true);
        return new RereDiffMatrix(gradTensor);
    }

    // ==================== AccGrad bridge (IDoubleMatrix → flat array) ====================

    /** Accumulates gradient from a 2D matrix. Used by FusedMatrixOps and external code. */
    void accGrad(IDoubleMatrix grad) {
        double[][] gd = grad.getData();
        int rows = gd.length;
        int cols = rows > 0 ? gd[0].length : 0;
        double[] flat = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(gd[i], 0, flat, i * cols, cols);
        }
        tensor.accGrad(flat);
    }

    /** Takes ownership of a freshly-allocated 2D array, flattening into tensor gradient. */
    void accGradDirect(double[][] data) {
        int rows = data.length;
        int cols = rows > 0 ? data[0].length : 0;
        double[] flat = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, flat, i * cols, cols);
        }
        tensor.accGrad(flat);
    }

    // ==================== Matrix operations ====================

    @Override
    public IDiffMatrix matmul(IDiffMatrix other) {
        RereDiffMatrix o = unwrapMatrix(other);
        return wrap((RereDiffTensor) tensor.mmul(o.tensor));
    }

    @Override
    public IDiffVector matmul(IDiffVector vector) {
        RereDiffVector v = RereDiffVector.resolveToRere(vector);
        // mmul requires at least 2D inputs. Reshape vector [n] → [n, 1],
        // compute [r,c] @ [c,1] → [r,1], then squeeze back to [r].
        RereDiffTensor v2d = (RereDiffTensor) v.tensor.reshape((int) v.tensor.totalSize(), 1);
        RereDiffTensor result2d = (RereDiffTensor) tensor.mmul(v2d);
        // result is [r,1] — squeeze to [r]
        if (result2d.shape()[1] == 1) {
            RereDiffTensor result1d = (RereDiffTensor) result2d.reshape((int) result2d.shape()[0]);
            return new RereDiffVector(result1d);
        }
        return new RereDiffVector(result2d);
    }

    @Override
    public IDiffMatrix transpose() {
        return wrap((RereDiffTensor) tensor.transpose());
    }

    // ==================== Element-wise with another matrix ====================

    @Override
    public IDiffMatrix add(IDiffMatrix other) {
        RereDiffMatrix o = unwrapMatrix(other);
        return wrap((RereDiffTensor) tensor.add(o.tensor));
    }

    @Override
    public IDiffMatrix sub(IDiffMatrix other) {
        RereDiffMatrix o = unwrapMatrix(other);
        return wrap((RereDiffTensor) tensor.sub(o.tensor));
    }

    @Override
    public IDiffMatrix mul(IDiffMatrix other) {
        RereDiffMatrix o = unwrapMatrix(other);
        return wrap((RereDiffTensor) tensor.mul(o.tensor));
    }

    @Override
    public IDiffMatrix div(IDiffMatrix other) {
        RereDiffMatrix o = unwrapMatrix(other);
        return wrap((RereDiffTensor) tensor.div(o.tensor));
    }

    // ==================== Scalar arithmetic ====================

    @Override
    public IDiffMatrix add(double scalar) {
        return wrap((RereDiffTensor) tensor.add(scalar));
    }

    @Override
    public IDiffMatrix sub(double scalar) {
        return wrap((RereDiffTensor) tensor.sub(scalar));
    }

    @Override
    public IDiffMatrix mul(double scalar) {
        return wrap((RereDiffTensor) tensor.mul(scalar));
    }

    @Override
    public IDiffMatrix div(double scalar) {
        return wrap((RereDiffTensor) tensor.div(scalar));
    }

    @Override
    public IDiffMatrix rsub(double scalar) {
        return wrap((RereDiffTensor) tensor.rsub(scalar));
    }

    @Override
    public IDiffMatrix rdiv(double scalar) {
        return wrap((RereDiffTensor) tensor.rdiv(scalar));
    }

    // ==================== Unary ====================

    @Override
    public IDiffMatrix neg() {
        return wrap((RereDiffTensor) tensor.neg());
    }

    @Override
    public IDiffMatrix pow(double n) {
        return wrap((RereDiffTensor) tensor.pow(n));
    }

    // ==================== Element-wise math ====================

    @Override
    public IDiffMatrix exp() {
        return wrap((RereDiffTensor) tensor.exp());
    }

    @Override
    public IDiffMatrix log() {
        return wrap((RereDiffTensor) tensor.log());
    }

    @Override
    public IDiffMatrix sigmoid() {
        return wrap((RereDiffTensor) tensor.sigmoid());
    }

    @Override
    public IDiffMatrix relu() {
        return wrap((RereDiffTensor) tensor.relu());
    }

    @Override
    public IDiffMatrix tanh() {
        return wrap((RereDiffTensor) tensor.tanh());
    }

    @Override
    public IDiffMatrix sqrt() {
        return wrap((RereDiffTensor) tensor.sqrt());
    }

    @Override
    public IDiffMatrix square() {
        return wrap((RereDiffTensor) tensor.square());
    }

    @Override
    public IDiffMatrix abs() {
        return wrap((RereDiffTensor) tensor.abs());
    }

    @Override
    public IDiffMatrix gelu() {
        return wrap((RereDiffTensor) tensor.gelu());
    }

    @Override
    public IDiffMatrix leakyRelu(double alpha) {
        return wrap((RereDiffTensor) tensor.leakyRelu(alpha));
    }

    @Override
    public IDiffMatrix elu(double alpha) {
        return wrap((RereDiffTensor) tensor.elu(alpha));
    }

    @Override
    public IDiffMatrix selu() {
        return wrap((RereDiffTensor) tensor.selu());
    }

    @Override
    public IDiffMatrix silu() {
        return wrap((RereDiffTensor) tensor.silu());
    }

    @Override
    public IDiffMatrix mish() {
        return wrap((RereDiffTensor) tensor.mish());
    }

    @Override
    public IDiffMatrix softplus(double beta) {
        return wrap((RereDiffTensor) tensor.softplus(beta));
    }

    @Override
    public IDiffMatrix hardtanh(double minVal, double maxVal) {
        return wrap((RereDiffTensor) tensor.hardtanh(minVal, maxVal));
    }

    @Override
    public IDiffMatrix clamp(double min, double max) {
        return wrap((RereDiffTensor) tensor.clamp(min, max));
    }

    // ==================== Normalization ====================

    @Override
    public IDiffMatrix layerNorm(IDiffVector gamma, IDiffVector beta, double eps) {
        RereDiffVector gr = RereDiffVector.resolveToRere(gamma);
        RereDiffVector br = RereDiffVector.resolveToRere(beta);
        return wrap((RereDiffTensor) tensor.layerNorm(gr.tensor, br.tensor, eps));
    }

    @Override
    public IDiffMatrix batchNorm(IDiffVector gamma, IDiffVector beta, double eps) {
        RereDiffVector gr = RereDiffVector.resolveToRere(gamma);
        RereDiffVector br = RereDiffVector.resolveToRere(beta);
        return wrap((RereDiffTensor) tensor.batchNorm(gr.tensor, br.tensor, eps));
    }

    @Override
    public IDiffMatrix dropout(double p) {
        return wrap((RereDiffTensor) tensor.dropout(p));
    }

    // ==================== Reductions ====================

    @Override
    public IDiffMatrix sum() {
        RereDiffTensor result = (RereDiffTensor) tensor.sum();
        // sum() returns shape [1]; reshape to 1x1 matrix
        int[] shape = tensor.shape();
        int rows = shape[0], cols = shape[1];
        // Copy data into 1x1 shape
        if (result.shape()[0] != 1) {
            // Scalar result — reshape
            return wrap((RereDiffTensor) result.reshape(1, 1));
        }
        return wrap(result);
    }

    @Override
    public IDiffMatrix mean() {
        int[] shape = tensor.shape();
        long n = (long) shape[0] * shape[1];
        RereDiffTensor summed = (RereDiffTensor) tensor.sum();
        RereDiffTensor result = (RereDiffTensor) summed.div((double) n);
        // reshape to 1x1
        if (result.shape().length > 0 && result.shape()[0] != 1) {
            return wrap((RereDiffTensor) result.reshape(1, 1));
        }
        return wrap(result);
    }

    @Override
    public IDiffVector sumAsVector() {
        RereDiffTensor result = (RereDiffTensor) tensor.sum();
        return new RereDiffVector(result);
    }

    @Override
    public IDiffVector meanAsVector() {
        int[] shape = tensor.shape();
        long n = (long) shape[0] * shape[1];
        RereDiffTensor summed = (RereDiffTensor) tensor.sum();
        RereDiffTensor result = (RereDiffTensor) summed.div((double) n);
        return new RereDiffVector(result);
    }

    @Override
    public IDiffVector sum(int axis) {
        if (axis != 0 && axis != 1) {
            throw new IllegalArgumentException("axis must be 0 or 1, got " + axis);
        }
        RereDiffTensor result = (RereDiffTensor) tensor.sum(axis, false);
        return new RereDiffVector(result);
    }

    @Override
    public IDiffVector max(int axis) {
        if (axis != 0 && axis != 1) {
            throw new IllegalArgumentException("axis must be 0 or 1, got " + axis);
        }
        RereDiffTensor result = (RereDiffTensor) tensor.max(axis, false);
        return new RereDiffVector(result);
    }

    // ==================== Broadcast arithmetic ====================

    @Override
    public IDiffMatrix sub(IDiffVector vec, int axis) {
        RereDiffVector v = RereDiffVector.resolveToRere(vec);
        if (axis != 0 && axis != 1) {
            throw new IllegalArgumentException("axis must be 0 or 1, got " + axis);
        }
        int[] shape = tensor.shape();
        if (axis == 1) {
            if (v.tensor.totalSize() != shape[0]) {
                throw new IllegalArgumentException(
                    "axis=1: vec length " + v.tensor.totalSize() + " != rows " + shape[0]);
            }
        } else {
            if (v.tensor.totalSize() != shape[1]) {
                throw new IllegalArgumentException(
                    "axis=0: vec length " + v.tensor.totalSize() + " != cols " + shape[1]);
            }
        }
        // Reshape vector for broadcasting: axis=0 → [1, cols], axis=1 → [rows, 1]
        RereDiffTensor vReshaped;
        if (axis == 0) {
            vReshaped = (RereDiffTensor) v.tensor.reshape(1, shape[1]);
        } else {
            vReshaped = (RereDiffTensor) v.tensor.reshape(shape[0], 1);
        }
        return wrap((RereDiffTensor) tensor.sub(vReshaped));
    }

    @Override
    public IDiffMatrix div(IDiffVector vec, int axis) {
        RereDiffVector v = RereDiffVector.resolveToRere(vec);
        if (axis != 0 && axis != 1) {
            throw new IllegalArgumentException("axis must be 0 or 1, got " + axis);
        }
        int[] shape = tensor.shape();
        if (axis == 1) {
            if (v.tensor.totalSize() != shape[0]) {
                throw new IllegalArgumentException(
                    "axis=1: vec length " + v.tensor.totalSize() + " != rows " + shape[0]);
            }
        } else {
            if (v.tensor.totalSize() != shape[1]) {
                throw new IllegalArgumentException(
                    "axis=0: vec length " + v.tensor.totalSize() + " != cols " + shape[1]);
            }
        }
        RereDiffTensor vReshaped;
        if (axis == 0) {
            vReshaped = (RereDiffTensor) v.tensor.reshape(1, shape[1]);
        } else {
            vReshaped = (RereDiffTensor) v.tensor.reshape(shape[0], 1);
        }
        return wrap((RereDiffTensor) tensor.div(vReshaped));
    }

    // ==================== Fused operations ====================

    @Override
    public IDiffVector softmaxCrossEntropy(IDiffMatrix oneHotLabels) {
        RereDiffMatrix y = unwrapMatrix(oneHotLabels);
        // softmaxCrossEntropy takes IDoubleTensor labels (constant, not differentiable)
        RereDiffTensor result = (RereDiffTensor) tensor.softmaxCrossEntropy(y.tensor.value(), 1);
        // Set export shape for graph export
        int[] shape = tensor.shape();
        result.setExportShape(new int[]{shape[0], shape[1]});
        return new RereDiffVector(result);
    }

    // ==================== Reshape ====================

    @Override
    public IDiffVector flatten() {
        RereDiffTensor result = (RereDiffTensor) tensor.flatten(0, 1);
        return new RereDiffVector(result);
    }

    @Override
    public IDiffMatrix reshape(int rows, int cols) {
        return wrap((RereDiffTensor) tensor.reshape(rows, cols));
    }

    // ==================== In-place operations ====================

    @Override
    public IDiffMatrix addInPlace(IDiffMatrix other) {
        // Delegate to add() — same semantics as RereDiffVector.addInPlace
        return add(other);
    }

    @Override
    public IDiffMatrix mulInPlace(double scalar) {
        if (!tensor.isLeaf()) {
            throw new IllegalStateException("mulInPlace only allowed on leaf variables");
        }
        RereDoubleTensor val = tensor.value();
        double[] data = val.toDoubleArray();
        for (int i = 0; i < data.length; i++) {
            data[i] *= scalar;
        }
        tensor.setValue(new RereDoubleTensor(data, val.shape()));
        tensor.setGradData(null);
        return this;
    }

    // ==================== Copy ====================

    @Override
    public IDiffMatrix copy() {
        return wrap((RereDiffTensor) tensor.clone());
    }

    @Override
    public IDiffMatrix detach() {
        return wrap((RereDiffTensor) tensor.detach());
    }

    // ==================== Comparison ====================

    @Override
    public boolean[][] ge(IMatrix<Double> other) {
        double[][] thisData = getValue().getData();
        double[][] otherData = ((IDoubleMatrix) other).getData();
        int rows = thisData.length;
        int cols = rows > 0 ? thisData[0].length : 0;
        boolean[][] result = new boolean[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = thisData[i][j] >= otherData[i][j];
            }
        }
        return result;
    }

    // ==================== Internal helpers ====================

    private static RereDiffMatrix unwrapMatrix(IDiffMatrix m) {
        if (m instanceof RereDiffMatrix rm) return rm;
        throw new IllegalArgumentException("Expected RereDiffMatrix, got: " + m.getClass().getSimpleName());
    }
}
