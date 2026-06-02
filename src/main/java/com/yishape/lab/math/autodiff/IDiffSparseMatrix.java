package com.yishape.lab.math.autodiff;

import java.io.Serializable;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.sparse.ISparseMatrix;

/**
 * Differentiable sparse matrix (reverse-mode AD).
 * 可微稀疏矩阵（反向模式自动微分）。
 *
 * <p>Primary operation is sparse-dense {@link #matmul(IDiffVector)}; gradients remain sparse.
 * 主要运算为稀疏-稠密矩阵向量乘；梯度保持稀疏结构。</p>
 */
public interface IDiffSparseMatrix extends Serializable {

    long serialVersionUID = 1L;

    ISparseMatrix getValue();

    ISparseMatrix getGradient();

    boolean isLeaf();

    void backward();

    void backward(ISparseMatrix initialGradient);

    void zeroGradient();

    // Core: sparse @ dense
    IDiffVector matmul(IDiffVector vector);

    // Element-wise sparse ops
    IDiffSparseMatrix add(IDiffSparseMatrix other);

    IDiffSparseMatrix sub(IDiffSparseMatrix other);

    IDiffSparseMatrix mul(double scalar);

    IDiffSparseMatrix div(double scalar);

    IDiffSparseMatrix elementwiseMul(IDiffSparseMatrix other);

    IDiffSparseMatrix negate();

    IDiffSparseMatrix transpose();

    // Activations
    IDiffSparseMatrix relu();

    IDiffSparseMatrix sigmoid();

    IDiffSparseMatrix tanh();

    IDiffSparseMatrix abs();

    IDiffVector sum();

    IDiffVector mean();

    IDiffSparseMatrix grad();
}
