package com.yishape.lab.math.autodiff.vmap;

import java.io.Serializable;

import com.yishape.lab.math.compute.IDoubleVectorComputer;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import com.yishape.lab.math.compute.ops.UniversalOperation;

/**
 * Batched compute primitives for vmap and deep learning.
 *
 * <p>Each method operates on flat stacked arrays (sample-major layout).
 * Element-wise operations are shape-agnostic and dispatch through the
 * GPU → HPC → SIMD → SISD chain. Reductions are sample-aware: they
 * reduce over the inner dimension D, producing N independent results.
 *
 * <p>Names follow two conventions: generic compute primitives use
 * {@code unaryOp}/{@code binaryOperate}/{@code batchReduce} matching
 * {@link IDoubleVectorComputer}, while DL-specific operations use
 * descriptive names ({@code batchSoftmax}, {@code batchLayerNorm},
 * {@code im2col}, {@code batchDropout}).
 *
 * @author lteb2
 */
public interface IVMap extends Serializable {

    // ---- Memory layout ops ----

    /**
     * Stack N arrays of identical length D into a single flat array of length N*D.
     * Layout: element j of sample i is at {@code flat[i * D + j]}.
     *
     * @param arrays non-empty, all sub-arrays same length
     * @return flat stacked array of length N*D
     * @throws IllegalArgumentException if arrays is null, empty, or sub-arrays differ in length
     */
    double[] stack(double[][] arrays);

    /**
     * Split a flat stacked array of length N*D back into N arrays of length D.
     *
     * @param flat flat array of length N*D
     * @param n    number of samples (outer dimension)
     * @param d    elements per sample (inner dimension)
     * @return double[n][d]
     */
    double[][] unstack(double[] flat, int n, int d);

    // ---- Generic compute primitives ----

    /**
     * Batched reduction: reduce over the inner dimension D independently for each
     * of N samples. GPU native batched reduction ({@code outer/inner}) when available;
     * falls back to per-row SIMD, then per-row scalar.
     *
     * @param flat flat array of length N*D (sample-major)
     * @param n    number of samples (outer dimension)
     * @param d    elements per sample (inner dimension)
     * @param op   reduction operation
     * @return double[n] with one result per sample
     */
    double[] batchReduce(double[] flat, int n, int d, ReduceOperation op);

    /**
     * Element-wise binary operation on flat arrays. Shape-agnostic.
     *
     * @param a  first operand
     * @param b  second operand (same length)
     * @param op binary operation
     * @return result array, same length
     */
    double[] binaryOperate(double[] a, double[] b, BinaryOperation op);

    /**
     * Element-wise binary operation on flat arrays. Shape-agnostic.
     *
     * @param a  first operand
     * @param b  second operand (same length)
     * @param op binary operation
     * @return result array, same length
     */
    default double[] elementwiseBinary(double[] a, double[] b, BinaryOperation op){
       return this.binaryOperate(a, b, op);
    }

    
    /**
     * Element-wise unary operation on a flat array. Shape-agnostic.
     * Covers activation functions (relu, gelu, sigmoid, tanh) and
     * math transforms (exp, log, sqrt, sin, cos).
     *
     * @param a     input array
     * @param op    unary operation
     * @param param additional parameter (e.g. power for POW); 0.0 if unused
     * @return result array, same length
     */
    double[] unaryOp(double[] a, UniversalOperation op, double param);

    /**
     * Batched flat row-major matrix multiply.
     * For b in 0..batch-1: C_b[m×n] = A_b[m×k] × B_b[k×n].
     * Dispatches through {@code DoubleFlatGemm.flatMmulBatched} (GPU → HPC → SIMD → scalar).
     *
     * @param a     flat array of length batch * m * k
     * @param b     flat array of length batch * k * n
     * @param batch number of independent matmul operations
     * @param m     rows per output
     * @param k     inner dimension
     * @param n     cols per output
     * @return flat array of length batch * m * n
     */
    double[] batchMatmul(double[] a, double[] b, int batch, int m, int k, int n);

    // ---- DL operations ----

    /**
     * Batched softmax: for each of N samples of dimension D, compute
     * {@code softmax(x_i)} with numerical stability (subtract max).
     *
     * @param flat  flat array of length N*D (sample-major)
     * @param batch number of samples N
     * @param dim   elements per sample D
     * @return flat array of length N*D with per-sample softmax applied
     */
    double[] batchSoftmax(double[] flat, int batch, int dim);

    /**
     * Batched layer normalization: for each of N samples of dimension D,
     * normalize to zero mean and unit variance, then apply scale gamma
     * and shift beta.
     *
     * @param x     flat array of length N*D (sample-major)
     * @param gamma scale parameter of length D (shared across batch)
     * @param beta  shift parameter of length D (shared across batch)
     * @param batch number of samples N
     * @param dim   elements per sample D
     * @param eps   epsilon for numerical stability
     * @return flat array of length N*D with per-sample layer norm applied
     */
    double[] batchLayerNorm(double[] x, double[] gamma, double[] beta, int batch, int dim, float eps);

    /**
     * Unfold image patches into a column matrix (im2col).
     * Convolution weights can then be multiplied with the result via
     * {@link #batchMatmul} for batched convolution.
     *
     * @param input  flat array of length C*H*W (channel-major within sample)
     * @param C      input channels
     * @param H      input height
     * @param W      input width
     * @param kH     kernel height
     * @param kW     kernel width
     * @param stride convolution stride
     * @param pad    zero-padding size
     * @return column matrix of shape C*kH*kW × outH*outW, row-major flat
     */
    double[] im2col(double[] input, int C, int H, int W, int kH, int kW, int stride, int pad);

    /**
     * Batched dropout: for each element, randomly sets it to zero with
     * probability {@code p} and scales survivors by {@code 1/(1-p)}.
     * Uses a deterministic RNG seeded by {@code seed + elementIndex}
     * for reproducibility.
     *
     * @param flat flat input array
     * @param p    dropout probability (0 ≤ p < 1)
     * @param seed random seed for reproducible mask generation
     * @return array of same length with dropout mask applied
     */
    double[] batchDropout(double[] flat, double p, long seed);
}
