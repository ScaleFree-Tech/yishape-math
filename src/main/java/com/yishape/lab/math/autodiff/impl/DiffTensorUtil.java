package com.yishape.lab.math.autodiff.impl;

import java.util.Arrays;
import java.util.function.DoubleBinaryOperator;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.compute.ops.UniversalOperation;

/**
 * Shared utility methods and symbolic backward factories for differentiable tensor operations.
 *
 * <p>All methods are static and operate on raw arrays and shapes, independent of any
 * specific {@link RereDiffTensor} instance. This class consolidates helper functions
 * that were previously scattered across RereDiffTensor's utility section, eliminating
 * duplication across extracted operation classes.
 */
public final class DiffTensorUtil {

    private DiffTensorUtil() { /* utility class */ }

    // ==================== BinaryBackward interface ====================

    /** Functional interface for binary operation gradient computation. */
    @FunctionalInterface
    public interface BinaryBackward {
        double apply(double grad, double aVal, double bVal);
    }

    // ==================== Shape utilities ====================

    /**
     * Compute the reduced shape after reducing along a dimension.
     *
     * @param shape    original shape
     * @param dim      reduction dimension
     * @param keepdim  whether to keep the dimension as size 1
     * @return the shape after reduction
     */
    public static int[] reducedShape(int[] shape, int dim, boolean keepdim) {
        if (keepdim) {
            int[] r = shape.clone();
            r[dim] = 1;
            return r;
        }
        // Reducing a 1D tensor produces a scalar (shape [1]), not an empty shape.
        // Empty shape is invalid for TensorShape (throws "Shape cannot be null or empty").
        if (shape.length == 1) return new int[]{1};
        int[] r = new int[shape.length - 1];
        int idx = 0;
        for (int i = 0; i < shape.length; i++) if (i != dim) r[idx++] = shape[i];
        return r;
    }

    /** Compute total number of elements from a shape array. */
    public static long computeSize(int[] shape) {
        long size = 1;
        for (int d : shape) size *= d;
        return size;
    }

    // ==================== Index utilities ====================

    /** Convert a flat index to multi-dimensional indices. */
    public static int[] unlinearizeInt(int flat, int[] shape) {
        int[] idx = new int[shape.length];
        int remaining = flat;
        for (int j = shape.length - 1; j >= 0; j--) {
            idx[j] = remaining % shape[j];
            remaining /= shape[j];
        }
        return idx;
    }

    /** Convert multi-dimensional indices to a flat index (row-major). */
    public static int flatIndex(int[] indices, int[] shape) {
        int idx = 0;
        int stride = 1;
        for (int j = shape.length - 1; j >= 0; j--) {
            idx += indices[j] * stride;
            stride *= shape[j];
        }
        return idx;
    }

    // ==================== Broadcast utilities ====================

    /**
     * Read a value from a broadcasted source array at the given result index.
     * Standard NumPy broadcast semantics: dimensions of size 1 are stretched.
     */
    public static double broadcastGetFlat(int[] resultIdx, double[] srcData, int[] srcShape, int[] resultShape) {
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

    /**
     * Map a result index to a flat index in the source array under broadcast alignment.
     * Left-aligned (DL/PyTorch) broadcast semantics: dimensions of size 1 are stretched.
     *
     * <p>For rank-1 sources, the broadcast axis is determined by comparing the source dim
     * size against the output's first and last dimensions (same heuristic as HPC's
     * {@code broadcast_is_left_aligned}):</p>
     * <ul>
     *   <li>Matches output first dim → left-aligned (contiguous blocks along dim 0)</li>
     *   <li>Matches output last dim  → right-aligned (cyclic along last dim)</li>
     *   <li>Both match (square)       → either works; use dim 0</li>
     *   <li>Neither match             → fallback to last dim (legacy)</li>
     * </ul>
     */
    public static int flatIndexFromBroadcast(int[] resultIdx, int[] srcShape, int[] resultShape) {
        // Rank-1 source: determine broadcast axis by dim matching
        if (srcShape.length == 1) {
            if (resultShape.length == 1) {
                return resultIdx[0] % srcShape[0];
            }
            int firstDim = resultShape[0];
            int lastDim = resultShape[resultShape.length - 1];
            if (srcShape[0] == firstDim && srcShape[0] == lastDim) {
                // Both match (square or repeated dims) — either axis equivalent
                return resultIdx[0] % srcShape[0];
            } else if (srcShape[0] == firstDim) {
                // Left-aligned: rank-1 dim matches first output dim → contiguous blocks
                return resultIdx[0] % srcShape[0];
            } else if (srcShape[0] == lastDim) {
                // Right-aligned: rank-1 dim matches last output dim → cyclic
                return resultIdx[resultShape.length - 1] % srcShape[0];
            }
            // Fallback: dim doesn't match either end; use last dim (legacy behavior)
            return resultIdx[resultShape.length - 1] % srcShape[0];
        }
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

    /**
     * Broadcast src data (in srcShape) to dst array (in dstShape).
     * Standard NumPy broadcast semantics: dimensions of size 1 are stretched.
     */
    public static void broadcastTo(double[] src, int[] srcShape, double[] dst, int[] dstShape) {
        int diff = dstShape.length - srcShape.length;
        long dstSize = 1;
        for (int d : dstShape) dstSize *= d;
        if (dstSize > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                "broadcastTo: tensor too large for int-indexed broadcast (" + dstSize + " elements)");
        }
        // §7a exception: broadcast requires per-element index mapping (non-contiguous access).
        // No accelerated alternative exists; DoubleVectorComputer operates on contiguous arrays.
        for (long flat = 0; flat < dstSize; flat++) {
            int[] dstIdx = unlinearizeInt((int) flat, dstShape);
            int srcFlat = flatIndexFromBroadcast(dstIdx, srcShape, dstShape);
            dst[(int) flat] = src[srcFlat];
        }
    }

    /**
     * Sum-reduce a broadcasted gradient array back to the original (non-broadcasted) shape.
     * For each element in the broadcasted array, maps it to the corresponding original index
     * and accumulates the gradient.
     */
    public static void unbroadcastSum(double[] gradBC, int[] bcShape, double[] gradOrig, int[] origShape) {
        long bcSize = 1;
        for (int d : bcShape) bcSize *= d;
        // Guard: int-indexed broadcast cannot handle >2^31 elements (silent int overflow)
        if (bcSize > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                "unbroadcastSum: tensor too large for int-indexed broadcast (" + bcSize + " elements)");
        }
        // §7a exception: unbroadcast requires per-element index mapping (non-contiguous access).
        // No accelerated alternative exists; DoubleVectorComputer operates on contiguous arrays.
        for (long bcFlat = 0; bcFlat < bcSize; bcFlat++) {
            int[] bcIdx = unlinearizeInt((int) bcFlat, bcShape);
            int origFlat = flatIndexFromBroadcast(bcIdx, origShape, bcShape);
            gradOrig[origFlat] += gradBC[(int) bcFlat];
        }
    }

    // ==================== Op tag utilities ====================

    /** Map common element-wise op tags to {@link UniversalOperation} for SIMD forward acceleration. */
    public static UniversalOperation tagToUniversalOp(String tag) {
        return switch (tag) {
            case "exp"     -> UniversalOperation.EXP;
            case "log"     -> UniversalOperation.LOG;
            case "sqrt"    -> UniversalOperation.SQRT;
            case "relu"    -> UniversalOperation.RELU;
            case "sigmoid" -> UniversalOperation.SIGMOID;
            case "tanh"    -> UniversalOperation.TANH;
            case "abs"     -> UniversalOperation.ABS;
            case "sin"     -> UniversalOperation.SIN;
            case "cos"     -> UniversalOperation.COS;
            case "gelu"    -> UniversalOperation.GELU;
            case "tan"     -> UniversalOperation.TAN;
            // square, silu, mish, elu, leakyRelu, selu, softplus, hardtanh, clamp
            // have no corresponding UniversalOperation enum entry yet; fall back to scalar
            default        -> null;
        };
    }

    // ==================== Symbolic backward factories ====================

    /**
     * Creates a symbolic backward factor for unary ops, expressed as tensor operations
     * on {@code xRef} so that tape-of-tape AD (MixedMode.hvp) can propagate gradients
     * back to the primal variables.
     *
     * <p>For ops with non-zero second derivative (exp, log, sigmoid, etc.), returns a
     * proper tensor expression like {@code xRef.exp()}. For ops with zero second
     * derivative (relu, abs, etc.) or complex ops (gelu, mish), returns a tensor
     * that is numerically correct but connected to xRef via {@code xRef.mul(0).add(…)}
     * so backward() reaches xRef with zero gradient.</p>
     *
     * @return a differentiable tensor factor, never null
     */
    public static IDiffTensor symbolicUnaryFactor(String tag, RereDiffTensor xRef,
                                                   DoubleBinaryOperator backward,
                                                   double[] xData, double scalarParam) {
        int n = xData.length;
        int[] shape = xRef.shape();
        IDiffTensor factor = switch (tag) {
            case "exp"     -> xRef.exp();
            case "log"     -> xRef.reciprocal();
            case "sqrt"    -> xRef.pow(-0.5).mul(0.5);
            case "sin"     -> xRef.cos();
            case "cos"     -> xRef.sin().neg();
            case "tan"     -> { IDiffTensor c = xRef.cos(); yield c.pow(2).reciprocal(); }
            case "sigmoid" -> { IDiffTensor s = xRef.sigmoid(); yield s.mul(s.neg().add(1.0)); }
            case "tanh"    -> { IDiffTensor t = xRef.tanh(); yield t.square().neg().add(1.0); }
            case "square"  -> xRef.mul(2.0);
            case "silu"    -> {
                IDiffTensor s = xRef.sigmoid();
                yield s.add(xRef.mul(s).mul(s.neg().add(1.0)));
            }
            case "softplus" -> {
                double beta = Double.isNaN(scalarParam) ? 1.0 : scalarParam;
                yield xRef.mul(beta).sigmoid();
            }
            // Ops where second derivative is zero or not practically expressible:
            // relu, leakyRelu, elu, selu, gelu, mish, abs, neg, clamp, hardtanh
            default        -> null;
        };
        if (factor != null) return factor;
        // Fallback: constant factor connected to xRef via mul(0) so gradient can flow
        // (as zero) back to xRef. Correct for ops with zero second derivative.
        double[] f = new double[n];
        for (int i = 0; i < n; i++) f[i] = backward.applyAsDouble(1.0, xData[i]);
        return xRef.mul(0.0).add(IDiffTensor.constantTensor(f, shape));
    }

    /** Add symbolicBackwardFn that broadcasts scalar g to inputShape. */
    public static java.util.function.Function<IDiffTensor, IDiffTensor[]> sumGradFn(int[] inputShape) {
        int[] shapeCopy = inputShape.clone();
        long n = 1;
        for (int d : inputShape) n *= d;
        long totalN = n;
        return g -> {
            double[] ones = new double[Math.toIntExact(totalN)];
            Arrays.fill(ones, 1.0);
            return new IDiffTensor[]{ g.mul(IDiffTensor.constantTensor(ones, shapeCopy)) };
        };
    }

    /** Add symbolicBackwardFn that broadcasts g to inputShape, multiplied by element-wise factor. */
    public static java.util.function.Function<IDiffTensor, IDiffTensor[]> broadcastGradFn(int[] inputShape, double[] factor) {
        int[] shapeCopy = inputShape.clone();
        double[] factorCopy = factor.clone();
        return g -> new IDiffTensor[]{ g.mul(IDiffTensor.constantTensor(factorCopy, shapeCopy)) };
    }

    /** Add symbolicBackwardFn for sum along dim: unsqueeze scalar g and multiply by factor.
     *  Connects factor to xRef via mul(0) so tape-of-tape gradients can flow back. */
    public static java.util.function.Function<IDiffTensor, IDiffTensor[]> dimSumGradFn(
            int[] inputShape, int dim, double[] factor, RereDiffTensor xRef) {
        int[] shapeCopy = inputShape.clone();
        double[] factorCopy = factor.clone();
        int dimCopy = dim;
        RereDiffTensor xRefCopy = xRef;
        return g -> {
            IDiffTensor expanded = g.unsqueeze(dimCopy);
            IDiffTensor factorTensor = xRefCopy.mul(0.0)
                .add(IDiffTensor.constantTensor(factorCopy, shapeCopy));
            return new IDiffTensor[]{ expanded.mul(factorTensor) };
        };
    }

    /** Add symbolicBackwardFn for scalar op: g * backward(1, x). */
    public static java.util.function.Function<IDiffTensor, IDiffTensor[]> scalarOpGradFn(
            DoubleBinaryOperator backward, double[] xData, int[] shape) {
        double[] xCopy = xData.clone();
        int[] shapeCopy = shape.clone();
        return g -> {
            int n = xCopy.length;
            double[] factor = new double[n];
            for (int i = 0; i < n; i++) factor[i] = backward.applyAsDouble(1.0, xCopy[i]);
            return new IDiffTensor[]{ g.mul(IDiffTensor.constantTensor(factor, shapeCopy)) };
        };
    }

    /** Symbolic backward for same-shape binary ops: returns gradient for each requiresGrad input.
     *  Connects factors to tensor references via mul(0) so tape-of-tape gradients flow back. */
    public static java.util.function.Function<IDiffTensor, IDiffTensor[]> binarySameSymbolicFn(
            int n, BinaryBackward gradA, BinaryBackward gradB,
            double[] aData, double[] bData, int[] shape,
            boolean hasA, boolean hasB,
            RereDiffTensor aRef, RereDiffTensor bRef) {
        double[] aCopy = aData.clone();
        double[] bCopy = bData.clone();
        int[] shapeCopy = shape.clone();
        RereDiffTensor aRefCopy = aRef;
        RereDiffTensor bRefCopy = bRef;
        return g -> {
            double[] factA = new double[n];
            double[] factB = new double[n];
            for (int i = 0; i < n; i++) {
                factA[i] = gradA.apply(1.0, aCopy[i], bCopy[i]);
                factB[i] = gradB.apply(1.0, aCopy[i], bCopy[i]);
            }
            // Connect to input tensors via mul(0) so gradient can flow (as zero for add/sub).
            IDiffTensor tA = hasA ? g.mul(aRefCopy.mul(0.0).add(IDiffTensor.constantTensor(factA, shapeCopy))) : null;
            IDiffTensor tB = hasB ? g.mul(bRefCopy.mul(0.0).add(IDiffTensor.constantTensor(factB, shapeCopy))) : null;
            if (hasA && hasB) return new IDiffTensor[]{ tA, tB };
            return new IDiffTensor[]{ hasA ? tA : tB };
        };
    }

    /** Symbolic backward for broadcast binary ops: scatter-reduce gradient factor to original shape.
     *  Connects factors to tensor references via mul(0) so tape-of-tape gradients flow back. */
    public static java.util.function.Function<IDiffTensor, IDiffTensor[]> binaryBroadcastSymbolicFn(
            int n, BinaryBackward gradA, BinaryBackward gradB,
            double[] bcA, double[] bcB,
            int[] sA, int[] sB, int[] resultShape,
            boolean hasA, boolean hasB,
            RereDiffTensor aRef, RereDiffTensor bRef) {
        double[] bcACopy = bcA.clone();
        double[] bcBCopy = bcB.clone();
        int[] sACopy = sA.clone();
        int[] sBCopy = sB.clone();
        int[] rShapeCopy = resultShape.clone();
        RereDiffTensor aRefCopy = aRef;
        RereDiffTensor bRefCopy = bRef;
        return g -> {
            if (hasA && hasB) {
                int aTotal = (int) computeSize(sACopy);
                int bTotal = (int) computeSize(sBCopy);
                double[] factA = new double[aTotal];
                double[] factB = new double[bTotal];
                for (int i = 0; i < n; i++) {
                    int[] idx = unlinearizeInt(i, rShapeCopy);
                    factA[flatIndexFromBroadcast(idx, sACopy, rShapeCopy)] += gradA.apply(1.0, bcACopy[i], bcBCopy[i]);
                    factB[flatIndexFromBroadcast(idx, sBCopy, rShapeCopy)] += gradB.apply(1.0, bcACopy[i], bcBCopy[i]);
                }
                return new IDiffTensor[]{
                    g.mul(aRefCopy.mul(0.0).add(IDiffTensor.constantTensor(factA, sACopy))),
                    g.mul(bRefCopy.mul(0.0).add(IDiffTensor.constantTensor(factB, sBCopy)))
                };
            } else if (hasA) {
                int aTotal = (int) computeSize(sACopy);
                double[] factA = new double[aTotal];
                for (int i = 0; i < n; i++) {
                    int[] idx = unlinearizeInt(i, rShapeCopy);
                    factA[flatIndexFromBroadcast(idx, sACopy, rShapeCopy)] += gradA.apply(1.0, bcACopy[i], bcBCopy[i]);
                }
                return new IDiffTensor[]{
                    g.mul(aRefCopy.mul(0.0).add(IDiffTensor.constantTensor(factA, sACopy)))
                };
            } else {
                int bTotal = (int) computeSize(sBCopy);
                double[] factB = new double[bTotal];
                for (int i = 0; i < n; i++) {
                    int[] idx = unlinearizeInt(i, rShapeCopy);
                    factB[flatIndexFromBroadcast(idx, sBCopy, rShapeCopy)] += gradB.apply(1.0, bcACopy[i], bcBCopy[i]);
                }
                return new IDiffTensor[]{
                    g.mul(bRefCopy.mul(0.0).add(IDiffTensor.constantTensor(factB, sBCopy)))
                };
            }
        };
    }
    // ==================== diagEmbed helpers ====================

    /** Decompose a flat row-major index into multi-dimensional coordinates. */
    static int[] decomposeFlatIndex(long flat, int[] shape) {
        int[] coord = new int[shape.length];
        long rem = flat;
        for (int i = shape.length - 1; i >= 0; i--) {
            coord[i] = (int)(rem % shape[i]);
            rem /= shape[i];
        }
        return coord;
    }

    /** Build flat output index for diagEmbed: batchCoords with (row,col) at dim1/dim2. */
    static int buildDiagEmbedOutputIndex(int[] batchCoord, int row, int col,
                                         int dim1, int dim2, int[] outShape) {
        int[] coord = new int[outShape.length];
        for (int i = 0, bi = 0; i < coord.length; i++) {
            if (i == dim1) coord[i] = row;
            else if (i == dim2) coord[i] = col;
            else coord[i] = batchCoord[bi++];
        }
        return flatIndex(coord, outShape);
    }
}
