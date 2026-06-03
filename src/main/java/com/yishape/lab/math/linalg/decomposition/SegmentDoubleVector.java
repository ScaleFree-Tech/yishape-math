package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.BinaryReduceOperation;

/**
 * Lightweight segment view over a {@code double[]} backing array with offset and length.
 * Provides dot product, AXPY, and in-place scale operations that route through
 * {@link DoubleVectorComputer} for automatic HPC / SIMD / SISD dispatch with fallback.
 *
 * <h3>Design notes for maintainers</h3>
 * <ul>
 *   <li>This class deliberately does NOT implement {@code IVector<Double>} —
 *       a full IVector implementation would require ~200 methods and create
 *       unnecessary maintenance burden. It exposes only the three operations
 *       needed by decomposition and solver hot loops.</li>
 *   <li>Each operation copies the segment into a contiguous temporary array
 *       before calling the computer. This is intentional: the computer layer
 *       (SIMDDoubleComputer / SISDDoubleComputer) expects contiguous arrays.
 *       The copy cost is O(segmentLength) and is amortized by SIMD speedup
 *       for segments larger than ~200 elements.</li>
 *   <li>This class is intentionally placed in the {@code decomposition} package,
 *       NOT in {@code linalg} — it is an internal implementation detail of the
 *       decomposition/solver subsystem, not a general-purpose utility.</li>
 *   <li>Do NOT import or use this class from outside the decomposition and
 *       decomposition.solver packages. It is not part of the public API.</li>
 * </ul>
 *
 * @author RereMouse
 * @since 2.0
 */
public final class SegmentDoubleVector {

    /** Backing array. Operations read from and write to this array. */
    private final double[] base;
    /** Start index within {@link #base} for this segment. */
    private final int offset;
    /** Number of elements in this segment. */
    private final int length;

    /**
     * Each SegmentDoubleVector holds its own computer instance.
     * {@link DoubleVectorComputer} is lightweight (it only creates the
     * concrete SIMD/SISD/GPU computer on first use via lazy initialization).
     */
    private final DoubleVectorComputer computer;

    /**
     * Create a segment view over {@code base[offset..offset+length-1]}.
     *
     * @param base   backing array (not copied — referenced directly)
     * @param offset start index in the backing array
     * @param length number of elements
     */
    public SegmentDoubleVector(double[] base, int offset, int length) {
        this.base = base;
        this.offset = offset;
        this.length = length;
        this.computer = new DoubleVectorComputer();
    }

    /**
     * Create a segment view over the entire backing array.
     *
     * @param base backing array (not copied — referenced directly)
     */
    public SegmentDoubleVector(double[] base) {
        this(base, 0, base.length);
    }

    /** Number of elements in this segment. */
    public int size() {
        return length;
    }

    /**
     * Dot product with another segment. Both segments must have the same length.
     * <p>
     * Routes through {@link DoubleVectorComputer#binaryReduceOperate} which
     * auto-dispatches HPC → SIMD → SISD based on segment size and availability.
     * </p>
     *
     * @param other another segment of equal length
     * @return dot product Σ this[i] * other[i]
     */
    public double dot(SegmentDoubleVector other) {
        double[] a = toContiguous();
        double[] b = other.toContiguous();
        return computer.binaryReduceOperate(a, b, BinaryReduceOperation.DOT);
    }

    /**
     * In-place AXPY: {@code this[i] += alpha * other[i]} for all i.
     * <p>
     * Routes through {@link DoubleVectorComputer#binaryOperate} for SIMD/HPC dispatch,
     * then writes the result back to the backing array segment.
     * </p>
     * <p>
     * Optimized fast paths: {@code alpha == 1.0} uses a single ADD operation;
     * {@code alpha == -1.0} uses a single SUBTRACT operation.
     * Other values use MULTIPLY followed by ADD.
     * </p>
     *
     * @param alpha scalar multiplier
     * @param other another segment of equal length, not mutated
     */
    public void axpy(double alpha, SegmentDoubleVector other) {
        double[] a = toContiguous();
        double[] b = other.toContiguous();
        double[] result;
        if (alpha == 1.0) {
            result = computer.binaryOperate(a, b, BinaryOperation.ADD);
        } else if (alpha == -1.0) {
            result = computer.binaryOperate(a, b, BinaryOperation.SUBTRACT);
        } else {
            double[] scaled = computer.binaryOperate(b, alpha, BinaryOperation.MULTIPLY);
            result = computer.binaryOperate(a, scaled, BinaryOperation.ADD);
        }
        System.arraycopy(result, 0, base, offset, length);
    }

    /**
     * In-place scalar division: {@code this[i] /= alpha} for all i.
     * <p>
     * Routes through {@link DoubleVectorComputer#binaryOperate} with DIVIDE operation.
     * </p>
     *
     * @param alpha divisor (must not be zero)
     * @throws ArithmeticException if alpha is zero
     */
    public void divideInPlace(double alpha) {
        double[] a = toContiguous();
        double[] result = computer.binaryOperate(a, alpha, BinaryOperation.DIVIDE);
        System.arraycopy(result, 0, base, offset, length);
    }

    /**
     * Copies the segment from {@code base[offset..offset+length-1]} into a new
     * contiguous array suitable for passing to the computer layer.
     */
    private double[] toContiguous() {
        double[] a = new double[length];
        System.arraycopy(base, offset, a, 0, length);
        return a;
    }
}
