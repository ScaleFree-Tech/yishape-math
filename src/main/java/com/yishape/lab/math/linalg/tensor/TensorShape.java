package com.yishape.lab.math.linalg.tensor;

import java.util.Arrays;

public class TensorShape {
    private final int[] shape;
    private final int rank;
    private final long totalSize;
    private final int[] strides;

    public TensorShape(int... shape) {
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Shape cannot be null or empty");
        }
        for (int dim : shape) {
            if (dim < 0) {
                throw new IllegalArgumentException("Shape dimensions cannot be negative: " + dim);
            }
        }
        this.shape = shape.clone();
        this.rank = shape.length;
        this.totalSize = computeTotalSize(shape);
        this.strides = computeCStrides(shape);
    }

    // internal constructor accepting custom strides
    TensorShape(int[] shape, int[] strides) {
        this.shape = shape.clone();
        this.rank = shape.length;
        this.totalSize = computeTotalSize(shape);
        this.strides = strides.clone();
    }

    private static long computeTotalSize(int[] shape) {
        long size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    public static int[] computeCStrides(int[] shape) {
        int[] strides = new int[shape.length];
        if (shape.length > 0) {
            strides[shape.length - 1] = 1;
            for (int i = shape.length - 2; i >= 0; i--) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        return strides;
    }

    public int rank() {
        return rank;
    }

    public int[] shape() {
        return shape.clone();
    }

    public int dim(int axis) {
        if (axis < 0) axis += rank;
        return shape[axis];
    }

    public long totalSize() {
        return totalSize;
    }

    public int[] strides() {
        return strides.clone();
    }

    public int stride(int axis) {
        if (axis < 0) axis += rank;
        return strides[axis];
    }

    /**
     * 检查当前 strides 是否等于 C-order.
     */
    public boolean isContiguous() {
        int[] cs = computeCStrides(shape);
        return Arrays.equals(strides, cs);
    }

    public boolean isScalar() {
        return rank == 0;
    }

    public boolean isVector() {
        return rank == 1;
    }

    public boolean isMatrix() {
        return rank == 2;
    }

    public boolean isCube() {
        return rank == 3;
    }

    public long linearIndex(int... indices) {
        if (indices.length != rank) {
            throw new IllegalArgumentException("Number of indices must match rank: " + indices.length + " vs " + rank);
        }
        long index = 0;
        for (int i = 0; i < rank; i++) {
            int idx = indices[i];
            if (idx < 0 || idx >= shape[i]) {
                throw new IndexOutOfBoundsException("Index out of bounds at axis " + i + ": " + idx + " >= " + shape[i]);
            }
            index += idx * (long) strides[i];
        }
        return index;
    }

    public int[] unlinearizeIndex(long linearIndex) {
        if (linearIndex < 0 || linearIndex >= totalSize) {
            throw new IndexOutOfBoundsException("Linear index out of bounds: " + linearIndex);
        }
        int[] indices = new int[rank];
        long remaining = linearIndex;
        for (int i = 0; i < rank; i++) {
            indices[i] = (int) (remaining / strides[i]);
            remaining = remaining % strides[i];
        }
        return indices;
    }

    public TensorShape reshape(int... newShape) {
        // 处理 -1 自动推断
        newShape = inferReshape(newShape);
        long newSize = computeTotalSize(newShape);
        if (newSize != totalSize) {
            throw new IllegalArgumentException("Cannot reshape from " + totalSize + " to " + newSize);
        }
        return new TensorShape(newShape);
    }

    /**
     * 推断 reshape 中的 -1 占位符.
     */
    public int[] inferReshape(int... partialShape) {
        int[] result = partialShape.clone();
        int minusOneIdx = -1;
        long known = 1;
        for (int i = 0; i < partialShape.length; i++) {
            if (partialShape[i] == -1) {
                if (minusOneIdx >= 0) {
                    throw new IllegalArgumentException("Only one -1 is allowed in reshape");
                }
                minusOneIdx = i;
            } else if (partialShape[i] == 0) {
                // 0 表示保持该维度不变: 仅在 reshape 中有效
                result[i] = shape[i];
                known *= shape[i];
            } else {
                known *= partialShape[i];
            }
        }
        if (minusOneIdx >= 0) {
            result[minusOneIdx] = (int) (totalSize / known);
            if (known * result[minusOneIdx] != totalSize) {
                throw new IllegalArgumentException("Shape " + Arrays.toString(partialShape)
                    + " is invalid for totalSize " + totalSize);
            }
        }
        return result;
    }

    public TensorShape permute(int... axes) {
        if (axes.length != rank) {
            throw new IllegalArgumentException("Axes length must match rank");
        }
        int[] newShape = new int[rank];
        int[] newStrides = new int[rank];
        for (int i = 0; i < rank; i++) {
            int ax = axes[i];
            if (ax < 0 || ax >= rank) {
                throw new IllegalArgumentException("Invalid axis: " + ax);
            }
            newShape[i] = shape[ax];
            newStrides[i] = strides[ax];
        }
        return new TensorShape(newShape, newStrides);
    }

    public TensorShape squeeze(int... axes) {
        if (axes.length == 0) {
            int[] newShape = Arrays.stream(shape).filter(d -> d != 1).toArray();
            if (newShape.length == 0) newShape = new int[]{1};
            return new TensorShape(newShape);
        }
        boolean[] remove = new boolean[rank];
        for (int ax : axes) {
            int a = ax < 0 ? ax + rank : ax;
            if (shape[a] != 1) {
                throw new IllegalArgumentException("Cannot squeeze dimension " + a + " with size " + shape[a]);
            }
            remove[a] = true;
        }
        int[] newShape = new int[rank - axes.length];
        int[] newStrides = new int[rank - axes.length];
        int idx = 0;
        for (int i = 0; i < rank; i++) {
            if (!remove[i]) {
                newShape[idx] = shape[i];
                newStrides[idx] = strides[i];
                idx++;
            }
        }
        return newShape.length == 0 ? new TensorShape(1) : new TensorShape(newShape, newStrides);
    }

    public TensorShape unsqueeze(int axis) {
        if (axis < 0) axis = rank + 1;
        if (axis < 0 || axis > rank) {
            throw new IllegalArgumentException("Invalid axis: " + axis);
        }
        int[] newShape = new int[rank + 1];
        int[] newStrides = new int[rank + 1];
        for (int i = 0; i < axis; i++) {
            newShape[i] = shape[i];
            newStrides[i] = strides[i];
        }
        newShape[axis] = 1;
        newStrides[axis] = 0; // 广播友好
        for (int i = axis; i < rank; i++) {
            newShape[i + 1] = shape[i];
            newStrides[i + 1] = strides[i];
        }
        return new TensorShape(newShape, newStrides);
    }

    /**
     * 将低秩 shape 左对齐到目标秩（用于广播）.
     */
    public TensorShape alignTo(int targetRank) {
        if (rank >= targetRank) return this;
        int[] newShape = new int[targetRank];
        int[] newStrides = new int[targetRank];
        int diff = targetRank - rank;
        for (int i = 0; i < diff; i++) {
            newShape[i] = 1;
            newStrides[i] = 0; // broadcast
        }
        for (int i = 0; i < rank; i++) {
            newShape[diff + i] = shape[i];
            newStrides[diff + i] = strides[i];
        }
        return new TensorShape(newShape, newStrides);
    }

    /**
     * 计算广播后的 shape（自动左对齐不同秩）.
     */
    public int[] broadcastShape(TensorShape other) {
        return broadcastShape(this.shape, other.shape);
    }

    /**
     * 静态广播方法: 左对齐后逐维取 max，任一维须为 1.
     *
     * <p><b>⚠️ Left-alignment trap:</b> Broadcasting uses NumPy/PyTorch-style
     * left-alignment — the smaller-rank shape is padded with 1s on the <b>left</b>.
     * This means {@code [64]} vs {@code [4, 4, 4]} becomes
     * {@code [1, 1, 64]} vs {@code [4, 4, 4]}, and axis 2 fails because
     * 4 ≠ 64 and neither is 1.</p>
     *
     * <p><b>Workaround:</b> Explicitly reshape to matching rank, e.g.,
     * {@code reshape(1, 1, 64)} before the operation.</p>
     */
    public static int[] broadcastShape(int[] shapeA, int[] shapeB) {
        int maxRank = Math.max(shapeA.length, shapeB.length);
        int[] a = alignLeft(shapeA, maxRank);
        int[] b = alignLeft(shapeB, maxRank);
        int[] result = new int[maxRank];
        for (int i = 0; i < maxRank; i++) {
            if (a[i] == b[i]) {
                result[i] = a[i];
            } else if (a[i] == 1) {
                result[i] = b[i];
            } else if (b[i] == 1) {
                result[i] = a[i];
            } else {
                String origA = shapeToString(shapeA);
                String origB = shapeToString(shapeB);
                String alignA = shapeToString(a);
                String alignB = shapeToString(b);
                throw new IllegalArgumentException(
                    "Cannot broadcast " + origA + " with " + origB + ":\n" +
                    "  Left-aligned: " + alignA + " vs " + alignB + "\n" +
                    "  Axis " + i + ": " + a[i] + " != " + b[i] + " (neither is 1)\n" +
                    "  Hint: reshape to matching rank before the operation, e.g.,\n" +
                    "  reshape(" + shapeToString(prependOnes(shapeA, maxRank - shapeA.length)) + ") or\n" +
                    "  reshape(" + shapeToString(prependOnes(shapeB, maxRank - shapeB.length)) + ")");
            }
        }
        return result;
    }

    /** Format a shape array as [d0, d1, ...]. */
    private static String shapeToString(int[] shape) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(shape[i]);
        }
        return sb.append("]").toString();
    }

    /** Prepend n ones to a shape array (for hint generation). */
    private static int[] prependOnes(int[] shape, int n) {
        int[] result = new int[shape.length + n];
        for (int i = 0; i < n; i++) result[i] = 1;
        System.arraycopy(shape, 0, result, n, shape.length);
        return result;
    }

    private static int[] alignLeft(int[] shape, int targetRank) {
        if (shape.length == targetRank) return shape.clone();
        int[] result = new int[targetRank];
        int diff = targetRank - shape.length;
        for (int i = 0; i < diff; i++) result[i] = 1;
        for (int i = 0; i < shape.length; i++) result[diff + i] = shape[i];
        return result;
    }

    /**
     * 子区域 shape（用于 slice/narrow）.
     */
    public TensorShape sliceShape(int axis, int start, int end) {
        int[] newShape = shape.clone();
        newShape[axis] = end - start;
        return new TensorShape(newShape, strides);
    }

    public boolean isCompatibleWith(TensorShape other) {
        if (rank != other.rank) return false;
        for (int i = 0; i < rank; i++) {
            if (shape[i] != other.shape[i] && shape[i] != 1 && other.shape[i] != 1) {
                return false;
            }
        }
        return true;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TensorShape that = (TensorShape) o;
        if (rank != that.rank) return false;
        return Arrays.equals(shape, that.shape);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(shape);
    }

    @Override
    public String toString() {
        if (rank == 0) return "TensorShape[]";
        StringBuilder sb = new StringBuilder("TensorShape[");
        for (int i = 0; i < rank; i++) {
            sb.append(shape[i]);
            if (i < rank - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}
