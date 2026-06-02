package com.yishape.lab.math.linalg.decomposition.impl;

/**
 * LAPACK DGEQP3 + DLAQPS 意图的列主元 QR：在 {@link RereQrcpDgeqp3Decomposition} 递推范数基础上，
 * 对尾随列的范数更新按宽度 {@code columnTileSize} 分条遍历，改善大 {@code n} 时缓存局部性；
 * Householder 与数值结果与 DGEQP3 式实现一致。完整 Level-3 块反射（DLARFB 类）可在此类上继续扩展。
 */
public class RereQrcpDlaqpsDecomposition extends RereQrcpDgeqp3Decomposition {

    /** 尾随列范数更新的列条带宽度（类 LAPACK 面板宽度对尾随更新的分块）。 */
    private final int columnTileSize;

    public RereQrcpDlaqpsDecomposition() {
        this(32);
    }

    public RereQrcpDlaqpsDecomposition(int columnTileSize) {
        super();
        this.columnTileSize = Math.max(1, columnTileSize);
    }

    public RereQrcpDlaqpsDecomposition(double threshold, int columnTileSize) {
        super(threshold);
        this.columnTileSize = Math.max(1, columnTileSize);
    }

    public RereQrcpDlaqpsDecomposition(double epsilon, int maxIterations, int columnTileSize) {
        super(epsilon, maxIterations);
        this.columnTileSize = Math.max(1, columnTileSize);
    }

    public RereQrcpDlaqpsDecomposition(double threshold, double normRecurrenceEpsilon, int columnTileSize) {
        super(threshold, normRecurrenceEpsilon);
        this.columnTileSize = Math.max(1, columnTileSize);
    }

    public RereQrcpDlaqpsDecomposition(double epsilon, int maxIterations, double normRecurrenceEpsilon, int columnTileSize) {
        super(epsilon, maxIterations, normRecurrenceEpsilon);
        this.columnTileSize = Math.max(1, columnTileSize);
    }

    public int getColumnTileSize() {
        return columnTileSize;
    }

    @Override
    protected void updatePartialColumnNormsLapackStyle(int minor, double[][] matrix) {
        int n = matrix.length;
        final int tile = columnTileSize;
        for (int j0 = minor + 1; j0 < n; j0 += tile) {
            int j1 = Math.min(j0 + tile, n);
            for (int j = j0; j < j1; j++) {
                LAPACK_updateColumnNormEstimateForIndex(j, minor, matrix);
            }
        }
    }
}
