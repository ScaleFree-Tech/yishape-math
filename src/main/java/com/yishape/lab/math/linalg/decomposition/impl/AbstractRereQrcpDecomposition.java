package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.IQrcpDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.QRDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.QrcpDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.QrcpLeastSquaresTolerantSolver;
import com.yishape.lab.util.Tuple2;

/**
 * 列主元 QR 基类：{@code A·P = Q·R}，子类实现选主元策略；提供严格 {@link #getSolver()} 与
 * xGELSY 风格 {@link #createTolerantLeastSquaresSolver(double)}。
 */
public abstract class AbstractRereQrcpDecomposition extends RereQRDecomposition implements IQrcpDecomposition {

    protected int[] columnPivot;
    private IMatrix<Double> cachedColumnPermutation;

    protected AbstractRereQrcpDecomposition() {
        super();
    }

    protected AbstractRereQrcpDecomposition(double threshold) {
        super(threshold);
    }

    protected AbstractRereQrcpDecomposition(double epsilon, int maxIterations) {
        super(epsilon, maxIterations);
    }

    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon, int maxIterations) {
        cachedColumnPermutation = null;
        columnPivot = null;
        return super.decompose(matrix, epsilon, maxIterations);
    }

    @Override
    protected void decompose(double[][] matrix) {
        columnPivot = new int[matrix.length];
        for (int i = 0; i < columnPivot.length; i++) {
            columnPivot[i] = i;
        }
        onBeforeQrcpSteps(matrix);
        final int lim = Math.min(matrix.length, matrix[0].length);
        for (int minor = 0; minor < lim; minor++) {
            performHouseholderReflection(minor, matrix);
        }
    }

    /** 在第一次 Householder 之前调用（例如初始化列范数工作区）。 */
    protected void onBeforeQrcpSteps(double[][] matrix) {
    }

    protected static void swapColumns(int i, int j, double[][] qrt, int[] pivots) {
        if (i == j) {
            return;
        }
        double[] tc = qrt[i];
        qrt[i] = qrt[j];
        qrt[j] = tc;
        int tp = pivots[i];
        pivots[i] = pivots[j];
        pivots[j] = tp;
    }

    @Override
    public int[] getColumnPivot() {
        if (columnPivot == null) {
            throw new IllegalStateException("Decomposition not performed yet");
        }
        return columnPivot.clone();
    }

    @Override
    public IMatrix<Double> getColumnPermutation() {
        if (columnPivot == null) {
            throw new IllegalStateException("Decomposition not performed yet");
        }
        if (cachedColumnPermutation == null) {
            int n = columnPivot.length;
            IMatrix<Double> p = Linalg.zeros(n, n);
            for (int j = 0; j < n; j++) {
                p.put(columnPivot[j], j, 1.0);
            }
            cachedColumnPermutation = p;
        }
        return cachedColumnPermutation;
    }

    @Override
    public IDecompositionSolver getSolver() {
        if (qrt == null || columnPivot == null) {
            throw new IllegalStateException("Decomposition not performed yet");
        }
        QRDecompositionSolver inner = new QRDecompositionSolver(qrt, rDiag, threshold);
        return new QrcpDecompositionSolver(inner, columnPivot);
    }

    @Override
    public IDecompositionSolver createTolerantLeastSquaresSolver(double rankTolerance) {
        if (qrt == null || columnPivot == null) {
            throw new IllegalStateException("Decomposition not performed yet");
        }
        if (rankTolerance < 0.0) {
            throw new IllegalArgumentException("rankTolerance must be non-negative");
        }
        return new QrcpLeastSquaresTolerantSolver(qrt, rDiag, columnPivot.clone(), rankTolerance, threshold);
    }
}
