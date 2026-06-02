package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * 对 {@code A·P = Q·R}（列主元 QR）的最小二乘 / 方阵求解：先得到 pivoted 坐标下的 {@code Z}（满足 {@code Q·R·Z ≈ B}），
 * 再还原 {@code X = P·Z}，使得 {@code A·X ≈ B}。
 */
public class QrcpDecompositionSolver implements IDecompositionSolver {

    private final QRDecompositionSolver pivotedQr;
    private final int[] columnPivot;

    public QrcpDecompositionSolver(QRDecompositionSolver pivotedQr, int[] columnPivot) {
        this.pivotedQr = pivotedQr;
        this.columnPivot = columnPivot;
    }

    @Override
    public boolean isNonSingular() {
        return pivotedQr.isNonSingular();
    }

    @Override
    public IVector<Double> solve(IVector<Double> b) {
        IVector<Double> z = pivotedQr.solve(b);
        double[] zd = z.toDoubleArray();
        int n = columnPivot.length;
        double[] xd = new double[n];
        for (int i = 0; i < n; i++) {
            xd[columnPivot[i]] = zd[i];
        }
        return Linalg.vector(xd);
    }

    @Override
    public IMatrix<Double> solve(IMatrix<Double> b) {
        IMatrix<Double> z = pivotedQr.solve(b);
        int n = columnPivot.length;
        int nrhs = z.cols();
        double[][] x = new double[n][nrhs];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nrhs; j++) {
                x[columnPivot[i]][j] = z.get(i, j);
            }
        }
        return Linalg.matrix(x);
    }

    @Override
    public IMatrix<Double> getInverse() {
        int n = columnPivot.length;
        return solve(Linalg.eye(n));
    }
}
