package com.yishape.lab.math.linalg.decomposition.impl.support.lasd;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.support.DiagonalPlusRankOneSymmetricEigenSolver;
import com.yishape.lab.math.util.RerePrecision;

/**
 * Reference-LAPACK {@code dlasd3.f}（SQRE=0），用
 * {@link DiagonalPlusRankOneSymmetricEigenSolver} 替代对 {@code DLASD4} 的逐列调用；
 * 左/右向量更新对满块使用 {@code U2(:,1:K)*Q} 与 {@code Q^T*VT2(1:K,:)}，与 Fortran 在 K=2 时等价，
 * 对更大 K 在 CTOT 仅产生平凡分块时亦等价（满稠密 Q）。
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public final class Dlasd3Sqre0Eigen {

    private Dlasd3Sqre0Eigen() {
    }

    /**
     * @param k       有效秩（secular 次数）
     * @param n       {@code uOut} 对应的左向量维度
     * @param m       {@code vtOut} 对应的右向量维度（列）
     * @param dsigma  dlasd2 后：{@code dsigma[1]=0}，{@code dsigma[2..K]} 为极点（升序 σ）
     * @param z       长度 n+1，secular 更新向量；将读 {@code z[1..k]}
     * @param u2      dlasd2 输出工作区（左向量块）
     * @param vt2     dlasd2 输出工作区（右向量块）
     * @param uOut    会被覆盖前 k 列：{@code uOut} n×n（1-based 逻辑行列）
     * @param vtOut   会被覆盖前 k 行：{@code vtOut} m×n（行 1..m，列 1..n）
     * @param dOut    长度 k+1，1-based 写出升序奇异值 {@code dOut[1..k]}
     * @param tolAbs  收敛/判定阈值下界
     */
    public static void dlasd3(
            int k,
            int n,
            int m,
            double[] dsigma,
            double[] z,
            double[][] u2,
            double[][] vt2,
            double[][] uOut,
            double[][] vtOut,
            double[] dOut,
            double tolAbs) {
        if (k == 1) {
            dOut[1] = Math.abs(z[1]);
            for (int i = 1; i <= m; i++) {
                vtOut[1][i] = vt2[1][i];
            }
            if (z[1] > 0) {
                for (int i = 1; i <= n; i++) {
                    uOut[i][1] = u2[i][1];
                }
            } else {
                for (int i = 1; i <= n; i++) {
                    uOut[i][1] = -u2[i][1];
                }
            }
            return;
        }

        double znrm = 0.0;
        for (int i = 1; i <= k; i++) {
            znrm += z[i] * z[i];
        }
        if (znrm < RerePrecision.SAFE_MIN * 16) {
            throw new IllegalStateException("dlasd3: zero z norm");
        }
        double rho = znrm;
        double inv = 1.0 / Math.sqrt(znrm);
        double[] dSq = new double[k];
        double[] zHat = new double[k];
        for (int i = 0; i < k; i++) {
            double sig = dsigma[i + 1];
            dSq[i] = sig * sig;
            zHat[i] = z[i + 1] * inv;
        }

        DiagonalPlusRankOneSymmetricEigenSolver.EigenResult er =
                DiagonalPlusRankOneSymmetricEigenSolver.solve(dSq, zHat, rho, Math.max(tolAbs, 1e-14));

        double[] sigAsc = new double[k];
        IMatrix<Double> qm = Linalg.zeros(k, k);
        for (int j = 0; j < k; j++) {
            int jc = k - 1 - j;
            double lam = er.eigenvalues[jc];
            if (lam < 0 && lam > -1e-10 * (1 + Math.abs(rho))) {
                lam = 0;
            }
            sigAsc[j] = Math.sqrt(Math.max(0, lam));
            dOut[j + 1] = sigAsc[j];
            for (int i = 0; i < k; i++) {
                qm.set(i, j, er.eigenvectors[i][jc]);
            }
        }

        IMatrix<Double> u2sub = Linalg.zeros(n, k);
        for (int r = 1; r <= n; r++) {
            for (int c = 1; c <= k; c++) {
                u2sub.set(r - 1, c - 1, u2[r][c]);
            }
        }
        IMatrix<Double> uLeft = u2sub.mmul(qm);
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < k; c++) {
                uOut[r + 1][c + 1] = uLeft.get(r, c);
            }
        }

        IMatrix<Double> vt2sub = Linalg.zeros(k, m);
        for (int r = 1; r <= k; r++) {
            for (int c = 1; c <= m; c++) {
                vt2sub.set(r - 1, c - 1, vt2[r][c]);
            }
        }
        IMatrix<Double> vtTop = qm.transposeNew().mmul(vt2sub);
        for (int r = 0; r < k; r++) {
            for (int c = 0; c < m; c++) {
                vtOut[r + 1][c + 1] = vtTop.get(r, c);
            }
        }
    }
}
