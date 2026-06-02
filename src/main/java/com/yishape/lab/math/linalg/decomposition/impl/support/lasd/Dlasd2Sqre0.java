package com.yishape.lab.math.linalg.decomposition.impl.support.lasd;

import com.yishape.lab.math.linalg.decomposition.impl.support.GivensDrotImatrix;
import com.yishape.lab.math.util.RerePrecision;

/**
 * Reference-LAPACK {@code dlasd2.f}，SQRE=0（M=N）。除 Java 数组第 0 格未使用外，下标 1..N 与 Fortran 一致。
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public final class Dlasd2Sqre0 {

    private static final double ZERO = 0.0;
    private static final double ONE = 1.0;
    private static final double TWO = 2.0;
    private static final double EIGHT = 8.0;

    private Dlasd2Sqre0() {
    }

    /**
     * @param idxq 入口须按 LAPACK 约定：前半 {@code 1..NL} 为 {@code 1..NL}；后半 {@code NL+2..N} 为
     *            {@code i-NL-1}（见 {@link LapackDlasdIdxq#initIdxqBeforeDlasd2}）
     * @return secular 维数 K（deflation 后）
     */
    public static int dlasd2(
            int nl,
            int nr,
            double alpha,
            double beta,
            double[] d,
            double[] z,
            double[][] u,
            double[][] vt,
            double[] dsigma,
            double[][] u2,
            double[][] vt2,
            int[] idxp,
            int[] idx,
            int[] idxc,
            int[] idxq,
            int[] coltyp) {
        int n = nl + nr + 1;
        int m = n;
        int nlp1 = nl + 1;
        int nlp2 = nl + 2;

        double z1 = alpha * vt[nlp1][nlp1];
        z[1] = z1;
        for (int i = nl; i >= 1; i--) {
            z[i + 1] = alpha * vt[i][nlp1];
            d[i + 1] = d[i];
            idxq[i + 1] = idxq[i] + 1;
        }
        for (int i = nlp2; i <= m; i++) {
            z[i] = beta * vt[i][nlp2];
        }
        for (int i = 2; i <= nlp1; i++) {
            coltyp[i] = 1;
        }
        for (int i = nlp2; i <= n; i++) {
            coltyp[i] = 2;
        }
        for (int i = nlp2; i <= n; i++) {
            idxq[i] += nlp1;
        }
        for (int i = 2; i <= n; i++) {
            dsigma[i] = d[idxq[i]];
            u2[i][1] = z[idxq[i]];
            idxc[i] = coltyp[idxq[i]];
        }
        double[] mergeA = new double[nl + nr + 2];
        for (int i = 1; i <= nl; i++) {
            mergeA[i] = dsigma[i + 1];
        }
        for (int i = 1; i <= nr; i++) {
            mergeA[nl + i] = dsigma[nl + 1 + i];
        }
        Dlamrg.dlamrg(nl, nr, mergeA, idx, 2);
        for (int i = 2; i <= n; i++) {
            int idxi = 1 + idx[i];
            d[i] = dsigma[idxi];
            z[i] = u2[idxi][1];
            coltyp[i] = idxc[idxi];
        }

        double eps = RerePrecision.MACHINE_EPSILON;
        double tol = Math.max(Math.abs(alpha), Math.abs(beta));
        tol = EIGHT * eps * Math.max(Math.abs(d[n]), tol);

        int k = 1;
        int k2 = n + 1;
        int j = 2;
        boolean branch120 = false;

        while (j <= n) {
            if (Math.abs(z[j]) <= tol) {
                k2--;
                idxp[k2] = j;
                coltyp[j] = 4;
                if (j == n) {
                    branch120 = true;
                    break;
                }
                j++;
            } else {
                break;
            }
        }

        if (!branch120) {
            if (j > n) {
                branch120 = true;
            } else {
                int jprev = j;
                while (true) {
                    j++;
                    if (j > n) {
                        k++;
                        u2[k][1] = z[jprev];
                        dsigma[k] = d[jprev];
                        idxp[k] = jprev;
                        break;
                    }
                    if (Math.abs(z[j]) <= tol) {
                        k2--;
                        idxp[k2] = j;
                        coltyp[j] = 4;
                    } else if ((d[j] - d[jprev]) <= tol) {
                        double s = z[jprev];
                        double c = z[j];
                        double tau = Math.hypot(c, s);
                        c = c / tau;
                        s = -s / tau;
                        z[j] = tau;
                        z[jprev] = ZERO;
                        int innerP = idx[jprev];
                        int inner = idx[j];
                        int idxjp = idxq[innerP + 1];
                        int idxjCol = idxq[inner + 1];
                        if (idxjp <= nlp1) {
                            idxjp--;
                        }
                        if (idxjCol <= nlp1) {
                            idxjCol--;
                        }
                        GivensDrotImatrix.applyColumns1Based(u, n, idxjp, idxjCol, c, s);
                        GivensDrotImatrix.applyRows1Based(vt, idxjp, idxjCol, m, c, s);
                        if (coltyp[j] != coltyp[jprev]) {
                            coltyp[j] = 3;
                        }
                        coltyp[jprev] = 4;
                        k2--;
                        for (int jp = jprev; jp <= j - 1; jp++) {
                            idxp[k2 + j - 1 - jp] = jp;
                        }
                        jprev = j;
                    } else {
                        k++;
                        u2[k][1] = z[jprev];
                        dsigma[k] = d[jprev];
                        idxp[k] = jprev;
                        jprev = j;
                    }
                }
            }
        }

        int[] ctot = new int[5];
        for (int jj = 2; jj <= n; jj++) {
            ctot[coltyp[jj]]++;
        }
        int[] psm = new int[5];
        psm[1] = 2;
        psm[2] = 2 + ctot[1];
        psm[3] = psm[2] + ctot[2];
        psm[4] = psm[3] + ctot[3];
        for (int jj = 2; jj <= n; jj++) {
            int jp = idxp[jj];
            int ct = coltyp[jp];
            idxc[psm[ct]] = jj;
            psm[ct]++;
        }
        for (int jj = 2; jj <= n; jj++) {
            int jp = idxp[jj];
            dsigma[jj] = d[jp];
            int inner = idx[idxp[idxc[jj]]];
            int idxj = idxq[inner + 1];
            if (idxj <= nlp1) {
                idxj--;
            }
            for (int row = 1; row <= n; row++) {
                u2[row][jj] = u[row][idxj];
            }
            for (int col = 1; col <= m; col++) {
                vt2[jj][col] = vt[idxj][col];
            }
        }

        dsigma[1] = ZERO;
        double hlftol = tol / TWO;
        if (Math.abs(dsigma[2]) <= hlftol) {
            dsigma[2] = hlftol;
        }
        if (Math.abs(z1) <= tol) {
            z[1] = tol;
        } else {
            z[1] = z1;
        }
        for (int ii = 2; ii <= k; ii++) {
            z[ii] = u2[ii][1];
        }

        for (int row = 1; row <= n; row++) {
            for (int c = 1; c <= n; c++) {
                u2[row][c] = ZERO;
            }
        }
        u2[nlp1][1] = ONE;
        for (int i = 1; i <= m; i++) {
            vt2[1][i] = vt[nlp1][i];
        }

        if (n > k) {
            System.arraycopy(dsigma, k + 1, d, k + 1, n - k);
            for (int row = 1; row <= n; row++) {
                for (int c = 1; c <= n - k; c++) {
                    u[row][k + c] = u2[row][k + c];
                }
            }
            for (int row = 1; row <= n - k; row++) {
                for (int col = 1; col <= m; col++) {
                    vt[k + row][col] = vt2[k + row][col];
                }
            }
        }

        for (int t = 1; t <= 4; t++) {
            coltyp[t] = ctot[t];
        }
        return k;
    }
}
