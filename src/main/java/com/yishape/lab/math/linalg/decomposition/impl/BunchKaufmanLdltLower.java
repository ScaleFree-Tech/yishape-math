package com.yishape.lab.math.linalg.decomposition.impl;

/**
 * LAPACK {@code DSYTF2} / {@code DSYTRS}, {@code UPLO='L'}：对称不定 Bunch–Kaufman
 * {@code L·D·Lᵀ} 分解与三角求解（稠密、0 基下标）。
 * <p>
 * {@code ipiv} 约定：1×1 主元时 {@code ipiv[k] ≥ 0} 为与行 {@code k} 交换的行号；
 * 2×2 主元时 {@code ipiv[k] = ipiv[k+1] = -(pivotRow+1)}（与 Fortran 的 {@code -KP} 一致，{@code KP} 为 1 基）。
 * </p>
 */
public final class BunchKaufmanLdltLower {

    private static final double ALPHA = (1.0 + Math.sqrt(17.0)) / 8.0;

    private BunchKaufmanLdltLower() {
    }

    /**
     * @return {@code info}：0 成功；{@code >0} 时为主元列（1 基，与 LAPACK 一致）
     */
    public static int dsytf2Lower(double[][] a, int n, int[] ipiv) {
        if (n <= 0) {
            return 0;
        }
        int info = 0;
        int k = 0;
        while (k < n) {
            int kstep = 1;
            double absakk = Math.abs(a[k][k]);
            double colmax;
            int imax = k;
            if (k < n - 1) {
                imax = k + 1;
                colmax = Math.abs(a[k + 1][k]);
                for (int i = k + 2; i < n; i++) {
                    double v = Math.abs(a[i][k]);
                    if (v > colmax) {
                        colmax = v;
                        imax = i;
                    }
                }
            } else {
                colmax = 0.0;
            }

            int kp = k;
            if (max(absakk, colmax) == 0.0 || Double.isNaN(absakk)) {
                if (info == 0) {
                    info = k + 1;
                }
                kp = k;
            } else {
                if (absakk >= ALPHA * colmax) {
                    kp = k;
                } else {
                    int jmax = k;
                    double rowmax = Math.abs(a[imax][k]);
                    for (int j = k + 1; j < imax; j++) {
                        double v = Math.abs(a[imax][j]);
                        if (v > rowmax) {
                            rowmax = v;
                            jmax = j;
                        }
                    }
                    if (imax < n - 1) {
                        jmax = imax + 1;
                        rowmax = Math.max(rowmax, Math.abs(a[jmax][imax]));
                        for (int i = imax + 2; i < n; i++) {
                            double v = Math.abs(a[i][imax]);
                            if (v > rowmax) {
                                rowmax = v;
                                jmax = i;
                            }
                        }
                    }
                    if (rowmax == 0.0 || Double.isNaN(rowmax)) {
                        kp = k;
                    } else if (absakk >= ALPHA * colmax * (colmax / rowmax)) {
                        kp = k;
                    } else if (Math.abs(a[imax][imax]) >= ALPHA * rowmax) {
                        kp = imax;
                    } else {
                        kp = imax;
                        kstep = 2;
                    }
                }
            }

            int kk = k + kstep - 1;
            if (kp != kk) {
                interchangeTrailingLower(a, n, k, kstep, kp, kk);
            }

            if (kstep == 1) {
                if (max(absakk, colmax) != 0.0 && !Double.isNaN(absakk) && k < n - 1) {
                    double d11 = 1.0 / a[k][k];
                    dsyrLowerTail(a, n, k, d11);
                    for (int i = k + 1; i < n; i++) {
                        a[i][k] *= d11;
                    }
                }
            } else {
                if (k < n - 2) {
                    double d21 = a[k + 1][k];
                    double d11 = a[k + 1][k + 1] / d21;
                    double d22 = a[k][k] / d21;
                    double t = 1.0 / (d11 * d22 - 1.0);
                    d21 = t / d21;
                    for (int j = k + 2; j < n; j++) {
                        double wk = d21 * (d11 * a[j][k] - a[j][k + 1]);
                        double wkp1 = d21 * (d22 * a[j][k + 1] - a[j][k]);
                        for (int i = j; i < n; i++) {
                            a[i][j] -= a[i][k] * wk + a[i][k + 1] * wkp1;
                        }
                        a[j][k] = wk;
                        a[j][k + 1] = wkp1;
                    }
                }
            }

            if (kstep == 1) {
                ipiv[k] = kp;
            } else {
                int np = -(kp + 1);
                ipiv[k] = np;
                ipiv[k + 1] = np;
            }
            k += kstep;
        }
        return info;
    }

    private static void interchangeTrailingLower(double[][] a, int n, int k, int kstep, int kp, int kk) {
        if (kp < n - 1) {
            for (int i = kp + 1; i < n; i++) {
                double tmp = a[i][kk];
                a[i][kk] = a[i][kp];
                a[i][kp] = tmp;
            }
        }
        for (int i = 0; i < kp - kk - 1; i++) {
            double tmp = a[kk + 1 + i][kk];
            a[kk + 1 + i][kk] = a[kp][kk + 1 + i];
            a[kp][kk + 1 + i] = tmp;
        }
        double t = a[kk][kk];
        a[kk][kk] = a[kp][kp];
        a[kp][kp] = t;
        if (kstep == 2) {
            t = a[k + 1][k];
            a[k + 1][k] = a[kp][k];
            a[kp][k] = t;
        }
    }

    private static void dsyrLowerTail(double[][] a, int n, int k, double negInvD) {
        for (int j = k + 1; j < n; j++) {
            for (int i = j; i < n; i++) {
                a[i][j] -= negInvD * a[i][k] * a[j][k];
            }
        }
    }

    private static double max(double a, double b) {
        return a > b ? a : b;
    }

    /**
     * 求解 {@code A·X=B}，{@code A} 已由 {@link #dsytf2Lower} 分解；{@code B} 被覆盖为 {@code X}。
     */
    public static void dsytrsLower(double[][] a, int n, int[] ipiv, double[][] b, int nrhs) {
        if (n == 0 || nrhs == 0) {
            return;
        }
        // L * D * X = B
        int k = 0;
        while (k < n) {
            if (k < ipiv.length && ipiv[k] >= 0) {
                int kp = ipiv[k];
                if (kp != k) {
                    swapRows(b, nrhs, k, kp);
                }
                if (k < n - 1) {
                    for (int j = 0; j < nrhs; j++) {
                        double bk = b[k][j];
                        for (int i = k + 1; i < n; i++) {
                            b[i][j] -= a[i][k] * bk;
                        }
                    }
                }
                double invDiag = 1.0 / a[k][k];
                for (int j = 0; j < nrhs; j++) {
                    b[k][j] *= invDiag;
                }
                k++;
            } else {
                int kp = -ipiv[k] - 1;
                if (kp != k + 1) {
                    swapRows(b, nrhs, k + 1, kp);
                }
                if (k < n - 2) {
                    for (int j = 0; j < nrhs; j++) {
                        double bk0 = b[k][j];
                        double bk1 = b[k + 1][j];
                        for (int i = k + 2; i < n; i++) {
                            b[i][j] -= a[i][k] * bk0 + a[i][k + 1] * bk1;
                        }
                    }
                }
                double akm1k = a[k + 1][k];
                double akm1 = a[k][k] / akm1k;
                double ak = a[k + 1][k + 1] / akm1k;
                double denom = akm1 * ak - 1.0;
                for (int j = 0; j < nrhs; j++) {
                    double bkm1 = b[k][j] / akm1k;
                    double bk = b[k + 1][j] / akm1k;
                    b[k][j] = (ak * bkm1 - bk) / denom;
                    b[k + 1][j] = (akm1 * bk - bkm1) / denom;
                }
                k += 2;
            }
        }

        // L^T * X = (previous X)
        k = n - 1;
        while (k >= 0) {
            if (ipiv[k] >= 0) {
                if (k < n - 1) {
                    for (int j = 0; j < nrhs; j++) {
                        double s = 0.0;
                        for (int i = k + 1; i < n; i++) {
                            s += b[i][j] * a[i][k];
                        }
                        b[k][j] -= s;
                    }
                }
                int kp = ipiv[k];
                if (kp != k) {
                    swapRows(b, nrhs, k, kp);
                }
                k--;
            } else {
                if (k < n - 1) {
                    for (int j = 0; j < nrhs; j++) {
                        double s0 = 0.0;
                        double s1 = 0.0;
                        for (int i = k + 1; i < n; i++) {
                            s0 += b[i][j] * a[i][k];
                            s1 += b[i][j] * a[i][k - 1];
                        }
                        b[k][j] -= s0;
                        b[k - 1][j] -= s1;
                    }
                }
                int kp = -ipiv[k] - 1;
                if (kp != k) {
                    swapRows(b, nrhs, k, kp);
                }
                k -= 2;
            }
        }
    }

    private static void swapRows(double[][] b, int nrhs, int r1, int r2) {
        if (r1 == r2) {
            return;
        }
        for (int j = 0; j < nrhs; j++) {
            double t = b[r1][j];
            b[r1][j] = b[r2][j];
            b[r2][j] = t;
        }
    }

    /**
     * 由分解块计算行列式（符号不变：对称行列交换成对出现）。
     */
    public static double determinantFromFactor(double[][] a, int n, int[] ipiv, double singularityEps) {
        double det = 1.0;
        int k = 0;
        while (k < n) {
            if (ipiv[k] >= 0) {
                double d = a[k][k];
                if (Math.abs(d) <= singularityEps) {
                    return 0.0;
                }
                det *= d;
                k++;
            } else {
                double a11 = a[k][k];
                double a22 = a[k + 1][k + 1];
                double a21 = a[k + 1][k];
                double d2 = a11 * a22 - a21 * a21;
                if (Math.abs(d2) <= singularityEps) {
                    return 0.0;
                }
                det *= d2;
                k += 2;
            }
        }
        return det;
    }
}
