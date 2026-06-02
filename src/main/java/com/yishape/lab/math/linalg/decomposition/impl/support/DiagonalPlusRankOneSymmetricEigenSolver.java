package com.yishape.lab.math.linalg.decomposition.impl.support;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.util.RerePrecision;
import com.yishape.lab.math.linalg.decomposition.impl.RereEigenDecomposition;
import com.yishape.lab.util.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 对角加秩一矩阵的特征对求解器 / Symmetric Eigenpairs Solver for Diagonal Plus Rank One Matrix
 * <p>
 * 求解 {@code M = diag(d) + ρ z zᵀ} 的特征值和特征向量。
 * Eigenvalues satisfy {@code 1 + ρ Σᵢ zᵢ²/(dᵢ − λ) = 0}.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 2.0
 */
public final class DiagonalPlusRankOneSymmetricEigenSolver {

    private DiagonalPlusRankOneSymmetricEigenSolver() {
    }

    /**
     * 与 {@link #solve(double[], double[], double, double)} 相同，使用 {@link IVector} 以复用 SIMD 友好的存储。
     */
    public static EigenResult solve(IVector<Double> dIn, IVector<Double> zIn, double rho, double tol) {
        int n = dIn.length();
        if (n != zIn.length()) {
            throw new IllegalArgumentException("d and z length mismatch");
        }
        double[] d = new double[n];
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            d[i] = dIn.get(i);
            z[i] = zIn.get(i);
        }
        return solve(d, z, rho, tol);
    }

    /**
     * @param tol absolute tolerance for deflation of small {@code z} entries and secular residuals
     * @return eigenvalues <strong>descending</strong>, eigenvectors as columns {@code Q[r][c]}
     */
    public static EigenResult solve(double[] dIn, double[] zIn, double rho, double tol) {
        if (dIn.length != zIn.length) {
            throw new IllegalArgumentException("d and z length mismatch");
        }
        int n = dIn.length;
        if (n == 0) {
            return new EigenResult(new double[0], new double[0][]);
        }

        double tolAbs = Math.max(tol, RerePrecision.MACHINE_EPSILON * 32);

        if (Math.abs(rho) < RerePrecision.SAFE_MIN * 16) {
            return solveDenseFull(dIn, zIn, rho, tolAbs);
        }

        if (n <= 256) {
            return solveDenseFull(dIn, zIn, rho, tolAbs);
        }

        List<Double> lambda = new ArrayList<>();
        List<double[]> vectors = new ArrayList<>();

        boolean[] fixed = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (Math.abs(zIn[i]) <= tolAbs * (1.0 + Math.abs(dIn[i]))) {
                lambda.add(dIn[i]);
                double[] e = new double[n];
                e[i] = 1.0;
                vectors.add(e);
                fixed[i] = true;
            }
        }

        int rem = n - lambda.size();
        if (rem == 0) {
            return sortDescending(lambda, vectors);
        }

        int[] origIdx = new int[rem];
        double[] dr = new double[rem];
        double[] zr = new double[rem];
        int p = 0;
        for (int i = 0; i < n; i++) {
            if (!fixed[i]) {
                origIdx[p] = i;
                dr[p] = dIn[i];
                zr[p] = zIn[i];
                p++;
            }
        }

        Integer[] ord = new Integer[rem];
        for (int i = 0; i < rem; i++) {
            ord[i] = i;
        }
        Arrays.sort(ord, (a, b) -> Double.compare(dr[a], dr[b]));

        double[] d = new double[rem];
        double[] z = new double[rem];
        int[] back = new int[rem];
        for (int i = 0; i < rem; i++) {
            d[i] = dr[ord[i]];
            z[i] = zr[ord[i]];
            back[i] = origIdx[ord[i]];
        }

        for (int i = 1; i < rem; i++) {
            if (d[i] <= d[i - 1] || d[i] - d[i - 1] < RerePrecision.MACHINE_EPSILON * (1.0 + Math.abs(d[i - 1]))) {
                d[i] = Math.nextUp(Math.max(d[i - 1], d[i]));
            }
        }
        double sep = Math.max(tolAbs * (1.0 + Math.max(Math.abs(d[0]), Math.abs(d[rem - 1]))),
                RerePrecision.SAFE_MIN * 16);

        double[] roots = new double[rem];
        if (rem == 1) {
            roots[0] = d[0] + rho * z[0] * z[0];
        } else {
            double[] sec = secularRootsMultiBracket(d, z, rho, rem, sep, tolAbs);
            if (sec == null || sec.length != rem) {
                return solveDenseFull(dIn, zIn, rho, tolAbs);
            }
            roots = sec;
        }

        for (int j = 0; j < rem; j++) {
            double[] q = new double[n];
            for (int t = 0; t < rem; t++) {
                int g = back[t];
                double den = d[t] - roots[j];
                if (Math.abs(den) < tolAbs * (1.0 + Math.abs(roots[j]))) {
                    den = Math.copySign(Math.max(tolAbs, RerePrecision.SAFE_MIN * 16), den == 0 ? 1.0 : den);
                }
                q[g] = z[t] / den;
            }
            double nv = nrm2(q);
            if (nv > 0) {
                for (int i = 0; i < n; i++) {
                    q[i] /= nv;
                }
            }
            lambda.add(roots[j]);
            vectors.add(q);
        }

        EigenResult raw = sortDescending(lambda, vectors);
        orthonormalize(raw.eigenvectors, tolAbs);
        if (!residualOk(dIn, zIn, rho, raw, tolAbs)) {
            return solveDenseFull(dIn, zIn, rho, tolAbs);
        }
        return raw;
    }

    private static EigenResult solveDenseFull(double[] dIn, double[] zIn, double rho, double tolAbs) {
        int n = dIn.length;
        IMatrix<Double> mat = Linalg.zeros(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                mat.set(i, j, rho * zIn[i] * zIn[j]);
            }
            mat.set(i, i, mat.get(i, i) + dIn[i]);
        }
        RereEigenDecomposition ed = new RereEigenDecomposition(Math.max(1e-15, tolAbs * 0.01), Math.max(500, n * 64));
        Tuple2<IVector<Double>, IMatrix<Double>> t = ed.decompose(mat, Math.max(tolAbs, 1e-14));
        double[] vals = new double[n];
        double[][] q = new double[n][n];
        for (int c = 0; c < n; c++) {
            vals[c] = t.getFirst().get(c);
            for (int r = 0; r < n; r++) {
                q[r][c] = t.getSecond().get(r, c);
            }
        }
        return new EigenResult(vals, q);
    }

    /**
     * Cauchy 交错：ρ&gt;0 时为相邻间隙 + 最右尾；ρ&lt;0 时为最左尾 + 相邻间隙。
     */
    private static double[] secularRootsMultiBracket(double[] d, double[] z, double rho, int rem, double sep, double tolAbs) {
        if (rem <= 0) {
            return new double[0];
        }
        double[] roots = new double[rem];
        if (rho > 0) {
            for (int k = 0; k < rem - 1; k++) {
                double lo = d[k] + sep;
                double hi = d[k + 1] - sep;
                if (!(lo < hi)) {
                    double eps = Math.max(tolAbs, RerePrecision.MACHINE_EPSILON * 8)
                            * (1.0 + Math.max(Math.abs(d[k]), Math.abs(d[k + 1])));
                    lo = d[k] + eps;
                    hi = d[k + 1] - eps;
                    if (!(lo < hi)) {
                        return null;
                    }
                }
                roots[k] = bracketScanBisect(d, z, rho, lo, hi, tolAbs, 160);
            }
            double loLast = d[rem - 1] + sep;
            double scale = 1.0 + Math.abs(rho) * nrm2(z) * nrm2(z);
            double hiLast = d[rem - 1] + Math.max(1.0, Math.abs(d[rem - 1]) * scale);
            roots[rem - 1] = bracketScanBisectTail(d, z, rho, loLast, hiLast, tolAbs);
        } else {
            double hiLeft = d[0] - sep;
            roots[0] = bracketScanBisectHead(d, z, rho, hiLeft, tolAbs);
            for (int k = 0; k < rem - 1; k++) {
                double lo = d[k] + sep;
                double hi = d[k + 1] - sep;
                if (!(lo < hi)) {
                    double eps = Math.max(tolAbs, RerePrecision.MACHINE_EPSILON * 8)
                            * (1.0 + Math.max(Math.abs(d[k]), Math.abs(d[k + 1])));
                    lo = d[k] + eps;
                    hi = d[k + 1] - eps;
                    if (!(lo < hi)) {
                        return null;
                    }
                }
                roots[k + 1] = bracketScanBisect(d, z, rho, lo, hi, tolAbs, 160);
            }
        }
        if (!secularResidualsOk(d, z, rho, roots, tolAbs)) {
            return null;
        }
        return roots;
    }

    private static boolean secularResidualsOk(double[] d, double[] z, double rho, double[] roots, double tolAbs) {
        for (double r : roots) {
            double f = secular(d, z, rho, r);
            if (Math.abs(f) > tolAbs * 64 * (1.0 + Math.abs(r))) {
                return false;
            }
        }
        return true;
    }

    private static boolean residualOk(double[] dIn, double[] zIn, double rho, EigenResult raw, double tolAbs) {
        int n = dIn.length;
        int m = raw.eigenvalues.length;
        if (m != n || raw.eigenvectors.length != n || (n > 0 && raw.eigenvectors[0].length != n)) {
            return false;
        }
        double lim = Math.max(tolAbs * 256, 1e-9) * (1.0 + Math.abs(rho));
        for (int c = 0; c < n; c++) {
            double lam = raw.eigenvalues[c];
            for (int r = 0; r < n; r++) {
                double sum = -lam * raw.eigenvectors[r][c];
                for (int j = 0; j < n; j++) {
                    double a = (j == r ? dIn[r] + rho * zIn[r] * zIn[j] : rho * zIn[r] * zIn[j]);
                    sum += a * raw.eigenvectors[j][c];
                }
                if (Math.abs(sum) > lim * nrm2Col(raw.eigenvectors, c)) {
                    return false;
                }
            }
        }
        return true;
    }

    private static double nrm2Col(double[][] q, int c) {
        double s = 0.0;
        for (double[] doubles : q) {
            double v = doubles[c];
            s += v * v;
        }
        return Math.sqrt(s);
    }

    private static double bracketScanBisectHead(double[] d, double[] z, double rho, double hi, double tol) {
        double fHi = secular(d, z, rho, hi);
        double lo = hi;
        double fLo = fHi;
        int exp = 0;
        while (fLo * fHi > 0 && exp < 160) {
            double step = (Math.abs(lo) + 1.0) * Math.pow(2.0, Math.min(exp, 100));
            lo = lo - step;
            fLo = secular(d, z, rho, lo);
            exp++;
        }
        if (fLo * fHi > 0) {
            return hi;
        }
        return bracketScanBisect(d, z, rho, lo, hi, tol, 160);
    }

    private static double bracketScanBisect(double[] d, double[] z, double rho, double lo, double hi, double tol, int samples) {
        if (!(lo < hi)) {
            return (lo + hi) * 0.5;
        }
        double prev = lo;
        double fPrev = secular(d, z, rho, prev);
        for (int s = 1; s <= samples; s++) {
            double x = lo + (hi - lo) * (s / (double) samples);
            double fx = secular(d, z, rho, x);
            if (fPrev * fx <= 0) {
                return bisectSecular(d, z, rho, prev, x, tol);
            }
            prev = x;
            fPrev = fx;
        }
        return (lo + hi) * 0.5;
    }

    private static double bracketScanBisectTail(double[] d, double[] z, double rho, double lo, double hi, double tol) {
        double fLo = secular(d, z, rho, lo);
        double hiC = hi;
        double fHi = secular(d, z, rho, hiC);
        int exp = 0;
        while (fLo * fHi > 0 && exp < 120) {
            hiC = hiC * 2.0 + Math.abs(lo) + 1.0;
            fHi = secular(d, z, rho, hiC);
            exp++;
        }
        if (fLo * fHi > 0) {
            return hiC;
        }
        return bracketScanBisect(d, z, rho, lo, hiC, tol, 128);
    }

    private static double bisectSecular(double[] d, double[] z, double rho, double lo, double hi, double tol) {
        double flo = secular(d, z, rho, lo);
        double fhi = secular(d, z, rho, hi);
        if (flo * fhi > 0) {
            double mid = (lo + hi) * 0.5;
            for (int s = 0; s < 40; s++) {
                double fm = secular(d, z, rho, mid);
                if (flo * fm <= 0) {
                    hi = mid;
                    fhi = fm;
                    break;
                }
                if (fm * fhi <= 0) {
                    lo = mid;
                    flo = fm;
                    break;
                }
                mid = (lo + hi) * 0.5;
            }
        }
        if (flo * fhi > 0) {
            return (lo + hi) * 0.5;
        }
        for (int it = 0; it < 200; it++) {
            double mid = (lo + hi) * 0.5;
            double fm = secular(d, z, rho, mid);
            if (Math.abs(fm) <= tol * (1.0 + Math.abs(mid)) || (hi - lo) <= tol * (1.0 + Math.abs(mid))) {
                return mid;
            }
            if (flo * fm <= 0) {
                hi = mid;
                fhi = fm;
            } else {
                lo = mid;
                flo = fm;
            }
        }
        return (lo + hi) * 0.5;
    }

    public static double secular(double[] d, double[] z, double rho, double lambda) {
        double s = 0.0;
        for (int i = 0; i < d.length; i++) {
            double den = d[i] - lambda;
            if (Math.abs(den) < RerePrecision.SAFE_MIN * 8) {
                den = Math.copySign(RerePrecision.SAFE_MIN * 8, den == 0 ? 1.0 : den);
            }
            double zi = z[i];
            s += (zi * zi) / den;
        }
        return 1.0 + rho * s;
    }

    private static double nrm2(double[] v) {
        double s = 0.0;
        for (double x : v) {
            s += x * x;
        }
        return Math.sqrt(s);
    }

    private static EigenResult sortDescending(List<Double> lambda, List<double[]> vectors) {
        int m = lambda.size();
        int n = vectors.get(0).length;
        Integer[] ord = new Integer[m];
        for (int i = 0; i < m; i++) {
            ord[i] = i;
        }
        Arrays.sort(ord, (a, b) -> Double.compare(lambda.get(b), lambda.get(a)));
        double[] vals = new double[m];
        double[][] q = new double[n][m];
        for (int c = 0; c < m; c++) {
            int o = ord[c];
            vals[c] = lambda.get(o);
            double[] col = vectors.get(o);
            for (int r = 0; r < n; r++) {
                q[r][c] = col[r];
            }
        }
        return new EigenResult(vals, q);
    }

    private static void orthonormalize(double[][] q, double tol) {
        int n = q.length;
        if (n == 0) {
            return;
        }
        int m = q[0].length;
        for (int j = 0; j < m; j++) {
            for (int i = 0; i < j; i++) {
                double dot = 0.0;
                for (int r = 0; r < n; r++) {
                    dot += q[r][i] * q[r][j];
                }
                for (int r = 0; r < n; r++) {
                    q[r][j] -= dot * q[r][i];
                }
            }
            double norm = 0.0;
            for (int r = 0; r < n; r++) {
                norm += q[r][j] * q[r][j];
            }
            norm = Math.sqrt(norm);
            if (norm > tol) {
                for (int r = 0; r < n; r++) {
                    q[r][j] /= norm;
                }
            }
        }
    }

    public static final class EigenResult {
        public final double[] eigenvalues;
        public final double[][] eigenvectors;

        public EigenResult(double[] eigenvalues, double[][] eigenvectors) {
            this.eigenvalues = eigenvalues;
            this.eigenvectors = eigenvectors;
        }
    }
}
