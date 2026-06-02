package com.yishape.lab.math.stats.distribution.multiv;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.Arrays;
import java.util.Random;

/**
 * 多元分布共享数值工具（χ²/F 分位数、二元边际 t 与多元正态的边际椭圆等）。
 * Shared numerical helpers (χ²/F quantiles, marginal-plane ellipses for Gaussian vs bivariate marginal-t).
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
final class MultivariateDistributionMath {

    private MultivariateDistributionMath() {
    }

    /**
     * χ² 分布分位数近似（Wilson–Hilferty），使得 P(χ²_df ≤ q) ≈ cumulativeProbability。
     */
    static double chiSquareQuantile(double cumulativeProbability, int degreesOfFreedom) {
        if (degreesOfFreedom <= 0) {
            throw new IllegalArgumentException("degreesOfFreedom must be positive");
        }
        if (cumulativeProbability <= 0 || cumulativeProbability >= 1) {
            throw new IllegalArgumentException("cumulativeProbability must be in (0,1)");
        }
        double z = normalInverseCdf(cumulativeProbability);
        double h = 2.0 / (9.0 * degreesOfFreedom);
        double inner = 1.0 - h + z * Math.sqrt(Math.max(h, 1e-15));
        return degreesOfFreedom * inner * inner * inner;
    }

    /**
     * Acklam 有理逼近的反标准正态分位数（公共领域常用实现）。
     */
    static double normalInverseCdf(double p) {
        if (p <= 0 || p >= 1) {
            throw new IllegalArgumentException("p must be in (0,1)");
        }

        final double[] a = {
                -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00
        };
        final double[] b = {
                -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                6.680131188771972e+01, -1.328068155288572e+01
        };
        final double[] c = {
                -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00
        };
        final double[] d = {
                7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                3.754408661907416e+00
        };

        double plow = 0.02425;
        double phigh = 1 - plow;

        if (p < plow) {
            double q = Math.sqrt(-2 * Math.log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
        if (phigh < p) {
            double q = Math.sqrt(-2 * Math.log(1 - p));
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }

        double q = p - 0.5;
        double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }

    /**
     * 在给定对称正定 2×2 协方差 Σ 下构造等高椭圆：(x-μ)ᵀΣ⁻¹(x-μ)=mahalanobisSquared。
     */
    static IMultivariateDistribution.ConfidenceEllipse confidenceEllipseFrom2x2MahalanobisSquared(
            IVector<Double> center2d,
            double cov00,
            double cov01,
            double cov11,
            double mahalanobisSquared) {
        if (mahalanobisSquared < 0) {
            throw new IllegalArgumentException("mahalanobisSquared must be non-negative");
        }
        double a = cov00;
        double b = cov01;
        double c = cov11;
        double trace = a + c;
        double det = a * c - b * b;
        double discriminant = Math.sqrt(Math.max(0.0, trace * trace - 4 * det));
        double eigenvalue1 = 0.5 * (trace + discriminant);
        double eigenvalue2 = 0.5 * (trace - discriminant);
        double majorAxis = Math.sqrt(Math.max(0.0, mahalanobisSquared * eigenvalue1));
        double minorAxis = Math.sqrt(Math.max(0.0, mahalanobisSquared * eigenvalue2));
        double angle = 0.5 * Math.atan2(2 * b, a - c);
        return new IMultivariateDistribution.ConfidenceEllipse(center2d, majorAxis, minorAxis, angle);
    }

    /**
     * 二元正态近似椭圆：(x-μ)ᵀΣ⁻¹(x-μ)=χ²₂(confidence 分位数)。
     */
    static IMultivariateDistribution.ConfidenceEllipse confidenceEllipseFrom2x2Covariance(
            IVector<Double> center2d,
            double cov00,
            double cov01,
            double cov11,
            double confidence) {
        if (confidence <= 0 || confidence >= 1) {
            throw new IllegalArgumentException("confidence must be in (0,1)");
        }
        double chi2 = chiSquareQuantile(confidence, 2);
        return confidenceEllipseFrom2x2MahalanobisSquared(center2d, cov00, cov01, cov11, chi2);
    }

    /**
     * 取均值与协方差在坐标 (i,j) 上的边际块，用于 d&gt;2 时在前两维指定平面上的椭圆可视化。
     */
    static IMultivariateDistribution.ConfidenceEllipse confidenceEllipseMarginalPlane(
            IVector<Double> mean,
            IMatrix<Double> covariance,
            int index0,
            int index1,
            double confidence) {
        if (mean.length() < 2 || covariance.rows() != mean.length()) {
            throw new IllegalArgumentException("mean/covariance dimension mismatch");
        }
        if (index0 < 0 || index1 < 0 || index0 >= mean.length() || index1 >= mean.length()) {
            throw new IllegalArgumentException("invalid marginal indices");
        }
        IVector<Double> center = Linalg.vector(new double[]{mean.get(index0), mean.get(index1)});
        double c00 = covariance.get(index0, index0);
        double c01 = covariance.get(index0, index1);
        double c11 = covariance.get(index1, index1);
        return confidenceEllipseFrom2x2Covariance(center, c00, c01, c11, confidence);
    }

    /**
     * 二元边际多元 t（ν&gt;2）：若 Σ_cov = ν/(ν−2)·Σ 为边际协方差，则
     * Q=(x−μ)ᵀΣ_cov⁻¹(x−μ) 满足 Q/2 ~ F_{2,ν−2}。
     * 等价于 Beta(1,(ν−2)/2) 闭式：取 x 使 I_x(1,b)=confidence，则 Q=(ν−2)·x/(1−x)，
     * 避免 {@link #fQuantile} 在极大非整数自由度时的数值不稳。
     */
    static double mahalanobisSquaredBivariateTMarginal(double confidence, double nu) {
        if (nu <= 2) {
            throw new IllegalArgumentException("degrees of freedom nu must exceed 2");
        }
        if (confidence <= 0 || confidence >= 1) {
            throw new IllegalArgumentException("confidence must be in (0,1)");
        }
        double b = (nu - 2) / 2.0;
        double om = Math.pow(1.0 - confidence, 1.0 / b);
        // I_x(1,b)=1-(1-x)^b = confidence ⇒ (1-x)^b = 1-confidence ⇒ 1-x = om
        double x = 1.0 - om;
        if (om <= 0 || x <= 0) {
            throw new IllegalStateException("degenerate bivariate-t marginal quantile");
        }
        return (nu - 2) * x / om;
    }

    /**
     * 二元边际多元 t（ν&gt;2）：若 Σ_cov = ν/(ν−2)·Σ 为边际协方差，则
     * (x−μ)ᵀΣ_cov⁻¹(x−μ)/2 ~ F_{2,ν−2}；置信轮廓取 Mahalanobis² = 2·F⁻¹(conf)，
     * 由 {@link #mahalanobisSquaredBivariateTMarginal} 稳定计算。
     */
    static IMultivariateDistribution.ConfidenceEllipse confidenceEllipseMarginalPlaneMultivariateT(
            IVector<Double> mean,
            IMatrix<Double> covariance,
            int index0,
            int index1,
            double confidence,
            double nu) {
        if (nu <= 2) {
            throw new IllegalArgumentException("degrees of freedom nu must exceed 2");
        }
        if (confidence <= 0 || confidence >= 1) {
            throw new IllegalArgumentException("confidence must be in (0,1)");
        }
        if (mean.length() < 2 || covariance.rows() != mean.length()) {
            throw new IllegalArgumentException("mean/covariance dimension mismatch");
        }
        if (index0 < 0 || index1 < 0 || index0 >= mean.length() || index1 >= mean.length()) {
            throw new IllegalArgumentException("invalid marginal indices");
        }
        IVector<Double> center = Linalg.vector(new double[]{mean.get(index0), mean.get(index1)});
        double c00 = covariance.get(index0, index0);
        double c01 = covariance.get(index0, index1);
        double c11 = covariance.get(index1, index1);
        double mahalSq = mahalanobisSquaredBivariateTMarginal(confidence, nu);
        return confidenceEllipseFrom2x2MahalanobisSquared(center, c00, c01, c11, mahalSq);
    }

    /** F_{d1,d2} 分位数近似：由 Beta 反函数得到（与 scipy.special.betaincinv 思路一致）。 */
    static double fQuantile(double cumulativeProbability, int d1, double d2) {
        if (d1 <= 0 || d2 <= 0) {
            throw new IllegalArgumentException("degrees of freedom must be positive");
        }
        if (cumulativeProbability <= 0 || cumulativeProbability >= 1) {
            throw new IllegalArgumentException("cumulativeProbability must be in (0,1)");
        }
        double x = incompleteBetaInverse(cumulativeProbability, d1 / 2.0, d2 / 2.0);
        return (d2 * x) / (d1 * (1 - x));
    }

    /** regula falsi 求 I_x(a,b)=p 的 x（单调）。 */
    static double incompleteBetaInverse(double p, double a, double b) {
        if (p <= 0 || p >= 1 || a <= 0 || b <= 0) {
            throw new IllegalArgumentException("invalid incompleteBetaInverse arguments");
        }
        double lo = 1e-14;
        double hi = 1 - 1e-14;
        double flo = incompleteBetaRegularized(lo, a, b) - p;
        double fhi = incompleteBetaRegularized(hi, a, b) - p;
        if (flo > 0 || fhi < 0) {
            throw new IllegalStateException("bracket failed for incompleteBetaInverse");
        }
        for (int it = 0; it < 80; it++) {
            double mid = hi - fhi * (hi - lo) / (fhi - flo + 1e-300);
            mid = Math.min(Math.max(mid, lo + 1e-15), hi - 1e-15);
            double fm = incompleteBetaRegularized(mid, a, b) - p;
            if (Math.abs(fm) < 1e-12) {
                return mid;
            }
            if (fm * flo < 0) {
                hi = mid;
                fhi = fm;
            } else {
                lo = mid;
                flo = fm;
            }
        }
        return 0.5 * (lo + hi);
    }

    /** 正则不完全 Beta I_x(a,b)，连分式（Numerical Recipes 思路）。 */
    static double incompleteBetaRegularized(double x, double a, double b) {
        if (x <= 0) {
            return 0;
        }
        if (x >= 1) {
            return 1;
        }
        double bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
        if (x < (a + 1) / (a + b + 2)) {
            return bt * betaContinuedFraction(x, a, b) / a;
        }
        return 1 - bt * betaContinuedFraction(1 - x, b, a) / b;
    }

    private static double betaContinuedFraction(double x, double a, double b) {
        final int maxIter = 200;
        final double eps = 3e-14;
        double qab = a + b;
        double qap = a + 1;
        double qam = a - 1;
        double c = 1;
        double d = 1 - qab * x / qap;
        if (Math.abs(d) < 1e-30) {
            d = 1e-30;
        }
        d = 1 / d;
        double h = d;
        for (int m = 1; m <= maxIter; m++) {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1 + aa * d;
            if (Math.abs(d) < 1e-30) {
                d = 1e-30;
            }
            c = 1 + aa / c;
            if (Math.abs(c) < 1e-30) {
                c = 1e-30;
            }
            d = 1 / d;
            h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1 + aa * d;
            if (Math.abs(d) < 1e-30) {
                d = 1e-30;
            }
            c = 1 + aa / c;
            if (Math.abs(c) < 1e-30) {
                c = 1e-30;
            }
            d = 1 / d;
            double del = d * c;
            h *= del;
            if (Math.abs(del - 1) <= eps) {
                break;
            }
        }
        return h;
    }

    static double logGamma(double x) {
        if (x <= 0) {
            throw new IllegalArgumentException("logGamma requires positive x");
        }
        double shift = 0;
        double z = x;
        while (z < 12) {
            shift -= Math.log(z);
            z += 1;
        }
        double inv = 1 / z;
        return shift + (z - 0.5) * Math.log(z) - z + 0.9189385332046727
                + inv / 12 - inv * inv / 360 + inv * inv * inv / 1260;
    }

    /** 对 conditionIndices 排序并同步 permute conditionValues，返回唯一性校验后的副本。 */
    static int[] sortConditionIndicesWithValues(int[] conditionIndices, IVector<Double> conditionValues,
                                                double[] sortedValuesOut) {
        int n = conditionIndices.length;
        if (n != conditionValues.length()) {
            throw new IllegalArgumentException("conditionIndices and conditionValues length mismatch");
        }
        Integer[] ord = new Integer[n];
        for (int i = 0; i < n; i++) {
            ord[i] = i;
        }
        Arrays.sort(ord, (i, j) -> Integer.compare(conditionIndices[i], conditionIndices[j]));
        int[] sortedIdx = new int[n];
        for (int i = 0; i < n; i++) {
            sortedIdx[i] = conditionIndices[ord[i]];
            sortedValuesOut[i] = conditionValues.get(ord[i]);
            if (i > 0 && sortedIdx[i] == sortedIdx[i - 1]) {
                throw new IllegalArgumentException("duplicate condition index: " + sortedIdx[i]);
            }
        }
        return sortedIdx;
    }

    static double[][] extractSubmatrix(IMatrix<Double> m, int[] rowIdx, int[] colIdx) {
        double[][] out = new double[rowIdx.length][colIdx.length];
        for (int i = 0; i < rowIdx.length; i++) {
            for (int j = 0; j < colIdx.length; j++) {
                out[i][j] = m.get(rowIdx[i], colIdx[j]);
            }
        }
        return out;
    }

    static double[] extractMean(IVector<Double> mean, int[] idx) {
        double[] out = new double[idx.length];
        for (int i = 0; i < idx.length; i++) {
            out[i] = mean.get(idx[i]);
        }
        return out;
    }

    static int[] complementIndices(int dimension, int[] sortedConditionIndices) {
        boolean[] mask = new boolean[dimension];
        for (int idx : sortedConditionIndices) {
            if (idx < 0 || idx >= dimension) {
                throw new IllegalArgumentException("condition index out of range: " + idx);
            }
            mask[idx] = true;
        }
        int cnt = 0;
        for (boolean b : mask) {
            if (!b) {
                cnt++;
            }
        }
        int[] rem = new int[cnt];
        int k = 0;
        for (int i = 0; i < dimension; i++) {
            if (!mask[i]) {
                rem[k++] = i;
            }
        }
        return rem;
    }

    static double digamma(double x) {
        if (x <= 0) {
            throw new IllegalArgumentException("digamma requires positive x");
        }
        if (x > 6.0) {
            return Math.log(x) - 1.0 / (2.0 * x) - 1.0 / (12.0 * x * x);
        }
        return digamma(x + 1.0) - 1.0 / x;
    }

    /**
     * KL(P‖Q) 蒙特卡洛估计，EP[log p − log q]。
     */
    static double klMonteCarlo(IMultivariateDistribution<Double> p,
                               IMultivariateDistribution<Double> q,
                               int samples,
                               Random rng) {
        if (p.getDimension() != q.getDimension()) {
            throw new IllegalArgumentException("dimension mismatch");
        }
        double sum = 0.0;
        for (int i = 0; i < samples; i++) {
            IVector<Double> x = p.sample();
            sum += p.logPdf(x) - q.logPdf(x);
        }
        return sum / samples;
    }

    /**
     * 切片 2-Wasserstein：随机投影下一维 W₂ 的样本平均（投影方向为单位高斯归一化）。
     */
    static double slicedWasserstein2(IMultivariateDistribution<Double> p,
                                     IMultivariateDistribution<Double> q,
                                     int dim,
                                     int samplesPerSide,
                                     int nProjections,
                                     Random rng) {
        if (p.getDimension() != dim || q.getDimension() != dim) {
            throw new IllegalArgumentException("dimension mismatch");
        }
        double[][] xs = new double[samplesPerSide][dim];
        double[][] ys = new double[samplesPerSide][dim];
        for (int i = 0; i < samplesPerSide; i++) {
            IVector<Double> a = p.sample();
            IVector<Double> b = q.sample();
            for (int j = 0; j < dim; j++) {
                xs[i][j] = a.get(j);
                ys[i][j] = b.get(j);
            }
        }
        double acc = 0.0;
        for (int r = 0; r < nProjections; r++) {
            double[] theta = randomUnitGaussianFlat(dim, rng);
            normalizeInPlace(theta);
            double[] px = new double[samplesPerSide];
            double[] py = new double[samplesPerSide];
            for (int i = 0; i < samplesPerSide; i++) {
                px[i] = dot(xs[i], theta);
                py[i] = dot(ys[i], theta);
            }
            Arrays.sort(px);
            Arrays.sort(py);
            double w2sq = 0.0;
            for (int i = 0; i < samplesPerSide; i++) {
                double d = px[i] - py[i];
                w2sq += d * d;
            }
            acc += Math.sqrt(w2sq / samplesPerSide);
        }
        return acc / nProjections;
    }

    private static double[] randomUnitGaussianFlat(int dim, Random rng) {
        double[] v = new double[dim];
        for (int i = 0; i < dim; i++) {
            v[i] = rng.nextGaussian();
        }
        return v;
    }

    private static void normalizeInPlace(double[] v) {
        double s = 0.0;
        for (double x : v) {
            s += x * x;
        }
        s = Math.sqrt(Math.max(s, 1e-30));
        for (int i = 0; i < v.length; i++) {
            v[i] /= s;
        }
    }

    private static double dot(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /** Wishart：Cov(W_ij,W_kl)=ν(V_ik V_jl + V_il V_jk)。 */
    static double wishartElementCovariance(double nu, IMatrix v, int i, int j, int k, int l) {
        return nu * (v.get(i, k) * v.get(j, l)
                + v.get(i, l) * v.get(j, k));
    }

    /**
     * 逆 Wishart（尺度 Ψ，自由度 ν）：与类内 {@code variance()} 公式一致的二阶混合矩。
     */
    static double inverseWishartElementCovariance(double nu, int p, IMatrix psi, int i, int j, int k, int l) {
        double denom = (nu - p) * (nu - p - 1) * (nu - p - 1) * (nu - p - 3);
        double term1 = 2 * psi.get(i, j) * psi.get(k, l);
        double term2 = (nu - p - 1) * (psi.get(i, k) * psi.get(j, l)
                + psi.get(i, l) * psi.get(j, k));
        return (term1 + term2) / denom;
    }
}
