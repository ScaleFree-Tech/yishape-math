package com.yishape.lab.math.optimize.linpg.highs;

import com.yishape.lab.math.compute.hpc.HpcAbiCodes;
import com.yishape.lab.math.compute.hpc.HpcDenseLpResult;
import com.yishape.lab.math.compute.hpc.HpcOptionalRuntime;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;

/**
 * 将 {@link ILinProgSolver} / MIP 调用参数转为 HiGHS 稠密形式（经 {@link HpcOptionalRuntime}），并转换结果码。
 */
@SuppressWarnings("rawtypes")
final class HighsNativeAdapt {

    static final class DenseLpPack {
        final int n;
        final double[] c;
        final double[][] aUb;
        final double[] bUb;
        final double[][] aEq;
        final double[] bEq;

        DenseLpPack(int n, double[] c, double[][] aUb, double[] bUb, double[][] aEq, double[] bEq) {
            this.n = n;
            this.c = c;
            this.aUb = aUb;
            this.bUb = bUb;
            this.aEq = aEq;
            this.bEq = bEq;
        }
    }

    private HighsNativeAdapt() {
    }

    static boolean hasNoncontinuous(int[] integrality) {
        if (integrality == null) {
            return false;
        }
        for (int v : integrality) {
            if (v != 0) {
                return true;
            }
        }
        return false;
    }

    static DenseLpPack packDenseOrNull(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq) {
        if (c == null || c.length() == 0) {
            return null;
        }
        int n = c.length();
        int mUb = A_ub == null ? 0 : A_ub.rows();
        int mEq = A_eq == null ? 0 : A_eq.rows();
        if (mUb == 0 && mEq == 0) {
            return null;
        }
        if (mUb > 0 && (b_ub == null || b_ub.length() < mUb)) {
            return null;
        }
        if (mEq > 0 && (b_eq == null || b_eq.length() < mEq)) {
            return null;
        }
        if (mUb > 0 && A_ub.cols() != n) {
            return null;
        }
        if (mEq > 0 && A_eq.cols() != n) {
            return null;
        }
        double[] cArr = c.toDoubleArray();
        double[][] aUb = mUb == 0 ? null : A_ub.toDoubleArray();
        double[] bUbArr = mUb == 0 ? new double[0] : vectorPrefix(b_ub, mUb);
        double[][] aEq = mEq == 0 ? null : A_eq.toDoubleArray();
        double[] bEqArr = mEq == 0 ? new double[0] : vectorPrefix(b_eq, mEq);
        return new DenseLpPack(n, cArr, aUb, bUbArr, aEq, bEqArr);
    }

    static HpcDenseLpResult runDenseNonnegative(DenseLpPack p, int[] integrality) {
        if (integrality != null && hasNoncontinuous(integrality)) {
            if (!HpcOptionalRuntime.isMixedIntegerLpNativeAvailable()) {
                return new HpcDenseLpResult(HpcAbiCodes.BAD_DIMENSION, Double.NaN, new double[0]);
            }
            HpcDenseLpResult r = HpcOptionalRuntime.lpMixedIntegerNonnegative(
                    p.c, p.aUb, p.bUb, p.aEq, p.bEq, integrality);
            return r != null ? r : lpBadDim();
        }
        HpcDenseLpResult r = HpcOptionalRuntime.lpNonnegative(p.c, p.aUb, p.bUb, p.aEq, p.bEq);
        return r != null ? r : lpBadDim();
    }

    private static HpcDenseLpResult lpBadDim() {
        return new HpcDenseLpResult(HpcAbiCodes.BAD_DIMENSION, Double.NaN, new double[0]);
    }

    static boolean shouldRetryWithFallback(int status) {
        return status != HpcAbiCodes.OK
                && status != HpcAbiCodes.LP_INFEASIBLE
                && status != HpcAbiCodes.LP_UNBOUNDED;
    }

    static OptResult toOptResult(HpcDenseLpResult r, int n) {
        if (r == null) {
            return null;
        }
        int st = r.status();
        if (st == HpcAbiCodes.OK) {
            return new OptResult.Builder(r.objective(), Linalg.vector(r.x()))
                    .converged(true)
                    .convergenceReason("HiGHS optimal")
                    .build();
        }
        if (st == HpcAbiCodes.LP_INFEASIBLE) {
            return new OptResult.Builder(Double.NaN, Linalg.zeros(n))
                    .converged(false)
                    .convergenceReason("HiGHS infeasible")
                    .build();
        }
        if (st == HpcAbiCodes.LP_UNBOUNDED) {
            return new OptResult.Builder(Double.NEGATIVE_INFINITY, Linalg.zeros(n))
                    .converged(false)
                    .convergenceReason("HiGHS unbounded (minimization)")
                    .build();
        }
        return null;
    }

    private static double[] vectorPrefix(IVector v, int len) {
        double[] a = new double[len];
        for (int i = 0; i < len; i++) {
            a[i] = v.get(i);
        }
        return a;
    }
}
