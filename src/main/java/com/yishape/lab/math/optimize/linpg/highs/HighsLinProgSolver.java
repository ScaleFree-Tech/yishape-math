package com.yishape.lab.math.optimize.linpg.highs;

import java.util.Objects;

import com.yishape.lab.math.compute.hpc.HpcDenseLpResult;
import com.yishape.lab.math.compute.hpc.HpcOptionalRuntime;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;

/**
 * 优先通过 HiGHS（经 yishape-math-hpc）求解与 {@link ILinProgSolver} 约定的非负稠密 LP；
 * 扩展未在 classpath、原生不可用、或返回需重试的错误码时回退到指定的 Java 求解器。
 */
@SuppressWarnings("rawtypes")
public final class HighsLinProgSolver implements ILinProgSolver {

    private final ILinProgSolver fallback;

    public HighsLinProgSolver() {
        this(new RereSimplexLinProgSolver());
    }

    public HighsLinProgSolver(ILinProgSolver fallback) {
        this.fallback = Objects.requireNonNull(fallback, "fallback");
    }

    @Override
    public OptResult solve(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq, IVector initX) {
        HighsNativeAdapt.DenseLpPack pack = HighsNativeAdapt.packDenseOrNull(c, A_ub, b_ub, A_eq, b_eq);
        if (pack == null || !HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return fallback.solve(c, A_ub, b_ub, A_eq, b_eq, initX);
        }
        HpcDenseLpResult r = HighsNativeAdapt.runDenseNonnegative(pack, null);
        if (HighsNativeAdapt.shouldRetryWithFallback(r.status())) {
            return fallback.solve(c, A_ub, b_ub, A_eq, b_eq, initX);
        }
        OptResult converted = HighsNativeAdapt.toOptResult(r, pack.n);
        if (converted == null) {
            return fallback.solve(c, A_ub, b_ub, A_eq, b_eq, initX);
        }
        return converted;
    }
}
