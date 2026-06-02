package com.yishape.lab.math.optimize.linpg.highs;

import com.yishape.lab.math.compute.hpc.HpcDenseLpResult;
import com.yishape.lab.math.compute.hpc.HpcOptionalRuntime;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;
import com.yishape.lab.math.optimize.linpg.RereIntegerProg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;

/**
 * 优先使用 HiGHS MILP（经 yishape-math-hpc）；扩展或 MILP 不可用时回退到 {@link RereIntegerProg}。
 */
@SuppressWarnings("rawtypes")
public class HighsIntegerProg extends RereIntegerProg {

    public HighsIntegerProg() {
        super(new RereSimplexLinProgSolver());
    }

    /**
     * @param baseForBranchBound 回退时用于 LP 松弛的子问题求解器
     */
    public HighsIntegerProg(ILinProgSolver baseForBranchBound) {
        super(baseForBranchBound);
    }

    @Override
    public OptResult solve(IVector c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq, IVector initX) {
        HighsNativeAdapt.DenseLpPack pack = HighsNativeAdapt.packDenseOrNull(c, A_ub, b_ub, A_eq, b_eq);
        if (pack == null || !HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return super.solve(c, A_ub, b_ub, A_eq, b_eq, initX);
        }
        int[] flags = new int[pack.n];
        fillMixedIntegerIntegralityFlags(pack.n, flags);
        HpcDenseLpResult r = HighsNativeAdapt.runDenseNonnegative(pack, flags);
        if (HighsNativeAdapt.shouldRetryWithFallback(r.status())) {
            return super.solve(c, A_ub, b_ub, A_eq, b_eq, initX);
        }
        OptResult converted = HighsNativeAdapt.toOptResult(r, pack.n);
        if (converted == null) {
            return super.solve(c, A_ub, b_ub, A_eq, b_eq, initX);
        }
        return converted;
    }
}
