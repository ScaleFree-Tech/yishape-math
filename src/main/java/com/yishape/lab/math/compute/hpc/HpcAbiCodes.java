package com.yishape.lab.math.compute.hpc;

/**
 * 与 {@code yishape_math_rust} / {@code com.yishape.lab.math.hpc.YishapeHpcStatus} 保持数值一致的状态码。
 * yishape-math 主模块不依赖 yishape-math-hpc 构件，故在此镜像常量。
 */
public final class HpcAbiCodes {

    private HpcAbiCodes() {
    }

    public static final int OK = 0;
    public static final int LP_INFEASIBLE = 1;
    public static final int LP_UNBOUNDED = 2;

    public static final int NULL_POINTER = -1;
    public static final int BAD_DIMENSION = -2;
    public static final int DECOMPOSITION_FAILED = -3;
    public static final int SOLVER_FAILED = -4;
    public static final int NOT_POSITIVE_DEFINITE = -5;
    public static final int HNSW_INVALID_HANDLE = -6;
    public static final int HNSW_DUPLICATE_ID = -7;
    public static final int HNSW_DIMENSION_MISMATCH = -8;
    public static final int HNSW_NOT_FOUND = -9;
    public static final int HNSW_PANIC = -10;
}
