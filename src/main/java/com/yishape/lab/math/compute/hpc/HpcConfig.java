package com.yishape.lab.math.compute.hpc;

/**
 * yishape-math-hpc 原生加速配置（系统属性）。阈值依据本仓库
 * {@code benchmarks/PERFORMANCE_COMPARISON_REPORT.md} 同机粗测（Java 25 + faer）。
 */
public final class HpcConfig {

    /**
     * 总开关：{@code false} 时不尝试调用原生（默认 {@code true}）。
     */
    public static final String PROP_ENABLE = "yishape.hpc";

    /**
     * 双精度矩阵乘的最小乘积维度阈值 {@code m×n×p}，低于则走 Java 实现。
     * 默认约 {@code 100³}：在 n≥256 的方阵乘上本机 faer 已稳定快于纯 Java，更小矩阵保留 JVM 路径以降低 FFM 固定成本。
     */
    public static final String PROP_GEMM_MIN_FLOPS = "yishape.hpc.gemm.minFlops";

    static final long DEFAULT_GEMM_MIN_FLOPS = 1_000_000L;

    /**
     * 对称正定 Cholesky（下三角 L）走原生的最小阶数 {@code n}（仅限方阵）。
     * 本机基准在 {@code n≤512} 时常为纯 Java 更快（打包与调用开销）；约 {@code n≥768} 起 faer 更占优。
     */
    public static final String PROP_CHOLESKY_MIN_DIM = "yishape.hpc.cholesky.minDim";

    static final int DEFAULT_CHOLESKY_MIN_DIM = 768;

    /**
     * 对 {@code m×n} 矩阵尝试原生 SVD 的最小元素积 {@code m·n}。
     * 与 {@link com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecompBlas2} 中
     * {@code size = m*n > 10_000} 触发极慢分治路径对齐，避免在中等规模上卡死纯 Java。
     */
    public static final String PROP_SVD_MIN_TOTAL_ELEMENTS = "yishape.hpc.svd.minTotalElements";

    static final long DEFAULT_SVD_MIN_TOTAL_ELEMENTS = 10_001L;

    /**
     * SVD 时 {@code U} 与 {@code V^T} 缓冲元素上限 {@code m×k + n×n}（瘦 U 为 {@code m×min(m,n)}）；超出则回退 Java。
     */
    public static final String PROP_SVD_MAX_U_VT_ELEMENTS = "yishape.hpc.svd.maxUPlusVtElements";

    static final long DEFAULT_SVD_MAX_U_VT_ELEMENTS = 20_000_000L;

    /**
     * 方阵稠密求解 {@code Ax=b}（{@code YishapeHpc.solveSquare}）的最小阶数 {@code n}；低于则走 Java LU，避免 FFI 定常开销。
     */
    public static final String PROP_SOLVE_MIN_DIM = "yishape.hpc.solve.minDim";

    static final int DEFAULT_SOLVE_MIN_DIM = 512;

    /**
     * 旧版统一阈值：若<strong>未</strong>设置 {@link #PROP_CHOLESKY_MIN_DIM}，则仍读取本属性作为 Cholesky 下限
     * （便于沿用 {@code -Dyishape.hpc.decomp.minDim=} 的脚本）。<strong>不参与</strong> SVD 元素积判定。
     */
    public static final String PROP_DECOMP_MIN_DIM = "yishape.hpc.decomp.minDim";

    // Cached system property values — parsed once at class load, never re-read.
    private static final boolean CACHED_ENABLE = Boolean.parseBoolean(System.getProperty(PROP_ENABLE, "true"));
    private static final boolean CACHED_HNSW_ENABLED = Boolean.parseBoolean(System.getProperty("yishape.hpc.hnswEnabled", "true"));
    private static final long CACHED_GEMM_MIN_FLOPS;
    private static final int CACHED_CHOLESKY_MIN_DIM;
    private static final long CACHED_SVD_MIN_TOTAL_ELEMENTS;

    static {
        long gemm = DEFAULT_GEMM_MIN_FLOPS;
        try {
            gemm = Math.max(0L, Long.parseLong(System.getProperty(PROP_GEMM_MIN_FLOPS, Long.toString(DEFAULT_GEMM_MIN_FLOPS))));
        } catch (NumberFormatException ignored) {}
        CACHED_GEMM_MIN_FLOPS = gemm;

        Integer chol = parseNonNegativeInt(System.getProperty(PROP_CHOLESKY_MIN_DIM));
        if (chol == null) chol = parseNonNegativeInt(System.getProperty(PROP_DECOMP_MIN_DIM));
        CACHED_CHOLESKY_MIN_DIM = chol != null ? chol : DEFAULT_CHOLESKY_MIN_DIM;

        long svd = DEFAULT_SVD_MIN_TOTAL_ELEMENTS;
        try {
            svd = Math.max(0L, Long.parseLong(System.getProperty(PROP_SVD_MIN_TOTAL_ELEMENTS, Long.toString(DEFAULT_SVD_MIN_TOTAL_ELEMENTS))));
        } catch (NumberFormatException ignored) {}
        CACHED_SVD_MIN_TOTAL_ELEMENTS = svd;
    }

    private HpcConfig() {
    }

    /**
     * 总开关判定：运行时开关 {@link HpcSwitch} 优先于系统属性 {@code -Dyishape.hpc=}。
     *
     * <p>逻辑：先检查 {@code HpcSwitch}（运行时热切换），若被禁用直接返回 {@code false}；
     * 再回退到系统属性（JVM 启动时配置）。</p>
     */
    public static boolean allowAttempts() {
        if (!HpcSwitch.isEnabled()) {
            return false;
        }
        return CACHED_ENABLE;
    }

    /**
     * HNSW 原生索引是否启用。检查系统属性 {@code yishape.hpc.hnswEnabled}（默认 true）。
     */
    public static boolean isHnswEnabled() {
        return CACHED_HNSW_ENABLED;
    }

    public static long gemmMinFlops() {
        return CACHED_GEMM_MIN_FLOPS;
    }

    /**
     * Cholesky 使用 faer 的最小 {@code n}。
     */
    static int choleskyMinDim() {
        return CACHED_CHOLESKY_MIN_DIM;
    }

    /**
     * 原生 SVD 的最小 {@code m·n}。
     */
    static long svdMinTotalElements() {
        return CACHED_SVD_MIN_TOTAL_ELEMENTS;
    }

    static long svdMaxUPlusVtElements() {
        try {
            long v = Long.parseLong(System.getProperty(PROP_SVD_MAX_U_VT_ELEMENTS,
                    Long.toString(DEFAULT_SVD_MAX_U_VT_ELEMENTS)));
            return Math.max(0L, v);
        } catch (NumberFormatException e) {
            return DEFAULT_SVD_MAX_U_VT_ELEMENTS;
        }
    }

    /**
     * 原生 {@code solveSquare} 的最小 {@code n}（方阵）。
     */
    static int solveMinDim() {
        Integer v = parseNonNegativeInt(System.getProperty(PROP_SOLVE_MIN_DIM));
        if (v != null) {
            return v;
        }
        return DEFAULT_SOLVE_MIN_DIM;
    }

    // ===================== v0.5.0 新增阈值 =====================

    public static final String PROP_EIGEN_MIN_DIM = "yishape.hpc.eigen.minDim";
    static final int DEFAULT_EIGEN_MIN_DIM = 200;

    /**
     * 非对称特征值分解走原生的最小阶数 n（仅限方阵）。
     * 本机基准在 n=200 时 faer 的多位移 QR + Schur 分解路径慢于纯 Java 的直接 QR 迭代，
     * 故默认设为 300；对称特征值不受此影响。
     */
    public static final String PROP_EIGEN_NONSYMMETRIC_MIN_DIM = "yishape.hpc.eigen.nonsymmetric.minDim";
    static final int DEFAULT_EIGEN_NONSYMMETRIC_MIN_DIM = 300;

    public static final String PROP_QR_MIN_TOTAL_ELEMENTS = "yishape.hpc.qr.minTotalElements";
    static final long DEFAULT_QR_MIN_TOTAL_ELEMENTS = 10_001L;

    public static final String PROP_LU_MIN_DIM = "yishape.hpc.lu.minDim";
    static final int DEFAULT_LU_MIN_DIM = 512;

    public static final String PROP_INVERSE_MIN_DIM = "yishape.hpc.inverse.minDim";
    static final int DEFAULT_INVERSE_MIN_DIM = 200;

    // ===================== DL convolution thresholds =====================

    /**
     * im2col / conv2d 使用 HPC 的最小元素数 {@code C*H*W}；低于则走 Java 实现。
     */
    public static final String PROP_CONV_MIN_ELEMENTS = "yishape.hpc.conv.minElements";

    static final long DEFAULT_CONV_MIN_ELEMENTS = 4096L;

    /**
     * flat dgemm 使用 HPC 的最小乘积维度阈值 {@code m×n×k}。
     */
    public static final String PROP_FLAT_GEMM_MIN_FLOPS = "yishape.hpc.flatGemm.minFlops";

    static final long DEFAULT_FLAT_GEMM_MIN_FLOPS = 100_000L;

    // ===================== DL scan thresholds =====================

    /**
     * Mamba selective scan 使用 HPC 的最小元素数 {@code L * innerDim}；低于则走 Java 实现。
     */
    public static final String PROP_MAMBA_MIN_ELEMENTS = "yishape.hpc.mamba.minElements";

    static final long DEFAULT_MAMBA_MIN_ELEMENTS = 4096L;

    // ===================== DL RNN thresholds =====================

    /**
     * LSTM fused gate 使用 HPC 的最小元素数 {@code inputSize}；低于则走 Java 实现。
     */
    public static final String PROP_RNN_MIN_ELEMENTS = "yishape.hpc.rnn.minElements";

    static final long DEFAULT_RNN_MIN_ELEMENTS = 128L;

    // ===================== DL loss thresholds =====================

    /**
     * CTCLoss 使用 HPC 的最小元素数 {@code T * C}；低于则走 Java 实现。
     */
    public static final String PROP_CTC_MIN_ELEMENTS = "yishape.hpc.ctc.minElements";

    static final long DEFAULT_CTC_MIN_ELEMENTS = 4096L;

    // ===================== HNSW =====================

    /**
     * HNSW 原生索引开关：{@code false} 时强制回退到暴力扫描实现（默认 {@code true}）。
     */
    public static final String PROP_HNSW_ENABLED = "yishape.hpc.hnswEnabled";

    static int eigenMinDim() {
        Integer v = parseNonNegativeInt(System.getProperty(PROP_EIGEN_MIN_DIM));
        return v != null ? v : DEFAULT_EIGEN_MIN_DIM;
    }

    static int eigenNonsymmetricMinDim() {
        Integer v = parseNonNegativeInt(System.getProperty(PROP_EIGEN_NONSYMMETRIC_MIN_DIM));
        return v != null ? v : DEFAULT_EIGEN_NONSYMMETRIC_MIN_DIM;
    }

    static long qrMinTotalElements() {
        try {
            long v = Long.parseLong(System.getProperty(PROP_QR_MIN_TOTAL_ELEMENTS, Long.toString(DEFAULT_QR_MIN_TOTAL_ELEMENTS)));
            return Math.max(0L, v);
        } catch (NumberFormatException e) {
            return DEFAULT_QR_MIN_TOTAL_ELEMENTS;
        }
    }

    static int luMinDim() {
        Integer v = parseNonNegativeInt(System.getProperty(PROP_LU_MIN_DIM));
        if (v != null) {
            return v;
        }
        v = parseNonNegativeInt(System.getProperty(PROP_DECOMP_MIN_DIM));
        return v != null ? v : DEFAULT_LU_MIN_DIM;
    }

    static int inverseMinDim() {
        Integer v = parseNonNegativeInt(System.getProperty(PROP_INVERSE_MIN_DIM));
        return v != null ? v : DEFAULT_INVERSE_MIN_DIM;
    }

    private static Integer parseNonNegativeInt(String raw) {
        if (raw == null || raw.isBlank()) {
            return null;
        }
        try {
            return Math.max(0, Integer.parseInt(raw.trim()));
        } catch (NumberFormatException e) {
            return null;
        }
    }

    // ===================== DL convolution config getters =====================

    static long convMinElements() {
        try {
            long v = Long.parseLong(System.getProperty(PROP_CONV_MIN_ELEMENTS, Long.toString(DEFAULT_CONV_MIN_ELEMENTS)));
            return Math.max(0L, v);
        } catch (NumberFormatException e) {
            return DEFAULT_CONV_MIN_ELEMENTS;
        }
    }

    static long flatGemmMinFlops() {
        try {
            long v = Long.parseLong(System.getProperty(PROP_FLAT_GEMM_MIN_FLOPS, Long.toString(DEFAULT_FLAT_GEMM_MIN_FLOPS)));
            return Math.max(0L, v);
        } catch (NumberFormatException e) {
            return DEFAULT_FLAT_GEMM_MIN_FLOPS;
        }
    }

    static long mambaMinElements() {
        try {
            long v = Long.parseLong(System.getProperty(PROP_MAMBA_MIN_ELEMENTS, Long.toString(DEFAULT_MAMBA_MIN_ELEMENTS)));
            return Math.max(0L, v);
        } catch (NumberFormatException e) {
            return DEFAULT_MAMBA_MIN_ELEMENTS;
        }
    }

    static long rnnMinElements() {
        try {
            long v = Long.parseLong(System.getProperty(PROP_RNN_MIN_ELEMENTS, Long.toString(DEFAULT_RNN_MIN_ELEMENTS)));
            return Math.max(0L, v);
        } catch (NumberFormatException e) {
            return DEFAULT_RNN_MIN_ELEMENTS;
        }
    }

    static long ctcMinElements() {
        try {
            long v = Long.parseLong(System.getProperty(PROP_CTC_MIN_ELEMENTS, Long.toString(DEFAULT_CTC_MIN_ELEMENTS)));
            return Math.max(0L, v);
        } catch (NumberFormatException e) {
            return DEFAULT_CTC_MIN_ELEMENTS;
        }
    }

    // ===================== Element-wise / activation thresholds =====================

    /**
     * 激活函数 / 逐元素运算使用 HPC 的最小元素数；低于则走 Java SIMD/SISD。
     * 当前 HPC Rust 端为标量实现，FFI 调用开销约 1-5μs，仅在大数组上有收益。
     */
    public static final String PROP_ACTIVATION_MIN_ELEMENTS = "yishape.hpc.activation.minElements";

    static final int DEFAULT_ACTIVATION_MIN_ELEMENTS = 1_000_000;

    public static int activationMinElements() {
        Integer v = parseNonNegativeInt(System.getProperty(PROP_ACTIVATION_MIN_ELEMENTS));
        return v != null ? v : DEFAULT_ACTIVATION_MIN_ELEMENTS;
    }
}
