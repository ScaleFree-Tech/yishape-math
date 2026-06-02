package com.yishape.lab.math.compute.hpc;

import java.util.ArrayList;
import java.util.List;

/**
 * HPC（Rust 原生加速）健康检查报告。
 *
 * <p>由 {@link HpcHealthChecker#check()} 生成，涵盖扩展 JAR 可用性、
 * 原生运行时状态以及各功能模块（矩阵乘、L-BFGS、分解）的端到端正确性验证结果。</p>
 *
 * @since 0.5.0
 * @see HpcHealthChecker
 */
public final class HpcHealthReport {

    private final boolean extensionPresent;
    private final boolean nativeRuntimeAvailable;
    private final boolean overallHealthy;
    private final boolean matMulHealthy;
    private final boolean lbfgsHealthy;
    private final boolean choleskyHealthy;
    private final boolean svdHealthy;
    private final boolean hnswHealthy;
    private final List<String> details;

    HpcHealthReport(boolean extensionPresent,
                    boolean nativeRuntimeAvailable,
                    boolean overallHealthy,
                    boolean matMulHealthy,
                    boolean lbfgsHealthy,
                    boolean choleskyHealthy,
                    boolean svdHealthy,
                    boolean hnswHealthy,
                    List<String> details) {
        this.extensionPresent = extensionPresent;
        this.nativeRuntimeAvailable = nativeRuntimeAvailable;
        this.overallHealthy = overallHealthy;
        this.matMulHealthy = matMulHealthy;
        this.lbfgsHealthy = lbfgsHealthy;
        this.choleskyHealthy = choleskyHealthy;
        this.svdHealthy = svdHealthy;
        this.hnswHealthy = hnswHealthy;
        this.details = List.copyOf(details);
    }

    /** 扩展 JAR ({@code yishape-math-hpc}) 是否在 classpath 上并成功解析。 */
    public boolean isExtensionPresent() {
        return extensionPresent;
    }

    /** Rust 原生运行时（gosh-lbfgs / faer 等）是否已加载且可用。 */
    public boolean isNativeRuntimeAvailable() {
        return nativeRuntimeAvailable;
    }

    /** 整体健康：扩展存在、原生可用、且各功能验证全部通过。 */
    public boolean isOverallHealthy() {
        return overallHealthy;
    }

    /** 矩阵乘法端到端验证是否通过。 */
    public boolean isMatMulHealthy() {
        return matMulHealthy;
    }

    /** L-BFGS 端到端验证是否通过。 */
    public boolean isLbfgsHealthy() {
        return lbfgsHealthy;
    }

    /** Cholesky 分解端到端验证是否通过。 */
    public boolean isCholeskyHealthy() {
        return choleskyHealthy;
    }

    /** SVD 端到端验证是否通过。 */
    public boolean isSvdHealthy() {
        return svdHealthy;
    }

    /** HNSW 向量索引端到端验证是否通过。 */
    public boolean isHnswHealthy() {
        return hnswHealthy;
    }

    /** 详细诊断信息（按发生顺序）。 */
    public List<String> getDetails() {
        return details;
    }

    /**
     * 生成易读的多行摘要。
     *
     * <pre>{@code
     * HPC Health Report
     * =================
     * Extension present        : true
     * Native runtime available : true
     * Overall healthy          : true
     *   MatMul    : PASS
     *   L-BFGS    : PASS
     *   Cholesky  : PASS
     *   SVD       : PASS
     * }</pre>
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("HPC Health Report\n");
        sb.append("=================\n");
        sb.append("Extension present        : ").append(extensionPresent).append('\n');
        sb.append("Native runtime available : ").append(nativeRuntimeAvailable).append('\n');
        sb.append("Overall healthy          : ").append(overallHealthy).append('\n');
        if (extensionPresent && nativeRuntimeAvailable) {
            sb.append("  MatMul    : ").append(matMulHealthy ? "PASS" : "FAIL").append('\n');
            sb.append("  L-BFGS    : ").append(lbfgsHealthy ? "PASS" : "FAIL").append('\n');
            sb.append("  Cholesky  : ").append(choleskyHealthy ? "PASS" : "FAIL").append('\n');
            sb.append("  SVD       : ").append(svdHealthy ? "PASS" : "FAIL").append('\n');
            sb.append("  HNSW      : ").append(hnswHealthy ? "PASS" : "FAIL").append('\n');
        }
        if (!details.isEmpty()) {
            sb.append("Details:\n");
            for (String d : details) {
                sb.append("  ").append(d).append('\n');
            }
        }
        return sb.toString();
    }
}
