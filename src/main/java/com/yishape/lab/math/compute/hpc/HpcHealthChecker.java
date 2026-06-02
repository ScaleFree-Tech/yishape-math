package com.yishape.lab.math.compute.hpc;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;

import java.util.ArrayList;
import java.util.List;

import com.yishape.lab.util.YishapeLogger;

/**
 * HPC（Rust 原生加速）健康检查器。
 *
 * <p>提供<strong>一键式</strong>端到端检测：不仅检查扩展 JAR 是否存在、
 * 原生运行时是否可加载，还会实际发起小规模计算并与预期结果对比，
 * 确保 Rust 端功能正确、结果可信。</p>
 *
 * <h3>使用示例</h3>
 * <pre>{@code
 * // 1. 最简单用法：一键检查，打印报告
 * HpcHealthReport report = HpcHealthChecker.check();
 * System.out.println(report);
 *
 * // 2. 根据结果做分支
 * if (report.isOverallHealthy()) {
 *     // 放心使用 HPC 加速
 * } else if (report.isExtensionPresent()) {
 *     // JAR 存在但功能异常，查看 details 定位问题
 *     report.getDetails().forEach(System.err::println);
 * } else {
 *     // HPC 完全不可用，回退纯 Java
 * }
 * }</pre>
 *
 * <p><strong>线程安全</strong>：{@code check()} 是只读操作，可并发调用。
 * <strong>异常安全</strong>：即使 Rust 端崩溃或返回异常数据，也不会抛出，
 * 而是将异常信息写入 {@link HpcHealthReport#getDetails()}。</p>
 *
 * @since 0.5.0
 * @see HpcHealthReport
 */
public final class HpcHealthChecker {

    private HpcHealthChecker() {
    }

    /**
     * 执行完整的 HPC 健康检查。
     *
     * <p>检查项（按顺序）：</p>
     * <ol>
     *   <li>扩展 JAR 是否存在</li>
     *   <li>原生运行时是否可用</li>
     *   <li>矩阵乘法端到端正确性（小规模 4×4）</li>
     *   <li>L-BFGS 端到端正确性（简单二次函数）</li>
     *   <li>Cholesky 分解端到端正确性（4×4 SPD 矩阵）</li>
     *   <li>SVD 端到端正确性（4×3 矩阵）</li>
     * </ol>
     *
     * @return 不可变的健康报告
     */
    public static HpcHealthReport check() {
        List<String> details = new ArrayList<>();

        // 1. 扩展存在性
        boolean extensionPresent = HpcOptionalRuntime.isExtensionPresent();
        details.add("Extension present: " + extensionPresent);

        if (!extensionPresent) {
            return new HpcHealthReport(
                    false, false, false,
                    false, false, false, false, false,
                    details);
        }

        // 2. 原生运行时可用性
        boolean nativeRuntimeAvailable;
        try {
            nativeRuntimeAvailable = HpcOptionalRuntime.isNativeRuntimeAvailable();
        } catch (Throwable t) {
            nativeRuntimeAvailable = false;
            details.add("Native runtime check failed: " + t.getMessage());
        }
        details.add("Native runtime available: " + nativeRuntimeAvailable);

        if (!nativeRuntimeAvailable) {
            return new HpcHealthReport(
                    true, false, false,
                    false, false, false, false, false,
                    details);
        }

        // 3. 矩阵乘法验证（绕过阈值，直接调用可选运行时）
        boolean matMulHealthy = checkMatMul(details);

        // 4. L-BFGS 验证
        boolean lbfgsHealthy = checkLbfgs(details);

        // 5. Cholesky 验证
        boolean choleskyHealthy = checkCholesky(details);

        // 6. SVD 验证
        boolean svdHealthy = checkSvd(details);

        // 7. HNSW 验证
        boolean hnswHealthy = checkHnsw(details);

        boolean overallHealthy = matMulHealthy && lbfgsHealthy && choleskyHealthy && svdHealthy && hnswHealthy;
        details.add("Overall healthy: " + overallHealthy);

        return new HpcHealthReport(
                true, true, overallHealthy,
                matMulHealthy, lbfgsHealthy, choleskyHealthy, svdHealthy, hnswHealthy,
                details);
    }

    // ===================== 各模块细粒度检查 =====================

    private static boolean checkMatMul(List<String> details) {
        try {
            double[][] a = {
                    {1, 2, 3, 4},
                    {5, 6, 7, 8},
                    {9, 10, 11, 12},
                    {13, 14, 15, 16}
            };
            double[][] b = {
                    {1, 0, 0, 0},
                    {0, 1, 0, 0},
                    {0, 0, 1, 0},
                    {0, 0, 0, 1}
            };
            // 期望 C = A * I = A
            double[][] c = HpcOptionalRuntime.tryMatMul(a, b);
            if (c == null) {
                details.add("MatMul: HPC returned null (function unavailable)");
                return false;
            }
            // 验证维度
            if (c.length != 4 || c[0].length != 4) {
                details.add("MatMul: dimension mismatch");
                return false;
            }
            // 验证数值
            double maxErr = 0;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    maxErr = Math.max(maxErr, Math.abs(c[i][j] - a[i][j]));
                }
            }
            if (maxErr > 1e-9) {
                details.add("MatMul: numerical error = " + maxErr);
                return false;
            }
            details.add("MatMul: PASS");
            return true;
        } catch (Throwable t) {
            details.add("MatMul: exception - " + t.getClass().getSimpleName() + " " + t.getMessage());
            return false;
        }
    }

    private static boolean checkLbfgs(List<String> details) {
        try {
            // f(x) = 0.5 * (x1^2 + 2*x2^2), min at (0, 0)
            IObjectiveFunction obj = x -> {
                double x1 = x.get(0);
                double x2 = x.get(1);
                return 0.5 * (x1 * x1 + 2 * x2 * x2);
            };
            IGradientFunction grd = x -> {
                double x1 = x.get(0);
                double x2 = x.get(1);
                return Linalg.vector(new double[]{x1, 2 * x2});
            };
            double[] initX = {10.0, -5.0};
            HpcOptimizers.RLbfgsResult result = HpcOptimizers.tryLBFGS(
                    initX, 10, 1e-8, 100, obj, grd);
            if (result == null) {
                details.add("L-BFGS: HPC returned null");
                return false;
            }
            if (!result.ok()) {
                details.add("L-BFGS: status = " + result.status());
                return false;
            }
            double[] sol = result.x();
            if (sol == null || sol.length != 2) {
                details.add("L-BFGS: invalid solution dimension");
                return false;
            }
            double err = Math.max(Math.abs(sol[0]), Math.abs(sol[1]));
            if (err > 1e-4) {
                details.add("L-BFGS: solution error = " + err);
                return false;
            }
            details.add("L-BFGS: PASS");
            return true;
        } catch (Throwable t) {
            details.add("L-BFGS: exception - " + t.getClass().getSimpleName() + " " + t.getMessage());
            return false;
        }
    }

    private static boolean checkCholesky(List<String> details) {
        try {
            // 4×4 SPD matrix: A = L * L^T
            double[][] l = {
                    {2, 0, 0, 0},
                    {1, 3, 0, 0},
                    {0, 1, 2, 0},
                    {1, 0, 1, 1}
            };
            double[][] a = new double[4][4];
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    double sum = 0;
                    for (int k = 0; k < 4; k++) {
                        sum += l[i][k] * l[j][k];
                    }
                    a[i][j] = sum;
                }
            }
            HpcOptionalRuntime.RCholesky r = HpcOptionalRuntime.cholesky(a);
            if (r == null) {
                details.add("Cholesky: HPC returned null");
                return false;
            }
            if (!r.ok()) {
                details.add("Cholesky: status = " + r.status());
                return false;
            }
            double[][] lLower = r.lLower();
            if (lLower == null || lLower.length != 4) {
                details.add("Cholesky: invalid result dimension");
                return false;
            }
            // 验证 L * L^T ≈ A
            double maxErr = 0;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    double sum = 0;
                    for (int k = 0; k <= Math.min(i, j); k++) {
                        sum += lLower[i][k] * lLower[j][k];
                    }
                    maxErr = Math.max(maxErr, Math.abs(sum - a[i][j]));
                }
            }
            if (maxErr > 1e-8) {
                details.add("Cholesky: reconstruction error = " + maxErr);
                return false;
            }
            details.add("Cholesky: PASS");
            return true;
        } catch (Throwable t) {
            details.add("Cholesky: exception - " + t.getClass().getSimpleName() + " " + t.getMessage());
            return false;
        }
    }

    private static boolean checkSvd(List<String> details) {
        try {
            // 3×2 矩阵
            double[][] a = {
                    {3, 2},
                    {2, 3},
                    {2, -2}
            };
            HpcOptionalRuntime.RSvd r = HpcOptionalRuntime.svd(a);
            if (r == null) {
                details.add("SVD: HPC returned null");
                return false;
            }
            if (!r.ok()) {
                details.add("SVD: status = " + r.status());
                return false;
            }
            double[][] u = r.u();
            double[] s = r.singularValues();
            double[][] vt = r.vt();
            if (u == null || s == null || vt == null) {
                details.add("SVD: null component in result");
                return false;
            }
            // 验证 U * Σ * V^T ≈ A
            int m = 3;
            int n = 2;
            int k = Math.min(m, n);
            double maxErr = 0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (int p = 0; p < k; p++) {
                        sum += u[i][p] * s[p] * vt[p][j];
                    }
                    maxErr = Math.max(maxErr, Math.abs(sum - a[i][j]));
                }
            }
            if (maxErr > 1e-7) {
                details.add("SVD: reconstruction error = " + maxErr);
                return false;
            }
            details.add("SVD: PASS");
            return true;
        } catch (Throwable t) {
            details.add("SVD: exception - " + t.getClass().getSimpleName() + " " + t.getMessage());
            return false;
        }
    }

    private static boolean checkHnsw(List<String> details) {
        try {
            if (!HpcOptionalRuntime.isHnswNativeAvailable()) {
                details.add("HNSW: not available (native library missing or version too old)");
                return false;
            }
            // 4 个 3D 向量，id 为 "0","1","2","3"
            float[][] data = {
                    {1.0f, 0.0f, 0.0f},
                    {0.0f, 1.0f, 0.0f},
                    {0.0f, 0.0f, 1.0f},
                    {1.0f, 1.0f, 1.0f}
            };
            long[] ids = {0L, 1L, 2L, 3L};
            Long handle = HpcOptionalRuntime.hnswBuildF32(3, flatten(data), ids, 0, 16, 200, 50);
            if (handle == null || handle <= 0) {
                details.add("HNSW: build failed");
                return false;
            }
            try {
                // 验证 size
                Integer sz = HpcOptionalRuntime.hnswSize(handle);
                if (sz == null || sz != 4) {
                    details.add("HNSW: size mismatch, expected 4, got " + sz);
                    return false;
                }
                // 查询靠近 [1,0,0] 的向量，期望第一个返回 id 0
                float[] query = {1.0f, 0.0f, 0.0f};
                HpcOptionalRuntime.RHnswSearch res = HpcOptionalRuntime.hnswSearchF32(handle, query, 2);
                if (res == null || !res.ok() || res.found() == 0) {
                    details.add("HNSW: search failed or empty result");
                    return false;
                }
                if (res.ids()[0] != 0L) {
                    details.add("HNSW: nearest mismatch, expected id 0, got " + res.ids()[0]);
                    return false;
                }
                // 验证 get
                float[] buf = new float[3];
                Integer rc = HpcOptionalRuntime.hnswGetF32(handle, 0L, buf);
                if (rc == null || rc != 0) {
                    details.add("HNSW: get failed, rc=" + rc);
                    return false;
                }
                if (Math.abs(buf[0] - 1.0f) > 1e-5f || Math.abs(buf[1]) > 1e-5f || Math.abs(buf[2]) > 1e-5f) {
                    details.add("HNSW: get vector mismatch");
                    return false;
                }
                // 验证 add
                float[] newVec = {0.5f, 0.5f, 0.0f};
                Integer addRc = HpcOptionalRuntime.hnswAddF32(handle, 4L, newVec);
                if (addRc == null || addRc != 0) {
                    details.add("HNSW: add failed, rc=" + addRc);
                    return false;
                }
                Integer szAfter = HpcOptionalRuntime.hnswSize(handle);
                if (szAfter == null || szAfter != 5) {
                    details.add("HNSW: size after add mismatch, expected 5, got " + szAfter);
                    return false;
                }
                details.add("HNSW: PASS");
                return true;
            } finally {
                HpcOptionalRuntime.hnswFree(handle);
            }
        } catch (Throwable t) {
            details.add("HNSW: exception - " + t.getClass().getSimpleName() + " " + t.getMessage());
            return false;
        }
    }

    private static float[] flatten(float[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        float[] flat = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, flat, i * cols, cols);
        }
        return flat;
    }
    
    
    
    private static final YishapeLogger log = YishapeLogger.getLogger(HpcHealthChecker.class);

    public static void main(String[] args) {
        var report = HpcHealthChecker.check();
        log.info("{}", report);
    }
}
