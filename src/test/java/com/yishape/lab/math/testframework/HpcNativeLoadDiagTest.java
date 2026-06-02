package com.yishape.lab.math.testframework;

import com.yishape.lab.math.compute.hpc.HpcOptimizers;
import org.junit.jupiter.api.Test;

/**
 * 精确诊断 HPC native 加载失败的根因。
 */
public class HpcNativeLoadDiagTest {

    @Test
    void diagnoseNativeLoadFailure() {
        System.out.println("=== Native Load Diagnostic ===");
        System.out.println("os.name: " + System.getProperty("os.name"));
        System.out.println("os.arch: " + System.getProperty("os.arch"));
        System.out.println("java.library.path: " + System.getProperty("java.library.path"));

        // 1. HPC class/jar 是否存在
        System.out.println("\n--- Step 1: Classpath check ---");
        try {
            Class<?> c = Class.forName("com.yishape.lab.math.hpc.YishapeHpc");
            System.out.println("YishapeHpc class: FOUND at " + c.getProtectionDomain().getCodeSource().getLocation());
        } catch (ClassNotFoundException e) {
            System.out.println("YishapeHpc class: NOT FOUND - " + e);
            return;
        }

        // 2. 直接访问 YishapeMathRust
        System.out.println("\n--- Step 2: YishapeMathRust availability ---");
        try {
            Class<?> rustClass = Class.forName("com.yishape.lab.math.hpc.internal.YishapeMathRust");
            System.out.println("YishapeMathRust class: FOUND at " + rustClass.getProtectionDomain().getCodeSource().getLocation());
        } catch (ClassNotFoundException e) {
            System.out.println("YishapeMathRust class: NOT FOUND - " + e);
        }

        // 3. HpcOptimizers 状态
        System.out.println("\n--- Step 3: HpcOptimizers status ---");
        System.out.println("isExtensionPresent: " + HpcOptimizers.isExtensionPresent());
        System.out.println("isNativeRuntimeAvailable: " + HpcOptimizers.isNativeRuntimeAvailable());

        // 4. 尝试手动触发 YishapeMathRust.isNativeAvailable (会触发 init)
        System.out.println("\n--- Step 4: Manual native init attempt ---");
        try {
            Class<?> rustClass = Class.forName("com.yishape.lab.math.hpc.internal.YishapeMathRust");
            java.lang.reflect.Method m = rustClass.getMethod("isNativeAvailable");
            boolean available = (Boolean) m.invoke(null);
            System.out.println("isNativeAvailable (manual): " + available);
        } catch (Exception e) {
            System.out.println("Manual init FAILED: " + e);
            Throwable cause = e.getCause();
            while (cause != null) {
                System.out.println("  caused by: " + cause.getClass().getName() + ": " + cause.getMessage());
                cause = cause.getCause();
            }
        }

        // 5. 尝试直接调用 tryLBFGS 看返回了什么
        System.out.println("\n--- Step 5: Direct tryLBFGS test ---");
        double[] x = {0.0, 0.0};
        var obj = new com.yishape.lab.math.optimize.IObjectiveFunction() {
            public double computeObjective(com.yishape.lab.math.linalg.IVector v) { return 0.0; }
        };
        var grd = new com.yishape.lab.math.optimize.IGradientFunction() {
            public com.yishape.lab.math.linalg.IVector computeGradient(com.yishape.lab.math.linalg.IVector v) {
                return com.yishape.lab.math.linalg.Linalg.vector(new double[]{0.0, 0.0});
            }
        };
        var result = HpcOptimizers.tryLBFGS(x, 10, 1e-6, 100, obj, grd);
        if (result == null) {
            System.out.println("tryLBFGS returned null");
        } else {
            System.out.println("tryLBFGS status: " + result.status() + " (OK=" + result.ok() + ")");
            System.out.println("tryLBFGS fx: " + result.fx());
        }

        System.out.println("\n=== Diagnostic Complete ===");
    }
}
