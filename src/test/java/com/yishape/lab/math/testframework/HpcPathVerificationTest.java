package com.yishape.lab.math.testframework;

import com.yishape.lab.math.compute.hpc.HpcOptimizers;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.newton.RustLBFGS;
import org.junit.jupiter.api.Test;

/**
 * 验证 RustLBFGS/RustOWLQN 是否真的走了 HPC 路径还是静默回退到了 RereLBFGS。
 */
public class HpcPathVerificationTest {

    @Test
    void verifyHpcPathStatus() {
        System.out.println("=== HPC Path Verification ===");
        System.out.println("isExtensionPresent: " + HpcOptimizers.isExtensionPresent());
        System.out.println("isNativeRuntimeAvailable: " + HpcOptimizers.isNativeRuntimeAvailable());

        // 检查 RustLBFGS 的类初始化是否成功加载了 HPC 类
        try {
            Class<?> hpcClass = Class.forName("com.yishape.lab.math.hpc.YishapeHpc");
            System.out.println("YishapeHpc class loaded: " + hpcClass.getName());
        } catch (ClassNotFoundException e) {
            System.out.println("YishapeHpc class NOT on classpath: " + e.getMessage());
        }

        // 检查 classpath 中是否存在 yishape-math-hpc jar
        String classpath = System.getProperty("java.class.path");
        boolean hasHpcJar = classpath.contains("yishape-math-hpc");
        System.out.println("yishape-math-hpc in classpath: " + hasHpcJar);
        if (hasHpcJar) {
            for (String part : classpath.split(System.getProperty("path.separator"))) {
                if (part.contains("yishape-math-hpc")) {
                    System.out.println("  -> " + part);
                }
            }
        }

        System.out.println("=== Verdict ===");
        if (HpcOptimizers.isExtensionPresent()) {
            System.out.println("HPC EXTENSION IS ACTIVE - RustLBFGS WILL USE HPC PATH");
        } else {
            System.out.println("HPC EXTENSION NOT AVAILABLE - RustLBFGS FALLS BACK TO RereLBFGS");
        }
    }
}
