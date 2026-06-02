package com.yishape.lab.math.testframework;

import org.junit.jupiter.api.Test;
import java.io.*;
import java.nio.file.*;

/**
 * 直接提取 jar 中的 DLL 并尝试 System.load，看具体的 UnsatisfiedLinkError 消息。
 */
public class HpcDllLoadTest {

    @Test
    void diagnoseDllLoadError() throws Exception {
        System.out.println("=== DLL Load Diagnostic ===");

        // 1. Extract DLL from jar
        String resourcePath = "META-INF/native-libs/windows-x86_64/yishape_math_rust.dll";
        System.out.println("Looking for resource: " + resourcePath);

        InputStream is = getClass().getClassLoader().getResourceAsStream(resourcePath);
        if (is == null) {
            // try from YishapeHpc's classloader
            Class<?> hpcClass = Class.forName("com.yishape.lab.math.hpc.YishapeHpc");
            is = hpcClass.getClassLoader().getResourceAsStream(resourcePath);
            System.out.println("  from YishapeHpc classloader: " + (is != null ? "found" : "not found"));
        }

        if (is == null) {
            System.out.println("DLL resource NOT FOUND in classpath!");
            return;
        }
        System.out.println("DLL resource FOUND in classpath");

        // 2. Extract to temp
        Path tempDir = Files.createTempDirectory("hpc_diag_");
        Path dllPath = tempDir.resolve("yishape_math_rust.dll");
        Files.copy(is, dllPath, StandardCopyOption.REPLACE_EXISTING);
        System.out.println("Extracted to: " + dllPath);
        System.out.println("File size: " + Files.size(dllPath) + " bytes");

        // 3. Try System.load
        System.out.println("\nAttempting System.load(\"" + dllPath + "\")...");
        try {
            System.load(dllPath.toString());
            System.out.println("SUCCESS! DLL loaded without error.");
        } catch (UnsatisfiedLinkError e) {
            System.out.println("FAILED: " + e.getClass().getName());
            System.out.println("  Message: " + e.getMessage());

            // Get nested cause for dependency info
            Throwable t = e.getCause();
            while (t != null) {
                System.out.println("  Caused by: " + t.getClass().getName() + ": " + t.getMessage());
                t = t.getCause();
            }

            // Also try to get more info about missing DLLs using Dependency Walker style
            System.out.println("\nCommon causes on Windows:");
            System.out.println("  - Missing MSVC runtime (vcruntime140.dll, msvcp140.dll)");
            System.out.println("  - Missing Rust runtime dependencies");
            System.out.println("  - DLL compiled for different architecture");
            System.out.println("  - Missing LAPACK/BLAS DLLs");
            System.out.println("  - Missing libgcc_s_seh or similar MinGW dependencies");
        }

        // 4. Cleanup
        try { Files.deleteIfExists(dllPath); } catch (Exception ignored) {}
        try { Files.deleteIfExists(tempDir); } catch (Exception ignored) {}

        System.out.println("\n=== Done ===");
    }
}
