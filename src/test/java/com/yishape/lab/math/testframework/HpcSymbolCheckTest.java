package com.yishape.lab.math.testframework;

import org.junit.jupiter.api.Test;
import java.io.*;
import java.lang.foreign.*;
import java.nio.file.*;
import java.util.Optional;

/**
 * 检查加载的 DLL 暴露了哪些 L-BFGS 相关符号。
 */
public class HpcSymbolCheckTest {

    @Test
    void checkLbfgsSymbols() throws Exception {
        // Load DLL
        String resourcePath = "META-INF/native-libs/windows-x86_64/yishape_math_rust.dll";
        InputStream is = getClass().getClassLoader().getResourceAsStream(resourcePath);
        Path dllPath = Files.createTempDirectory("hpc_sym_").resolve("yishape_math_rust.dll");
        Files.copy(is, dllPath, StandardCopyOption.REPLACE_EXISTING);
        System.load(dllPath.toString());

        SymbolLookup lookup = SymbolLookup.loaderLookup();

        // Check all expected exported symbols (prefixed with yishape_hpc_)
        String[] lbfgsSyms = {
            "yishape_hpc_lbfgs_minimize",
            "yishape_hpc_owlqn_minimize",
            "yishape_hpc_abi_version",
            "test_hello",
            "solve_linear_system",
            "yishape_hpc_dense_svd_col_major",
            "yishape_hpc_symmetric_eigen_col_major",
            "yishape_hpc_eigen_nonsymmetric_col_major",
            "yishape_hpc_dense_cholesky_lower_col_major",
            "yishape_hpc_dense_qr_col_major",
            "yishape_hpc_dense_lu_col_major",
            "yishape_hpc_dense_inverse_col_major",
            "yishape_hpc_dgemm_col_major",
            "yishape_hpc_lp_minimize_nonneg",
            "yishape_hpc_hnsw_build_f32",
        };

        System.out.println("=== L-BFGS Symbol Check ===");
        for (String sym : lbfgsSyms) {
            System.out.printf("%-45s ", sym);
            Optional<MemorySegment> addr = lookup.find(sym);
            if (addr.isPresent()) {
                System.out.println("FOUND  (addr=" + addr.get().address() + ")");
            } else {
                System.out.println("NOT FOUND");
            }
        }

        // Also dump all yishape_* symbols
        System.out.println("\n=== All yishape_* symbols (heuristic scan) ===");
        String[] commonPrefixes = {
            "yishape_", "lbfgs_", "owlqn_", "gosh_", "test_", "solve_"
        };
        for (String prefix : commonPrefixes) {
            for (int i = 0; i < 200; i++) {
                String testSym = prefix + i;
                lookup.find(testSym).ifPresent(addr ->
                    System.out.println("  " + testSym + " -> " + addr.address()));
            }
        }

        System.out.println("(SymbolLookup doesn't support enumeration; use dumpbin /exports on the DLL file directly)");

        // Cleanup
        try { Files.deleteIfExists(dllPath); } catch (Exception ignored) {}

        System.out.println("\n=== Try full YishapeMathRust initialization ===");
        try {
            Class<?> rustClass = Class.forName("com.yishape.lab.math.hpc.internal.YishapeMathRust");
            java.lang.reflect.Method m = rustClass.getMethod("isNativeAvailable");
            // Reset init state via reflection to force re-init
            java.lang.reflect.Field initStateField = rustClass.getDeclaredField("initState");
            initStateField.setAccessible(true);
            int oldState = initStateField.getInt(null);
            System.out.println("Old initState: " + oldState);

            // Try init again (already failed, initState=2, won't re-init)
            boolean available = (Boolean) m.invoke(null);
            System.out.println("isNativeAvailable: " + available);
            System.out.println("initState after: " + initStateField.getInt(null));

            if (!available && oldState == 2) {
                System.out.println("\ninitState=2 means initialization FAILED previously.");
                System.out.println("The error was caught but we can't see what it was without modifying the code.");
                System.out.println("Run the DLL through `dumpbin /exports` or `nm` to verify symbols match what YishapeMathRust expects.");
            }
        } catch (Exception e) {
            System.out.println("Error: " + e);
        }

        System.out.println("\nDLL path was: " + dllPath.getParent() + " (cleaned up)");
    }
}
