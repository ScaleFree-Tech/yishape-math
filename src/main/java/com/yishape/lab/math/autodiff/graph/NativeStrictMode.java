package com.yishape.lab.math.autodiff.graph;

/**
 * Development-mode guard: when {@code -Dyishape.strictNative=true} is set,
 * any native (HPC/GPU) execution failure that would silently return NaN and
 * fall back to CPU instead throws {@link NativeExecutionException}.
 *
 * <h3>Motivation</h3>
 * <p>In production, GPU/HPC are best-effort accelerators: if they fail, the
 * framework falls back to CPU transparently. This is correct for deployment
 * but <b>disastrous for development</b> — unsupported ops, Rust panics, and
 * protocol mismatches all manifest as silent NaN → CPU fallback, hiding bugs
 * that only surface as "weird, the GPU doesn't seem to help."</p>
 *
 * <h3>Usage</h3>
 * <pre>{@code
 *   mvn test -Dyishape.strictNative=true -Dtest=ProtocolContractTest
 * }</pre>
 *
 * <p>When enabled, any native graph returning NaN throws immediately, so
 * the developer sees exactly which op/backend failed and why.</p>
 *
 * <h3>What still returns NaN even in strict mode</h3>
 * <ul>
 *   <li>GPU/HPC not available (no hardware, no native library)</li>
 *   <li>Graph below minimum-element threshold</li>
 *   <li>Explicit cooldown / rate-limit skip</li>
 * </ul>
 *
 * @since 0.5.0
 */
public final class NativeStrictMode {

    private static final boolean STRICT = Boolean.parseBoolean(
        System.getProperty("yishape.strictNative", "false"));

    private NativeStrictMode() {}

    /** True when {@code -Dyishape.strictNative=true} is active. */
    public static boolean isStrict() {
        return STRICT;
    }

    /**
     * If strict mode is active, throws {@link NativeExecutionException} with
     * the given message. Otherwise returns {@code Double.NaN}.
     *
     * @param backend  "GPU" or "HPC"
     * @param reason   human-readable description of what went wrong
     * @return        never returns normally in strict mode (always throws)
     */
    public static double failOrNaN(String backend, String reason) {
        if (STRICT) {
            throw new NativeExecutionException(backend, reason);
        }
        return Double.NaN;
    }

    /**
     * If strict mode is active, throws {@link NativeExecutionException}.
     * Otherwise returns {@code Double.NaN}.
     *
     * @param backend  "GPU" or "HPC"
     * @param format   printf-style format string
     * @param args     format arguments
     * @return        never returns normally in strict mode (always throws)
     */
    public static double failOrNaN(String backend, String format, Object... args) {
        if (STRICT) {
            throw new NativeExecutionException(backend, String.format(format, args));
        }
        return Double.NaN;
    }

    // ── Exception class ──────────────────────────────────────────────────

    /**
     * Thrown when native (HPC/GPU) graph execution fails in strict mode.
     */
    public static final class NativeExecutionException extends RuntimeException {
        private final String backend;

        NativeExecutionException(String backend, String reason) {
            super("[" + backend + "-STRICT] " + reason
                + " (set -Dyishape.strictNative=false to fall back to CPU)");
            this.backend = backend;
        }

        public String backend() { return backend; }
    }
}
