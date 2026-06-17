package com.yishape.lab.math.autodiff.support;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

import com.yishape.lab.util.YishapeLogger;

/**
 * Low-overhead profiler for tracking {@code toDoubleArray()} call sites in
 * the autodiff delegate layer.
 *
 * <p>Enabled via system property {@code -Dyishape.profile.toDoubleArray=true}
 * or {@link #enable()}. When enabled, each call to {@link #record()} captures
 * the caller's class and method via {@link StackWalker} and increments a
 * per-site counter. When disabled, {@code record()} is a no-op (single
 * volatile read, no allocation).
 *
 * <p>Usage — instrument the single choke point:
 * <pre>{@code
 * // In RereDoubleTensor.toDoubleArray():
 * if (ToDoubleArrayProfiler.ENABLED) ToDoubleArrayProfiler.record();
 * }</pre>
 *
 * <p>Dump top-N callers via {@link #dumpTopN(int)} (debug level) or
 * {@link #dumpTopN(int, Appendable)} (custom output).
 */
public final class ToDoubleArrayProfiler {

    private static final YishapeLogger log = YishapeLogger.getLogger(ToDoubleArrayProfiler.class);

    /** Master switch — single volatile read on the hot path. */
    public static volatile boolean ENABLED = Boolean.getBoolean("yishape.profile.toDoubleArray");

    private static final ConcurrentHashMap<String, LongAdder> COUNTS = new ConcurrentHashMap<>();

    private static final StackWalker WALKER = StackWalker.getInstance(StackWalker.Option.RETAIN_CLASS_REFERENCE);

    private ToDoubleArrayProfiler() {}

    /** Enable profiling. Safe to call multiple times. */
    public static void enable() { ENABLED = true; }

    /** Disable profiling and clear counters. */
    public static void disable() { ENABLED = false; }

    /**
     * Record a toDoubleArray call, capturing the caller location.
     * No-op when {@link #ENABLED} is {@code false}.
     */
    public static void record() {
        if (!ENABLED) return;
        String caller = resolveCaller();
        COUNTS.computeIfAbsent(caller, k -> new LongAdder()).increment();
    }

    /**
     * Record with an explicit tag (avoiding StackWalker overhead).
     * Use this for manual instrumentation when the caller is statically known.
     */
    public static void record(String tag) {
        if (!ENABLED) return;
        COUNTS.computeIfAbsent(tag, k -> new LongAdder()).increment();
    }

    /** @return the number of distinct call sites recorded */
    public static int siteCount() {
        return COUNTS.size();
    }

    /** Clear all counters. */
    public static void reset() {
        COUNTS.clear();
    }

    /**
     * Dump the top-N call sites to the debug log.
     * @param n maximum number of entries to dump
     */
    public static void dumpTopN(int n) {
        if (COUNTS.isEmpty()) return;
        List<Map.Entry<String, LongAdder>> sorted = sortedEntries(n);
        long total = sorted.stream().mapToLong(e -> e.getValue().sum()).sum();
        StringBuilder sb = new StringBuilder(512);
        sb.append("=== toDoubleArray Profile (top ").append(Math.min(n, sorted.size()))
          .append(" of ").append(COUNTS.size()).append(" sites, ").append(total).append(" total calls) ===\n");
        int rank = 1;
        for (var e : sorted) {
            long count = e.getValue().sum();
            double pct = total > 0 ? 100.0 * count / total : 0;
            sb.append(String.format("  %2d. %6d (%5.1f%%)  %s\n", rank++, count, pct, e.getKey()));
        }
        log.info(sb.toString());
    }

    private static List<Map.Entry<String, LongAdder>> sortedEntries(int n) {
        List<Map.Entry<String, LongAdder>> list = new ArrayList<>(COUNTS.entrySet());
        list.sort(Map.Entry.<String, LongAdder>comparingByValue(
            Comparator.comparingLong(LongAdder::sum)).reversed());
        return list.subList(0, Math.min(n, list.size()));
    }

    /**
     * Walk the stack to find the first caller outside this profiler class
     * and outside {@code RereDoubleTensor.toDoubleArray()}.
     */
    private static String resolveCaller() {
        return WALKER.walk(frames -> frames
            .filter(f -> !f.getClassName().equals(ToDoubleArrayProfiler.class.getName()))
            .filter(f -> !f.getClassName().contains("RereDoubleTensor"))
            .skip(0) // skip the immediate wrapper if any
            .findFirst()
            .map(f -> shortName(f.getClassName()) + "." + f.getMethodName())
            .orElse("unknown"));
    }

    /** Strip package prefix for readability. */
    private static String shortName(String fqn) {
        int dot = fqn.lastIndexOf('.');
        return dot >= 0 ? fqn.substring(dot + 1) : fqn;
    }
}
