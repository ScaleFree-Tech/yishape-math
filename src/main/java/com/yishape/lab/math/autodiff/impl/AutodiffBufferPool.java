package com.yishape.lab.math.autodiff.impl;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.IdentityHashMap;
import java.util.Map;

/**
 * Thread-local double[] buffer pool for autodiff fused ops.
 * <p>
 * <b>Bucketing strategy (v2, POOL-1 + POOL-2):</b>
 * <ul>
 *   <li><b>Slab region</b> (&le; 64 KB = 8192 doubles): dense buckets at 256-element
 *       granularity — request 100 → alloc 256 (not 128). Reduces waste from 50% to
 *       &lt; 25% per bucket in the hot small-array regime.</li>
 *   <li><b>Power-of-2 region</b> (&gt; 8192): classic 2ⁿ buckets. Large arrays are rare
 *       enough that per-bucket overhead dominates fragmentation concern.</li>
 *   <li><b>Non-power-of-2 recovery</b>: {@code release()} finds the largest bucket whose
 *       allocate-size &le; {@code buf.length}, so any size array can be recycled. Previously
 *       non-power-of-2 arrays were dropped for GC.</li>
 * </ul>
 *
 * <p>Buffers acquired from this pool are zeroed before reuse.
 * Callers must only release buffers they acquired (or freshly allocated ones of known size).</p>
 *
 * <p>Typical usage:
 * <pre>
 *   double[] buf = AutodiffBufferPool.acquire(n);
 *   // ... use buf ...
 *   AutodiffBufferPool.release(buf);
 * </pre></p>
 */
final class AutodiffBufferPool {

    // --- slab region ---
    private static final int SLAB_SIZE = 256;            // 256 doubles = 2 KB
    private static final int SLAB_MAX_ELEMS = 8192;      // 8192 doubles = 64 KB
    private static final int SLAB_BUCKETS = SLAB_MAX_ELEMS / SLAB_SIZE; // 32 buckets
    // --- power-of-2 region ---
    private static final int POW2_BUCKETS = 22; // 2^13 (=8192, first pow2 bucket) .. 2^27 (=128M)
    private static final int TOTAL_BUCKETS = SLAB_BUCKETS + POW2_BUCKETS; // 32 + 22 = 54

    private static final int MAX_PER_BUCKET = 4;

    private static final ThreadLocal<Deque<double[]>[]> POOLS = ThreadLocal.withInitial(() -> {
        @SuppressWarnings("unchecked")
        Deque<double[]>[] qs = new Deque[TOTAL_BUCKETS];
        for (int i = 0; i < TOTAL_BUCKETS; i++) qs[i] = new ArrayDeque<>(MAX_PER_BUCKET);
        return qs;
    });

    /**
     * Tracks buffers acquired but not yet released per thread.
     * Used for leak detection and cleanup after backward graph execution.
     */
    private static final ThreadLocal<Map<double[], Boolean>> LEASED = ThreadLocal.withInitial(IdentityHashMap::new);

    private AutodiffBufferPool() {}

    /**
     * Acquire a buffer of at least {@code minSize} elements.
     * Returns a zeroed array (either from pool or freshly allocated).
     */
    static double[] acquire(int minSize) {
        if (minSize <= 0) return new double[0];
        int bucket = bucketFor(minSize);
        if (bucket < 0) {
            return new double[minSize]; // too large for pool, allocate exact
        }
        int allocSize = allocSizeForBucket(bucket);
        Deque<double[]> q = POOLS.get()[bucket];
        double[] buf = q.pollFirst();
        if (buf != null && buf.length >= minSize) {
            Arrays.fill(buf, 0.0);
            LEASED.get().put(buf, Boolean.TRUE);
            return buf;
        }
        // Allocate slab-aligned or exact — never round up to next power-of-2
        buf = new double[allocSize];
        LEASED.get().put(buf, Boolean.TRUE);
        return buf;
    }

    /**
     * Release a buffer back to the pool.
     * Finds the largest bucket whose allocate-size ≤ buf.length so non-power-of-2
     * arrays can be recycled (POOL-2 fix). Previously these were dropped for GC.
     */
    static void release(double[] buf) {
        if (buf == null || buf.length == 0) return;
        LEASED.get().remove(buf);
        int bucket = releaseBucketFor(buf.length);
        if (bucket < 0) return; // too large or too small to pool
        Deque<double[]> q = POOLS.get()[bucket];
        if (q.size() < MAX_PER_BUCKET) {
            q.offerLast(buf);
        }
    }

    /**
     * Release all buffers acquired by this thread that were not explicitly released.
     * Called after backward graph execution to prevent leaks from exception paths.
     */
    static void cleanupThread() {
        Map<double[], Boolean> leased = LEASED.get();
        if (!leased.isEmpty()) {
            // Copy keys before iterating: release() calls leased.remove() which would
            // ConcurrentModify the keySet being iterated.
            for (double[] buf : leased.keySet().toArray(new double[0][])) {
                release(buf);
            }
            leased.clear();
        }
    }

    // --- bucket mapping ---

    /**
     * Map {@code minSize} to a bucket index for acquisition.
     * Returns -1 when size exceeds the maximum poolable size.
     */
    private static int bucketFor(int minSize) {
        if (minSize <= SLAB_MAX_ELEMS) {
            // Slab: bucket 0..31. bucket = ceil(minSize / 256) - 1, clamped to 0..31
            int b = (minSize + SLAB_SIZE - 1) / SLAB_SIZE - 1;
            return Math.max(0, Math.min(b, SLAB_BUCKETS - 1));
        }
        // Power-of-2: bucket 32..53. Find smallest 2^n >= minSize.
        int p = 32 - Integer.numberOfLeadingZeros(minSize - 1); // ceil(log2(minSize))
        if (p > 27) return -1; // > 128M doubles (~1 GB), don't pool
        return SLAB_BUCKETS + (p - 13); // 2^13 = 8192 → bucket 32
    }

    /**
     * Map a released buffer's length to the largest bucket whose allocate-size ≤ length.
     * This is how non-power-of-2 arrays get recycled — they land in a smaller bucket
     * and may be used for smaller requests, accepting controlled fragmentation.
     */
    private static int releaseBucketFor(int length) {
        if (length <= 0) return -1;
        // Match acquire bucket mapping, then walk down while alloc size exceeds buffer length.
        int b = bucketFor(length);
        if (b < 0) return -1;
        while (b > 0 && allocSizeForBucket(b) > length) {
            b--;
        }
        return b;
    }

    /** Size allocated for a given bucket index. */
    private static int allocSizeForBucket(int bucket) {
        if (bucket < SLAB_BUCKETS) {
            return (bucket + 1) * SLAB_SIZE;
        }
        return 1 << (bucket - SLAB_BUCKETS + 13);
    }

    // --- diagnostics (package-private for testing) ---

    static int bucketCount() { return TOTAL_BUCKETS; }
    static int slabBuckets() { return SLAB_BUCKETS; }
    static int pow2Buckets() { return POW2_BUCKETS; }
    static int slabSize() { return SLAB_SIZE; }
    static int slabMaxElems() { return SLAB_MAX_ELEMS; }

    /** Clear all per-thread pools. Safe to call from shutdown hooks. */
    static void resetThreadLocals() {
        POOLS.remove();
    }
}
