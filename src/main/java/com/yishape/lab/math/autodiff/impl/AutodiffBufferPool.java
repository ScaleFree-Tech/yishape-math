package com.yishape.lab.math.autodiff.impl;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

/**
 * Thread-local double[] buffer pool for autodiff fused ops.
 * Organized by power-of-2 size buckets to minimize allocation overhead
 * during training loops.
 *
 * <p>Buffers acquired from this pool are zeroed before reuse.
 * Callers must only release buffers they acquired (or freshly allocated ones of known size).</p>
 *
 * <p>Typical usage in FusedOps:
 * <pre>
 *   double[] buf = AutodiffBufferPool.acquire(n);
 *   // ... use buf ...
 *   AutodiffBufferPool.release(buf);
 * </pre></p>
 */
final class AutodiffBufferPool {

    private static final int MAX_BUCKET = 28; // supports up to 2^27 = 128M elements (~1GB for double)
    private static final int MAX_PER_BUCKET = 4; // per-thread cache cap per bucket

    private static final ThreadLocal<Deque<double[]>[]> POOLS = ThreadLocal.withInitial(() -> {
        @SuppressWarnings("unchecked")
        Deque<double[]>[] qs = new Deque[MAX_BUCKET];
        for (int i = 0; i < MAX_BUCKET; i++) qs[i] = new ArrayDeque<>(MAX_PER_BUCKET);
        return qs;
    });

    private AutodiffBufferPool() {}

    /**
     * Acquire a buffer of at least {@code minSize} elements.
     * Returns a zeroed array (either from pool or freshly allocated).
     */
    static double[] acquire(int minSize) {
        if (minSize <= 0) return new double[0];
        int bucket = bucketFor(minSize);
        if (bucket >= MAX_BUCKET) {
            return new double[minSize];
        }
        int allocSize = 1 << bucket;
        Deque<double[]> q = POOLS.get()[bucket];
        double[] buf = q.pollFirst();
        if (buf != null && buf.length >= minSize) {
            Arrays.fill(buf, 0.0);
            return buf;
        }
        return new double[allocSize];
    }

    /**
     * Release a buffer back to the pool.
     * Only buffers whose length is an exact power of 2 are cached;
     * others are silently dropped (let GC handle them).
     */
    static void release(double[] buf) {
        if (buf == null || buf.length == 0) return;
        int bucket = bucketFor(buf.length);
        if ((1 << bucket) != buf.length) return; // not power-of-2, drop
        if (bucket >= MAX_BUCKET) return; // too large, drop
        Deque<double[]> q = POOLS.get()[bucket];
        if (q.size() < MAX_PER_BUCKET) {
            q.offerLast(buf);
        }
    }

    /** Returns the bucket index: smallest n such that 2^n >= size. */
    private static int bucketFor(int size) {
        return 32 - Integer.numberOfLeadingZeros(size - 1);
    }
}
