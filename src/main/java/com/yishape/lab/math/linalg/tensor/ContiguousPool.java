package com.yishape.lab.math.linalg.tensor;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Thread-local pool of double[] buffers for contiguous() materialization.
 * Power-of-2 bucketed, max 64 buffers per thread.
 */
public final class ContiguousPool {

    private static final int MAX_PER_BUCKET = 64;

    private static final ThreadLocal<Deque<double[]>[]> POOLS = ThreadLocal.withInitial(() -> {
        @SuppressWarnings("unchecked")
        Deque<double[]>[] buckets = new Deque[32]; // 2^0 through 2^31
        return buckets;
    });

    private ContiguousPool() {}

    public static double[] acquire(int size) {
        int bucket = 32 - Integer.numberOfLeadingZeros(Math.max(size, 1) - 1);
        if (bucket >= 32) return new double[size];
        Deque<double[]>[] buckets = POOLS.get();
        Deque<double[]> q = buckets[bucket];
        if (q != null && !q.isEmpty()) {
            double[] buf = q.pollFirst();
            if (buf != null) return buf;
        }
        return new double[1 << bucket];
    }

    public static void release(double[] buf) {
        if (buf == null) return;
        int bucket = 32 - Integer.numberOfLeadingZeros(buf.length - 1);
        if (bucket >= 32) return;
        Deque<double[]>[] buckets = POOLS.get();
        Deque<double[]> q = buckets[bucket];
        if (q == null) {
            q = new ArrayDeque<>(MAX_PER_BUCKET);
            buckets[bucket] = q;
        }
        if (q.size() < MAX_PER_BUCKET) {
            q.addLast(buf);
        }
    }

    public static void clear() {
        Deque<double[]>[] buckets = POOLS.get();
        for (int i = 0; i < buckets.length; i++) {
            buckets[i] = null;
        }
    }
}
