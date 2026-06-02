package com.yishape.lab.math.ml.neighbor;

import com.yishape.lab.math.vecidx.SearchHit;
import com.yishape.lab.math.vecidx.VecSearchOption;
import com.yishape.lab.math.vecidx.distance.EuclideanMetric;
import com.yishape.lab.math.vecidx.impl.*;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.function.Supplier;
import com.yishape.lab.math.vecidx.IFloatVecIdx;

/**
 * Vecidx 各实现性能对比基准测试。
 */
public class VecidxBenchmark {

    static final int DIM = 128;
    static final int N = 100_000;
    static final int Q = 20;
    static final int K = 10;

    static final Random RAND = new Random(42);

    public static void main(String[] args) {
        System.out.printf("=== Vecidx Benchmark: dim=%d, N=%,d, queries=%d, k=%d ===%n%n", DIM, N, Q, K);

        // --- Generate data ---
        long mem0 = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        System.out.printf("Generating %,d random vectors...%n", N);
        long tGen0 = System.nanoTime();

        float[][] data = new float[N][DIM];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < DIM; j++) data[i][j] = RAND.nextFloat() * 2 - 1;
        }
        String[] ids = new String[N];
        for (int i = 0; i < N; i++) ids[i] = String.valueOf(i);

        float[][] queries = new float[Q][DIM];
        for (int i = 0; i < Q; i++) {
            for (int j = 0; j < DIM; j++) queries[i][j] = RAND.nextFloat() * 2 - 1;
        }

        long tGen1 = System.nanoTime();
        long mem1 = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        System.out.printf("  Generated in %.2f s, data memory ~%,d MB%n",
                (tGen1 - tGen0) / 1_000_000_000.0, (mem1 - mem0) / (1024 * 1024));

        VecSearchOption exactOpts = VecSearchOption.EXACT;
        VecSearchOption approxOpts = VecSearchOption.DEFAULT;

        // --- Ground truth ---
        System.out.println("\n--- Computing ground truth ---");
        List<List<SearchHit>> groundTruth;
        {
            long t0 = System.nanoTime();
            try (BruteForceFloatVecIdx bf = new BruteForceFloatVecIdx(data, ids, EuclideanMetric.FLOAT)) {
                groundTruth = new java.util.ArrayList<>(Q);
                for (int i = 0; i < Q; i++) groundTruth.add(bf.search(queries[i], K));
            }
            long t1 = System.nanoTime();
            System.out.printf("  Done in %.2f s%n", (t1 - t0) / 1_000_000_000.0);
        }

        // --- Each method: build once, then query + recall ---
        run("BruteForce (exact)", () -> new BruteForceFloatVecIdx(data, ids, EuclideanMetric.FLOAT),
                queries, groundTruth);
        run("KD-Tree   (exact)", () -> new KdTreeFloatVecIdx(data, ids, EuclideanMetric.FLOAT, exactOpts),
                queries, groundTruth);
        run("HNSW      (Rust)", () -> new RustHnswFloatVecIdx(data, ids, EuclideanMetric.FLOAT, approxOpts),
                queries, groundTruth);
        run("JavaHNSW  (Java)", () -> new RereHnswFloatVecIdx(data, ids, EuclideanMetric.FLOAT, approxOpts),
                queries, groundTruth);
        run("LSH       (approx)", () -> new LshFloatVecIdx(data, ids, EuclideanMetric.FLOAT, approxOpts),
                queries, groundTruth);
        run("PQ        (approx)", () -> new PqFloatVecIdx(data, ids, EuclideanMetric.FLOAT, approxOpts),
                queries, groundTruth);
        run("PQ+HNSW   (approx)", () -> new PqHnswFloatVecIdx(data, ids, EuclideanMetric.FLOAT, approxOpts),
                queries, groundTruth);

        System.out.println("\nDone.");
    }

    static void run(String label, Supplier<IFloatVecIdx> builder, float[][] queries,
                    List<List<SearchHit>> groundTruth) {
        System.out.println();

        // 1. Build
        System.gc();
        long tB0 = System.nanoTime();
        IFloatVecIdx idx = builder.get();
        long tB1 = System.nanoTime();
        double buildS = (tB1 - tB0) / 1_000_000_000.0;
        System.out.printf("-- %s --%n", label);
        System.out.printf("  Build:        %8.2f s  (size=%,d)%n", buildS, idx.size());

        // 2. Query benchmark
        System.gc();
        long t0 = System.nanoTime();
        for (float[] q : queries) idx.search(q, K);
        long t1 = System.nanoTime();
        double queryMs = (t1 - t0) / 1_000_000.0;
        System.out.printf("  Query %dx%d:   %8.2f ms  (%.2f ms/q)%n", Q, K, queryMs, queryMs / Q);

        // 3. Recall
        double sumOverlap = 0;
        for (int i = 0; i < queries.length; i++) {
            List<SearchHit> res = idx.search(queries[i], K);
            Set<String> gtSet = new HashSet<>(groundTruth.get(i).stream().map(SearchHit::id).toList());
            long matches = res.stream().map(SearchHit::id).filter(gtSet::contains).count();
            sumOverlap += (double) matches / K;
        }
        System.out.printf("  Recall@%d:      %.4f%n", K, sumOverlap / queries.length);

        // 4. Close
        if (idx instanceof AutoCloseable c) {
            try { c.close(); } catch (Exception ignored) {}
        }
    }
}
