package com.yishape.lab.math.ml.neighbor;

import com.yishape.lab.math.vecidx.VI;
import com.yishape.lab.math.vecidx.VecSearchOption;
import com.yishape.lab.math.vecidx.SearchHit;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.vecidx.impl.BruteForceFloatVecIdx;
import com.yishape.lab.math.vecidx.impl.PqHnswDoubleVecIdx;
import com.yishape.lab.math.vecidx.impl.PqHnswFloatVecIdx;
import com.yishape.lab.math.vecidx.distance.EuclideanMetric;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import com.yishape.lab.math.vecidx.IFloatVecIdx;
import com.yishape.lab.math.vecidx.IDoubleVecIdx;
import com.yishape.lab.math.vecidx.IMutableVecIdx;

/**
 * PQ + HNSW 组合向量索引的端到端集成测试。
 *
 * <p>验证 {@link PqHnswFloatVecIdx} 与 {@link PqHnswDoubleVecIdx}
 * 的构建、搜索、增删语义正确，且召回率满足近似索引预期。</p>
 */
class PqHnswVectorIndexIntegrationTest {

    @Test
    void floatIndexBuildAndSearch() {
        float[][] data = {
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 1.0f},
                {1.0f, 1.0f, 1.0f}
        };
        String[] ids = {"a", "b", "c", "d"};
        VecSearchOption opts = VecSearchOption.DEFAULT;

        IFloatVecIdx idx = VI.buildPqHnswFloat(data, ids, EuclideanMetric.FLOAT, opts);
        assertTrue(idx instanceof PqHnswFloatVecIdx);
        assertEquals(3, idx.dimensions());
        assertEquals(4, idx.size());
        assertTrue(idx.isApproximate());

        float[] query = {1.0f, 0.1f, 0.0f};
        List<SearchHit> hits = idx.search(query, 2, null, null);
        assertEquals(2, hits.size());
        assertEquals("a", hits.get(0).id());
        assertTrue(hits.get(0).distance() >= 0.0);

        IVector<Float> vec = idx.getVector("a");
        assertNotNull(vec);
        assertEquals(1.0f, vec.get(0), 1e-4f);

        idx.close();
    }

    @Test
    void doubleIndexBuildAndSearch() {
        double[][] data = {
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
                {0.0, 0.0, 1.0},
                {1.0, 1.0, 1.0}
        };
        String[] ids = {"d0", "d1", "d2", "d3"};
        VecSearchOption opts = VecSearchOption.DEFAULT;

        IDoubleVecIdx idx = VI.buildPqHnswDouble(data, ids, EuclideanMetric.DOUBLE, opts);
        assertTrue(idx instanceof PqHnswDoubleVecIdx);
        assertEquals(3, idx.dimensions());
        assertEquals(4, idx.size());

        double[] query = {1.0, 0.1, 0.0};
        List<SearchHit> hits = idx.search(query, 1, null, null);
        assertEquals(1, hits.size());
        assertEquals("d0", hits.get(0).id());
        assertTrue(hits.get(0).distance() >= 0.0);

        idx.close();
    }

    @Test
    void opqFloatBuildAndSearch() {
        float[][] data = {
                {1.0f, 0.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 0.0f, 1.0f},
                {1.0f, 1.0f, 1.0f, 1.0f}
        };
        String[] ids = {"a", "b", "c", "d", "e"};
        VecSearchOption opts = VecSearchOption.DEFAULT;

        IFloatVecIdx idx = VI.buildOpqHnswFloat(data, ids, EuclideanMetric.FLOAT, opts);
        assertTrue(idx instanceof PqHnswFloatVecIdx);
        assertEquals(4, idx.dimensions());
        assertEquals(5, idx.size());

        float[] query = {1.0f, 0.1f, 0.0f, 0.0f};
        List<SearchHit> hits = idx.search(query, 2, null, null);
        assertEquals(2, hits.size());
        assertEquals("a", hits.get(0).id());

        idx.close();
    }

    @Test
    void mutableAddAndRemove() {
        float[][] data = {
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 1.0f},
                {1.0f, 1.0f, 0.0f},
                {0.0f, 1.0f, 1.0f},
                {1.0f, 0.0f, 1.0f},
                {0.5f, 0.5f, 0.5f},
                {0.2f, 0.8f, 0.3f}
        };
        String[] ids = {"a", "b", "c", "d", "e", "f", "g", "h"};
        VecSearchOption opts = VecSearchOption.DEFAULT;

        IFloatVecIdx idx = VI.buildPqHnswFloat(data, ids, EuclideanMetric.FLOAT, opts);
        IMutableVecIdx<Float> mut = (IMutableVecIdx<Float>) idx;

        mut.add("new", IVector.of(new float[]{0.05f, 0.05f, 0.95f}));
        assertEquals(9, idx.size());
        assertTrue(mut.contains("new"));

        float[] query = {0.0f, 0.0f, 1.0f};
        List<SearchHit> hits = idx.search(query, 2, null, null);
        assertEquals(2, hits.size());
        // "new" is closest to query, "c" is exact match
        Set<String> top2 = Set.of(hits.get(0).id(), hits.get(1).id());
        assertTrue(top2.contains("new"), "top-2 should contain newly added vector near query");

        assertTrue(mut.remove("b"));
        assertEquals(8, idx.size());
        assertNull(idx.getVector("b"));

        idx.close();
    }

    @Test
    void recallVsBruteForce() {
        int n = 500;
        int dim = 16;
        Random rand = new Random(42);

        float[][] data = new float[n][dim];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                data[i][j] = rand.nextFloat();
            }
        }
        String[] ids = new String[n];
        for (int i = 0; i < n; i++) {
            ids[i] = String.valueOf(i);
        }

        VecSearchOption opts = VecSearchOption.DEFAULT;

        IFloatVecIdx approx = VI.buildPqHnswFloat(data, ids, EuclideanMetric.FLOAT, opts);
        IFloatVecIdx exact = new BruteForceFloatVecIdx(data, ids, EuclideanMetric.FLOAT);

        int queries = 20;
        int k = 10;
        double totalRecall = 0.0;

        for (int q = 0; q < queries; q++) {
            float[] query = new float[dim];
            for (int j = 0; j < dim; j++) {
                query[j] = rand.nextFloat();
            }

            List<SearchHit> approxHits = approx.search(query, k, null, null);
            List<SearchHit> exactHits = exact.search(query, k, null, null);

            Set<String> exactSet = new HashSet<>();
            for (SearchHit h : exactHits) {
                exactSet.add(h.id());
            }

            int overlap = 0;
            for (SearchHit h : approxHits) {
                if (exactSet.contains(h.id())) {
                    overlap++;
                }
            }
            totalRecall += (double) overlap / k;
        }

        double avgRecall = totalRecall / queries;
        assertTrue(avgRecall >= 0.60,
                "PQ+HNSW average recall should be >= 0.60, got " + avgRecall);

        approx.close();
        exact.close();
    }

    @Test
    void emptyIndexSearchReturnsEmpty() {
        VecSearchOption opts = VecSearchOption.DEFAULT;
        PqHnswFloatVecIdx idx = new PqHnswFloatVecIdx(4, EuclideanMetric.FLOAT, opts, false);
        assertEquals(0, idx.size());

        float[] query = {1.0f, 0.0f, 0.0f, 0.0f};
        List<SearchHit> hits = idx.search(query, 5, null, null);
        assertTrue(hits.isEmpty());

        idx.close();
    }

    @Test
    void filterAndExclude() {
        float[][] data = {
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 1.0f},
                {0.5f, 0.5f, 0.5f}
        };
        String[] ids = {"a", "b", "c", "d"};
        VecSearchOption opts = VecSearchOption.DEFAULT;

        IFloatVecIdx idx = VI.buildPqHnswFloat(data, ids, EuclideanMetric.FLOAT, opts);

        float[] query = {1.0f, 0.0f, 0.0f};

        // exclude "a"
        List<SearchHit> hits = idx.search(query, 1, List.of("a"), null);
        assertEquals(1, hits.size());
        assertEquals("d", hits.get(0).id());

        // filter predicate
        List<SearchHit> hits2 = idx.search(query, 1, null, id -> id.equals("c"));
        assertEquals(1, hits2.size());
        assertEquals("c", hits2.get(0).id());

        idx.close();
    }
}
