package com.yishape.lab.math.ml.neighbor;

import com.yishape.lab.math.vecidx.MetricType;
import com.yishape.lab.math.vecidx.VecSearchOption;
import com.yishape.lab.math.vecidx.SearchHit;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.vecidx.impl.RereHnswDoubleVecIdx;
import com.yishape.lab.math.vecidx.impl.RereHnswFloatVecIdx;
import com.yishape.lab.math.vecidx.distance.CosineMetric;
import com.yishape.lab.math.vecidx.distance.EuclideanMetric;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import com.yishape.lab.math.vecidx.IDisMetric;
import com.yishape.lab.math.vecidx.IFloatVecIdx;

/**
 * 纯 Java HNSW 向量索引的单元测试（不依赖 Rust FFI）。
 */
class JavaHnswVectorIndexTest {

    @Test
    void floatIndexEuclideanSmoke() {
        float[][] data = {
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 1.0f},
                {1.0f, 1.0f, 1.0f}
        };
        String[] ids = {"a", "b", "c", "d"};
        VecSearchOption opts = VecSearchOption.DEFAULT;

        RereHnswFloatVecIdx idx = new RereHnswFloatVecIdx(
                data, ids, EuclideanMetric.FLOAT, opts);

        assertEquals(3, idx.dimensions());
        assertEquals(4, idx.size());
        assertTrue(idx.isApproximate());
        assertTrue(idx.isConcurrent());

        // search near [1,0,0]
        float[] query = {1.0f, 0.1f, 0.0f};
        List<SearchHit> hits = idx.search(query, 2, null, null);
        assertEquals(2, hits.size());
        assertEquals("a", hits.get(0).id());
        assertTrue(hits.get(0).distance() >= 0.0);

        // get vector
        IVector<Float> vec = idx.getVector("a");
        assertNotNull(vec);
        assertEquals(1.0f, vec.get(0), 1e-4f);

        // add
        idx.add("e", IVector.of(new float[]{0.5f, 0.5f, 0.5f}));
        assertEquals(5, idx.size());
        assertTrue(idx.contains("e"));

        // search after add
        List<SearchHit> hits2 = idx.search(query, 3, null, null);
        assertTrue(hits2.size() >= 1);

        idx.close();
    }

    @Test
    void floatIndexCosineSmoke() {
        float[][] data = {
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f},
                {1.0f, 1.0f, 0.0f}
        };
        String[] ids = {"x", "y", "z"};
        VecSearchOption opts = VecSearchOption.DEFAULT;

        RereHnswFloatVecIdx idx = new RereHnswFloatVecIdx(
                data, ids, CosineMetric.FLOAT, opts);

        float[] query = {1.0f, 0.0f, 0.0f};
        List<SearchHit> hits = idx.search(query, 1, null, null);
        assertEquals(1, hits.size());
        assertEquals("x", hits.get(0).id());
        assertTrue(hits.get(0).distance() >= 0.0 && hits.get(0).distance() <= 2.0);

        idx.close();
    }

    @Test
    void floatIndexWithExcludesAndFilter() {
        float[][] data = {
                {1.0f, 0.0f},
                {0.0f, 1.0f},
                {0.5f, 0.5f},
                {0.9f, 0.1f}
        };
        String[] ids = {"0", "1", "2", "3"};
        VecSearchOption opts = VecSearchOption.DEFAULT;

        RereHnswFloatVecIdx idx = new RereHnswFloatVecIdx(
                data, ids, EuclideanMetric.FLOAT, opts);

        float[] query = {1.0f, 0.0f};

        // exclude "0"
        Set<String> excludes = new HashSet<>();
        excludes.add("0");
        List<SearchHit> hits = idx.search(query, 1, excludes, null);
        assertEquals(1, hits.size());
        assertEquals("3", hits.get(0).id()); // [0.9, 0.1] is next closest

        // filter
        List<SearchHit> hits2 = idx.search(query, 1, null, id -> id.equals("1"));
        assertEquals(1, hits2.size());
        assertEquals("1", hits2.get(0).id());

        idx.close();
    }

    @Test
    void doubleIndexSmoke() {
        double[][] data = {
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
                {0.0, 0.0, 1.0}
        };
        String[] ids = {"d0", "d1", "d2"};
        VecSearchOption opts = VecSearchOption.DEFAULT;

        RereHnswDoubleVecIdx idx = new RereHnswDoubleVecIdx(
                data, ids, EuclideanMetric.DOUBLE, opts);

        assertEquals(3, idx.size());

        double[] query = {1.0, 0.1, 0.0};
        List<SearchHit> hits = idx.search(query, 1, null, null);
        assertEquals(1, hits.size());
        assertEquals("d0", hits.get(0).id());

        IVector<Double> vec = idx.getVector("d0");
        assertNotNull(vec);
        assertEquals(1.0, vec.get(0), 1e-4);

        idx.close();
    }

    @Test
    void unsupportedMetricThrows() {
        float[][] data = {{1.0f, 0.0f}};
        String[] ids = {"0"};
        VecSearchOption opts = VecSearchOption.DEFAULT;

        IDisMetric<Float> manhattan = new IDisMetric<>() {
            @Override
            public MetricType type() { return MetricType.MANHATTAN; }
            @Override
            public String name() { return "manhattan"; }
            @Override
            public boolean isSimilarity() { return false; }
            @Override
            public double compute(IVector<Float> a, IVector<Float> b) {
                double sum = 0;
                for (int i = 0; i < a.length(); i++) {
                    sum += Math.abs(a.get(i) - b.get(i));
                }
                return sum;
            }
        };

        assertThrows(IllegalArgumentException.class, () ->
                new RereHnswFloatVecIdx(data, ids, manhattan, opts));
    }

    @Test
    void removeAndSearchStillWorks() {
        float[][] data = {
                {1.0f, 0.0f},
                {0.0f, 1.0f},
                {0.5f, 0.5f}
        };
        String[] ids = {"a", "b", "c"};
        VecSearchOption opts = VecSearchOption.DEFAULT;

        RereHnswFloatVecIdx idx = new RereHnswFloatVecIdx(
                data, ids, EuclideanMetric.FLOAT, opts);
        assertEquals(3, idx.size());

        // 删除 "a"
        assertTrue(idx.remove("a"));
        assertEquals(2, idx.size());
        assertFalse(idx.contains("a"));

        // 删除后搜索应正常工作
        float[] query = {1.0f, 0.0f};
        List<SearchHit> hits = idx.search(query, 1, java.util.List.of(), null);
        assertEquals(1, hits.size());
        assertEquals("c", hits.get(0).id()); // [0.5, 0.5] 比 [0, 1] 更接近 [1, 0]

        // 重复删除返回 false
        assertFalse(idx.remove("a"));

        idx.close();
    }

    @Test
    void removeEntryPointAndContinue() {
        float[][] data = {
                {1.0f, 0.0f},
                {0.0f, 1.0f}
        };
        String[] ids = {"a", "b"};
        VecSearchOption opts = VecSearchOption.DEFAULT;

        RereHnswFloatVecIdx idx = new RereHnswFloatVecIdx(
                data, ids, EuclideanMetric.FLOAT, opts);

        // 删除 entry point（通常是第一个插入的节点 "a"）
        assertTrue(idx.remove("a"));
        assertEquals(1, idx.size());

        // 剩余节点应仍可搜索
        float[] query = {0.0f, 1.0f};
        List<SearchHit> hits = idx.search(query, 1, java.util.List.of(), null);
        assertEquals(1, hits.size());
        assertEquals("b", hits.get(0).id());

        idx.close();
    }

    @Test
    void clearAndReuse() {
        float[][] data = {{1.0f, 0.0f}, {0.0f, 1.0f}};
        String[] ids = {"0", "1"};
        VecSearchOption opts = VecSearchOption.DEFAULT;

        RereHnswFloatVecIdx idx = new RereHnswFloatVecIdx(
                data, ids, EuclideanMetric.FLOAT, opts);
        assertEquals(2, idx.size());

        idx.clear();
        assertEquals(0, idx.size());
        assertFalse(idx.contains("0"));

        // add after clear
        idx.add("2", IVector.of(new float[]{0.5f, 0.5f}));
        assertEquals(1, idx.size());
        assertTrue(idx.contains("2"));

        idx.close();
    }

    @Test
    void largeDatasetRecallVsBruteForce() {
        int n = 500;
        int dim = 16;
        float[][] data = new float[n][dim];
        String[] ids = new String[n];
        java.util.Random rand = new java.util.Random(42);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                data[i][j] = rand.nextFloat();
            }
            ids[i] = String.valueOf(i);
        }

        VecSearchOption opts = VecSearchOption.DEFAULT;

        RereHnswFloatVecIdx hnsw = new RereHnswFloatVecIdx(
                data, ids, EuclideanMetric.FLOAT, opts);
        IFloatVecIdx brute = new com.yishape.lab.math.vecidx.impl.BruteForceFloatVecIdx(
                data, ids, EuclideanMetric.FLOAT);

        // 用 10 个随机 query 测试召回率
        int correct = 0;
        int total = 0;
        for (int q = 0; q < 10; q++) {
            float[] query = new float[dim];
            for (int j = 0; j < dim; j++) {
                query[j] = rand.nextFloat();
            }
            List<SearchHit> hnswHits = hnsw.search(query, 10, java.util.List.of(), null);
            List<SearchHit> bruteHits = brute.search(query, 10, java.util.List.of(), null);

            Set<String> bruteTop = new HashSet<>();
            for (SearchHit h : bruteHits) {
                bruteTop.add(h.id());
            }
            for (SearchHit h : hnswHits) {
                if (bruteTop.contains(h.id())) {
                    correct++;
                }
                total++;
            }
        }

        double recall = (double) correct / total;
        assertTrue(recall >= 0.80,
                "HNSW 召回率应不低于 80%，实际: " + recall);

        hnsw.close();
    }
}
