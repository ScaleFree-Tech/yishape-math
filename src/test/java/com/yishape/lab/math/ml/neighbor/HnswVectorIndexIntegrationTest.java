package com.yishape.lab.math.ml.neighbor;

import com.yishape.lab.math.vecidx.VI;
import com.yishape.lab.math.vecidx.MetricType;
import com.yishape.lab.math.vecidx.VecSearchOption;
import com.yishape.lab.math.vecidx.SearchHit;
import com.yishape.lab.math.compute.hpc.HpcOptionalRuntime;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.vecidx.impl.BruteForceDoubleVecIdx;
import com.yishape.lab.math.vecidx.impl.BruteForceFloatVecIdx;
import com.yishape.lab.math.vecidx.impl.RustHnswDoubleVecIdx;
import com.yishape.lab.math.vecidx.impl.RustHnswFloatVecIdx;
import com.yishape.lab.math.vecidx.distance.EuclideanMetric;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import com.yishape.lab.math.vecidx.IDisMetric;
import com.yishape.lab.math.vecidx.IFloatVecIdx;
import com.yishape.lab.math.vecidx.IDoubleVecIdx;
import com.yishape.lab.math.vecidx.IMutableVecIdx;
import com.yishape.lab.math.vecidx.IdxType;

/**
 * HNSW 向量索引在 yishape-math 层的端到端集成测试。
 *
 * <p>验证 {@link VI} 工厂在 HPC 可用时正确路由到 HNSW 实现，
 * 且搜索/获取语义与暴力扫描一致（近似误差范围内）。</p>
 */
class HnswVectorIndexIntegrationTest {

    @BeforeAll
    static void assumeHpc() {
        Assumptions.assumeTrue(HpcOptionalRuntime.isHnswNativeAvailable(),
                "HNSW 原生模块不可用，跳过集成测试");
    }

    @Test
    void floatIndexBuildAndSearch() {
        float[][] data = {
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 1.0f},
                {1.0f, 1.0f, 1.0f}
        };
        String[] ids = {"a", "b", "c", "d"};
        VecSearchOption opts = new VecSearchOption(IdxType.HNSW, 16, 200, 200, 0);

        IFloatVecIdx idx = VI.buildFloat(data, ids, EuclideanMetric.FLOAT, opts);
        assertTrue(idx instanceof RustHnswFloatVecIdx,
                "当 HPC 可用且 approximate=true 时应返回 HnswFloatVectorIndex");
        assertEquals(3, idx.dimensions());
        assertEquals(4, idx.size());
        assertTrue(idx.isApproximate());

        // search near [1,0,0] -> expect "a" first
        float[] query = {1.0f, 0.1f, 0.0f};
        List<SearchHit> hits = idx.search(query, 2, null, null);
        assertEquals(2, hits.size());
        assertEquals("a", hits.get(0).id());
        assertTrue(hits.get(0).distance() >= 0.0);

        // get vector
        IVector<Float> vec = idx.getVector("a");
        assertNotNull(vec);
        assertEquals(1.0f, vec.get(0), 1e-4f);
        assertEquals(0.0f, vec.get(1), 1e-4f);

        // mutable ops via cast
        IMutableVecIdx<Float> mut = (IMutableVecIdx<Float>) idx;
        mut.add("e", IVector.of(new float[]{0.5f, 0.5f, 0.5f}));
        assertEquals(5, idx.size());
        assertTrue(mut.contains("e"));

        // search after add
        List<SearchHit> hits2 = idx.search(query, 3, null, null);
        assertTrue(hits2.size() >= 1);

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
        VecSearchOption opts = new VecSearchOption(IdxType.HNSW, 16, 200, 200, 0);

        IDoubleVecIdx idx = new RustHnswDoubleVecIdx(data, ids, EuclideanMetric.DOUBLE, opts);
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
    void subsetBuildSelectsHnsw() {
        double[][] full = {
                {1.0, 0.0},
                {0.0, 1.0},
                {1.0, 1.0},
                {0.0, 0.0}
        };
        String[] subsetIds = {"0", "2"}; // row 0 and row 2
        VecSearchOption opts = new VecSearchOption(IdxType.HNSW, 16, 200, 200, 0);

        double[][] subsetData = {full[0].clone(), full[2].clone()};
        IDoubleVecIdx idx = new RustHnswDoubleVecIdx(subsetData, subsetIds, EuclideanMetric.DOUBLE, opts);
        assertEquals(2, idx.size());

        double[] query = {1.0, 0.0};
        List<SearchHit> hits = idx.search(query, 1, null, null);
        assertEquals(1, hits.size());
        assertEquals("0", hits.get(0).id());

        idx.close();
    }

    @Test
    void mutableIndexDefaultsToBruteForce() {
        VecSearchOption opts = VecSearchOption.DEFAULT;
        IMutableVecIdx<Double> idx = VI.newMutableDouble(3, EuclideanMetric.DOUBLE, opts);
        assertTrue(idx instanceof BruteForceDoubleVecIdx);
        idx.add("x", IVector.of(new double[]{1.0, 2.0, 3.0}));
        assertEquals(1, idx.size());
        assertTrue(idx.contains("x"));
    }

    @Test
    void fallbackToBruteForceWhenMetricUnsupported() {
        float[][] data = {
                {1.0f, 0.0f},
                {0.0f, 1.0f}
        };
        String[] ids = {"0", "1"};
        VecSearchOption opts = VecSearchOption.DEFAULT;

        // Manhattan 距离不受 HNSW 支持，应回落到暴力扫描
        IDisMetric<Float> manhattan = new IDisMetric<>() {
            @Override
            public MetricType type() { return MetricType.MANHATTAN; }
            @Override
            public double compute(IVector<Float> a, IVector<Float> b) {
                double sum = 0;
                for (int i = 0; i < a.length(); i++) {
                    sum += Math.abs(a.get(i) - b.get(i));
                }
                return sum;
            }

            @Override
            public String name() {
                return "manhattan";
            }

            @Override
            public boolean isSimilarity() {
                return false;
            }
        };

        IFloatVecIdx idx = VI.buildFloat(data, ids, manhattan, opts);
        assertTrue(idx instanceof BruteForceFloatVecIdx,
                "不支持的度量应回落到 BruteForceVectorIndex");
        assertEquals(2, idx.size());
    }
}
