package com.yishape.lab.math.ml.dml.triplet;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.vecidx.SearchHit;
import com.yishape.lab.math.vecidx.VI;
import com.yishape.lab.math.vecidx.VecSearchOption;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import com.yishape.lab.math.vecidx.IDoubleVecIdx;

/**
 * 从带标签样本批量构造 {@link Triplet}，供对角 DML（{@code ddml}）等使用。
 *
 * <p>流程概览：按标签分组 → 在每类内在子空间上做近似最近邻（可选 HNSW 等，见
 * {@link com.yishape.lab.math.vecidx.VecSearchOption}）→ 与异类池组合锚点–同类–异类三元组；
 * 可与并行开关在样本量较大时启用流式并行。</p>
 *
 * <p><strong>精确近邻保证</strong>：默认使用 {@link VecSearchOption#DEFAULT}
 * 时，若底层选用 HNSW 等近似索引，则通过 oversampling + 精排（{@link com.yishape.lab.math.vecidx.IVecIdx#searchExact}，
 * {@code factor=10}）恢复精确 Top-1 近邻。DML 三元组构建推荐在
 * {@link VecSearchOption#DEFAULT} 索引上调用 {@code searchExact(..., factor=10)} 以获得更高召回率，
 * 而非直接使用 {@link VecSearchOption#EXACT}（会禁用 ANN 加速）。</p>
 *
 * <p>实现意图与参考 Julia 模块 {@code TripletModule.jl#build_triplets} 对齐（距离与权重语义以
 * {@link Triplet} 字段为准）。</p>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Schultz, M., &amp; Joachims, T. (2003). Learning a distance metric from relative comparisons.
 *       In <em>NeurIPS</em> 16.</li>
 *   <li>Weinberger, K. Q., &amp; Saul, L. K. (2009). Distance metric learning for large margin nearest
 *       neighbor classification. <em>JMLR</em>, 10, 207–244.（目标近邻与间隔思路。）</li>
 * </ul>
 */
@SuppressWarnings("rawtypes")
public final class TripletBuilder {

    /** 与参考实现一致：剔除「同类段 − 异类段」间隔最大的前若干条，抑制明显离群三元组。 */
    private static final int NOISE_TRIM = 5;
    private static final double ROU = Triplet.defaultRou();

    private static final int MIN_PARALLEL_CLASSES = 3;
    private static final int MIN_PARALLEL_ANCHORS = 96;

    private TripletBuilder() {
    }

    /**
     * 自特征矩阵与字符串标签构造三元组；使用默认近邻策略、单线程。
     *
     * @param features    行样本；元素类型可为 {@link Double} 等可转为 {@code double} 的类型
     * @param labels      与行等长的离散标签
     * @param maxTriplets 结果列表长度上界（蒸馏后截断）
     */
    public static List<Triplet> build(IMatrix features, String[] labels, int maxTriplets) {
        return build(features, labels, maxTriplets, false, VecSearchOption.DEFAULT);
    }

    /**
     * 同上，可指定是否在「类间」或「类内锚点数足够大」时并行。
     */
    public static List<Triplet> build(IMatrix features, String[] labels, int maxTriplets, boolean parallel) {
        return build(features, labels, maxTriplets, parallel, VecSearchOption.DEFAULT);
    }

    /**
     * 完整参数版本：近邻索引类型/HNSW 等由 {@link VecSearchOption} 控制。
     */
    public static List<Triplet> build(IMatrix features, String[] labels, int maxTriplets, boolean parallel,
            VecSearchOption nnOpts) {
        if (features == null || labels == null) {
            throw new IllegalArgumentException("features 与 labels 不能为 null");
        }
        Objects.requireNonNull(nnOpts, "nnOpts");
        int nRows = features.getRowNum();
        if (labels.length != nRows) {
            throw new IllegalArgumentException("labels 长度须等于样本数");
        }

        Map<String, List<Integer>> byLabel = new LinkedHashMap<>();
        for (int i = 0; i < nRows; i++) {
            String la = labels[i];
            byLabel.computeIfAbsent(la, k -> new ArrayList<>()).add(i);
        }

        double[][] x = materializeFeatures(features);

        boolean parallelAcrossClasses = parallel && byLabel.size() >= MIN_PARALLEL_CLASSES;

        VecSearchOption vnFinal = nnOpts;

        List<Triplet> triplets;
        if (parallelAcrossClasses) {
            triplets = byLabel.entrySet().parallelStream()
                    .flatMap(ent -> tripletsForClass(ent.getValue(), x, byLabel, ent.getKey(), false, vnFinal)
                            .stream())
                    .collect(Collectors.toCollection(ArrayList::new));
        } else {
            triplets = new ArrayList<>();
            for (Map.Entry<String, List<Integer>> ent : byLabel.entrySet()) {
                boolean anchorParallel = parallel && ent.getValue().size() >= MIN_PARALLEL_ANCHORS;
                triplets.addAll(tripletsForClass(ent.getValue(), x, byLabel, ent.getKey(), anchorParallel,
                        vnFinal));
            }
        }

        // 先去噪（间隔最大者），再按权重保留至多 maxTriplets 条
        if (triplets.size() > NOISE_TRIM) {
            triplets.sort(Comparator.comparingDouble((Triplet t) -> t.ijDis() - t.jkDis()).reversed());
            triplets = new ArrayList<>(triplets.subList(NOISE_TRIM, triplets.size()));
        }

        if (triplets.size() > maxTriplets) {
            triplets.sort(Comparator.comparingDouble(Triplet::weight).reversed());
            triplets = new ArrayList<>(triplets.subList(0, maxTriplets));
        }

        return triplets;
    }

    /**
     * 标签为 {@link IVector} 时的便捷重载（经 {@link DmlArrays#stringLabels} 转换）。
     */
    public static List<Triplet> build(IMatrix features, IVector labels, int maxTriplets) {
        return build(features, labels, maxTriplets, false, VecSearchOption.DEFAULT);
    }

    /** {@link #build(IMatrix, String[], int, boolean)} 的向量标签版。 */
    public static List<Triplet> build(IMatrix features, IVector labels, int maxTriplets, boolean parallel) {
        return build(features, labels, maxTriplets, parallel, VecSearchOption.DEFAULT);
    }

    /** {@link #build(IMatrix, String[], int, boolean, VectorSearchOptions)} 的向量标签版。 */
    public static List<Triplet> build(IMatrix features, IVector labels, int maxTriplets, boolean parallel,
            VecSearchOption nnOpts) {
        return build(features, DmlArrays.stringLabels(labels), maxTriplets, parallel, nnOpts);
    }

    private static double[][] materializeFeatures(IMatrix features) {
        Objects.requireNonNull(features, "features");
        return features.toDoubleArray();
    }

    private static List<Triplet> tripletsForClass(List<Integer> idxClass, double[][] x,
            Map<String, List<Integer>> byLabel, String selfLabelKey, boolean anchorParallel,
            VecSearchOption nnOpts) {
        int sc = idxClass.size();
        if (sc < 2) {
            return new ArrayList<>(0);
        }
        int[] otherRows = otherClassRowIndices(byLabel, selfLabelKey);
        if (otherRows.length == 0) {
            return new ArrayList<>(0);
        }

        int[] sameRows = listToRowIndexArray(idxClass);
        String[] sameIds = intArrayToStringIds(sameRows);
        String[] otherIds = intArrayToStringIds(otherRows);
        IDoubleVecIdx sameNn =
                VI.buildDoubleSubset(x, sameIds, nnOpts);
        IDoubleVecIdx otherNn = VI.buildDoubleSubset(x, otherIds, nnOpts);

        if (!anchorParallel) {
            ArrayList<Triplet> acc = new ArrayList<>(Math.max(8, sc));
            for (int pos = 0; pos < sc; pos++) {
                Triplet t = tripletAt(idxClass.get(pos), x, sameNn, otherNn);
                if (t != null) {
                    acc.add(t);
                }
            }
            return acc;
        }

        return IntStream.range(0, sc).parallel()
                .mapToObj(pos -> tripletAt(idxClass.get(pos), x, sameNn, otherNn))
                .filter(Objects::nonNull)
                .collect(Collectors.toCollection(ArrayList::new));
    }

    private static int[] listToRowIndexArray(List<Integer> idxClass) {
        int n = idxClass.size();
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = idxClass.get(i);
        }
        return a;
    }

    private static int[] otherClassRowIndices(Map<String, List<Integer>> byLabel, String excludeKey) {
        int total = 0;
        for (Map.Entry<String, List<Integer>> en : byLabel.entrySet()) {
            if (en.getKey().equals(excludeKey)) {
                continue;
            }
            total += en.getValue().size();
        }
        int[] out = new int[total];
        int p = 0;
        for (Map.Entry<String, List<Integer>> en : byLabel.entrySet()) {
            if (en.getKey().equals(excludeKey)) {
                continue;
            }
            for (int ix : en.getValue()) {
                out[p++] = ix;
            }
        }
        return out;
    }

    private static String[] intArrayToStringIds(int[] rows) {
        String[] ids = new String[rows.length];
        for (int i = 0; i < rows.length; i++) {
            ids[i] = String.valueOf(rows[i]);
        }
        return ids;
    }

    private static Triplet tripletAt(int rowJ, double[][] x,
            IDoubleVecIdx sameNn, IDoubleVecIdx otherNn) {
        try {
            double[] xj = x[rowJ];
            List<SearchHit> hSame = sameNn.searchExact(xj, 1, 10,
                    Set.of(String.valueOf(rowJ)), null);
            if (hSame.isEmpty()) {
                return null;
            }
            int rowI = Integer.parseInt(hSame.get(0).id());
            double ijDis = hSame.get(0).distance();

            List<SearchHit> hOther = otherNn.searchExact(xj, 1, 10, Set.of(), null);
            if (hOther.isEmpty()) {
                return null;
            }
            int rowOther = Integer.parseInt(hOther.get(0).id());
            double jkDis = hOther.get(0).distance();

            Triplet t = new Triplet(
                    Linalg.vector(x[rowI]),
                    Linalg.vector(x[rowJ]),
                    Linalg.vector(x[rowOther]),
                    ijDis,
                    jkDis,
                    1.0);
            t.setWeight(Triplet.computeWeight(t.ijDis(), t.jkDis(), ROU));
            return t;
        } catch (RuntimeException ignored) {
            return null;
        }
    }
}
