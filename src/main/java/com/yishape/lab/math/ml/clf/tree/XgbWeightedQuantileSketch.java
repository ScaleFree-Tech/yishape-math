package com.yishape.lab.math.ml.clf.tree;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Hessian加权直方图分箱边界构造 / Hessian-Weighted Histogram Bin Boundary Construction
 * <p>
 * 实现Hessian加权的直方图分箱边界构造，用于对齐梯度提升树常用的weighted quantile候选划分。
 * 当前实现：对{@code sampleIndices}指向的行，按特征值排序后合并相等取值，
 * 在累积Hessian质量上等间隔取{@code maxBin}个箱边界（离散加权分位点），适合中小规模稠密矩阵；
 * 与streaming sketch相比更简单且可作为数值基准。
 * </p>
 * <p>
 * Implements Hessian-weighted histogram bin boundary construction for weighted quantile
 * candidate split points used in gradient boosting trees. Current implementation: for rows
 * pointed to by {@code sampleIndices}, sorts by feature values and merges equal values,
 * then takes {@code maxBin} bin boundaries at equal intervals on cumulative Hessian mass
 * (discrete weighted quantiles), suitable for small to medium dense matrices;
 * simpler than streaming sketch and can serve as numerical benchmark.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public final class XgbWeightedQuantileSketch implements Serializable {

    private static final long serialVersionUID = 1L;

    private static final double EPS = 1e-15;

    private XgbWeightedQuantileSketch() {
    }

    /**
     * @param features       特征矩阵（全局行号）
     * @param hessians       与海森矩阵列对齐的一维向量（与建树所用的单行 Hessian 一致）
     * @param sampleIndices  参与本棵树加权分位的全局行索引（通常为子采样）
     * @param maxBin         箱数 B（≥2），输出每条边界长度为 {@code B + 1}
     * @return {@code edges[j][b]} 为第 {@code j} 列第 {@code b} 个箱左边界风格阈值链：
     *         样本划入 bin {@code k} 满足 {@code edges[k] < v <= edges[k+1]}（首尾稍作外延）
     */
    public static double[][] buildEdges(IMatrix features, IVector hessians, int[] sampleIndices, int maxBin) {
        int cols = features.cols();
        int b = Math.max(2, maxBin);
        double[][] edges = new double[cols][b + 1];
        for (int j = 0; j < cols; j++) {
            buildColumn(features, hessians, sampleIndices, j, b, edges[j]);
        }
        return edges;
    }

    private static void buildColumn(IMatrix features, IVector hessians, int[] rows, int col, int maxBin, double[] e) {
        int n = rows.length;
        if (n == 0) {
            Arrays.fill(e, 0.0);
            return;
        }

        Integer[] ord = new Integer[n];
        for (int i = 0; i < n; i++) {
            ord[i] = i;
        }
        Arrays.sort(ord, Comparator.comparingDouble(i -> features.get(rows[i], col)));

        List<Double> vx = new ArrayList<>();
        List<Double> hw = new ArrayList<>();
        for (int i = 0; i < n; ) {
            double val = features.get(rows[ord[i]], col);
            double sw = 0.0;
            int j = i;
            while (j < n && Double.compare(features.get(rows[ord[j]], col), val) == 0) {
                sw += Math.max(EPS, hessians.get(rows[ord[j]]));
                j++;
            }
            vx.add(val);
            hw.add(sw);
            i = j;
        }

        int m = vx.size();
        double vmin = vx.get(0);
        double vmax = vx.get(m - 1);
        double padLo = 1e-9 * (Math.abs(vmin) + 1.0);
        double padHi = 1e-9 * (Math.abs(vmax) + 1.0);

        if (m == 1 || vmax <= vmin + 1e-15) {
            uniformDegenerate(vmin, vmax, padLo, padHi, maxBin, e);
            return;
        }

        double[] cum = new double[m + 1];
        for (int i = 0; i < m; i++) {
            cum[i + 1] = cum[i] + hw.get(i);
        }
        double total = cum[m];
        if (total < EPS) {
            uniformDegenerate(vmin, vmax, padLo, padHi, maxBin, e);
            return;
        }

        e[0] = vmin - padLo;
        for (int k = 1; k < maxBin; k++) {
            double target = total * k / maxBin;
            int idx = 0;
            while (idx < m - 1 && cum[idx + 1] < target - EPS) {
                idx++;
            }
            e[k] = vx.get(idx);
        }
        e[maxBin] = vmax + padHi;
        monotonize(e, maxBin);
    }

    private static void uniformDegenerate(double vmin, double vmax, double padLo, double padHi, int maxBin, double[] e) {
        e[0] = vmin - padLo;
        double span = Math.max(vmax - vmin + padLo + padHi, padLo + padHi);
        for (int k = 1; k < maxBin; k++) {
            e[k] = e[0] + span * k / maxBin;
        }
        e[maxBin] = vmax + padHi;
        monotonize(e, maxBin);
    }

    private static void monotonize(double[] e, int maxBin) {
        for (int b = 1; b <= maxBin; b++) {
            if (e[b] <= e[b - 1]) {
                e[b] = e[b - 1] + 1e-12 * (Math.abs(e[b - 1]) + 1.0);
            }
        }
    }
}
