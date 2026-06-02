package com.yishape.lab.math.ml.dml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * DML 共享的<strong>数组整理</strong>：特征矩阵导出为原始行、标签编码为连续类别索引、
 * 向量标签与 {@code String[]} 流水线之间的桥接。
 *
 * <p>编码规则：{@link #classIndices(String[])} 按标签<strong>首次出现顺序</strong>分配
 * {@code 0 … C−1}，保证可重复运行下映射稳定（基于 {@link LinkedHashMap} 的插入顺序）。</p>
 *
 * @see ISupervisedDml
 * @see DmlMetric
 */
public final class DmlArrays {

    private DmlArrays() {
    }

    /**
     * 将 {@link IMatrix} 行拷贝为原生 {@code double[][]}，便于仍使用数组内核的算法。
     *
     * @param features 元素类型为 {@link Double} 的矩阵，行为样本
     * @return 新分配的二维数组，{@code result[i][j] = features.get(i,j)}
     */
    public static double[][] featureRows(IMatrix<Double> features) {
        Objects.requireNonNull(features, "features");
        return features.toDoubleArray();
    }

    /**
     * 逐元素字符串化，供依赖 {@code String[]} 标签 API 的路径使用（不分配类别编号）。
     *
     * @param labels 与样本行对齐的向量；元素通过 {@link String#valueOf(Object)} 转成标签串
     * @return 与 {@code labels} 等长的新数组
     */
    public static String[] stringLabels(IVector<?> labels) {
        Objects.requireNonNull(labels, "labels");
        String[] ls = new String[labels.length()];
        for (int i = 0; i < ls.length; i++) {
            ls[i] = String.valueOf(labels.get(i));
        }
        return ls;
    }

    /**
     * 将离散标签映射为稠密类别索引 {@code 0 … C−1}（按首次出现顺序）。
     *
     * @param labels 训练标签，{@code null} 禁止
     * @return 与 {@code labels} 等长的整型数组
     */
    public static int[] classIndices(String[] labels) {
        Objects.requireNonNull(labels, "labels");
        Map<String, Integer> map = new LinkedHashMap<>();
        for (String s : labels) {
            map.putIfAbsent(s, map.size());
        }
        int[] y = new int[labels.length];
        for (int i = 0; i < labels.length; i++) {
            y[i] = map.get(labels[i]);
        }
        return y;
    }

    /**
     * 先 {@link #stringLabels} 再 {@link #classIndices(String[])}，使向量标签与字符串标签路径一致。
     *
     * @param labels 任意元素类型（经字符串化后参与稳定映射）
     * @return 类别索引数组
     */
    public static int[] classIndices(IVector<?> labels) {
        return classIndices(stringLabels(labels));
    }
}
