package com.yishape.lab.math.ml.dml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import java.util.Objects;

/**
 * 监督型距离度量学习的统一<strong>拟合契约</strong>：实现者保存算法族与超参数，
 * {@link #fit} 仅消费特征与标签，产出 {@link DmlMetric}。
 *
 * <p><strong>包结构</strong>：具体算法位于 {@code dml} 下各子包（如 {@code fisher}、{@code ddml}、
 * {@code triplet}）；共性类型为本包根目录中的 {@link DmlMetric}、{@link MetricForm}、
 * {@link DmlArrays} 及本接口。</p>
 *
 * <h2>推荐使用顺序</h2>
 * <ol>
 *   <li><strong>首选</strong>：{@link com.yishape.lab.math.ml.ML#dml}（{@link com.yishape.lab.math.ml.DmlWrapper}）
 *       上的无参 / 重载工厂——与 {@link com.yishape.lab.math.ml.ClfWrapper}、{@link com.yishape.lab.math.ml.DrWrapper}
 *       并列，返回类型仅为本接口。</li>
 *   <li><strong>深调参与实验</strong>：{@code new 具体算法类().setXxx(...).fit(...)}（仍在 {@code dml} 子包内，可与上类互相替换引用）。</li>
 *   <li><strong>静态 {@code fit}</strong>：各实现类上的 {@code static fit} 仅保留为<strong>一行脚本</strong>便利，新应用代码不宜作为主体 API。</li>
 * </ol>
 * <p>详细分层约定见 {@link com.yishape.lab.math.ml.DmlWrapper} 类注释。</p>
 *
 * <h2>标签约定</h2>
 * <p>{@link IVector} 标签经 {@link DmlArrays#stringLabels} / {@link DmlArrays#classIndices}
 * 转为类别索引；字符串标签经稳定映射编码为 {@code 0..C−1}。</p>
 *
 * @see DmlMetric
 * @see DmlArrays
 */
public interface ISupervisedDml {

    /**
     * 从数值/装箱等任意可 {@link String#valueOf} 的标签向量拟合度量。
     *
     * @param features 行样本，{@link IMatrix#getColNum()} 为输入维
     * @param labels   与行同长的训练标签
     * @return 非 null 的拟合结果
     */
    public default DmlMetric fit(IMatrix<Double> features, IVector<?> labels) {
        Objects.requireNonNull(labels, "labels");
        return fit(features, DmlArrays.stringLabels(labels));
    }

    /**
     * 从离散标签（字符串键）拟合度量。
     *
     * @param features 行样本
     * @param labels   与行同长；编码顺序为首次出现顺序
     * @return 非 null 的拟合结果
     */
    DmlMetric fit(IMatrix<Double> features, String[] labels);
}
