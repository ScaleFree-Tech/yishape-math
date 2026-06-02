package com.yishape.lab.math.ml;

import com.yishape.lab.math.ml.preprocessing.*;
import com.yishape.lab.math.ml.preprocessing.RereNormalizer.Norm;
import com.yishape.lab.math.ml.preprocessing.RerePowerTransformer.Method;
import com.yishape.lab.math.ml.preprocessing.RereQuantileTransformer.OutputDistribution;
import com.yishape.lab.math.ml.preprocessing.RereBucketizer.Strategy;

/**
 * 数据预处理（Preprocessing）工厂：作为 {@link ML#preproc} 的<strong>唯一推荐入口</strong>。
 *
 * <h2>设计约定（与分类 / 距离度量学习 Wrapper 一致）</h2>
 * <ol>
 *   <li><strong>对外类型</strong>：所有工厂方法仅声明返回 {@link ITransform}（或 {@link IRereScaler}），不暴露具体实现类名。</li>
 *   <li><strong>默认与重载</strong>：无参版本 = 库内审慎默认；<strong>对结果影响大、且在教程里常被调的内容</strong>通过方法重载显式传入
 *       （与 {@code logisticRegression(l1,l2)}、{@code kNN(k)} 同级）。</li>
 *   <li><strong>何时只用本类</strong>：业务代码、流水线、需以多态持有「某一种预处理器」时——一律
 *       {@code ML.preproc.xxx(...)}。</li>
 *   <li><strong>何时 {@code new} 实现类</strong>：仅当需要<strong>尚未在本 Wrapper 中设重载或文档未承诺覆盖</strong>的调参项时，
 *       在 {@code com.yishape.lab.math.ml.preprocessing} 子包中
 *       {@code new FooPreprocessor().setRareKnob(...).fit(...)}；
 *       实现类仍实现 {@link ITransform}，可与门面产出类型互换。</li>
 *   <li><strong>统一接口</strong>：所有预处理器均实现 {@link ITransform}（支持 {@code fit / transform / fitTransform}），
 *       其中带缩放功能的还实现 {@link IRereScaler}（额外支持 {@code inverseTransform}）。</li>
 * </ol>
 *
 * @see ML#preproc
 * @see ITransform
 * @see IRereScaler
 * @see com.yishape.lab.math.ml.preprocessing
 */
public final class PreprocWrapper {

    // ==================== Scaler 系列 ====================

    /**
     * Z-Score 标准化（均值=0，标准差=1）。
     *
     * @return 可逆预处理器
     * @see RereStandardScaler
     */
    public IRereScaler<Double> standardScaler() {
        return new RereStandardScaler();
    }

    /**
     * Z-Score 标准化，可选是否除以标准差。
     *
     * @param withStd 为 true 时除以标准差（标准 Z-Score）；为 false 时仅中心化（减均值）
     * @return 可逆预处理器
     */
    public IRereScaler<Double> standardScaler(boolean withStd) {
        return new RereStandardScaler(withStd);
    }

    /**
     * Min-Max 缩放，默认目标范围 [0, 1]。
     *
     * @return 可逆预处理器
     * @see RereMinMaxScaler
     */
    public IRereScaler<Double> minMaxScaler() {
        return new RereMinMaxScaler();
    }

    /**
     * Min-Max 缩放，指定目标范围。
     *
     * @param targetMin 目标最小值
     * @param targetMax 目标最大值
     * @return 可逆预处理器
     */
    public IRereScaler<Double> minMaxScaler(double targetMin, double targetMax) {
        return new RereMinMaxScaler(targetMin, targetMax);
    }

    /**
     * 鲁棒缩放，使用中位数和四分位距（IQR），对异常值具有较强的鲁棒性。
     *
     * @return 可逆预处理器
     * @see RereRobustScaler
     */
    public IRereScaler<Double> robustScaler() {
        return new RereRobustScaler();
    }

    /**
     * 按最大绝对值缩放，使特征值位于 [-1, 1] 区间，常用于稀疏数据。
     *
     * @return 可逆预处理器
     * @see RereMaxAbsScaler
     */
    public IRereScaler<Double> maxAbsScaler() {
        return new RereMaxAbsScaler();
    }

    // ==================== Normalizer ====================

    /**
     * 逐样本（行）归一化，默认 L2 范数。
     *
     * @return 预处理器
     * @see RereNormalizer
     */
    public ITransform<Double> normalizer() {
        return new RereNormalizer();
    }

    /**
     * 逐样本（行）归一化，指定范数类型。
     *
     * @param norm 范数类型：{@code L1}（元素绝对值之和为1）、{@code L2}（欧几里得范数为1）、{@code MAX}（最大绝对值为1）
     * @return 预处理器
     */
    public ITransform<Double> normalizer(Norm norm) {
        return new RereNormalizer(norm);
    }

    // ==================== PolynomialFeatures ====================

    /**
     * 多项式特征生成，默认 degree=2。
     *
     * @return 预处理器
     * @see RerePolynomialFeatures
     */
    public ITransform<Double> polynomialFeatures() {
        return new RerePolynomialFeatures();
    }

    /**
     * 多项式特征生成，指定多项式阶数。
     *
     * @param degree 多项式阶数（须 >= 1）
     * @return 预处理器
     */
    public ITransform<Double> polynomialFeatures(int degree) {
        return new RerePolynomialFeatures(degree);
    }

    /**
     * 多项式特征生成，完整配置。
     *
     * @param degree          多项式阶数（须 >= 1）
     * @param includeBias     是否包含偏置项（常数列 1）
     * @param interactionOnly  是否仅生成交互项（不生成高阶项）
     * @return 预处理器
     */
    public ITransform<Double> polynomialFeatures(int degree, boolean includeBias, boolean interactionOnly) {
        return new RerePolynomialFeatures(degree, includeBias, interactionOnly);
    }

    // ==================== Binarizer ====================

    /**
     * 阈值二值化，默认阈值 0.0。
     *
     * @return 预处理器
     * @see RereBinarizer
     */
    public ITransform<Double> binarizer() {
        return new RereBinarizer();
    }

    /**
     * 阈值二值化，指定阈值。
     *
     * @param threshold 二值化阈值：{@code x >= threshold} → 1，否则 0
     * @return 预处理器
     */
    public ITransform<Double> binarizer(double threshold) {
        return new RereBinarizer(threshold);
    }

    // ==================== OneHotEncoder ====================

    /**
     * 独热编码，将分类特征编码为独热格式。
     *
     * @return 预处理器
     * @see RereOneHotEncoder
     */
    public ITransform<Double> oneHotEncoder() {
        return new RereOneHotEncoder();
    }

    // ==================== LabelBinarizer ====================

    /**
     * 标签二值化，将标签列表编码为二值矩阵。
     *
     * @return 预处理器
     * @see RereLabelBinarizer
     */
    public ITransform<Double> labelBinarizer() {
        return new RereLabelBinarizer();
    }

    // ==================== PowerTransformer ====================

    /**
     * 幂变换，默认 Yeo-Johnson 方法（支持正负数据），使数据趋向正态分布。
     *
     * @return 预处理器
     * @see RerePowerTransformer
     */
    public ITransform<Double> powerTransformer() {
        return new RerePowerTransformer();
    }

    /**
     * 幂变换，指定方法。
     *
     * @param method {@code YEO_JOHNSON}（支持任意值）或 {@code BOX_COX}（仅支持正值）
     * @return 预处理器
     */
    public ITransform<Double> powerTransformer(Method method) {
        return new RerePowerTransformer(method);
    }

    // ==================== QuantileTransformer ====================

    /**
     * 分位数变换，默认均匀分布，1000 分位数。
     *
     * @return 预处理器
     * @see RereQuantileTransformer
     */
    public ITransform<Double> quantileTransformer() {
        return new RereQuantileTransformer();
    }

    /**
     * 分位数变换，指定分位数数量。
     *
     * @param nQuantiles 分位数数量（须 > 0）
     * @return 预处理器
     */
    public ITransform<Double> quantileTransformer(int nQuantiles) {
        return new RereQuantileTransformer(nQuantiles);
    }

    /**
     * 分位数变换，指定输出分布类型。
     *
     * @param outputDistribution {@code UNIFORM}（均匀分布）或 {@code NORMAL}（正态分布）
     * @return 预处理器
     */
    public ITransform<Double> quantileTransformer(OutputDistribution outputDistribution) {
        return new RereQuantileTransformer(outputDistribution);
    }

    /**
     * 分位数变换，完整配置。
     *
     * @param nQuantiles        分位数数量（须 > 0）
     * @param outputDistribution 输出分布类型
     * @return 预处理器
     */
    public ITransform<Double> quantileTransformer(int nQuantiles, OutputDistribution outputDistribution) {
        return new RereQuantileTransformer(outputDistribution, nQuantiles);
    }

    // ==================== KernelCenterer ====================

    /**
     * 核矩阵中心化，对核矩阵（Gram matrix）进行中心化处理。
     *
     * @return 预处理器
     * @see RereKernelCenterer
     */
    public ITransform<Double> kernelCenterer() {
        return new RereKernelCenterer();
    }

    // ==================== Bucketizer ====================

    /**
     * 分箱（离散化），默认等宽分箱，5 个分箱。
     *
     * @return 预处理器
     * @see RereBucketizer
     */
    public ITransform<Double> bucketizer() {
        return new RereBucketizer();
    }

    /**
     * 分箱（离散化），指定分箱数量，默认等宽分箱。
     *
     * @param nBins 分箱数量（须 > 0）
     * @return 预处理器
     */
    public ITransform<Double> bucketizer(int nBins) {
        return new RereBucketizer(nBins);
    }

    /**
     * 分箱（离散化），完整配置。
     *
     * @param strategy 分箱策略：{@code FIXED_WIDTH}（等宽）或 {@code QUANTILE}（等频）
     * @param nBins   分箱数量（须 > 0）
     * @return 预处理器
     */
    public ITransform<Double> bucketizer(Strategy strategy, int nBins) {
        return new RereBucketizer(strategy, nBins);
    }

}
