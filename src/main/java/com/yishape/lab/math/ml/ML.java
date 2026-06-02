package com.yishape.lab.math.ml;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.ml.clf.ensemble.EnsembleClassifier;

/**
 * 算法工具类，提供各种机器学习算法的统一入口。
 *
 * 该类通过静态 wrapper 字段组织算法工厂，主要入口如下：
 * <ul>
 * <li>分类算法：{@link #clf} —
 * {@link ClfWrapper#logisticRegression()}、{@link ClfWrapper#randomForest()}、
 * {@link ClfWrapper#decisionTree()}、{@link ClfWrapper#linearSvm()}、
 * {@link ClfWrapper#xGboost()}、{@link ClfWrapper#kNN(int)}、
 * {@link ClfWrapper#ensembleClassifier(EnsembleClassifier.EnsembleStrategy, long)} 等</li>
 * <li>回归算法：{@link #reg} —
 * {@link RegWrapper#linear()} 及带正则化的快捷构造</li>
 * <li>降维：{@link #dr} —
 * {@link DrWrapper#pca(int)}、{@link DrWrapper#svd(int)}、
 * {@link DrWrapper#tsne(int)}、{@link DrWrapper#umap(int)} 等</li>
 * <li>聚类：{@link #clu} —
 * {@link CluWrapper#kMeans(int)}、{@link CluWrapper#gmm(int)} 等</li>
 * <li>距离度量学习：{@link #dml} —
 * {@link DmlWrapper#diagDml()} 等（含自主研发 DDML 算法）</li>
 * <li>数据预处理：{@link #preproc} —
 * {@link PreprocWrapper#standardScaler()}、{@link PreprocWrapper#minMaxScaler()} 等</li>
 * </ul>
 *
 * <p>
 * 使用示例：</p>
 * <pre>{@code
 * // 分类
 * IClassifier lr = ML.clf.logisticRegression();
 * IClassifier rf = ML.clf.randomForest();
 *
 * // 回归
 * IRegression reg = ML.reg.linear(0.0, 0.1);
 *
 * // 降维
 * IMatrix Z = ML.dr.pca(2).fitTransform(X);
 *
 * // 聚类
 * var clusters = ML.clu.kMeans(3).fit(X).getResult();
 *
 * // 距离度量学习
 * var metricLearner = ML.dml.diagDml().fit(X, y).getResult();
 * }</pre>
 *
 * @author lteb2
 * @since 1.0
 */
public class ML {

    private static final YishapeLogger log = YishapeLogger.getLogger(ML.class);

    public static ClfWrapper clf = new ClfWrapper();

    public static DrWrapper dr = new DrWrapper();

    public static RegWrapper reg = new RegWrapper();

    public static CluWrapper clu = new CluWrapper();

    /** 监督距离度量学习：工厂见 {@link DmlWrapper}（仅返回 {@link com.yishape.lab.math.ml.dml.ISupervisedDml}，设计约定见该类注释）。 */
    public static DmlWrapper dml = new DmlWrapper();
    /**
     * 数据预处理
     */
    public static PreprocWrapper preproc = new PreprocWrapper();

}
