package model_zoo.yishape_ml_iris;

import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.ml.ML;
import com.yishape.lab.math.ml.dimreduce.RerePCA;
import com.yishape.lab.math.plot.Plots;
import model_zoo.yishape_common.ProjectDataPaths;

/**
 * 使用 {@code data/iris.csv}：逻辑回归多分类、分类指标、K 折交叉验证与 PCA 降维示例。
 */
public class YishapeMlIrisExample {

    public static void main(String[] args) throws Exception {
        System.out.println("======== YiShape-Math：机器学习（Iris） =========");

        String path = ProjectDataPaths.resolveDataCsv("iris.csv");
        var df = DataFrame.readCsv(path);
        System.out.println("数据形状（行, 列）: " + df.rows() + " x " + df.cols());
        System.out.println("列名: " + df.getColumnNames());

        var features = df.sliceColumn(0, -1).toMatrix();
        var labels = df.getColumn(-1).toStringArray();

        var cv = ML.kFoldCrossValidation(ML.logisticRegression(0.0, 0.0), features, labels, 5);
        System.out.println("5 折交叉验证:\n" + cv);

        var clf = ML.logisticRegression(0.0, 0.0);
        var fitResult = clf.fit(features, labels);
        System.out.println("训练结果摘要:\n" + fitResult);

        var metrics = ML.classificationMetrics(clf, features, labels);
        System.out.println("训练集分类指标:\n" + metrics);
        System.out.printf("准确率: %.4f%n", metrics.getAccuracy());

        var umap = ML.umapDimReducer();
        var reduced = umap.dimensionReduction(features, 2);
        Plots.scatter(reduced.getColumn(0), reduced.getColumn(1)).show();
        System.out.println("UMAP 降至 2 维后的矩阵形状: " + reduced.getRowNum() + " x " + reduced.getColNum());

        System.out.println("======== 示例结束（Iris） =========");
    }
}
