package model_zoo.yishape_ml_boston;

import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.ml.ML;
import model_zoo.yishape_common.ProjectDataPaths;

/**
 * 使用 {@code data/boston_housing.csv}：以全部特征对房价 {@code y} 做带 L2 正则的线性回归，并报告训练集 RMSE。
 */
public class YishapeMlBostonExample {

    public static void main(String[] args) throws Exception {
        System.out.println("======== YiShape-Math：机器学习（Boston 回归） =========");

        String path = ProjectDataPaths.resolveDataCsv("boston_housing.csv");
        var df = DataFrame.readCsv(path);
        int n = df.rows();
        var x = df.sliceColumn(0, -1).toMatrix();//支持负索引切片，-1表示最后一列(不包含，同numpy)
        var y = df.getColumn(-1).toVec();

        var lr = ML.linearRegression(0.0, 0.1);
        var regResult = lr.fit(x, y);
        System.out.println("优化损失（带正则）: " + regResult.getLoss());
        System.out.println("偏置项: " + regResult.getBias());
        System.out.println("特征权重维度: " + regResult.getWeights());

        System.out.printf("训练集 RMSE: %.4f%n", regResult.getRmse());
        System.out.printf("训练集 R2: %.4f%n", regResult.getR2Score());

        System.out.println("======== 示例结束（Boston） =========");
    }


}
