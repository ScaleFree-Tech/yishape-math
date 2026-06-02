package model_zoo.yishape_stats_wine;

import com.yishape.lab.math.data.Column;
import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.stats.Stats;
import com.yishape.lab.math.stats.anova.ANOVAResult;
import model_zoo.yishape_common.ProjectDataPaths;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

/**
 * 使用 {@code data/wine.csv}：正态分布、皮尔逊相关、均值置信区间、单样本 t 检验与单因素方差分析（按酒类标签分组）。
 */
public class YishapeStatsWineExample {

    public static void main(String[] args) throws Exception {
        System.out.println("======== YiShape-Math：统计学（Wine） =========");

        String path = ProjectDataPaths.resolveDataCsv("wine.csv");
        var df = DataFrame.readCsv(path);

        var x1 = df.getColumnByName("x1");
        var x2 = df.getColumnByName("x2");
        
        var vx1 = x1.toVec();
        var vx2 = x2.toVec();
        System.out.printf("x1 与 x2 的皮尔逊相关系数: %.4f%n", Stats.corr(vx1, vx2));
        System.out.printf("x1 与 x2 的协方差: %.6f%n", Stats.cov(vx1, vx2));

        var n01 = Stats.norm(0, 1);
        System.out.printf("N(0,1) 在 1.96 处的 PDF=%.4f, CDF=%.4f%n", n01.pdf(1.96), n01.cdf(1.96));

        var ci = Stats.estimator.estimateMeanIntevalWithT(vx1, 0.95);
        System.out.printf("x1 的 95%% 均值置信区间 (t): [%.4f, %.4f]%n", ci.getFirst(), ci.getSecond());

        double x1Mean = vx1.mean();
        var ttest = Stats.tester.testMeanEqualWithT(x1Mean, vx1, 0.95);
        System.out.printf("单样本 t 检验（H0: μ = 样本均值 %.6f）: p=%.4g, 通过? %s%n",
                x1Mean, ttest.p, ttest.pass);

        IVector g1 = groupFeatureByLabel(df, "x1", 1);
        IVector g2 = groupFeatureByLabel(df, "x1", 2);
        IVector g3 = groupFeatureByLabel(df, "x1", 3);
        ANOVAResult anova = Stats.anova.performOneWayANOVA(g1, g2, g3);
        printAnova(anova, g1.length(), g2.length(), g3.length());

        System.out.println("======== 示例结束（Wine 统计） =========");
    }

    private static IDoubleVector groupFeatureByLabel(DataFrame df, String featureName, int labelValue) {
        Column lab = df.getColumnByName("label");
        Column feat = df.getColumnByName(featureName);
        List<Double> vals = new ArrayList<>();
        for (int i = 0; i < df.rows(); i++) {
            int lv = ((Number) lab.getData().get(i)).intValue();
            if (lv == labelValue) {
                vals.add(((Number) feat.getData().get(i)).doubleValue());
            }
        }
        return IDoubleVector.of(vals.stream().mapToDouble(Double::doubleValue).toArray());
    }

    private static void printAnova(ANOVAResult r, int n1, int n2, int n3) {
        try {
            Field fField = ANOVAResult.class.getDeclaredField("fStatistic");
            fField.setAccessible(true);
            double fStat = fField.getDouble(r);
            int k = 3;
            int n = n1 + n2 + n3;
            var fDist = Stats.f(k - 1, n - k);
            double p = 1.0 - fDist.cdf(fStat);
            if (p < 0) {
                p = 0.0;
            }
            String pStr = (p == 0.0 || p < 1e-12) ? "< 1e-12（F 很大时数值下溢）" : String.format("%.6e", p);
            System.out.printf("单因素 ANOVA（x1 按标签 1/2/3）: F=%.4f, p=%s%n", fStat, pStr);
        } catch (ReflectiveOperationException e) {
            System.out.println("单因素 ANOVA 已执行（无法读取 F 统计量时略过数值输出）。");
        }
    }
}
