package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.ml.cls.knn.RereKnn;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;
import com.yishape.lab.math.ml.metric.CrossValidation;

/**
 *
 * @author lteb2
 */
public class LRTestKnn {

    public static void main(String args[]) {

//        String path = "C:\\Users\\lteb2\\Downloads\\d9c2cb80-3944-4f82-b884-93cad3e586fc.csv";
        String path = "G:\\电子科技大学-工作\\商务统计_2025\\data\\iris.csv";
        try {
            var df = DataFrame.readCsv(path);
            var feature = df.sliceColumn(1, -1).toMatrix();
            var labels = df.get(df.getColumnCount()-1).toStringArray();
            System.out.println(feature);
            System.out.println(labels);
            var lr = new RereKnn();
            var res = lr.fit(feature, labels);
            System.out.println(res);
            var predicted = lr.predictBatch(feature);
            ClassificationMetrics metrics = ClassificationMetrics.compute(labels, predicted);
            System.out.println(metrics);
            
            var result = CrossValidation.kFoldCrossValidation(lr, feature, labels, 3);
            System.out.println(result);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
