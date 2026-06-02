package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.ml.ML;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;

/**
 *
 * @author lteb2
 */
public class LRTest {

    public static void main(String args[]) {
String path = "G:\\电子科技大学-工作\\商务统计_2025\\data\\iris.csv";
//        String path = "F:\\电子科技大学工作\\商务统计_2025\\data\\iris.csv";
        try {
            var df = DataFrame.readCsv(path);
            var feature = df.sliceColumn(0, -1).toMatrix();
            var labels = df.get(df.getColumnCount()-1).toStringArray();
            System.out.println(feature);
            System.out.println(labels);
            var lr = ML.clf.logisticRegression(0.0,0.0);
            var res = lr.fit(feature, labels);
            System.out.println(res);
            String label = lr.predict(feature.getRow(0));
            System.out.println(label);
            var predicted = lr.predictBatch(feature);
            ClassificationMetrics metrics = ML.clf.classificationMetrics(lr,feature,labels);
            System.out.println(metrics);
            
            var result = ML.clf.kFoldCrossValidation(lr, feature, labels, 3);
            System.out.println(result);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
