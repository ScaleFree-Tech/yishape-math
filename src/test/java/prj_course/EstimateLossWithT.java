package prj_course;

import com.yishape.lab.math.stats.Stats;
import com.yishape.lab.math.linalg.*;

public class EstimateLossWithT {

public static void main(String[] args){
    double[] samples = new double[]{
        4.8,5.2,3.9,7.6,6.3,5.6
    };
    var vec = Linalg.vector(samples);
    double sigma = vec.stdValue(1);
     var tp = Stats.estimator.estimateMeanIntevalWithT(vec);
//    var tp = Stats.estimator.estimateMeanIntevalWithZ(vec,sigma);
    System.out.println(tp);
    double a = tp._1;
    double b = tp._2;
    double m = vec.meanValue();
    double t = (a+b+4*m)/6.0;
    System.out.println("工期："+t);
}


}
