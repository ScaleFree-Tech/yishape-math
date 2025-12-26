package prj_course;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.stats.Stats;

public class EstimateLoss {

public static void main(String[] args){
    double[] losses = new double[]{4.0,5.8,3.2,5.6,9.4,5.6};
    IVector<Double> vec = Linalg.vector(losses);
    double mean = vec.mean();
    double std = vec.std();
    int n = vec.size();
    double confidence = 0.95;
    double z_upper = Stats.norm().ppf(1-(1.0-confidence)/2.0);
    double z_lower = Stats.norm().ppf((1.0-confidence)/2.0);

    double u_lower = mean - z_upper*std/Math.sqrt(n);
    double u_upper = mean - z_lower*std/Math.sqrt(n);
    System.out.println("Mean:"+mean);
    System.out.println("Std:"+std);
    System.out.println("estimated interval:"+u_lower+","+u_upper);

}


    
}
