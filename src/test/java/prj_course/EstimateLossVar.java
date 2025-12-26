package prj_course;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.stats.Stats;

public class EstimateLossVar {
    
public static void main(String[] args){
    double[] losses = new double[]{4.0,5.8,3.2,5.6,9.4,5.6};
    IVector<Double> vec = Linalg.vector(losses);
    double mean = vec.mean();
    double s2 = vec.var(1);
    int n = vec.size();
    double confidence = 0.95;
    double chi2_upper = Stats.chi2(n-1).ppf(1-(1.0-confidence)/2.0);
    double chi2_lower = Stats.chi2(n-1).ppf((1.0-confidence)/2.0);

    double sigma2_lower = (n-1)*s2/chi2_upper;
    double sigma2_upper = (n-1)*s2/chi2_lower;
    double sigma_lower = Math.sqrt(sigma2_lower);
    double sigma_upper = Math.sqrt(sigma2_upper);
    System.out.println("Mean:"+mean);
    System.out.println("Sample varance:"+s2);
    System.out.println("estimated sigma interval:"+sigma_lower+","+sigma_upper);

}


}
