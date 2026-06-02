package prj_course;

import com.yishape.lab.math.linalg.Linalg;

/**
 *
 * @author lteb2
 */
public class Test2 {

    public static void main(String args[]) {
        double[][] data = {{1.0, 2.0}, {4.0, 8.0}};
        var A = Linalg.matrix(data);
        double[] bb = {1.0,4.0};
        var b = Linalg.vector(bb);
//        var res1 = Linalg.solve(A, b);
//        System.out.println(res1);
        var res2 = Linalg.lstsq(A, b);
        System.out.println(res2);
    }

}
