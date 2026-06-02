package prj_course;

import com.yishape.lab.math.linalg.Linalg;

/**
 *
 * @author lteb2
 */
public class Test1 {

    public static void main(String args[]) {

        var mat = Linalg.randn(10, 10);
        var inv = mat.inv();
        var pinv = mat.pinv();
        var r1 = mat.mmul(inv);
        var r2 = mat.mmul(pinv);
        System.out.println(r1+"\n");
        System.out.println(r2);

    }

}
