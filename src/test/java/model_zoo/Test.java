package model_zoo;

import com.yishape.lab.math.linalg.Linalg;

public class Test {
    public static void main(String args[]){
        var mm = Linalg.rand(3,4,1);
        mm = Linalg.matrix(new double[][]{
            {1,2,3,4},
            {5,6,7,8},
            {9,10,11,12}
        });
        var tp = mm.svd();
        System.out.println(tp._2);
    }
}
