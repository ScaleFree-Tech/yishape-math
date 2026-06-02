/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.yishape.lab.math.stats.testing;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.stats.Stats;

/**
 *
 * @author lteb2
 */
public class TestTAndZ {

    public static void main(String args[]) {
        double[] ds = new double[] { 2.3, 4.5, 2.4, 1.8, 3.2 };
        IVector vec = Linalg.vector(ds);
        double sigma = vec.stdValue(1);
        var tp1 = Stats.estimator.estimateMeanIntevalWithT(vec);
        var tp2 = Stats.estimator.estimateMeanIntevalWithZ(vec, sigma);
        System.out.println(tp1);
        System.out.println(tp2);
    }

}
