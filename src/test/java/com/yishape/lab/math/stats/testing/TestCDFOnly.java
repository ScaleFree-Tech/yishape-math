/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.yishape.lab.math.stats.testing;

import com.yishape.lab.math.stats.Stats;

/**
 *
 * @author lteb2
 */
public class TestCDFOnly {

    public static void main(String args[]) {
        // 创建T分布对象，自由度为4
        var tDist = Stats.t(4.0);
        System.out.println("T分布对象: " + tDist);
        
        // 测试CDF计算
        System.out.println("\n测试CDF计算:");
        double[] test_values = {-5.0, -2.776, -1.0, 0.0, 1.0, 2.776, 5.0};
        for (double t : test_values) {
            double cdf = tDist.cdf(t);
            System.out.printf("CDF(%6.3f) = %8.6f\n", t, cdf);
        }
        
        // 理论期望值:
        System.out.println("\n理论期望值:");
        System.out.println("CDF(-2.776) ≈ 0.025");
        System.out.println("CDF(0.0) = 0.5");
        System.out.println("CDF(2.776) ≈ 0.975");
    }
}