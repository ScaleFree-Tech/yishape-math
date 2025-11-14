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
public class TestTDetails {

    public static void main(String args[]) {
        // 创建T分布对象，自由度为4
        var tDist = Stats.t(4.0);
        System.out.println("T分布对象: " + tDist);
        
        // 测试已知的T分布临界值
        System.out.println("\n测试已知T分布临界值 (自由度=4):");
        System.out.println("理论值 t(0.975) ≈ 2.776");
        
        // 测试CDF值
        double t_critical = 2.776;
        double cdf_value = tDist.cdf(t_critical);
        System.out.println("实际CDF(" + t_critical + ") = " + cdf_value);
        
        // 测试ppf值
        double prob = 0.975;
        double ppf_value = tDist.ppf(prob);
        System.out.println("实际ppf(" + prob + ") = " + ppf_value);
        
        // 测试更多点
        System.out.println("\n更多测试点:");
        double[] test_points = {0.0, 1.0, 2.0, 2.776, 5.0};
        for (double point : test_points) {
            System.out.println("CDF(" + point + ") = " + tDist.cdf(point));
        }
        
        System.out.println("\n分位数测试:");
        double[] test_probs = {0.025, 0.5, 0.975};
        for (double prob_val : test_probs) {
            System.out.println("ppf(" + prob_val + ") = " + tDist.ppf(prob_val));
        }
    }
}