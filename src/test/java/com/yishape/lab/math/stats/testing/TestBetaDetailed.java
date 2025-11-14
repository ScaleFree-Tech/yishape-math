/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.yishape.lab.math.stats.testing;

import com.yishape.lab.math.RereMathUtil;

/**
 *
 * @author lteb2
 */
public class TestBetaDetailed {

    public static void main(String args[]) {
        // 测试不完全贝塔函数计算过程
        double df = 4.0;
        double t = 2.776;
        double a = df / 2.0;  // 2.0
        double b = 0.5;
        
        System.out.println("T分布参数:");
        System.out.println("自由度: " + df);
        System.out.println("t值: " + t);
        System.out.println("a = df/2: " + a);
        System.out.println("b = 0.5: " + b);
        
        // 计算x
        double x = df / (df + t * t);
        System.out.println("x = df/(df+t^2): " + x);
        
        // 计算不完全贝塔函数
        double betaValue = RereMathUtil.incompleteBeta(a, b, x);
        System.out.println("incompleteBeta(" + a + ", " + b + ", " + x + ") = " + betaValue);
        
        // 计算CDF
        double cdf = 1.0 - 0.5 * betaValue;
        System.out.println("CDF(" + t + ") = 1 - 0.5 * " + betaValue + " = " + cdf);
        
        System.out.println("\n期望结果:");
        System.out.println("CDF(2.776) 应该接近 0.975");
    }
}