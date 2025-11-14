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
public class TestBetaFunction {

    public static void main(String args[]) {
        // 测试不完全贝塔函数
        System.out.println("测试不完全贝塔函数:");
        
        // 测试一些已知值
        double a = 2.0;  // 自由度/2
        double b = 0.5;  // 1/2
        double x = 0.5;  // t^2/(自由度+t^2)
        
        double result = RereMathUtil.incompleteBeta(a, b, x);
        System.out.println("incompleteBeta(" + a + ", " + b + ", " + x + ") = " + result);
        
        // 对于T分布，t=2.776, 自由度=4
        // t^2 = 7.706
        // x = t^2 / (df + t^2) = 7.706 / (4 + 7.706) = 7.706 / 11.706 ≈ 0.658
        double df = 4.0;
        double t_val = 2.776;
        double t_squared = t_val * t_val;
        double x_val = t_squared / (df + t_squared);
        double a_val = df / 2.0;
        double b_val = 0.5;
        
        System.out.println("\nT分布参数:");
        System.out.println("自由度: " + df);
        System.out.println("t值: " + t_val);
        System.out.println("t^2: " + t_squared);
        System.out.println("x = t^2/(df+t^2): " + x_val);
        System.out.println("a = df/2: " + a_val);
        System.out.println("b = 0.5: " + b_val);
        
        double beta_result = RereMathUtil.incompleteBeta(a_val, b_val, x_val);
        System.out.println("incompleteBeta(" + a_val + ", " + b_val + ", " + x_val + ") = " + beta_result);
        
        // 根据T分布CDF公式: CDF(t) = 0.5 + 0.5 * sign(t) * incompleteBeta(a, b, x)
        double cdf_result = 0.5 + 0.5 * Math.signum(t_val) * beta_result;
        System.out.println("T分布CDF(" + t_val + ") = " + cdf_result);
        
        // 正确的CDF值应该接近0.975
        System.out.println("期望的CDF值 (0.975): " + 0.975);
    }
}