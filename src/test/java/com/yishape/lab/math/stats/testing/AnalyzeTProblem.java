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
public class AnalyzeTProblem {

    public static void main(String args[]) {
        double[] ds = new double[] { 2.3, 4.5, 2.4, 1.8, 3.2 };
        IVector vec = Linalg.vector(ds);
        
        // 基本统计信息
        double mean = (double)vec.mean();
        int n = vec.length();
        double s = (double)vec.std(1);
        System.out.println("样本均值: " + mean);
        System.out.println("样本大小: " + n);
        System.out.println("样本标准差: " + s);
        
        // T分布参数
        double df = n - 1;
        System.out.println("自由度: " + df);
        
        // 创建T分布对象
        var tDist = Stats.t(df);
        System.out.println("T分布对象: " + tDist);
        
        // 计算临界值
        double confidence = 0.95;
        double left = (1.0 - confidence) / 2.0;
        double right = 1 - left;
        System.out.println("左侧概率: " + left);
        System.out.println("右侧概率: " + right);
        
        double tLowBound = tDist.ppf(left);
        double tUpperBound = tDist.ppf(right);
        System.out.println("T分布左侧临界值: " + tLowBound);
        System.out.println("T分布右侧临界值: " + tUpperBound);
        
        // 计算置信区间
        double margin = tUpperBound * s / Math.sqrt(n);
        double lower = mean - margin;
        double upper = mean + margin;
        System.out.println("置信区间: (" + lower + ", " + upper + ")");
        
        // 验证T分布的ppf方法
        System.out.println("\n验证T分布ppf方法:");
        System.out.println("ppf(0.025): " + tDist.ppf(0.025));
        System.out.println("ppf(0.975): " + tDist.ppf(0.975));
        System.out.println("ppf(0.5): " + tDist.ppf(0.5));
        
        // 验证T分布的cdf方法
        System.out.println("\n验证T分布cdf方法:");
        System.out.println("cdf(-2.776): " + tDist.cdf(-2.776));
        System.out.println("cdf(0): " + tDist.cdf(0));
        System.out.println("cdf(2.776): " + tDist.cdf(2.776));
    }
}