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
public class TestTAndZFixed {

    public static void main(String args[]) {
        double[] ds = new double[] { 2.3, 4.5, 2.4, 1.8, 3.2 };
        IVector vec = Linalg.vector(ds);
        double sigma = (double) vec.std(1);
        
        // 修复后的实现
        var tp1 = Stats.estimator.estimateMeanIntevalWithT(vec);
        var tp2 = Stats.estimator.estimateMeanIntevalWithZ(vec, sigma);
        System.out.println("修复后T分布区间估计: " + tp1);
        System.out.println("修复后Z分布区间估计: " + tp2);
        
        // 手动计算验证
        double mean = (double)vec.mean();
        int n = vec.length();
        double s = (double)vec.std(1);
        System.out.println("样本均值: " + mean);
        System.out.println("样本大小: " + n);
        System.out.println("样本标准差: " + s);
        
        // T分布临界值计算 (自由度=4, 置信度=0.95)
        var tDist = Stats.t(n-1);
        double tCritical = tDist.ppf(0.975); // 双侧检验，0.025在上尾
        System.out.println("T分布临界值 (df=4, α=0.05): " + tCritical);
        
        // 置信区间手动计算
        double marginT = tCritical * s / Math.sqrt(n);
        double lowerT = mean - marginT;
        double upperT = mean + marginT;
        System.out.println("手动计算T分布置信区间: (" + lowerT + ", " + upperT + ")");
        
        // Z分布临界值计算 (置信度=0.95)
        var normDist = Stats.norm();
        double zCritical = normDist.ppf(0.975);
        System.out.println("Z分布临界值 (α=0.05): " + zCritical);
        
        // 置信区间手动计算
        double marginZ = zCritical * sigma / Math.sqrt(n);
        double lowerZ = mean - marginZ;
        double upperZ = mean + marginZ;
        System.out.println("手动计算Z分布置信区间: (" + lowerZ + ", " + upperZ + ")");
    }

}