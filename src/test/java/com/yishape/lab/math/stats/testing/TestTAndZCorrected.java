/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.yishape.lab.math.stats.testing;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.stats.Stats;
import com.yishape.lab.util.Tuple2;

/**
 *
 * @author lteb2
 */
public class TestTAndZCorrected {

    public static void main(String args[]) {
        double[] ds = new double[] { 2.3, 4.5, 2.4, 1.8, 3.2 };
        IVector vec = Linalg.vector(ds);
        double sigma = vec.stdValue(1);
        
        // 原始错误的实现
        var tp1 = Stats.estimator.estimateMeanIntevalWithT(vec);
        var tp2 = Stats.estimator.estimateMeanIntevalWithZ(vec, sigma);
        System.out.println("原始T分布区间估计: " + tp1);
        System.out.println("原始Z分布区间估计: " + tp2);
        
        // 修正后的实现
        var tp1_corrected = estimateMeanIntevalWithT(vec);
        var tp2_corrected = Stats.estimator.estimateMeanIntevalWithZ(vec, sigma);
        System.out.println("修正T分布区间估计: " + tp1_corrected);
        System.out.println("修正Z分布区间估计: " + tp2_corrected);
    }
    
    // 修正后的T分布区间估计方法
    public static Tuple2<Double,Double> estimateMeanIntevalWithT(IVector sample) {
        return estimateMeanIntevalWithT(sample, 0.95);
    }

    public static Tuple2<Double,Double> estimateMeanIntevalWithT(IVector sample, double confidence) {
        var t = Stats.t(sample.length() - 1);
        double left = (1.0 - confidence) / 2.0;
        double right = 1 - left;
        double tLowBound = t.ppf(left);
        double tUpperBound = t.ppf(right);
        double mean = sample.meanValue();
        int n = sample.length();
        double s = sample.stdValue(1);
        // 修正：使用正确的公式
        double margin = tUpperBound * s / Math.sqrt(n);
        double mLowBound = mean - margin;
        double mUpperBound = mean + margin;
        return new Tuple2(mLowBound, mUpperBound);
    }
}