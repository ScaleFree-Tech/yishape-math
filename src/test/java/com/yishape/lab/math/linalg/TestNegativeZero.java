package com.yishape.lab.math.linalg;

public class TestNegativeZero {
    public static void main(String[] args) {
        // 创建包含负零的向量
        double[] data = {-0.0, 0.0, 1.0, -1.0, 0.000001, -0.000001};
        IDoubleVector vector = new RereDoubleVector(data);
        
        System.out.println("向量内容:");
        System.out.println(vector.toString());
        
        // 验证负零的处理
        System.out.println("\n单独测试负零:");
        double negZero = -0.0;
        System.out.println("负零值: " + negZero);
        System.out.println("负零绝对值: " + Math.abs(negZero));
        System.out.println("负零与0比较: " + (negZero == 0.0));
    }
}