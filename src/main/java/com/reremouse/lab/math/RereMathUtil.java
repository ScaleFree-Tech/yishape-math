package com.reremouse.lab.math;

import java.util.Random;

/**
 * 数学工具类，提供各种数学相关的实用方法 包括类型转换、随机数生成等功能
 *
 * @author lteb2
 */
public class RereMathUtil {

    /**
     * 将float数组转换为double数组
     *
     * @param array 要转换的float数组
     * @return 转换后的double数组
     * @throws NullPointerException 如果输入数组为null
     */
    public static double[] floatToDouble(float[] array) {
        double[] darr = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            darr[i] = array[i];
        }
        return darr;
    }

    /**
     * 将double数组转换为float数组
     *
     * @param array 要转换的double数组
     * @return 转换后的float数组
     * @throws NullPointerException 如果输入数组为null
     */
    public static float[] doubleToFloat(double[] array) {
        float[] darr = new float[array.length];
        for (int i = 0; i < array.length; i++) {
            darr[i] = (float) array[i];
        }
        return darr;
    }

    /**
     * 将int数组转换为float数组
     *
     * @param array 要转换的int数组
     * @return 转换后的float数组
     * @throws NullPointerException 如果输入数组为null
     */
    public static float[] intToFloat(int[] array) {
        float[] darr = new float[array.length];
        for (int i = 0; i < array.length; i++) {
            darr[i] = (float) array[i];
        }
        return darr;
    }

    /**
     * 将float数组转换为int数组（截断小数部分）
     *
     * @param array 要转换的float数组
     * @return 转换后的int数组
     * @throws NullPointerException 如果输入数组为null
     */
    public static int[] floatToInt(float[] array) {
        int[] darr = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            darr[i] = (int) array[i];
        }
        return darr;
    }

    /**
     * 将Float包装类数组转换为float基本类型数组
     *
     * @param v 要转换的Float数组
     * @return 转换后的float数组
     * @throws NullPointerException 如果输入数组为null或包含null元素
     */
    public static float[] toPrimitive(Float[] v) {
        float[] vv = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            vv[i] = v[i];
        }
        return vv;
    }

    /**
     * 将Double包装类数组转换为double基本类型数组
     *
     * @param v 要转换的Double数组
     * @return 转换后的double数组
     * @throws NullPointerException 如果输入数组为null或包含null元素
     */
    public static double[] toPrimitive(Double[] v) {
        double[] vv = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            vv[i] = v[i];
        }
        return vv;
    }

    /**
     * 将Integer包装类数组转换为int基本类型数组
     *
     * @param v 要转换的Integer数组
     * @return 转换后的int数组
     * @throws NullPointerException 如果输入数组为null或包含null元素
     */
    public static int[] toPrimitive(Integer[] v) {
        int[] vv = new int[v.length];
        for (int i = 0; i < v.length; i++) {
            vv[i] = v[i];
        }
        return vv;
    }

    /**
     * 生成从start（包括）到end（不包括）之间的伪随机整数序列 使用指定的种子值，确保每次调用时使用相同的种子会产生相同的随机序列
     *
     * @param seed 随机种子，用于初始化随机数生成器
     * @param start 起始数（包括）
     * @param end 结束数（不包括）
     * @param num 要生成的随机整数数量
     * @return 包含num个随机整数的数组，每个整数都在[start, end)范围内
     * @throws IllegalArgumentException 如果start >= end 或 num < 0
     */
    public static int[] generateRandomInts(int seed, int start, int end, int num) {
        if (start >= end) {
            throw new IllegalArgumentException("start (" + start + ") must be less than end (" + end + ")");
        }
        if (num < 0) {
            throw new IllegalArgumentException("num (" + num + ") must be non-negative");
        }

        Random random = new Random(seed);
        int[] result = new int[num];
        int range = end - start;

        for (int i = 0; i < num; i++) {
            result[i] = start + random.nextInt(range);
        }

        return result;
    }

    /**
     * 生成从start（包括）到end（不包括）之间的随机整数序列 使用系统当前时间作为随机种子，每次调用产生不同的随机序列
     *
     * @param start 起始数（包括）
     * @param end 结束数（不包括）
     * @param num 要生成的随机整数数量
     * @return 包含num个随机整数的数组，每个整数都在[start, end)范围内
     * @throws IllegalArgumentException 如果start >= end 或 num < 0
     */
    public static int[] generateRandomInts(int start, int end, int num) {
        if (start >= end) {
            throw new IllegalArgumentException("start (" + start + ") must be less than end (" + end + ")");
        }
        if (num < 0) {
            throw new IllegalArgumentException("num (" + num + ") must be non-negative");
        }

        Random random = new Random();
        int[] result = new int[num];
        int range = end - start;

        for (int i = 0; i < num; i++) {
            result[i] = start + random.nextInt(range);
        }

        return result;
    }

    public static void main(String args[]) {
        var r1 = RereMathUtil.generateRandomInts(0, 0, 10, 5);
        for (var i : r1) {
            System.out.println(i);
        }

    }

}
