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

    // ========== 概率分布相关数学函数 / Probability Distribution Mathematical Functions ==========
    
    /**
     * 伽马函数的近似实现
     * Approximation implementation of gamma function
     * 
     * @param x 输入值 / Input value
     * @return 伽马函数值 / Gamma function value
     */
    public static double gamma(float x) {
        // Stirling's approximation for gamma function
        if (x < 0.5f) {
            return Math.PI / (Math.sin(Math.PI * x) * gamma(1.0f - x));
        }
        
        x = x - 1.0f;
        double result = 0.99999999999980993;
        double[] coefficients = {
            676.5203681218851, -1259.1392167224028, 771.32342877765313,
            -176.61502916214059, 12.507343278686905, -0.13857109526572012,
            9.9843695780195716e-6, 1.5056327351493116e-7
        };
        
        for (int i = 0; i < coefficients.length; i++) {
            result += coefficients[i] / (x + i + 1);
        }
        
        double t = x + coefficients.length - 0.5;
        return Math.sqrt(2 * Math.PI) * Math.pow(t, x + 0.5) * Math.exp(-t) * result;
    }
    
    /**
     * 不完全贝塔函数的近似实现
     * Approximation implementation of incomplete beta function
     * 
     * @param a 参数a / Parameter a
     * @param b 参数b / Parameter b
     * @param x 输入值 / Input value
     * @return 不完全贝塔函数值 / Incomplete beta function value
     */
    public static double incompleteBeta(float a, float b, float x) {
        if (x < 0.0f || x > 1.0f) {
            throw new IllegalArgumentException("x必须在[0,1]范围内 / x must be in range [0,1]");
        }
        
        if (x == 0.0f) return 0.0;
        if (x == 1.0f) return 1.0;
        
        // 使用连分数展开
        // Using continued fraction expansion
        double bt = Math.exp(gamma(a + b) - gamma(a) - gamma(b) + 
            a * Math.log(x) + b * Math.log(1.0 - x));
        
        if (x < (a + 1.0f) / (a + b + 2.0f)) {
            return bt * betaCF(a, b, x) / a;
        } else {
            return 1.0 - bt * betaCF(b, a, 1.0f - x) / b;
        }
    }
    
    /**
     * 贝塔函数的连分数展开
     * Continued fraction expansion for beta function
     */
    private static double betaCF(float a, float b, float x) {
        int maxIter = 100;
        double eps = 1e-10;
        
        double qab = a + b;
        double qap = a + 1.0;
        double qam = a - 1.0;
        double c = 1.0;
        double d = 1.0 - qab * x / qap;
        
        if (Math.abs(d) < eps) d = eps;
        d = 1.0 / d;
        double h = d;
        
        for (int m = 1; m <= maxIter; m++) {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (Math.abs(d) < eps) d = eps;
            c = 1.0 + aa / c;
            if (Math.abs(c) < eps) c = eps;
            d = 1.0 / d;
            h *= d * c;
            
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (Math.abs(d) < eps) d = eps;
            c = 1.0 + aa / c;
            if (Math.abs(c) < eps) c = eps;
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            
            if (Math.abs(del - 1.0) < eps) break;
        }
        
        return h;
    }
    
    /**
     * 误差函数的近似实现
     * Approximation implementation of error function
     * 
     * @param x 输入值 / Input value
     * @return 误差函数值 / Error function value
     */
    public static double erf(float x) {
        // Abramowitz and Stegun approximation
        double a1 =  0.254829592;
        double a2 = -0.284496736;
        double a3 =  1.421413741;
        double a4 = -1.453152027;
        double a5 =  1.061405429;
        double p  =  0.3275911;
        
        int sign = x < 0 ? -1 : 1;
        x = Math.abs(x);
        
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        
        return sign * y;
    }
    
    /**
     * 不完全伽马函数的近似实现
     * Approximation implementation of incomplete gamma function
     * 
     * @param a 参数a / Parameter a
     * @param x 输入值 / Input value
     * @return 不完全伽马函数值 / Incomplete gamma function value
     */
    public static double incompleteGamma(float a, float x) {
        if (x < 0.0f) {
            throw new IllegalArgumentException("x必须大于等于0 / x must be greater than or equal to 0");
        }
        
        if (x == 0.0f) return 0.0;
        if (a == 0.0f) return 1.0;
        
        // 使用简单的数值积分方法
        // Using simple numerical integration method
        return incompleteGammaIntegral(a, x);
    }
    
    /**
     * 使用数值积分计算不完全伽马函数
     * Calculate incomplete gamma function using numerical integration
     */
    private static double incompleteGammaIntegral(float a, float x) {
        // 使用梯形法则进行数值积分
        // Using trapezoidal rule for numerical integration
        int n = 1000; // 积分步数
        double h = x / n;
        double sum = 0.0;
        
        // 计算积分 ∫[0,x] t^(a-1) * e^(-t) dt
        for (int i = 0; i <= n; i++) {
            double t = i * h;
            if (t == 0.0 && a <= 1.0) {
                // 处理t=0时的奇点
                continue;
            }
            
            double weight = (i == 0 || i == n) ? 0.5 : 1.0;
            double integrand = Math.pow(t, a - 1.0) * Math.exp(-t);
            sum += weight * integrand;
        }
        
        return h * sum;
    }
    
    /**
     * 伽马函数的连分数展开
     * Continued fraction expansion for gamma function
     */
    private static double gammaCF(float a, float x) {
        int maxIter = 100;
        double eps = 1e-10;
        
        double b = x + 1.0 - a;
        double c = 1e30;
        double d = 1.0 / b;
        double h = d;
        
        for (int i = 1; i <= maxIter; i++) {
            double an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if (Math.abs(d) < eps) d = eps;
            c = b + an / c;
            if (Math.abs(c) < eps) c = eps;
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            
            if (Math.abs(del - 1.0) < eps) break;
        }
        
        return h;
    }
    
    /**
     * 逆正态累积分布函数的近似实现
     * Approximation implementation of inverse normal CDF
     * 
     * @param p 概率值，范围[0,1] / Probability value, range [0,1]
     * @return 逆正态CDF值 / Inverse normal CDF value
     */
    public static double inverseNormalCDF(float p) {
        if (p < 0.0f || p > 1.0f) {
            throw new IllegalArgumentException("概率值必须在[0,1]范围内 / Probability must be in range [0,1]");
        }
        
        if (p == 0.0f) return Double.NEGATIVE_INFINITY;
        if (p == 1.0f) return Double.POSITIVE_INFINITY;
        
        // Beasley-Springer-Moro algorithm
        double[] a = {0, -3.969683028665376e+01, 2.209460984245205e+02,
                     -2.759285104469687e+02, 1.383577518672690e+02,
                     -3.066479806614716e+01, 2.506628277459239e+00};
        
        double[] b = {0, -5.447609879822406e+01, 1.615858368580409e+02,
                     -1.556989798598866e+02, 6.680131188771972e+01,
                     -1.328068155288572e+01};
        
        double[] c = {0, -7.784894002430293e-03, -3.223964580411365e-01,
                     -2.400758277161838e+00, -2.549732539343734e+00,
                     4.374664141464968e+00, 2.938163982698783e+00};
        
        double[] d = {0, 7.784695709041462e-03, 3.224671290700398e-01,
                     2.445134137142996e+00, 3.754408661907416e+00};
        
        double x = 0.0;
        
        if (p < 0.02425) {
            // Lower tail
            double q = Math.sqrt(-2.0 * Math.log(p));
            x = (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
                ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0);
        } else if (p < 0.97575) {
            // Central region
            double q = p - 0.5;
            double r = q * q;
            x = (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q /
                (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1.0);
        } else {
            // Upper tail
            double q = Math.sqrt(-2.0 * Math.log(1.0 - p));
            x = -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
                ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0);
        }
        
        return x;
    }
    
    /**
     * 生成随机浮点数数组
     * Generate random float array
     * 
     * @param n 数组长度 / Array length
     * @return 随机浮点数数组 / Random float array
     */
    public static float[] generateRandomFloats(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("数组长度必须大于等于0 / Array length must be greater than or equal to 0");
        }
        
        Random random = new Random();
        float[] result = new float[n];
        for (int i = 0; i < n; i++) {
            result[i] = random.nextFloat();
        }
        return result;
    }
    
    /**
     * 生成随机浮点数数组（带种子）
     * Generate random float array with seed
     * 
     * @param seed 随机种子 / Random seed
     * @param n 数组长度 / Array length
     * @return 随机浮点数数组 / Random float array
     */
    public static float[] generateRandomFloats(int seed, int n) {
        if (n < 0) {
            throw new IllegalArgumentException("数组长度必须大于等于0 / Array length must be greater than or equal to 0");
        }
        
        Random random = new Random(seed);
        float[] result = new float[n];
        for (int i = 0; i < n; i++) {
            result[i] = random.nextFloat();
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
