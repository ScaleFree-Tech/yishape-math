package com.yishape.lab.math.util;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * 数值精度工具类，提供精确的浮点数比较功能
 * 参考Apache Commons Math的Precision类的源码实现
 * 在数值计算中什么叫相等？？？1.0E-19等不等于0？如果用"=="判断一定不等，但是从数值计算角度看必须相等。
 * 为避免以上问题，特创建此类处理此类问题
 * 2018.7于电子科大东二院
 * 
 * @author lteb2
 */
public class RerePrecision {
    
    /**
     * 默认的绝对误差
     */
    private static final double EPSILON = 1e-15;
    
    /**
     * 默认的最大ULP数（Units in the Last Place）
     */
    private static final int MAX_ULPS = 1;
    
    /**
     * 机器精度 - IEEE 754双精度浮点数的机器epsilon
     * 这是最大的相对舍入误差
     */
    public static final double MACHINE_EPSILON;
    
    /**
     * 安全最小值，使得 1 / SAFE_MIN 不会溢出
     * 在IEEE 754算术中，这也是最小的标准化数字 2^-1022
     */
    public static final double SAFE_MIN;
    
    /** IEEE754表示中的指数偏移量 */
    private static final long EXPONENT_OFFSET = 1023L;
    
    /** 用于按字典序排列有符号双精度数的偏移量 */
    private static final long SGN_MASK = 0x8000000000000000L;
    
    /** 正零 */
    private static final double POSITIVE_ZERO = 0d;
    
    /** 正零的位表示 */
    private static final long POSITIVE_ZERO_DOUBLE_BITS = Double.doubleToRawLongBits(+0.0);
    
    /** 负零的位表示 */
    private static final long NEGATIVE_ZERO_DOUBLE_BITS = Double.doubleToRawLongBits(-0.0);
    
    static {
        // 计算机器精度：2^-53
        MACHINE_EPSILON = Double.longBitsToDouble((EXPONENT_OFFSET - 53L) << 52);
        
        // 计算安全最小值：2^-1022
        SAFE_MIN = Double.longBitsToDouble((EXPONENT_OFFSET - 1022L) << 52);
    }
    
    /**
     * 私有的构造函数，防止实例化
     */
    private RerePrecision() {
        // 工具类，不需要实例化
    }
    
    /**
     * 比较两个双精度浮点数，使用默认的绝对误差
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @return 如果x &lt; y返回-1，如果x &gt; y返回1，如果相等返回0
     */
    public static int compareTo(double x, double y) {
        return compareTo(x, y, EPSILON);
    }
    
    /**
     * 比较两个双精度浮点数，使用指定的绝对误差
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @param eps 绝对误差
     * @return 如果x &lt; y返回-1，如果x &gt; y返回1，如果相等返回0
     */
    public static int compareTo(double x, double y, double eps) {
        if (equals(x, y, eps)) {
            return 0;
        } else if (x < y) {
            return -1;
        } else {
            return 1;
        }
    }
    
    /**
     * 比较两个双精度浮点数，使用指定的最大ULP数
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @param maxUlps 最大ULP数
     * @return 如果x &lt; y返回-1，如果x &gt; y返回1，如果相等返回0
     */
    public static int compareTo(double x, double y, int maxUlps) {
        if (equals(x, y, maxUlps)) {
            return 0;
        } else if (x < y) {
            return -1;
        } else {
            return 1;
        }
    }
    
    /**
     * 检查两个双精度浮点数是否相等，使用默认的绝对误差
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @return 如果相等返回true，否则返回false
     */
    public static boolean equals(double x, double y) {
        return equals(x, y, MAX_ULPS);
    }
    
    /**
     * 检查两个双精度浮点数是否相等，使用指定的绝对误差
     * 参考Apache Commons Math3的实现，先检查ULP相等性，再检查绝对误差
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @param eps 绝对误差
     * @return 如果相等返回true，否则返回false
     */
    public static boolean equals(double x, double y, double eps) {
        return equals(x, y, MAX_ULPS) || Math.abs(y - x) <= eps;
    }
    
    /**
     * 检查两个双精度浮点数是否相等，使用指定的绝对误差和最大ULP数
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @param eps 绝对误差
     * @param maxUlps 最大ULP数
     * @return 如果相等返回true，否则返回false
     */
    public static boolean equals(double x, double y, double eps, int maxUlps) {
        // 处理无穷大的情况
        if (Double.isInfinite(x) || Double.isInfinite(y)) {
            return x == y;
        }
        
        // 处理NaN的情况
        if (Double.isNaN(x) || Double.isNaN(y)) {
            return false; // NaN不等于任何值，包括它自己
        }
        
        // 首先检查ULP误差
        if (equals(x, y, maxUlps)) {
            return true;
        }
        
        // 然后检查绝对误差
        return Math.abs(x - y) <= eps;
    }
    
    /**
     * 检查两个双精度浮点数是否相等，使用指定的最大ULP数
     * 改进的ULP比较算法，参考Apache Commons Math3的实现
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @param maxUlps 最大ULP数
     * @return 如果相等返回true，否则返回false
     */
    public static boolean equals(double x, double y, int maxUlps) {
        // 获取两个数的位表示
        final long xInt = Double.doubleToRawLongBits(x);
        final long yInt = Double.doubleToRawLongBits(y);
        
        final boolean isEqual;
        if (((xInt ^ yInt) & SGN_MASK) == 0L) {
            // 数字符号相同，没有溢出风险
            isEqual = Math.abs(xInt - yInt) <= maxUlps;
        } else {
            // 数字符号相反，需要小心处理溢出
            final long deltaPlus;
            final long deltaMinus;
            if (xInt < yInt) {
                deltaPlus = yInt - POSITIVE_ZERO_DOUBLE_BITS;
                deltaMinus = xInt - NEGATIVE_ZERO_DOUBLE_BITS;
            } else {
                deltaPlus = xInt - POSITIVE_ZERO_DOUBLE_BITS;
                deltaMinus = yInt - NEGATIVE_ZERO_DOUBLE_BITS;
            }
            
            if (deltaPlus > maxUlps) {
                isEqual = false;
            } else {
                isEqual = deltaMinus <= (maxUlps - deltaPlus);
            }
        }
        
        return isEqual && !Double.isNaN(x) && !Double.isNaN(y);
    }
    
    /**
     * 检查两个双精度浮点数是否相等，使用相对误差
     * 改进的相对误差比较，参考Apache Commons Math3的实现
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @param eps 相对误差
     * @return 如果相等返回true，否则返回false
     */
    public static boolean equalsWithRelativeTolerance(double x, double y, double eps) {
        if (equals(x, y, MAX_ULPS)) {
            return true;
        }
        
        final double absoluteMax = Math.max(Math.abs(x), Math.abs(y));
        final double relativeDifference = Math.abs((x - y) / absoluteMax);
        
        return relativeDifference <= eps;
    }
    
    /**
     * 检查两个双精度浮点数是否相等，包括NaN的情况
     * 如果两个值都是NaN，则认为相等
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @return 如果相等或都是NaN返回true，否则返回false
     */
    public static boolean equalsIncludingNaN(double x, double y) {
        return (x != x || y != y) ? !(x != x ^ y != y) : equals(x, y, MAX_ULPS);
    }
    
    /**
     * 检查两个双精度浮点数是否相等，包括NaN的情况，使用指定的绝对误差
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @param eps 绝对误差
     * @return 如果相等或都是NaN返回true，否则返回false
     */
    public static boolean equalsIncludingNaN(double x, double y, double eps) {
        return equalsIncludingNaN(x, y) || (Math.abs(y - x) <= eps);
    }
    
    /**
     * 检查两个双精度浮点数是否相等，包括NaN的情况，使用指定的最大ULP数
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @param maxUlps 最大ULP数
     * @return 如果相等或都是NaN返回true，否则返回false
     */
    public static boolean equalsIncludingNaN(double x, double y, int maxUlps) {
        return (x != x || y != y) ? !(x != x ^ y != y) : equals(x, y, maxUlps);
    }
    
    /**
     * 检查一个双精度浮点数是否等于0，使用默认的绝对误差
     * 
     * @param x 要检查的值
     * @return 如果等于0返回true，否则返回false
     */
    public static boolean equalsZero(double x) {
        return equalsZero(x, EPSILON);
    }
    
    /**
     * 检查一个双精度浮点数是否等于0，使用指定的绝对误差
     * 
     * @param x 要检查的值
     * @param eps 绝对误差
     * @return 如果等于0返回true，否则返回false
     */
    public static boolean equalsZero(double x, double eps) {
        return Math.abs(x) <= eps;
    }
    
    /**
     * 检查一个双精度浮点数是否大于另一个，使用指定的绝对误差
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @param eps 绝对误差
     * @return 如果x > y返回true，否则返回false
     */
    public static boolean isGreaterThan(double x, double y, double eps) {
        return compareTo(x, y, eps) > 0;
    }
    
    /**
     * 检查一个双精度浮点数是否小于另一个，使用指定的绝对误差
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @param eps 绝对误差
     * @return 如果x &lt; y返回true，否则返回false
     */
    public static boolean isLessThan(double x, double y, double eps) {
        return compareTo(x, y, eps) < 0;
    }
    
    /**
     * 检查一个双精度浮点数是否大于等于另一个，使用指定的绝对误差
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @param eps 绝对误差
     * @return 如果x >= y返回true，否则返回false
     */
    public static boolean isGreaterThanOrEqual(double x, double y, double eps) {
        return compareTo(x, y, eps) >= 0;
    }
    
    /**
     * 检查一个双精度浮点数是否小于等于另一个，使用指定的绝对误差
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @param eps 绝对误差
     * @return 如果x &lt;= y返回true，否则返回false
     */
    public static boolean isLessThanOrEqual(double x, double y, double eps) {
        return compareTo(x, y, eps) <= 0;
    }
    
    /**
     * 检查一个双精度浮点数是否大于另一个，使用默认的绝对误差
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @return 如果x > y返回true，否则返回false
     */
    public static boolean isGreaterThan(double x, double y) {
        return isGreaterThan(x, y, EPSILON);
    }
    
    /**
     * 检查一个双精度浮点数是否小于另一个，使用默认的绝对误差
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @return 如果x &lt; y返回true，否则返回false
     */
    public static boolean isLessThan(double x, double y) {
        return isLessThan(x, y, EPSILON);
    }
    
    /**
     * 检查一个双精度浮点数是否大于等于另一个，使用默认的绝对误差
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @return 如果x >= y返回true，否则返回false
     */
    public static boolean isGreaterThanOrEqual(double x, double y) {
        return isGreaterThanOrEqual(x, y, EPSILON);
    }
    
    /**
     * 检查一个双精度浮点数是否小于等于另一个，使用默认的绝对误差
     * 
     * @param x 第一个值
     * @param y 第二个值
     * @return 如果x &lt;= y返回true，否则返回false
     */
    public static boolean isLessThanOrEqual(double x, double y) {
        return isLessThanOrEqual(x, y, EPSILON);
    }
    
    /**
     * 获取默认的epsilon值
     * 
     * @return 默认的epsilon值
     */
    public static double getDefaultEpsilon() {
        return EPSILON;
    }
    
    /**
     * 获取默认的最大ULP数
     * 
     * @return 默认的最大ULP数
     */
    public static int getDefaultMaxUlps() {
        return MAX_ULPS;
    }
    
    /**
     * 获取机器精度
     * 
     * @return 机器精度
     */
    public static double getMachineEpsilon() {
        return MACHINE_EPSILON;
    }
    
    /**
     * 获取安全最小值
     * 
     * @return 安全最小值
     */
    public static double getSafeMin() {
        return SAFE_MIN;
    }
    
    /**
     * 四舍五入到某某位小数
     * @param value
     * @param places
     * @return 
     */
    public static double roundToDecimalPlaces(double value, int places) {
    return new BigDecimal(value)
            .setScale(places, RoundingMode.HALF_UP)
            .doubleValue();
}
}