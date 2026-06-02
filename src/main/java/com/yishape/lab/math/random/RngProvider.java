package com.yishape.lab.math.random;

/**
 * 随机数生成器抽象接口。替换 java.util.Random 以获得更好的统计质量和线程安全性。
 *
 * <p>实现要求：
 * <ul>
 *   <li>{@link #nextDouble()} 返回 [0, 1) 均匀分布</li>
 *   <li>{@link #nextGaussian()} 返回 N(0,1) 正态分布（使用 Box-Muller 或 Ziggurat）</li>
 *   <li>{@link #split()} 分叉出新生成器，状态与原生成器无关（适合并行）</li>
 *   <li>{@link #copy()} 深拷贝当前状态</li>
 * </ul>
 *
 * @author lteb2
 * @since 0.6.0
 */
public interface RngProvider {

    /** 返回 64 位随机整数 */
    long nextLong();

    /** 返回 [0, 1) 均匀分布的 double */
    double nextDouble();

    /** 返回 [0, 1) 均匀分布的 float */
    default float nextFloat() {
        return (float) ((nextLong() >>> 40) * 0x1.0p-24f);
    }

    /** 返回 N(0,1) 标准正态分布 */
    double nextGaussian();

    /** 填充字节数组 */
    default void nextBytes(byte[] bytes) {
        for (int i = 0; i < bytes.length; ) {
            long r = nextLong();
            for (int j = 0; j < 8 && i < bytes.length; j++, i++) {
                bytes[i] = (byte) r;
                r >>>= 8;
            }
        }
    }

    /** 返回 [0, bound) 的均匀随机整数 */
    default int nextInt(int bound) {
        if (bound <= 0) throw new IllegalArgumentException("bound must be positive");
        long bits, val;
        do {
            bits = nextLong() & 0x7FFFFFFFFFFFFFFFL;
            val = bits % bound;
        } while (bits - val + (bound - 1) < 0);
        return (int) val;
    }

    /** 分叉出新生成器，状态与当前生成器无关（并行友好） */
    RngProvider split();

    /** 深拷贝当前生成器（保存/恢复状态） */
    RngProvider copy();

    /** 序列化状态 */
    long[] getState();

    /** 反序列化状态 */
    void setState(long[] state);
}
