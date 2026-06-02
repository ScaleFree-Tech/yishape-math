package com.yishape.lab.math.linalg.decomposition.impl.support.lasd;

/**
 * LAPACK DLASD2 入口对 {@code IDXQ} 的要求（见 {@code dlasd2.f} 对 {@code IDXQ} 的说明：
 * 前半在 DO 10 中已逐次后移；后半在进入 DO 50 前须为「局部偏移」，使得加上 {@code NLP1=NL+1} 后得到全局列下标）。
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public final class LapackDlasdIdxq {

    private LapackDlasdIdxq() {
    }

    /**
     * @param nl   上块行维 NL
     * @param n    N = NL+NR+1
     * @param idxq 1..n，将被覆写为 dlasd2 期望的入口值
     */
    public static void initIdxqBeforeDlasd2(int nl, int n, int[] idxq) {
        int nlp2 = nl + 2;
        for (int i = 1; i <= nl; i++) {
            idxq[i] = i;
        }
        for (int i = nlp2; i <= n; i++) {
            idxq[i] = i - nl - 1;
        }
    }
}
