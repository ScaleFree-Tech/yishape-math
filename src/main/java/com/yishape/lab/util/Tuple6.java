package com.yishape.lab.util;

import java.io.Serializable;

/**
 * 六元组 / Sextuple
 *
 * @author lteb2
 * @param <E1> 第一个元素类型 / First element type
 * @param <E2> 第二个元素类型 / Second element type
 * @param <E3> 第三个元素类型 / Third element type
 * @param <E4> 第四个元素类型 / Fourth element type
 * @param <E5> 第五个元素类型 / Fifth element type
 * @param <E6> 第六个元素类型 / Sixth element type
 * @version 1.0
 * @since 1.0
 */
public class Tuple6<E1, E2, E3, E4, E5, E6> extends Tuple5<E1, E2, E3, E4, E5> implements Serializable {

    public E6 _6;

    /**
     * 创建六元组 / Create Tuple6
     *
     * @param a 第一个元素 / First element
     * @param b 第二个元素 / Second element
     * @param c 第三个元素 / Third element
     * @param d 第四个元素 / Fourth element
     * @param e 第五个元素 / Fifth element
     * @param f 第六个元素 / Sixth element
     */
    public Tuple6(E1 a, E2 b, E3 c, E4 d, E5 e, E6 f) {
        super(a, b, c, d, e);
        this._6 = f;
    }

    /**
     * 获取第六个元素 / Get sixth element
     * @return 第六个元素 / Sixth element
     */
    public E6 getSixth() {
        return this._6;
    }
}