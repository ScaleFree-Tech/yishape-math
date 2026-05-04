package com.yishape.lab.util;

import java.io.Serializable;

/**
 * 四元组 / Quadruple
 * @author lteb2
 * @param <E1> 第一个元素类型 / First element type
 * @param <E2> 第二个元素类型 / Second element type
 * @param <E3> 第三个元素类型 / Third element type
 * @param <E4> 第四个元素类型 / Fourth element type
 * @version 1.0
 * @since 1.0
 */
public class Tuple4<E1, E2, E3,E4> extends Tuple3<E1, E2,E3> implements Serializable{

    public E4 _4;

    /**
     * 创建四元组 / Create Tuple4
     *
     * @param a 第一个元素 / First element
     * @param b 第二个元素 / Second element
     * @param c 第三个元素 / Third element
     * @param d 第四个元素 / Fourth element
     * @param bilingual 双语标记 / Bilingual marker
     */
    public Tuple4(E1 a, E2 b, E3 c,E4 d) {
        super(a, b,c);
        this._4 = d;
    }

    /**
     * 获取第四个元素 / Get fourth element
     * @return 第四个元素 / Fourth element
     */
    public E4 getFourth() {
        return this._4;
    }
}