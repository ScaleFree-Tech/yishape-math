package com.yishape.lab.util;

import java.io.Serializable;

/**
 * 五元组 / Quintuple
 * @author lteb2
 * @param <E1> 第一个元素类型 / First element type
 * @param <E2> 第二个元素类型 / Second element type
 * @param <E3> 第三个元素类型 / Third element type
 * @param <E4> 第四个元素类型 / Fourth element type
 * @param <E5> 第五个元素类型 / Fifth element type
 * @version 1.0
 * @since 1.0
 */
public class Tuple5<E1, E2, E3,E4,E5> extends Tuple4<E1, E2,E3,E4> implements Serializable{

    public E5 _5;

    /**
     * 创建五元组 / Create Tuple5
     *
     * @param a 第一个元素 / First element
     * @param b 第二个元素 / Second element
     * @param c 第三个元素 / Third element
     * @param d 第四个元素 / Fourth element
     * @param e 第五个元素 / Fifth element
     * @param bilingual 双语标记 / Bilingual marker
     */
    public Tuple5(E1 a, E2 b, E3 c,E4 d,E5 e) {
        super(a, b,c,d);
        this._5 = e;
    }

    /**
     * 获取第五个元素 / Get fifth element
     * @return 第五个元素 / Fifth element
     */
    public E5 getFifth() {
        return this._5;
    }
}