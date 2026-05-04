package com.yishape.lab.util;

import java.io.Serializable;

/**
 * 七元组 / Septuple
 *
 * @author lteb2
 * @param <E1> 第一个元素类型 / First element type
 * @param <E2> 第二个元素类型 / Second element type
 * @param <E3> 第三个元素类型 / Third element type
 * @param <E4> 第四个元素类型 / Fourth element type
 * @param <E5> 第五个元素类型 / Fifth element type
 * @param <E6> 第六个元素类型 / Sixth element type
 * @param <E7> 第七个元素类型 / Seventh element type
 * @version 1.0
 * @since 1.0
 */
public class Tuple7<E1, E2, E3, E4, E5, E6,E7> extends Tuple6<E1, E2, E3, E4, E5, E6> implements Serializable {

    public E7 _7;

    /**
     * 创建七元组 / Create Tuple7
     *
     * @param a 第一个元素 / First element
     * @param b 第二个元素 / Second element
     * @param c 第三个元素 / Third element
     * @param d 第四个元素 / Fourth element
     * @param e 第五个元素 / Fifth element
     * @param f 第六个元素 / Sixth element
     * @param g 第七个元素 / Seventh element
     * @param bilingual 双语标记 / Bilingual marker
     */
    public Tuple7(E1 a, E2 b, E3 c, E4 d, E5 e, E6 f, E7 g) {
        super(a, b, c, d, e,f);
        this._7 = g;
    }

    /**
     * 获取第七个元素 / Get seventh element
     * @return 第七个元素 / Seventh element
     */
    public E7 getSeventh() {
        return this._7;
    }
}