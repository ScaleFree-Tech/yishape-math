package com.yishape.lab.util;

import java.io.Serializable;

/**
 * 九元组 / Nonuple
 *
 * @author lteb2
 * @param <E1> 第一个元素类型 / First element type
 * @param <E2> 第二个元素类型 / Second element type
 * @param <E3> 第三个元素类型 / Third element type
 * @param <E4> 第四个元素类型 / Fourth element type
 * @param <E5> 第五个元素类型 / Fifth element type
 * @param <E6> 第六个元素类型 / Sixth element type
 * @param <E7> 第七个元素类型 / Seventh element type
 * @param <E8> 第八个元素类型 / Eighth element type
 * @param <E9> 第九个元素类型 / Ninth element type
 * @version 1.0
 * @since 1.0
 */
public class Tuple9<E1, E2, E3, E4, E5, E6, E7, E8, E9> extends Tuple8<E1, E2, E3, E4, E5, E6, E7, E8> implements Serializable {

    public E9 _9;

    /**
     * 创建九元组 / Create Tuple9
     *
     * @param a 第一个元素 / First element
     * @param b 第二个元素 / Second element
     * @param c 第三个元素 / Third element
     * @param d 第四个元素 / Fourth element
     * @param e 第五个元素 / Fifth element
     * @param f 第六个元素 / Sixth element
     * @param g 第七个元素 / Seventh element
     * @param h 第八个元素 / Eighth element
     * @param i 第九个元素 / Ninth element
     */
    public Tuple9(E1 a, E2 b, E3 c, E4 d, E5 e, E6 f, E7 g, E8 h, E9 i) {
        super(a, b, c, d, e, f, g, h);
        this._9 = i;
    }

    /**
     * 获取第九个元素 / Get ninth element
     * @return 第九个元素 / Ninth element
     */
    public E9 getNinth() {
        return this._9;
    }
}