package com.yishape.lab.util;

import java.io.Serializable;

/**
 * 八元组 / Tuple with 8 Elements
 * <p>
 * 表示包含8个元素的元组。
 * Represents a tuple with 8 elements.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class Tuple8<E1, E2, E3, E4, E5, E6,E7,E8> extends Tuple7<E1, E2, E3, E4, E5, E6,E7> implements Serializable {

    public E8 _8;

    /**
     * 创建八元组 / Create Tuple8
     *
     * @param a 第一个元素 / First element
     * @param b 第二个元素 / Second element
     * @param c 第三个元素 / Third element
     * @param d 第四个元素 / Fourth element
     * @param e 第五个元素 / Fifth element
     * @param f 第六个元素 / Sixth element
     * @param g 第七个元素 / Seventh element
     * @param h 第八个元素 / Eighth element
     */
    public Tuple8(E1 a, E2 b, E3 c, E4 d, E5 e, E6 f, E7 g,E8 h) {
        super(a, b, c, d, e,f,g);
        this._8 = h;
    }
    
    /**
     * 获取第八个元素 / Get eighth element
     * @return 第八个元素 / Eighth element
     */
    public E8 getEighth() {
        return this._8;
    }
}