package com.yishape.lab.util;

import java.io.Serializable;

/**
 * 三元组 / Triple
 *
 * @author lteb2
 * @param <E1> 第一个元素类型 / First element type
 * @param <E2> 第二个元素类型 / Second element type
 * @param <E3> 第三个元素类型 / Third element type
 * @version 1.0
 * @since 1.0
 */
public class Tuple3<E1, E2, E3> extends Tuple2<E1, E2> implements Serializable {

    public E3 _3;

    /**
     * 创建三元组 / Create Tuple3
     *
     * @param a 第一个元素 / First element
     * @param b 第二个元素 / Second element
     * @param c 第三个元素 / Third element
     */
    public Tuple3(E1 a, E2 b, E3 c) {
        super(a, b);
        this._3 = c;
    }

    @Override
    public String toString() {
        return _1 + ", " + _2 + ", " + _3;
    }


    /**
     * 获取第三个元素 / Get third element
     * @return 第三个元素 / Third element
     */
    public E3 getThird() {
        return this._3;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Tuple3)) return false;
        if (!super.equals(o)) return false;
        Tuple3<?, ?, ?> t = (Tuple3<?, ?, ?>) o;
        return java.util.Objects.equals(_3, t._3);
    }

    @Override
    public int hashCode() {
        return java.util.Objects.hash(_1, _2, _3);
    }
}