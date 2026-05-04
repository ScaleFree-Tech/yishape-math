package com.yishape.lab.util;

import java.io.Serializable;

/**
 * 二元组 / Pair
 *
 * @author lteb2
 * @param <E1> 第一个元素类型 / First element type
 * @param <E2> 第二个元素类型 / Second element type
 * @version 1.0
 * @since 1.0
 */
public class Tuple2<E1, E2> implements Serializable{

    /** 第一个元素（写法类似Scala）/ First element (style similar to Scala) */
    public E1 _1;
    /** 第二个元素 / Second element */
    public E2 _2;

    public Tuple2() {
    }

    /**
     * 创建二元组 / Create Tuple2
     *
     * @param a 第一个元素 / First element
     * @param b 第二个元素 / Second element
     */
    public Tuple2(E1 a, E2 b) {
        this._1 = a;
        this._2 = b;
    }

    @Override
    public String toString() {
        return "("+this._1+", "+this._2+")";
    }

    /**
     * 获取第一个元素 / Get first element
     * @return 第一个元素 / First element
     */
    public E1 getFirst(){
        return this._1;
    }

    /**
     * 获取第二个元素 / Get second element
     * @return 第二个元素 / Second element
     */
    public E2 getSecond(){
        return this._2;
    }
}