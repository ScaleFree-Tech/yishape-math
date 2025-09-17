package com.reremouse.lab.util;

import java.io.Serializable;

/**
 * 三元组
 *
 * @author RereMouse
 */
public class Tuple3<E1, E2, E3> extends Tuple2<E1, E2> implements Serializable {

    public E3 _3;

    public Tuple3(E1 a, E2 b, E3 c) {
        super(a, b);
        this._3 = c;
    }

    @Override
    public String toString() {
        return _1 + ", " + _2 + ", " + _3;
    }

    public E1 getFirst() {
        return this._1;
    }

    public E2 getSecond() {
        return this._2;
    }

    public E3 getThird() {
        return this._3;
    }

}
