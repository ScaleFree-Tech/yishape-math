package com.reremouse.lab.util;

import java.io.Serializable;

/**
 * 四元组
 * @author lteb2
 */
public class Tuple4<E1, E2, E3,E4> extends Tuple3<E1, E2,E3> implements Serializable{

    public E4 _4;

    public Tuple4(E1 a, E2 b, E3 c,E4 d) {
        super(a, b,c);
        this._4 = d;
    }
    
    public E4 getFourth() {
        return this._4;
    }
}