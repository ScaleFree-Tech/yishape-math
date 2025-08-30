
package com.reremouse.lab.util;

import java.io.Serializable;

/**
 * 五元组
 * @author RereMouse
 */
public class Tuple5<E1, E2, E3,E4,E5> extends Tuple4<E1, E2,E3,E4> implements Serializable{

    public E5 _5;

    public Tuple5(E1 a, E2 b, E3 c,E4 d,E5 e) {
        super(a, b,c,d);
        this._5 = e;
    }
}
