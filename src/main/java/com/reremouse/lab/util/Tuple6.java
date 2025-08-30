package com.reremouse.lab.util;

import java.io.Serializable;

/**
 * 六元组
 *
 * @author RereMouse
 */
public class Tuple6<E1, E2, E3, E4, E5, E6> extends Tuple5<E1, E2, E3, E4, E5> implements Serializable {

    public E6 _6;

    public Tuple6(E1 a, E2 b, E3 c, E4 d, E5 e, E6 f) {
        super(a, b, c, d, e);
        this._6 = f;
    }
}
