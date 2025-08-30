package com.reremouse.lab.util;

import java.io.Serializable;

/**
 * 七元组
 *
 * @author RereMouse
 */
public class Tuple7<E1, E2, E3, E4, E5, E6,E7> extends Tuple6<E1, E2, E3, E4, E5, E6> implements Serializable {

    public E7 _7;

    public Tuple7(E1 a, E2 b, E3 c, E4 d, E5 e, E6 f, E7 g) {
        super(a, b, c, d, e,f);
        this._7 = g;
    }
}
