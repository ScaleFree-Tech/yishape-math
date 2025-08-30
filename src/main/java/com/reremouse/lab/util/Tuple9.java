package com.reremouse.lab.util;

import java.io.Serializable;

/**
 * 九元组
 *
 * @author RereMouse
 */
public class Tuple9<E1, E2, E3, E4, E5, E6, E7, E8, E9> extends Tuple8<E1, E2, E3, E4, E5, E6, E7, E8> implements Serializable {

    public E9 _9;

    public Tuple9(E1 a, E2 b, E3 c, E4 d, E5 e, E6 f, E7 g, E8 h, E9 i) {
        super(a, b, c, d, e, f, g, h);
        this._9 = i;
    }
}
