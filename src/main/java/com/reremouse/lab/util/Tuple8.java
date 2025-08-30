package com.reremouse.lab.util;

import java.io.Serializable;

/**
 * 八元组
 *
 * @author RereMouse
 */
public class Tuple8<E1, E2, E3, E4, E5, E6,E7,E8> extends Tuple7<E1, E2, E3, E4, E5, E6,E7> implements Serializable {

    public E8 _8;

    public Tuple8(E1 a, E2 b, E3 c, E4 d, E5 e, E6 f, E7 g,E8 h) {
        super(a, b, c, d, e,f,g);
        this._8 = h;
    }
}
