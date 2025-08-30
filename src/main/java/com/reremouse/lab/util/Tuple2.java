package com.reremouse.lab.util;

import java.io.Serializable;

/**
 * 二元组
 *
 * @author RereMouse
 * @param <E1>
 * @param <E2>
 */
public class Tuple2<E1, E2> implements Serializable{

    public E1 _1;//第一个元素，写法类似Scala
    public E2 _2;

    public Tuple2() {
    }

    public Tuple2(E1 a, E2 b) {
        this._1 = a;
        this._2 = b;
    }

    @Override
    public String toString() {
        return "("+this._1+", "+this._2+")"; 
    
    }
    
    
    
}
