package com.reremouse.lab.util;

import java.lang.reflect.Method;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 *
 * @author RereMouse
 */
public class RereCollectionUtil {

    /**
     * 从列表中删除某元素（按getId()匹配）
     *
     * @param <T> 列表元素类型
     * @param ls 列表
     * @param id 元素ID
     */
    public static <T> List<T> deleteElementFromList(List<T> ls, String id) {
        return (List<T>)deleteElementFromCollection(ls, id);
    }

    /**
     * 从集合中删除某元素（按getId()匹配）
     *
     * @param <T> 集合元素类型
     * @param ls 集合
     * @param id 元素ID
     */
    public static <T> Collection<T> deleteElementFromCollection(Collection<T> ls, String id) {
        try {
            Iterator<T> it = ls.iterator();
            while (it.hasNext()) {
                T x = it.next();
                Method m = x.getClass().getMethod("getId");
                String xid = m.invoke(x).toString();
                if (id.equals(xid)) {
                    it.remove();
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("本类要求集合元素必须有id属性");
        }
        return ls;
    }

    public static void collectionAsString() {

    }

    public static void main(String args[]) {

    }

}
