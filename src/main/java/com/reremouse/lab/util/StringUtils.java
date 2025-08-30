/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.reremouse.lab.util;

/**
 *
 * @author lteb2
 */
public class StringUtils {

    /**
     * 拼接数组中的元素为字符串，类似 org.apache.commons.lang3.StringUtils.join
     *
     * @param array 要拼接的数组
     * @param separator 分隔符（可以是空字符串）
     * @return 拼接后的字符串
     */
    public static String join(Object[] array, String separator) {
        if (array == null) {
            return null;
        }
        return join(array, separator, 0, array.length);
    }

    /**
     * 拼接数组中指定范围的元素为字符串
     *
     * @param array 要拼接的数组
     * @param separator 分隔符
     * @param startIndex 开始索引（包含）
     * @param endIndex 结束索引（不包含）
     * @return 拼接后的字符串
     */
    public static String join(Object[] array, String separator, int startIndex, int endIndex) {
        if (array == null) {
            return null;
        }

        if (separator == null) {
            separator = "";
        }

        StringBuilder sb = new StringBuilder();
        for (int i = startIndex; i < endIndex; i++) {
            if (i > startIndex) {
                sb.append(separator);
            }
            if (array[i] != null) {
                sb.append(array[i]);
            }
        }

        return sb.toString();
    }

    /**
     * 可变参数支持
     */
    public static String join(String separator, Object... array) {
        return join(array, separator);
    }

    public static String joinIgnoreNull(Object[] array, String separator) {
        if (array == null) {
            return null;
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < array.length; i++) {
            if (array[i] == null) {
                continue;
            }
            if (i > 0 && sb.length() > 0) {
                sb.append(separator);
            }
            sb.append(array[i]);
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        String[] arr = {"apple", "banana", "orange"};
        System.out.println(StringUtils.join(arr, ", ")); // 输出: apple, banana, orange

        Integer[] numbers = {1, 2, 3, 4, 5};
        System.out.println(StringUtils.join(numbers, "-")); // 输出: 1-2-3-4-5

        System.out.println(StringUtils.join(new String[]{"a", null, "b"}, ",")); // 输出: a,,b（保留 null）
    }

}
