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

    /**
     * 检查字符串是否为空白（null、空字符串或只包含空白字符）
     * 类似 org.apache.commons.lang3.StringUtils.isBlank
     *
     * @param str 要检查的字符串
     * @return 如果字符串为空白则返回true，否则返回false
     */
    public static boolean isBlank(String str) {
        if (str == null) {
            return true;
        }
        return str.trim().isEmpty();
    }

    /**
     * 检查字符串是否不为空白
     * 类似 org.apache.commons.lang3.StringUtils.isNotBlank
     *
     * @param str 要检查的字符串
     * @return 如果字符串不为空白则返回true，否则返回false
     */
    public static boolean isNotBlank(String str) {
        return !isBlank(str);
    }

    /**
     * 检查字符串是否为空（null或空字符串）
     * 类似 org.apache.commons.lang3.StringUtils.isEmpty
     *
     * @param str 要检查的字符串
     * @return 如果字符串为空则返回true，否则返回false
     */
    public static boolean isEmpty(String str) {
        return str == null || str.isEmpty();
    }

    /**
     * 检查字符串是否不为空
     * 类似 org.apache.commons.lang3.StringUtils.isNotEmpty
     *
     * @param str 要检查的字符串
     * @return 如果字符串不为空则返回true，否则返回false
     */
    public static boolean isNotEmpty(String str) {
        return !isEmpty(str);
    }

    public static void main(String[] args) {
        String[] arr = {"apple", "banana", "orange"};
        System.out.println(StringUtils.join(arr, ", ")); // 输出: apple, banana, orange

        Integer[] numbers = {1, 2, 3, 4, 5};
        System.out.println(StringUtils.join(numbers, "-")); // 输出: 1-2-3-4-5

        System.out.println(StringUtils.join(new String[]{"a", null, "b"}, ",")); // 输出: a,,b（保留 null）
        
        // 测试新增的字符串检查方法
        System.out.println("isBlank测试:");
        System.out.println("null: " + isBlank(null)); // true
        System.out.println("空字符串: " + isBlank("")); // true
        System.out.println("空白字符串: " + isBlank("   ")); // true
        System.out.println("正常字符串: " + isBlank("hello")); // false
    }

}
