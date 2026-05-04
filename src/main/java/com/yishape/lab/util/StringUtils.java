package com.yishape.lab.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 字符串工具类 / String Utility Class
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class StringUtils {

    private static final Logger log = LoggerFactory.getLogger(StringUtils.class);

    
    /**
     * 拼接数组中的元素为字符串，类似 org.apache.commons.lang3.StringUtils.join
     * Join array elements into a string, similar to org.apache.commons.lang3.StringUtils.join
     *
     * @param array 要拼接的数组 / Array to join
     * @param separator 分隔符（可以是空字符串） / Separator (can be empty string)
     * @return 拼接后的字符串 / Joined string
     */
    public static String join(Object[] array, String separator) {
        if (array == null) {
            return null;
        }
        return join(array, separator, 0, array.length);
    }
    
    /**
     * 拼接数组中指定范围的元素为字符串
     * Join elements in specified range of array into a string
     *
     * @param array 要拼接的数组 / Array to join
     * @param separator 分隔符 / Separator
     * @param startIndex 开始索引（包含） / Start index (inclusive)
     * @param endIndex 结束索引（不包含） / End index (exclusive)
     * @return 拼接后的字符串 / Joined string
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
     * 可变参数支持 / Variable arguments support
     *
     * @param separator 分隔符 / Separator
     * @param array 要拼接的数组 / Array to join
     * @return 拼接后的字符串 / Joined string
     */
    public static String join(String separator, Object... array) {
        return join(array, separator);
    }
    
    /**
     * 忽略null元素的拼接 / Join ignoring null elements
     *
     * @param array 要拼接的数组 / Array to join
     * @param separator 分隔符 / Separator
     * @return 拼接后的字符串 / Joined string
     */
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
     * Check if string is blank (null, empty string or contains only whitespace)
     * 类似 org.apache.commons.lang3.StringUtils.isBlank
     * Similar to org.apache.commons.lang3.StringUtils.isBlank
     *
     * @param str 要检查的字符串 / String to check
     * @return 如果字符串为空白则返回true，否则返回false / Returns true if string is blank, false otherwise
     */
    public static boolean isBlank(String str) {
        if (str == null) {
            return true;
        }
        return str.trim().isEmpty();
    }
    
    /**
     * 检查字符串是否不为空白
     * Check if string is not blank
     * 类似 org.apache.commons.lang3.StringUtils.isNotBlank
     * Similar to org.apache.commons.lang3.StringUtils.isNotBlank
     *
     * @param str 要检查的字符串 / String to check
     * @return 如果字符串不为空白则返回true，否则返回false / Returns true if string is not blank, false otherwise
     */
    public static boolean isNotBlank(String str) {
        return !isBlank(str);
    }
    
    /**
     * 检查字符串是否为空（null或空字符串）
     * Check if string is empty (null or empty string)
     * 类似 org.apache.commons.lang3.StringUtils.isEmpty
     * Similar to org.apache.commons.lang3.StringUtils.isEmpty
     *
     * @param str 要检查的字符串 / String to check
     * @return 如果字符串为空则返回true，否则返回false / Returns true if string is empty, false otherwise
     */
    public static boolean isEmpty(String str) {
        return str == null || str.isEmpty();
    }
    
    /**
     * 检查字符串是否不为空
     * Check if string is not empty
     * 类似 org.apache.commons.lang3.StringUtils.isNotEmpty
     * Similar to org.apache.commons.lang3.StringUtils.isNotEmpty
     *
     * @param str 要检查的字符串 / String to check
     * @return 如果字符串不为空则返回true，否则返回false / Returns true if string is not empty, false otherwise
     */
    public static boolean isNotEmpty(String str) {
        return !isEmpty(str);
    }
    
    /**
     * 主方法，用于测试 / Main method, used for testing
     *
     * @param args 命令行参数 / Command line arguments
     */
    public static void main(String[] args) {
        String[] arr = {"apple", "banana", "orange"};
        log.debug(StringUtils.join(arr, ", ")); // 输出: apple, banana, orange
        
        Integer[] numbers = {1, 2, 3, 4, 5};
        log.debug(StringUtils.join(numbers, "-")); // 输出: 1-2-3-4-5
        
        log.debug(StringUtils.join(new String[]{"a", null, "b"}, ",")); // 输出: a,,b（保留null位置）
        
        // 测试新增的字符串检查方法
        log.debug("isBlank测试:");
        log.debug("null: " + isBlank(null)); // true
        log.debug("空字符串: " + isBlank("")); // true
        log.debug("空白字符串: " + isBlank("   ")); // true
        log.debug("正常字符串: " + isBlank("hello")); // false
    }
}