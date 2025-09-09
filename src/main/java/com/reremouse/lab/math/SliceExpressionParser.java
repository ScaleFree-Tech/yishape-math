package com.reremouse.lab.math;

import java.util.regex.Pattern;

/**
 * 切片表达式解析器
 * Slice Expression Parser
 * <p>
 * 统一处理各种切片表达式，支持负数索引，类似numpy的切片语法
 * Unified handling of slice expressions with negative indexing support, similar to numpy slice syntax
 * </p>
 * 
 * @author RereMouse
 * @version 1.0
 */
public class SliceExpressionParser {
    
    /**
     * 切片表达式正则表达式
     * 支持格式：start:end:step 或 start:end 或 :end 或 start: 或 : 等
     * 支持负数索引
     */
    private static final Pattern SLICE_PATTERN = Pattern.compile("^(-?\\d*):(-?\\d*)(?::(-?\\d+))?$");
    
    /**
     * 切片结果类
     * Slice result class
     */
    public static class SliceResult {
        public final int start;
        public final int end;
        public final int step;
        public final int actualStart;
        public final int actualEnd;
        
        public SliceResult(int start, int end, int step, int actualStart, int actualEnd) {
            this.start = start;
            this.end = end;
            this.step = step;
            this.actualStart = actualStart;
            this.actualEnd = actualEnd;
        }
        
        @Override
        public String toString() {
            return String.format("SliceResult{start=%d, end=%d, step=%d, actualStart=%d, actualEnd=%d}", 
                               start, end, step, actualStart, actualEnd);
        }
    }
    
    /**
     * 解析切片表达式
     * Parse slice expression
     * 
     * @param sliceExpression 切片表达式，如 "1:3", ":-1", "::2", "1:-1:2" 等
     * @param maxSize 最大尺寸（用于处理负数索引）
     * @return 解析结果
     * @throws IllegalArgumentException 如果表达式格式不正确
     */
    public static SliceResult parse(String sliceExpression, int maxSize) {
        if (sliceExpression == null || sliceExpression.trim().isEmpty()) {
            throw new IllegalArgumentException("切片表达式不能为空 / Slice expression cannot be empty");
        }
        
        if (maxSize <= 0) {
            throw new IllegalArgumentException("最大尺寸必须大于0 / Max size must be greater than 0");
        }
        
        String trimmed = sliceExpression.trim();
        var matcher = SLICE_PATTERN.matcher(trimmed);
        
        if (!matcher.matches()) {
            throw new IllegalArgumentException("无效的切片表达式: " + sliceExpression + " / Invalid slice expression: " + sliceExpression);
        }
        
        // 解析start
        String startStr = matcher.group(1);
        int start = startStr.isEmpty() ? 0 : Integer.parseInt(startStr);
        
        // 解析end
        String endStr = matcher.group(2);
        int end = endStr.isEmpty() ? maxSize : Integer.parseInt(endStr);
        
        // 解析step
        String stepStr = matcher.group(3);
        int step = stepStr == null ? 1 : Integer.parseInt(stepStr);
        
        // 验证step
        if (step <= 0) {
            throw new IllegalArgumentException("步长必须大于0 / Step must be greater than 0");
        }
        
        // 处理负数索引
        int actualStart = start;
        int actualEnd = end;
        
        if (start < 0) {
            actualStart = maxSize + start;
        }
        if (end < 0) {
            actualEnd = maxSize + end;
        }
        
        // 边界检查
        if (actualStart < 0) {
            actualStart = 0;
        }
        if (actualEnd > maxSize) {
            actualEnd = maxSize;
        }
        if (actualStart > actualEnd) {
            actualStart = actualEnd;
        }
        
        return new SliceResult(start, end, step, actualStart, actualEnd);
    }
    
    /**
     * 解析简单的整数索引（支持负数）
     * Parse simple integer index (supports negative indexing)
     * 
     * @param indexExpression 索引表达式
     * @param maxSize 最大尺寸
     * @return 实际索引
     * @throws IllegalArgumentException 如果表达式格式不正确
     */
    public static int parseIndex(String indexExpression, int maxSize) {
        if (indexExpression == null || indexExpression.trim().isEmpty()) {
            throw new IllegalArgumentException("索引表达式不能为空 / Index expression cannot be empty");
        }
        
        if (maxSize <= 0) {
            throw new IllegalArgumentException("最大尺寸必须大于0 / Max size must be greater than 0");
        }
        
        try {
            int index = Integer.parseInt(indexExpression.trim());
            int actualIndex = index;
            
            if (index < 0) {
                actualIndex = maxSize + index;
            }
            
            if (actualIndex < 0 || actualIndex >= maxSize) {
                throw new IndexOutOfBoundsException("索引超出范围: " + index + " / Index out of bounds: " + index);
            }
            
            return actualIndex;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("无效的索引表达式: " + indexExpression + " / Invalid index expression: " + indexExpression);
        }
    }
    
    /**
     * 检查是否为切片表达式
     * Check if expression is a slice expression
     * 
     * @param expression 表达式
     * @return 是否为切片表达式
     */
    public static boolean isSliceExpression(String expression) {
        if (expression == null || expression.trim().isEmpty()) {
            return false;
        }
        return SLICE_PATTERN.matcher(expression.trim()).matches();
    }
    
    /**
     * 检查是否为简单索引表达式
     * Check if expression is a simple index expression
     * 
     * @param expression 表达式
     * @return 是否为简单索引表达式
     */
    public static boolean isIndexExpression(String expression) {
        if (expression == null || expression.trim().isEmpty()) {
            return false;
        }
        try {
            Integer.parseInt(expression.trim());
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }
    
    /**
     * 生成索引数组
     * Generate index array based on slice result
     * 
     * @param sliceResult 切片结果
     * @return 索引数组
     */
    public static int[] generateIndices(SliceResult sliceResult) {
        int count = 0;
        for (int i = sliceResult.actualStart; i < sliceResult.actualEnd; i += sliceResult.step) {
            count++;
        }
        
        int[] indices = new int[count];
        int idx = 0;
        for (int i = sliceResult.actualStart; i < sliceResult.actualEnd; i += sliceResult.step) {
            indices[idx++] = i;
        }
        
        return indices;
    }
}
